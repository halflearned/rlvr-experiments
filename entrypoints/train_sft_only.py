"""SFT-only training (no RL components - just supervised fine-tuning)."""

import argparse
import asyncio
import os
import time

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("NCCL_TIMEOUT", "90")

import torch
from transformers import AutoTokenizer

from rlvr_experiments.data import DataIterator, DATASET_LOADERS
from rlvr_experiments.ops import compute_logprobs
from rlvr_experiments.runtime import Runtime
from rlvr_experiments.tracer import trace_span
from rlvr_experiments.utils import set_seed, get_checkpoint_dir, upload_traces_to_s3


def _sft_loss(logits, response, padding_mask, prompt_lens, temperature=1.0):
    """SFT cross-entropy loss.

    logits: [B, seq_len, V] from model on full input_ids (prompt + completion + padding)
    response: [B, comp_len] completion token ids
    padding_mask: [B, comp_len] float mask (1 for real tokens, 0 for padding)
    prompt_lens: [B] prompt lengths
    """
    logprobs, _ = compute_logprobs(logits, response, prompt_lens=prompt_lens, temperature=temperature)
    mask = padding_mask.to(logprobs.device, dtype=torch.float32)
    return -(logprobs * mask).sum() / mask.sum().clamp_min(1.0)


async def main() -> None:
    run_start_time = time.perf_counter()

    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str)
    args = parser.parse_args()

    runtime = await Runtime.from_plan(args.config)
    plan = runtime.plan
    training = plan.training
    run_id = runtime.run_name or "sft_run"

    seed = plan.run.get("seed", 42)
    set_seed(seed)
    print(f"[init] Using seed: {seed}")

    # Only start the trainer role (no reference, no rollout)
    await runtime.start(wire=False)
    trainer = runtime.roles["trainer"]
    tracer = runtime.tracer

    # Resume from a checkpoint
    resume_step = training.get("resume_step", 0)
    if resume_step:
        trainer.version = resume_step
        print(f"[init] Resuming from step {resume_step}")

    tokenizer = AutoTokenizer.from_pretrained(**plan.tokenizer)
    pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    # Load SFT dataset
    sft_cfg = dict(plan.sft_data)
    sft_name = sft_cfg.pop("dataset")
    sft_load_fn = DATASET_LOADERS[sft_name]
    sft_ds = sft_load_fn(**sft_cfg)
    sft_iter_cfg = dict(plan.sft_data_iter)
    sft_iter_cfg.setdefault("skip_chat_template", True)
    sft_iter = DataIterator(sft_ds, tokenizer=tokenizer, **sft_iter_cfg)
    sft_count = sft_ds.count() if hasattr(sft_ds, 'count') else len(sft_ds)
    print(f"[init] SFT dataset: {sft_name} ({sft_count} examples)")

    checkpoint_dir, _ = get_checkpoint_dir()

    # Training parameters
    max_steps = training["max_steps"]
    checkpoint_interval = training["checkpoint_interval"]
    batch_size = training.get("sft_batch_size", training.get("prompts_per_forward_backward", 2))
    accumulation_steps = training.get("sft_accumulation_steps", training.get("prompts_per_optim_step", 64) // batch_size)
    micro_batch_size = training.get("sft", {}).get("micro_batch_size", batch_size)

    # Sequence length parameters
    seq_len_buckets = training.get("sft", {}).get("seq_len_buckets", training.get("seq_len_buckets", [768]))
    completion_len_buckets = training.get("sft", {}).get("completion_len_buckets", training.get("completion_len_buckets", [512]))

    print(f"[config] batch_size={batch_size}, accumulation_steps={accumulation_steps}, micro_batch_size={micro_batch_size}")
    print(f"[config] seq_len_buckets={seq_len_buckets}, completion_len_buckets={completion_len_buckets}")
    print(f"[config] max_steps={max_steps}, checkpoint_interval={checkpoint_interval}")

    def make_sft_batch(items):
        prompt_tokens = [torch.tensor(tokenizer.encode(i["template"], add_special_tokens=False)) for i in items]
        completion_tokens = [torch.tensor(tokenizer.encode(i["problem"]["completion"], add_special_tokens=False)) for i in items]

        max_prompt = max(len(p) for p in prompt_tokens)
        max_comp = max(len(c) for c in completion_tokens)

        # Snap completion length to nearest bucket
        for bucket in completion_len_buckets:
            if max_comp <= bucket:
                max_comp = bucket
                break
        else:
            max_comp = completion_len_buckets[-1]

        # Truncate completions that exceed max bucket
        completion_tokens = [c[:max_comp] for c in completion_tokens]

        # Snap total seq length to nearest bucket
        max_seq = max_prompt + max_comp
        for bucket in seq_len_buckets:
            if max_seq <= bucket:
                max_seq = bucket
                break
        else:
            max_seq = seq_len_buckets[-1]

        # If prompt + comp exceeds max_seq, truncate prompts (keep completions intact)
        if max_prompt + max_comp > max_seq:
            max_prompt = max_seq - max_comp

        # Pad ALL prompts to the same length
        padded_prompts = []
        for p in prompt_tokens:
            if len(p) > max_prompt:
                p = p[-max_prompt:]
            elif len(p) < max_prompt:
                p = torch.cat([p.new_full((max_prompt - len(p),), pad_token_id), p])
            padded_prompts.append(p)
        prompt_lens = torch.tensor([max_prompt] * len(items))

        # Build input_ids = padded_prompt + completion, pad to max_seq
        input_ids_list = []
        for p, c in zip(padded_prompts, completion_tokens):
            x = torch.cat([p, c])
            input_ids_list.append(torch.cat([x, x.new_full((max_seq - x.numel(),), pad_token_id)]))
        input_ids = torch.stack(input_ids_list)

        # completion_ids and mask
        completion_ids = torch.stack([torch.cat([x, x.new_full((max_comp - x.numel(),), pad_token_id)]) for x in completion_tokens])
        padding_mask = (completion_ids != pad_token_id).float()
        return input_ids, completion_ids, padding_mask, prompt_lens

    def next_batch(batch_size_n):
        items = []
        while len(items) < batch_size_n:
            item = sft_iter.get_next()
            if item is None:
                break
            items.append(item)
        return items or None

    accum_count = 0
    accum_loss = 0.0
    accum_ntokens = 0
    sft_epoch = 0

    for epoch in range(training.get("num_epochs") or 999999):
        if max_steps and trainer.version >= max_steps:
            break
        sft_iter.new_epoch(seed=seed + epoch)
        sft_epoch = 0
        print(f"\n[epoch {epoch}] starting")

        while True:
            items = next_batch(batch_size)
            if items is None:
                # Ran out of data in this epoch
                break

            sft_input_ids, sft_completion_ids, sft_mask, sft_prompt_lens = make_sft_batch(items)

            with trace_span("forward_backward_sft"):
                loss_sft, _ = await trainer.forward_backward(
                    _sft_loss,
                    sft_input_ids,
                    loss_args=(sft_completion_ids,),
                    loss_kwargs={"padding_mask": sft_mask, "prompt_lens": sft_prompt_lens, "temperature": 1.0},
                    scale_loss=1.0 / accumulation_steps,
                    micro_batch_size=micro_batch_size,
                )

            for item in items:
                sft_iter.mark_done(item["problem"]["prompt_id"])

            accum_count += 1
            accum_loss += float(loss_sft)
            accum_ntokens += sft_input_ids.numel()

            if accum_count < accumulation_steps:
                continue

            # Optim step
            with trace_span("optim_step"):
                grad_norm = await trainer.optim_step()
            avg_loss = accum_loss / accumulation_steps

            tracer.counter("metrics", {"loss": avg_loss, "grad_norm": grad_norm})
            titan_metrics = await trainer.log_metrics(avg_loss, grad_norm, accum_ntokens)
            if titan_metrics:
                tracer.counter("titan.metrics", titan_metrics)

            print(f"[epoch {epoch}] optim_step={trainer.version} loss={avg_loss:.4f} grad_norm={grad_norm:.4f}")

            if checkpoint_interval and trainer.version % checkpoint_interval == 0:
                ckpt_path = os.path.join(checkpoint_dir, f"{run_id}_step{trainer.version}")
                await trainer.export_to_hf(ckpt_path)
            if trainer.version % 10 == 0:
                upload_traces_to_s3(runtime.trace_dir, run_id)

            accum_count = 0
            accum_loss = 0.0
            accum_ntokens = 0

            if max_steps and trainer.version >= max_steps:
                break

    print("\n=== Training complete ===")

    final_ckpt_path = os.path.join(checkpoint_dir, f"{run_id}_final")
    print(f"Saving final checkpoint to {final_ckpt_path}")
    await trainer.export_to_hf(final_ckpt_path)

    upload_traces_to_s3(runtime.trace_dir, run_id)

    run_elapsed = time.perf_counter() - run_start_time
    print(f"\n=== Run Summary ===")
    print(f"Total time: {run_elapsed:.1f}s")

    import sys
    sys.exit(0)


if __name__ == "__main__":
    asyncio.run(main())
