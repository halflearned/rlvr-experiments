"""GRPO training (clean, single-file, minimal abstractions)."""

import argparse
import asyncio
import os
import time
import traceback
from typing import Any

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("NCCL_TIMEOUT", "90")

import torch
from transformers import AutoTokenizer

from rlvr_experiments.algorithms.grpo import RolloutSample, TrainSample, make_batch, RewardStats
from rlvr_experiments.data import DataIterator, DATASET_LOADERS, load_apps, load_mbpp, load_humaneval, load_gsm8k, load_math, load_deepscaler, load_dummy, load_ifeval, load_if_multi_constraints, load_mixed
from rlvr_experiments.losses import GRPOLoss, DrGRPOLoss, DAPOLoss, compute_grpo_advantages, compute_drgrpo_advantages
from rlvr_experiments.ops import compute_logprobs
from rlvr_experiments.rollout_logger import log_rollout
from rlvr_experiments.runtime import Runtime
from rlvr_experiments.sample_logger import log_sample
from rlvr_experiments.syncing import sync_titan_to_vllm
from rlvr_experiments.tracer import trace_span
from rlvr_experiments.utils import set_seed, get_checkpoint_dir, upload_traces_to_s3
from rlvr_experiments.verifiers import VerifierPool, APPSVerifier, MBPPVerifier, HumanEvalVerifier, MathVerifier, IFEvalVerifier, IFMultiConstraintsVerifier, MultiVerifier

DATASETS = {
    "apps": (load_apps, APPSVerifier),
    "humaneval": (load_humaneval, HumanEvalVerifier),
    "mbpp": (load_mbpp, MBPPVerifier),
    "gsm8k": (load_gsm8k, MathVerifier),
    "math": (load_math, MathVerifier),
    "deepscaler": (load_deepscaler, MathVerifier),
    "dummy": (load_dummy, MathVerifier),
    "ifeval": (load_ifeval, IFEvalVerifier),
    "if_multi_constraints": (load_if_multi_constraints, IFMultiConstraintsVerifier),
    "mixed": (load_mixed, MultiVerifier),
}


def _compute_schedule(training_cfg: dict) -> dict:
    p_rollout = training_cfg["prompts_per_rollout_sync"]
    p_reference = training_cfg["prompts_per_reference_sync"]
    p_optim = training_cfg["prompts_per_optim_step"]
    p_fwd_bwd = training_cfg["prompts_per_forward_backward"]
    accumulation_steps = p_optim // p_fwd_bwd
    sync_model_every = p_rollout // p_optim or 1
    sync_ref_every = p_reference // p_optim or 1
    return {
        "accumulation_steps": accumulation_steps,
        "sync_model_every": sync_model_every,
        "sync_ref_every": sync_ref_every,
        "max_staleness": training_cfg["max_staleness"],
    }


def _get_micro_batch_size(cfg: Any, seq_len: int) -> int:
    if isinstance(cfg, dict):
        for bucket in sorted(cfg.keys()):
            if seq_len <= bucket:
                return cfg[bucket]
        return cfg[max(cfg.keys())]
    return cfg


def _sft_loss(logits, response, padding_mask, prompt_lens, temperature=1.0):
    """SFT cross-entropy loss using the same input_ids approach as GRPO.

    logits: [B, seq_len, V] from model on full input_ids (prompt + completion + padding)
    response: [B, comp_len] completion token ids
    padding_mask: [B, comp_len] float mask (1 for real tokens, 0 for padding)
    prompt_lens: [B] prompt lengths
    """
    # Use compute_logprobs which handles DTensor correctly
    # The key: response must match the expected target_len
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
    sampling = plan.sampling
    roles = plan.roles
    run_id = runtime.run_name or "grpo_run"

    seed = plan.run.get("seed", 42)
    set_seed(seed)
    print(f"[init] Using seed: {seed}")

    await runtime.start()

    trainer = runtime.roles["trainer"]; reference = runtime.roles["reference"]; rollout = runtime.roles["rollout"]
    buffer = runtime.buffer; tracer = runtime.tracer

    tokenizer = AutoTokenizer.from_pretrained(**plan.tokenizer)
    pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    data_cfg = dict(plan.data)
    dataset_name = data_cfg.pop("dataset")
    load_fn, verifier_cls = DATASETS[dataset_name]
    data_iter_cfg = dict(plan.data_iter)
    data_iter_cfg.setdefault("skip_chat_template", True)
    if dataset_name == "mixed":
        ds, order = load_fn(**data_cfg)
        data_iter = DataIterator(ds, tokenizer=tokenizer, order=order, **data_iter_cfg)
    else:
        data_iter = DataIterator(load_fn(**data_cfg), tokenizer=tokenizer, **data_iter_cfg)

    verifier_cfg = plan.verifier
    verifier = VerifierPool(verifier_cls, **verifier_cfg)
    verify_completions = verifier.verify_completions

    sft_iter = None
    if hasattr(plan, "sft_data") and plan.sft_data:
        sft_cfg = dict(plan.sft_data)
        sft_name = sft_cfg.pop("dataset")
        sft_load_fn = DATASET_LOADERS[sft_name]
        sft_ds = sft_load_fn(**sft_cfg)
        sft_iter_cfg = dict(plan.sft_data_iter)
        sft_iter_cfg.setdefault("skip_chat_template", True)
        sft_iter = DataIterator(sft_ds, tokenizer=tokenizer, **sft_iter_cfg)
        print(f"[init] SFT enabled with dataset: {sft_name}")
    else:
        print("[init] SFT disabled (no sft_data in config)")

    loss_cfg = dict(plan.loss)
    loss_name = loss_cfg.pop("name", "drgrpo")
    if loss_name == "grpo":
        loss_fn = GRPOLoss(**loss_cfg); compute_advantages = compute_grpo_advantages
    elif loss_name == "drgrpo":
        loss_fn = DrGRPOLoss(**loss_cfg); compute_advantages = compute_drgrpo_advantages
    elif loss_name == "dapo":
        loss_fn = DAPOLoss(**loss_cfg); compute_advantages = compute_grpo_advantages
    else:
        raise ValueError(f"Unknown loss name: {loss_name}. Must be 'grpo' or 'drgrpo'.")
    print(f"[init] Using loss: {loss_name}")
    reward_stats = RewardStats()

    checkpoint_dir, _ = get_checkpoint_dir()

    schedule = _compute_schedule(training)
    accumulation_steps = schedule["accumulation_steps"]
    sync_model_every = schedule["sync_model_every"]
    sync_ref_every = schedule["sync_ref_every"]
    max_staleness = schedule["max_staleness"]

    max_completion_len = sampling["max_tokens"]
    seq_len_buckets = training["seq_len_buckets"]
    completion_len_buckets = training["completion_len_buckets"] or [max_completion_len]
    max_seq_len = seq_len_buckets[-1]

    # SFT-specific parameters (independent of GRPO)
    sft_training = training.get("sft", {})
    sft_seq_len_buckets = sft_training.get("seq_len_buckets", seq_len_buckets)
    sft_completion_len_buckets = sft_training.get("completion_len_buckets", completion_len_buckets)
    sft_micro_batch_size = sft_training.get("micro_batch_size", training["prompts_per_forward_backward"])
    sft_max_seq_len = sft_seq_len_buckets[-1]

    sampling_params = {**sampling, "logprobs": 0}
    policy_temperature = sampling_params.get("temperature", 1.0)
    rollout_max_model_len = roles["rollout"].config.get("max_model_len")
    rollout_timeout_s = training.get("rollout_timeout_s", 9999)
    max_concurrent_tasks = training.get("max_concurrent_tasks", 64)

    print(f"[config] accumulation_steps={accumulation_steps}, sync_model_every={sync_model_every}, sync_ref_every={sync_ref_every}")

    def mark_filtered(prompt_id: str, trainer_version: int, dataset: str, reason: str) -> None:
        buffer.stats.record_filtered(trainer_version)
        log_sample("filtered", prompt_id=prompt_id, version=trainer_version, reason=reason, dataset=dataset)
        data_iter.mark_done(prompt_id)

    def make_sft_batch(items):
        prompt_tokens = [torch.tensor(tokenizer.encode(i["template"], add_special_tokens=False)) for i in items]
        completion_tokens = [torch.tensor(tokenizer.encode(i["problem"]["completion"], add_special_tokens=False)) for i in items]

        max_prompt = max(len(p) for p in prompt_tokens)
        max_comp = max(len(c) for c in completion_tokens)

        # Snap completion length to nearest SFT bucket
        for bucket in sft_completion_len_buckets:
            if max_comp <= bucket:
                max_comp = bucket
                break
        else:
            max_comp = sft_completion_len_buckets[-1]

        # Truncate completions that exceed max bucket
        completion_tokens = [c[:max_comp] for c in completion_tokens]

        # Snap total seq length to nearest SFT bucket
        max_seq = max_prompt + max_comp
        for bucket in sft_seq_len_buckets:
            if max_seq <= bucket:
                max_seq = bucket
                break
        else:
            max_seq = sft_seq_len_buckets[-1]

        # If prompt + comp exceeds max_seq, truncate prompts (keep completions intact)
        if max_prompt + max_comp > max_seq:
            max_prompt = max_seq - max_comp

        # Pad ALL prompts to the same length so compute_logprobs DTensor path
        # sees a single group (all prompt_lens equal) and avoids shape mismatch.
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

        # completion_ids and mask - pad to bucketed max_comp
        completion_ids = torch.stack([torch.cat([x, x.new_full((max_comp - x.numel(),), pad_token_id)]) for x in completion_tokens])
        padding_mask = (completion_ids != pad_token_id).float()
        return input_ids, completion_ids, padding_mask, prompt_lens

    def next_sft_batch(batch_size):
        if sft_iter is None:
            return None
        items = []
        while len(items) < batch_size:
            item = sft_iter.get_next()
            if item is None:
                break
            items.append(item)
        return items or None

    async def compute_ref_logprobs(rollout_sample: RolloutSample) -> torch.Tensor:
        with trace_span("ref_logprobs"):
            n = rollout_sample.input_ids.size(0)
            mb = training.get("completions_per_micro_batch_reference") or n
            chunks = []
            for i in range(0, n, mb):
                chunk = await reference.compute_logprobs(
                    rollout_sample.input_ids[i:i+mb],
                    rollout_sample.completion_ids[i:i+mb],
                    torch.tensor([rollout_sample.prompt_len] * min(mb, n - i)),
                    temperature=policy_temperature,
                )
                chunks.append(chunk)
            return torch.cat(chunks, dim=0)

    async def produce_epoch():
        async def process_one(item):
            prompt_id = item["problem"].get("prompt_id", "unknown")
            dataset = item["problem"].get("dataset_name", item["problem"].get("verifier_type", "unknown"))

            per_sample_max = item["problem"].get("max_completion_len")
            effective_max_tokens = per_sample_max if isinstance(per_sample_max, int) else max_completion_len
            sp = {**sampling_params, "max_tokens": effective_max_tokens}

            if rollout_max_model_len is not None:
                prompt_tokens = len(tokenizer.encode(item["template"], add_special_tokens=False))
                headroom = rollout_max_model_len - prompt_tokens
                if headroom <= 0:
                    mark_filtered(prompt_id, rollout.trainer_version, dataset, reason="prompt_too_long")
                    return
                if sp.get("max_tokens") and sp["max_tokens"] > headroom:
                    sp = {**sp, "max_tokens": headroom}

            trainer_version = rollout.trainer_version
            verify_problem = dict(item["problem"])
            verify_problem.setdefault("prompt", item["prompt"])
            verify_problem.setdefault("template", item["template"])

            with trace_span("generate"):
                response = await asyncio.wait_for(rollout.generate_single(item["template"], **sp), timeout=rollout_timeout_s)
            completions = [out.text for out in response.outputs]
            rollout_sample = RolloutSample.from_vllm(response, pad_token_id, prompt_id=prompt_id)
            with trace_span("verify"):
                rewards = await verify_completions(verify_problem, completions)

            log_rollout(prompt_id=prompt_id, prompt=item["prompt"], completions=completions, rewards=rewards, trainer_version=trainer_version, trainer_version_after=rollout.trainer_version, dataset=dataset)

            if torch.tensor(rewards, dtype=torch.float32).std() < 1e-6:
                reward_stats.record(rewards, used=False)
                mark_filtered(prompt_id, trainer_version, dataset, reason="zero_variance")
                return
            if rollout_sample.input_ids.shape[1] > max_seq_len:
                reward_stats.record(rewards, used=False)
                mark_filtered(prompt_id, trainer_version, dataset, reason="seq_too_long")
                return

            ref_logprobs = await compute_ref_logprobs(rollout_sample)
            reward_stats.record(rewards, used=True)
            sample = TrainSample(rollout_sample, rewards, ref_logprobs, item_id=prompt_id, trainer_version=trainer_version, dataset=dataset)
            await buffer.put(sample, trainer_version, item_id=prompt_id)
            log_sample("buffered", prompt_id=prompt_id, version=trainer_version, dataset=dataset)

        async def safe_process_one(item):
            prompt_id = item["problem"].get("prompt_id", "unknown")
            try:
                await process_one(item)
            except asyncio.CancelledError:
                raise
            except Exception as e:
                trainer_version = rollout.trainer_version
                tb = traceback.format_exc()
                print(f"[producer] prompt failed {prompt_id}: {e}\n{tb}")
                buffer.stats.record_failed(trainer_version)
                log_sample("failed", prompt_id=prompt_id, version=trainer_version, error=str(e), traceback=tb)
                data_iter.mark_failed(prompt_id)

        async def worker() -> None:
            while True:
                item = await data_iter.get_next_async()
                if item is None:
                    return
                await safe_process_one(item)

        try:
            async with asyncio.TaskGroup() as tg:
                for _ in range(max_concurrent_tasks):
                    tg.create_task(worker())
        finally:
            await buffer.mark_done()

    async def batches(producer_task):
        pending_batch = []
        prompts_per_forward_backward = training["prompts_per_forward_backward"]
 
        def evict_stale(sample_list):
            if not sample_list:
                return sample_list
            min_acceptable_version = trainer.version - max_staleness
            fresh = []
            for s in sample_list:
                if s.trainer_version < min_acceptable_version:
                    buffer.stats.record_wasted(s.trainer_version)
                    log_sample("evicted", prompt_id=s.item_id, trained_at_step=trainer.version, trainer_version=s.trainer_version, reason="stale_after_pop")
                    data_iter.mark_pending(s.item_id)
                else:
                    fresh.append(s)
            return fresh

        def emit(sample_list):
            batch, stats = make_batch(sample_list, pad_token_id, seq_len_buckets=seq_len_buckets, completion_len_buckets=completion_len_buckets)
            item_ids = [s.item_id for s in sample_list]
            group_sizes = [len(s.rewards) for s in sample_list]
            trained_meta = [
                {
                    "item_id": s.item_id,
                    "trainer_version": s.trainer_version,
                    "dataset": s.dataset,
                    "n_tokens": sum(s.rollout.completion_lens),
                }
                for s in sample_list
            ]
            return batch, stats, item_ids, group_sizes, trained_meta

        while True:
            if producer_task.done() and producer_task.exception():
                raise producer_task.exception()
            if buffer.size() == 0 and data_iter.pending_count() == 0 and data_iter.in_flight_count() == len(pending_batch):
                entry = None
            else:
                entry = await buffer.pop()

            if entry is None:
                if pending_batch:
                    pending_batch = evict_stale(pending_batch)
                    if not pending_batch:
                        return
                    yield emit(pending_batch)
                return

            pending_batch.append(entry.item)

            if len(pending_batch) >= prompts_per_forward_backward:
                pending_batch = evict_stale(pending_batch)
                if len(pending_batch) < prompts_per_forward_backward:
                    continue
                yield emit(pending_batch)
                pending_batch = []

    accum_count = 0; accum_loss = 0.0; accum_ntokens = 0; accum_grpo = 0.0; accum_sft = 0.0
    trained_meta_accum = []

    sft_epoch = 0
    for epoch in range(training.get("num_epochs") or 999999):
        if training["max_steps"] and trainer.version >= training["max_steps"]:
            break
        data_iter.new_epoch(seed=seed + epoch)
        sft_epoch = 0
        if sft_iter is not None:
            sft_iter.new_epoch(seed=seed + epoch)
        producer = asyncio.create_task(produce_epoch())

        async for batch, stats, item_ids, group_sizes, trained_meta in batches(producer):
            accum_count += 1
            trained_meta_accum.extend(trained_meta)
            advantages = compute_advantages(batch.rewards, group_sizes=group_sizes)
            with trace_span("forward_backward"):
                if loss_name == "dapo":
                    _loss_args = (batch.completion_ids, batch.logprobs, advantages)
                    _loss_kwargs = {"padding_mask": batch.mask, "prompt_lens": batch.prompt_lens, "temperature": policy_temperature, "ref_logprobs": batch.ref_logprobs}
                else:
                    _loss_args = (batch.completion_ids, batch.ref_logprobs, batch.logprobs, advantages)
                    _loss_kwargs = {"padding_mask": batch.mask, "prompt_lens": batch.prompt_lens, "temperature": policy_temperature}
                loss_grpo, grpo_debug = await trainer.forward_backward(
                    loss_fn,
                    batch.input_ids,
                    loss_args=_loss_args,
                    loss_kwargs=_loss_kwargs,
                    scale_loss=training["alpha_grpo"] / accumulation_steps,
                    micro_batch_size=_get_micro_batch_size(training["completions_per_micro_batch"], stats.padded_seq_len),
                )
            sft_items = next_sft_batch(training["prompts_per_forward_backward"])
            if sft_items:
                print(f"[sft] making batch with {len(sft_items)} items", flush=True)
                sft_input_ids, sft_completion_ids, sft_mask, sft_prompt_lens = make_sft_batch(sft_items)
                print(f"[sft] batch shapes: input={sft_input_ids.shape}, comp={sft_completion_ids.shape}, mask={sft_mask.shape}", flush=True)
                with trace_span("forward_backward_sft"):
                    loss_sft, _ = await trainer.forward_backward(
                        _sft_loss,
                        sft_input_ids,
                        loss_args=(sft_completion_ids,),
                        loss_kwargs={"padding_mask": sft_mask, "prompt_lens": sft_prompt_lens, "temperature": 1.0},
                        scale_loss=training["alpha_sft"] / accumulation_steps,
                        micro_batch_size=sft_micro_batch_size,
                    )
                print(f"[sft] loss={float(loss_sft):.4f}", flush=True)
                for item in sft_items:
                    sft_iter.mark_done(item["problem"]["prompt_id"])
                if sft_iter.all_done():
                    sft_epoch += 1
                    sft_iter.new_epoch(seed=seed + epoch + sft_epoch)
            else:
                loss_sft = 0.0
                print("[sft] no items available", flush=True)

            accum_grpo += float(loss_grpo)
            accum_sft += float(loss_sft)
            accum_loss += training["alpha_grpo"] * loss_grpo + training["alpha_sft"] * loss_sft
            accum_ntokens += batch.input_ids.numel()
            del advantages
            for item_id in item_ids:
                data_iter.mark_done(item_id)
            if accum_count < accumulation_steps:
                continue

            with trace_span("optim_step"):
                grad_norm = await trainer.optim_step()
            avg_loss = accum_loss / accumulation_steps
            avg_grpo = accum_grpo / accumulation_steps
            avg_sft = accum_sft / accumulation_steps

            avg_reward = batch.rewards.mean().item()
            rw_metrics = reward_stats.get_metrics()
            stats.trace(tracer, step=trainer.version)
            tracer.counter("metrics", {"loss": avg_loss, "loss_grpo": avg_grpo, "loss_sft": avg_sft, "grad_norm": grad_norm, "avg_reward": avg_reward})
            tracer.counter("grpo.debug", grpo_debug)
            titan_metrics = await trainer.log_metrics(avg_loss, grad_norm, accum_ntokens)
            if titan_metrics:
                tracer.counter("titan.metrics", titan_metrics)
            tracer.counter("reward_stats", rw_metrics)

            for meta in trained_meta_accum:
                buffer.stats.record_used(meta["trainer_version"])
                log_sample("trained", prompt_id=meta["item_id"], trained_at_step=trainer.version, trainer_version=meta["trainer_version"], dataset=meta["dataset"], n_tokens=meta["n_tokens"])

            print(f"[epoch {epoch}] step={trainer.version} loss={avg_loss:.4f} grpo={avg_grpo:.4f} sft={avg_sft:.4f} grad_norm={grad_norm:.4f} reward={avg_reward:.2f} reward_all={rw_metrics.get('reward_overall', avg_reward):.2f}")

            if trainer.version % sync_ref_every == 0:
                await sync_titan_to_vllm(trainer, reference, abort_in_flight=training["abort_in_flight"], trainer_version=trainer.version, wire_dtype="float16")
            if trainer.version % sync_model_every == 0:
                await sync_titan_to_vllm(trainer, rollout, abort_in_flight=training["abort_in_flight"], trainer_version=trainer.version, wire_dtype="float16")
            if training["checkpoint_interval"] and trainer.version % training["checkpoint_interval"] == 0:
                ckpt_path = os.path.join(checkpoint_dir, f"{run_id}_step{trainer.version}")
                await trainer.export_to_hf(ckpt_path)
            if trainer.version % 10 == 0:
                upload_traces_to_s3(runtime.trace_dir, run_id)

            accum_count = 0; accum_loss = 0.0; accum_ntokens = 0; accum_grpo = 0.0; accum_sft = 0.0
            trained_meta_accum = []
            if training["max_steps"] and trainer.version >= training["max_steps"]:
                break

        producer.cancel()
        await asyncio.gather(producer, return_exceptions=True)
        buffer.reset()

    print("\n=== Training complete ===")
    await rollout.stop(abort=True)

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
