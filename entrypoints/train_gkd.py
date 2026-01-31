"""Simplified synchronous GKD (Generalized Knowledge Distillation) training.

No async buffer, no producer-consumer split. Each step:
  1. Take a batch of prompts
  2. Generate 1 completion per prompt via rollout
  3. Batch teacher top-k logprob computation (single call, not sequential)
  4. Forward through trainer, compute JSD loss, backward
  5. Optimizer step, sync weights, repeat
"""

import argparse
import asyncio
import os
import time

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("NCCL_TIMEOUT", "90")

import torch
from transformers import AutoTokenizer

from rlvr_experiments.algorithms.grpo import RolloutSample
from rlvr_experiments.data import DataIterator, load_gsm8k
from rlvr_experiments.ops import compute_logprobs
from rlvr_experiments.runtime import Runtime
from rlvr_experiments.syncing import sync_titan_to_vllm
from rlvr_experiments.tracer import trace_span
from rlvr_experiments.utils import set_seed, upload_traces_to_s3
from rlvr_experiments.verifiers import VerifierPool
from rlvr_experiments.verifiers.math import MathVerifier

# Import OPSD data structures and loss from train_opsd
from train_opsd import (
    OPSDJSDLoss,
    OPSDCompletion,
    OPSDBatch,
    make_opsd_batch,
    build_chatml_teacher_template,
)


def parse_teacher_topk_results(
    results: list[list[dict[int, float]]],
    completion_token_ids_list: list[list[int]],
    top_k: int,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Parse vLLM top-k logprob results into (topk_ids, topk_lps) tensors.

    Args:
        results: List (per sequence) of list (per position) of {token_id: logprob}.
        completion_token_ids_list: List of completion token ID lists (to know lengths).
        top_k: Number of top-k entries expected.

    Returns:
        List of (topk_ids, topk_lps) tuples, each [comp_len, K].
    """
    parsed = []
    for seq_idx, position_dicts in enumerate(results):
        comp_len = len(completion_token_ids_list[seq_idx])
        K = top_k

        topk_ids = torch.zeros(comp_len, K, dtype=torch.long)
        topk_lps = torch.full((comp_len, K), -100.0, dtype=torch.float32)

        for t, lp_dict in enumerate(position_dicts[:comp_len]):
            if not lp_dict:
                continue
            sorted_items = sorted(lp_dict.items(), key=lambda x: x[1], reverse=True)[:K]
            for k_idx, (tid, lp) in enumerate(sorted_items):
                topk_ids[t, k_idx] = tid
                topk_lps[t, k_idx] = lp

        parsed.append((topk_ids, topk_lps))
    return parsed


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
    run_id = runtime.run_name or "gkd_run"

    seed = plan.run.get("seed", 42)
    set_seed(seed)
    print(f"[init] seed={seed}")

    await runtime.start()

    trainer = runtime.roles["trainer"]
    rollout = runtime.roles["rollout"]
    teacher_model = runtime.roles.get("teacher", rollout)

    if "teacher" in runtime.roles:
        print(f"[init] Using separate teacher model (frozen, not synced)")
    else:
        print(f"[init] Using rollout as teacher (self-distillation)")

    resume_step = training.get("resume_step", 0)
    if resume_step:
        trainer.version = resume_step
        rollout.set_trainer_version(resume_step)
        print(f"[init] Resuming from step {resume_step}")

    tokenizer = AutoTokenizer.from_pretrained(**plan.tokenizer)
    pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    data_cfg = dict(plan.data)
    dataset_name = data_cfg.pop("dataset")
    data_iter_cfg = dict(plan.data_iter)
    data_iter_cfg.setdefault("skip_chat_template", True)

    if dataset_name == "gsm8k":
        data_iter = DataIterator(load_gsm8k(**data_cfg), tokenizer=tokenizer, **data_iter_cfg)
        verifier_cls = MathVerifier
    else:
        raise ValueError(f"train_gkd.py currently supports gsm8k only, got {dataset_name}")

    verifier = VerifierPool(verifier_cls, **plan.verifier)
    verify_completions = verifier.verify_completions
    tracer = runtime.tracer

    # --- Config ---
    batch_size = training["batch_size"]           # completions per optimizer step
    minibatch_size = training["minibatch_size"]   # completions per forward-backward
    assert batch_size % minibatch_size == 0
    accumulation_steps = batch_size // minibatch_size

    sync_model_every = training.get("sync_every_n_steps", 1)
    max_completion_len = sampling["max_tokens"]
    seq_len_buckets = training["seq_len_buckets"]
    completion_len_buckets = training["completion_len_buckets"] or [max_completion_len]

    # Force n=1 for simplified GKD
    sampling_params = {**sampling, "logprobs": 0, "n": 1}
    policy_temperature = sampling_params.get("temperature", 1.0)
    teacher_max_model_len = roles["teacher"].config.get("max_model_len") if "teacher" in roles else None
    teacher_use_chat_template = training.get("teacher_use_chat_template", True)
    teacher_top_k = training.get("teacher_top_k", 32)
    max_steps = training.get("max_steps", 300)
    checkpoint_interval = training.get("checkpoint_interval", 20)

    opsd_loss_type = training.get("opsd_loss_type", "jsd")
    jsd_alpha = training.get("jsd_alpha", 0.9)
    loss_fn = OPSDJSDLoss(alpha=jsd_alpha)

    print(f"[config] batch_size={batch_size} minibatch_size={minibatch_size} accum={accumulation_steps}")
    print(f"[config] teacher_top_k={teacher_top_k} jsd_alpha={jsd_alpha}")
    print(f"[config] teacher_use_chat_template={teacher_use_chat_template}")
    print(f"[config] max_steps={max_steps} checkpoint_interval={checkpoint_interval}")
    print(f"[config] sync_every={sync_model_every}")

    # --- Checkpoint dir ---
    run_dir = runtime.run_dir
    checkpoint_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    # ── Main synchronous loop ────────────────────────────────────────────

    for epoch in range(999999):
        if trainer.version >= max_steps:
            break
        data_iter.new_epoch(seed=seed + epoch)

        accum_count = 0
        accum_loss = 0.0
        accum_ntokens = 0
        accum_rewards = []

        while trainer.version < max_steps:
            # 1. Grab a minibatch of prompts
            items = []
            for _ in range(minibatch_size):
                item = await data_iter.get_next_async()
                if item is None:
                    break
                items.append(item)
            if not items:
                break

            step_t0 = time.perf_counter()

            # 2. Generate 1 completion per prompt (all in parallel via asyncio)
            with trace_span("generate", args={"n_prompts": len(items)}):
                gen_tasks = [
                    rollout.generate_single(item["template"], **sampling_params)
                    for item in items
                ]
                responses = await asyncio.gather(*gen_tasks)

            gen_t = time.perf_counter() - step_t0

            # 3. Verify rewards (for logging only — GKD doesn't filter by reward)
            completions_text = [resp.outputs[0].text for resp in responses]
            verify_problems = []
            for item in items:
                vp = dict(item["problem"])
                vp.setdefault("prompt", item["prompt"])
                vp.setdefault("template", item["template"])
                verify_problems.append(vp)

            with trace_span("verify"):
                reward_tasks = [
                    verify_completions(vp, [ct])
                    for vp, ct in zip(verify_problems, completions_text)
                ]
                reward_results = await asyncio.gather(*reward_tasks)
            rewards = [r[0] for r in reward_results]
            verify_t = time.perf_counter() - step_t0 - gen_t

            # 4. Build teacher inputs and get top-k logprobs in ONE batched call
            teacher_seqs = []
            teacher_prompt_lens = []
            completion_token_ids_list = []
            valid_indices = []  # which items have valid teacher input

            for i, (item, resp) in enumerate(zip(items, responses)):
                out = resp.outputs[0]
                comp_token_ids = list(out.token_ids)

                # Build teacher template
                if teacher_use_chat_template:
                    teacher_tmpl = build_chatml_teacher_template(item["prompt"])
                else:
                    teacher_tmpl = item["template"]

                teacher_prompt_ids = tokenizer.encode(teacher_tmpl, add_special_tokens=False)
                teacher_seq = teacher_prompt_ids + comp_token_ids
                teacher_plen = len(teacher_prompt_ids)

                # Check length fits
                if teacher_max_model_len and len(teacher_seq) > teacher_max_model_len:
                    continue

                teacher_seqs.append(teacher_seq)
                teacher_prompt_lens.append(teacher_plen)
                completion_token_ids_list.append(comp_token_ids)
                valid_indices.append(i)

            if not valid_indices:
                # All too long — skip this minibatch
                for item in items:
                    data_iter.mark_done(item["problem"].get("prompt_id", "unknown"))
                continue

            teacher_t0 = time.perf_counter()
            with trace_span("teacher_topk_logprobs_batch", args={"n_seqs": len(teacher_seqs)}):
                teacher_results = await teacher_model.get_logprobs_topk_single(
                    teacher_seqs, teacher_prompt_lens,
                    top_k=teacher_top_k, temperature=policy_temperature,
                )
            teacher_t = time.perf_counter() - teacher_t0

            # Parse results into tensors
            parsed = parse_teacher_topk_results(
                teacher_results, completion_token_ids_list, teacher_top_k
            )

            # 5. Build OPSDCompletion objects
            opsd_completions = []
            for j, orig_idx in enumerate(valid_indices):
                resp = responses[orig_idx]
                out = resp.outputs[0]
                prompt_ids = resp.prompt_token_ids
                prompt_len = len(prompt_ids)
                actual_len = len(out.token_ids)

                # Build input_ids (prompt + completion)
                full_seq = list(prompt_ids) + list(out.token_ids)
                input_ids = torch.tensor(full_seq, dtype=torch.long)
                completion_ids = torch.tensor(list(out.token_ids), dtype=torch.long)

                # Student logprobs from generation
                raw_lps = [out.logprobs[k][out.token_ids[k]].logprob for k in range(actual_len)]
                logprobs_t = torch.tensor(raw_lps, dtype=torch.float32)

                topk_ids, topk_lps = parsed[j]

                opsd_completions.append(OPSDCompletion(
                    input_ids=input_ids,
                    completion_ids=completion_ids,
                    logprobs=logprobs_t,
                    teacher_topk_ids=topk_ids[:actual_len],
                    teacher_topk_lps=topk_lps[:actual_len],
                    prompt_len=prompt_len,
                    completion_len=actual_len,
                    reward=rewards[orig_idx],
                    item_id=items[orig_idx]["problem"].get("prompt_id", "unknown"),
                    trainer_version=trainer.version,
                    dataset=dataset_name,
                ))

            if not opsd_completions:
                for item in items:
                    data_iter.mark_done(item["problem"].get("prompt_id", "unknown"))
                continue

            # 6. Make batch and forward-backward
            batch, stats = make_opsd_batch(
                opsd_completions, pad_token_id,
                seq_len_buckets=seq_len_buckets,
                completion_len_buckets=completion_len_buckets,
            )

            with trace_span("forward_backward"):
                loss, debug = await trainer.forward_backward(
                    loss_fn,
                    batch.input_ids,
                    loss_args=(batch.completion_ids, batch.teacher_topk_ids, batch.teacher_topk_lps),
                    loss_kwargs={"padding_mask": batch.mask, "prompt_lens": batch.prompt_lens, "temperature": policy_temperature},
                    scale_loss=1.0 / accumulation_steps,
                )

            fwd_t = time.perf_counter() - teacher_t0 - teacher_t

            accum_count += 1
            accum_loss += float(loss)
            accum_ntokens += sum(c.completion_len for c in opsd_completions)
            accum_rewards.extend(rewards)

            # Mark items done
            for item in items:
                data_iter.mark_done(item["problem"].get("prompt_id", "unknown"))

            print(f"  [fwd_bwd] accum={accum_count}/{accumulation_steps} "
                  f"n={len(opsd_completions)} seq={stats.padded_seq_len} "
                  f"gen={gen_t:.1f}s teacher={teacher_t:.1f}s fwd={fwd_t:.1f}s "
                  f"reward={sum(rewards)/len(rewards):.2f}", flush=True)

            if accum_count < accumulation_steps:
                continue

            # 7. Optimizer step
            with trace_span("optim_step"):
                grad_norm = await trainer.optim_step()

            avg_loss = accum_loss / accumulation_steps
            avg_reward = sum(accum_rewards) / len(accum_rewards) if accum_rewards else 0.0

            # Log metrics
            dbg = debug or {}
            tracer.counter("metrics", {"loss": avg_loss, "grad_norm": grad_norm, "avg_reward": avg_reward})
            if dbg:
                tracer.counter("opsd.debug", dbg)
            tracer.counter("reward_stats", {
                "reward_used": avg_reward,
                "reward_overall": avg_reward,
                "n_used": len(accum_rewards),
                "n_filtered": 0,
            })
            titan_metrics = await trainer.log_metrics(avg_loss, grad_norm, accum_ntokens)
            if titan_metrics:
                tracer.counter("titan.metrics", titan_metrics)

            step_elapsed = time.perf_counter() - step_t0
            print(f"[epoch {epoch}] step={trainer.version} loss={avg_loss:.4f} grad_norm={grad_norm:.4f} "
                  f"reward={avg_reward:.2f} mean_kl={dbg.get('mean_kl', 0):.4f} "
                  f"student_lp={dbg.get('mean_student_lp', 0):.4f} "
                  f"elapsed={step_elapsed:.1f}s", flush=True)

            # 8. Weight sync
            if trainer.version % sync_model_every == 0:
                with trace_span("sync"):
                    await sync_titan_to_vllm(
                        trainer, rollout,
                        abort_in_flight=training.get("abort_in_flight", False),
                        trainer_version=trainer.version,
                        wire_dtype="float16",
                    )

            # 9. Checkpoint
            if checkpoint_interval and trainer.version % checkpoint_interval == 0:
                ckpt_path = os.path.join(checkpoint_dir, f"{run_id}_step{trainer.version}")
                await trainer.export_to_hf(ckpt_path)
                print(f"  [checkpoint] saved {ckpt_path}", flush=True)

            if trainer.version % 10 == 0:
                upload_traces_to_s3(runtime.trace_dir, run_id)

            # Reset accumulators
            accum_count = 0
            accum_loss = 0.0
            accum_ntokens = 0
            accum_rewards = []

            if trainer.version >= max_steps:
                break

    # --- Final checkpoint ---
    print("\n=== Training complete ===")
    final_ckpt = os.path.join(checkpoint_dir, f"{run_id}_final")
    print(f"Saving final checkpoint to {final_ckpt}")
    await trainer.export_to_hf(final_ckpt)
    upload_traces_to_s3(runtime.trace_dir, run_id)

    run_elapsed = time.perf_counter() - run_start_time
    print(f"Total time: {run_elapsed:.1f}s")
    print(f"Steps completed: {trainer.version}")

    import sys
    sys.exit(0)


if __name__ == "__main__":
    asyncio.run(main())
