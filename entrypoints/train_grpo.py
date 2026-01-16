"""GRPO training."""

import asyncio
import argparse
import os
import time
import traceback

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from transformers import AutoTokenizer

from rlvr_experiments.algorithms.grpo import RolloutSample, TrainSample, make_batch, RewardStats
from rlvr_experiments.data import DataIterator, load_apps, load_mbpp, load_humaneval, load_gsm8k, load_math, load_dummy, load_ifeval, load_mixed, DATASET_LOADERS
from rlvr_experiments.losses import GRPOLoss, compute_advantages
from rlvr_experiments.rollout_logger import log_rollout
from rlvr_experiments.runtime import Runtime
from rlvr_experiments.sample_logger import log_sample
from rlvr_experiments.syncing import sync_titan_to_vllm
from rlvr_experiments.tracer import trace_span
from rlvr_experiments.utils import set_seed
from rlvr_experiments.verifiers import VerifierPool, APPSVerifier, MBPPVerifier, HumanEvalVerifier, MathVerifier, IFEvalVerifier, MultiVerifier

DATASETS = {
    "apps": (load_apps, APPSVerifier),
    "humaneval": (load_humaneval, HumanEvalVerifier),
    "mbpp": (load_mbpp, MBPPVerifier),
    "gsm8k": (load_gsm8k, MathVerifier),
    "math": (load_math, MathVerifier),
    "dummy": (load_dummy, MathVerifier),
    "ifeval": (load_ifeval, IFEvalVerifier),
    "mixed": (load_mixed, MultiVerifier),
}


async def main():
    run_start_time = time.perf_counter()

    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str)
    args = parser.parse_args()

    # =========================================================================
    # SETUP
    # =========================================================================
    runtime = await Runtime.from_plan(args.config)
    plan = runtime.plan
    run_name = plan.run.get("name", "grpo_run")

    # Set random seed for reproducibility
    seed = plan.run.get("seed", 42)
    set_seed(seed)
    print(f"[init] Using seed: {seed}")

    await runtime.start()

    trainer = runtime.roles["trainer"]
    reference = runtime.roles["reference"]
    rollout = runtime.roles["rollout"]
    buffer = runtime.buffer
    tracer = runtime.tracer

    tokenizer = AutoTokenizer.from_pretrained(**plan.tokenizer)
    pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    # Dataset + verifier
    data_cfg = dict(plan.data)
    dataset_name = data_cfg.pop("dataset")
    load_fn, verifier_cls = DATASETS[dataset_name]
    verifier = VerifierPool(verifier_cls, **plan.verifier)

    # load_mixed returns (dataset, order), other loaders return just dataset
    if dataset_name == "mixed":
        ds, order = load_fn(**data_cfg)
        data_iter = DataIterator(ds, tokenizer=tokenizer, order=order, **plan.data_iter)
    else:
        data_iter = DataIterator(load_fn(**data_cfg), tokenizer=tokenizer, **plan.data_iter)

    loss_fn = GRPOLoss(**plan.loss)
    reward_stats = RewardStats()

    # =========================================================================
    # CONFIG
    # =========================================================================
    num_epochs = plan.training.get("num_epochs")
    max_steps = plan.training.get("max_steps")
    max_staleness = plan.training.get("max_staleness", 0)
    abort_in_flight = plan.training.get("abort_in_flight", True)
    checkpoint_interval = plan.training.get("checkpoint_interval", 50)
    checkpoint_dir = os.environ.get("SM_MODEL_DIR", "/efs/rlvr-experiments/checkpoints")

    # Batching
    prompts_per_rollout_sync = plan.training.get("prompts_per_rollout_sync")
    prompts_per_reference_sync = plan.training.get("prompts_per_reference_sync")
    prompts_per_optim_step = plan.training.get("prompts_per_optim_step")
    prompts_per_forward_backward = plan.training.get("prompts_per_forward_backward")
    completions_per_micro_batch_cfg = plan.training.get("completions_per_micro_batch")
    completions_per_micro_batch_reference = plan.training.get("completions_per_micro_batch_reference")

    accumulation_steps = prompts_per_optim_step // prompts_per_forward_backward
    sync_model_every = prompts_per_rollout_sync // prompts_per_optim_step
    sync_ref_every = prompts_per_reference_sync // prompts_per_optim_step

    # Sequence length limits
    max_completion_len = plan.sampling.get("max_tokens", 512)
    completions_per_prompt = plan.sampling.get("n", 64)
    seq_len_buckets = plan.training.get("seq_len_buckets") or [768]
    completion_len_buckets = plan.training.get("completion_len_buckets") or [max_completion_len]
    max_seq_len = seq_len_buckets[-1]

    # Sampling params (strip logprobs for generation, add seed if not present)
    sampling_params = {**plan.sampling, "logprobs": 0}
    if "seed" not in sampling_params:
        sampling_params["seed"] = seed
    rollout_max_model_len = plan.roles.get("rollout").config.get("max_model_len")
    rollout_timeout_s = plan.training.get("rollout_timeout_s", 9999)

    print(f"[config] accumulation_steps={accumulation_steps}, "
          f"sync_model_every={sync_model_every}, sync_ref_every={sync_ref_every}")

    # =========================================================================
    # PRODUCER: generate -> verify -> ref_logprobs -> buffer
    # =========================================================================
    async def produce_epoch():
        async def process_one(item):
            """Process one prompt: generate completions, verify, compute ref logprobs, buffer."""
            prompt_id = item["problem"].get("prompt_id", "unknown")
            dataset = item["problem"].get("dataset_name", item["problem"].get("verifier_type", "unknown"))

            # Use per-sample max_completion_len if available and valid, else fall back to global
            per_sample_max = item["problem"].get("max_completion_len")
            effective_max_tokens = per_sample_max if isinstance(per_sample_max, int) else max_completion_len
            sp = {**sampling_params, "max_tokens": effective_max_tokens}

            # --- Preflight check: skip prompts too long for vLLM context ---
            if rollout_max_model_len is not None:
                prompt_tokens = len(tokenizer.encode(item["template"], add_special_tokens=False))
                headroom = rollout_max_model_len - prompt_tokens
                if headroom <= 0:
                    trainer_version = rollout.trainer_version
                    tracer.counter("skipped", {"prompt_too_long": 1})
                    buffer.stats.record_filtered(trainer_version)
                    log_sample("filtered", prompt_id=prompt_id, version=trainer_version, reason="prompt_too_long", dataset=dataset)
                    data_iter.mark_done(prompt_id)
                    return
                if sp.get("max_tokens") and sp["max_tokens"] > headroom:
                    sp = {**sp, "max_tokens": headroom}

            # --- Generate completions ---
            log_sample("generation_start", prompt_id=prompt_id, dataset=dataset)
            with trace_span("generate"):
                response = await asyncio.wait_for(
                    rollout.generate_single(item["template"], **sp),
                    timeout=rollout_timeout_s,
                )
            trainer_version = rollout.trainer_version
            completions = [out.text for out in response.outputs]
            rollout_sample = RolloutSample.from_vllm(response, pad_token_id)
            log_sample("generation_done", prompt_id=prompt_id, version=trainer_version, n_completions=len(completions), dataset=dataset)

            # --- Verify completions ---
            with trace_span("verify"):
                rewards = await verifier.verify_completions(item["problem"], completions)

            # --- Log rollout (even if filtered) ---
            log_rollout(
                prompt_id=prompt_id,
                prompt=item["prompt"],
                completions=completions,
                rewards=rewards,
                trainer_version=trainer_version,
                dataset=dataset,
            )

            # --- Filter: zero variance rewards (all correct or all wrong) ---
            if torch.tensor(rewards, dtype=torch.float32).std() < 1e-6:
                reward_stats.record(rewards, used=False)
                tracer.counter("skipped", {"zero_variance": 1})
                buffer.stats.record_filtered(trainer_version)
                log_sample("filtered", prompt_id=prompt_id, version=trainer_version, reason="zero_variance", dataset=dataset)
                data_iter.mark_done(prompt_id)
                return

            # --- Filter: sequence too long for trainer (defensive) ---
            if rollout_sample.input_ids.shape[1] > max_seq_len:
                reward_stats.record(rewards, used=False)
                tracer.counter("skipped", {"seq_too_long": 1})
                buffer.stats.record_filtered(trainer_version)
                log_sample("filtered", prompt_id=prompt_id, version=trainer_version, reason="seq_too_long", dataset=dataset)
                data_iter.mark_done(prompt_id)
                return

            # --- Compute reference logprobs (only for samples that pass filters) ---
            with trace_span("ref_logprobs"):
                n = rollout_sample.input_ids.size(0)
                mb = completions_per_micro_batch_reference or n
                chunks = []
                for i in range(0, n, mb):
                    chunk = await reference.compute_logprobs(
                        rollout_sample.input_ids[i:i+mb],
                        rollout_sample.completion_ids[i:i+mb],
                        torch.tensor([rollout_sample.prompt_len] * min(mb, n - i)),
                    )
                    chunks.append(chunk)
                ref_logprobs = torch.cat(chunks, dim=0)

            # --- Buffer the sample ---
            reward_stats.record(rewards, used=True)
            sample = TrainSample(
                rollout_sample,
                rewards,
                ref_logprobs,
                item_id=prompt_id,
                trainer_version=trainer_version,
                dataset=dataset,
            )
            await buffer.put(sample, trainer_version, item_id=prompt_id)
            log_sample("buffered", prompt_id=prompt_id, version=trainer_version, dataset=dataset)

        # --- Worker pool ---
        # Use a fixed pool of long-lived workers that repeatedly pulls from the iterator.
        # This avoids subtle busy-spin edge cases in manual task-dispatch loops.
        max_concurrent_tasks = plan.training.get("max_concurrent_tasks", 64)

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
                tracer.counter("failed", {"prompt_failed": 1})
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

    # =========================================================================
    # CONSUMER: drain buffer -> make batches -> yield for training
    # =========================================================================
    # trainer.version increments at each optimizer step. Used for staleness checks.
    # A sample is stale if it was generated with weights from version < (trainer.version - max_staleness).

    # Build micro-batch size lookup: can be int or dict {bucket: size}
    def get_micro_batch_size(completion_len: int) -> int:
        """Get micro-batch size for a given completion length bucket."""
        if isinstance(completions_per_micro_batch_cfg, dict):
            # Find the bucket that matches this completion length
            for bucket in sorted(completions_per_micro_batch_cfg.keys()):
                if completion_len <= bucket:
                    return completions_per_micro_batch_cfg[bucket]
            # Fall back to largest bucket's size
            return completions_per_micro_batch_cfg[max(completions_per_micro_batch_cfg.keys())]
        return completions_per_micro_batch_cfg


    async def batches(producer_task):
        """Yield batches from buffer, handling staleness eviction."""
        samples = []
        while True:
            # Propagate producer errors
            if producer_task.done() and producer_task.exception():
                raise producer_task.exception()

            entry = await buffer.pop()
            if entry is None:
                return  # Producer finished, buffer drained

            # --- Staleness check ---
            # Reject samples generated with weights too far behind current trainer version
            min_acceptable_version = trainer.version - max_staleness
            if entry.version < min_acceptable_version:
                buffer.stats.record_wasted(entry.version)
                tracer.counter("retry", {"stale_evicted": 1, "trainer_version": entry.version, "trained_at_step": trainer.version})
                log_sample("evicted", prompt_id=entry.item_id, trained_at_step=trainer.version,
                           trainer_version=entry.version, reason="stale")
                data_iter.mark_pending(entry.item_id)
                continue

            samples.append(entry.item)

            # --- Yield batch when we have enough samples ---
            if len(samples) >= prompts_per_forward_backward:
                batch, stats = make_batch(
                    samples,
                    pad_token_id,
                    seq_len_buckets=seq_len_buckets,
                    completion_len_buckets=completion_len_buckets,
                )
                # Log lag and dataset for each sample in batch
                lags = []
                dataset_counts = {}
                dataset_tokens = {}
                for s in samples:
                    lag = trainer.version - s.trainer_version
                    lags.append(lag)
                    sample_tokens = sum(s.rollout.completion_lens)
                    buffer.stats.record_used(s.trainer_version)
                    log_sample("trained", prompt_id=s.item_id, trained_at_step=trainer.version,
                               trainer_version=s.trainer_version, lag=lag, dataset=s.dataset,
                               n_tokens=sample_tokens)
                    dataset_counts[s.dataset] = dataset_counts.get(s.dataset, 0) + 1
                    dataset_tokens[s.dataset] = dataset_tokens.get(s.dataset, 0) + sample_tokens
                # Lag breakdown: count samples at each lag value
                lag_counts = {}
                for lag in lags:
                    lag_counts[lag] = lag_counts.get(lag, 0) + 1
                tracer.counter("batch.lag", {
                    "mean_lag": sum(lags) / len(lags) if lags else 0,
                    "max_lag": max(lags) if lags else 0,
                    "trained_at_step": trainer.version,
                    **{f"lag_{k}": v for k, v in lag_counts.items()},
                })
                tracer.counter("batch.datasets", {
                    "trained_at_step": trainer.version,
                    **dataset_counts,
                    **{f"{k}_tokens": v for k, v in dataset_tokens.items()},
                })
                item_ids = [s.item_id for s in samples]
                yield batch, stats, item_ids
                samples = []

    # =========================================================================
    # TRAINING LOOP
    # =========================================================================
    accum_count = 0
    accum_loss = 0.0
    accum_ntokens = 0

    for epoch in range(num_epochs or 999999):
        if max_steps and trainer.version >= max_steps:
            break

        data_iter.new_epoch(seed=seed + epoch)
        tracer.counter("epoch", {"epoch": epoch})
        producer = asyncio.create_task(produce_epoch())

        async for batch, stats, item_ids in batches(producer):
            accum_count += 1

            # Compute GRPO advantages (normalized within each prompt's completions)
            advantages = compute_advantages(batch.rewards, group_size=completions_per_prompt)

            # Forward/backward pass
            with trace_span("forward_backward"):
                loss, grpo_debug = await trainer.forward_backward(
                    loss_fn,
                    batch.input_ids,
                    loss_args=(batch.completion_ids, batch.ref_logprobs, batch.logprobs, advantages),
                    loss_kwargs={"padding_mask": batch.mask, "prompt_lens": batch.prompt_lens},
                    scale_loss=1.0 / accumulation_steps,
                    micro_batch_size=get_micro_batch_size(stats.padded_completion_len),
                )
            accum_loss += loss
            accum_ntokens += batch.input_ids.numel()
            del advantages

            # Mark samples as done after training
            for item_id in item_ids:
                data_iter.mark_done(item_id)

            # Gradient accumulation: wait for full batch before optimizer step
            if accum_count < accumulation_steps:
                continue

            # Optimizer step (auto-increments trainer.version)
            with trace_span("optim_step"):
                grad_norm = await trainer.optim_step()
            avg_loss = accum_loss / accumulation_steps

            # --- Logging ---
            avg_reward = batch.rewards.mean().item()
            rw_metrics = reward_stats.get_metrics()
            stats.trace(tracer)
            tracer.counter("metrics", {"loss": avg_loss, "grad_norm": grad_norm, "avg_reward": avg_reward})
            tracer.counter("grpo.debug", grpo_debug)
            titan_metrics = await trainer.log_metrics(avg_loss, grad_norm, accum_ntokens)
            if titan_metrics:
                tracer.counter("titan.metrics", titan_metrics)
            tracer.counter("reward_stats", rw_metrics)
            tracer.counter("vllm.metrics", await rollout.get_metrics())
            tracer.counter("health", {
                "buffer_size": buffer.size(),
                "pending": data_iter.pending_count(),
                "in_flight": data_iter.in_flight_count(),
                "done": data_iter.done_count(),
                "failed": data_iter.failed_count(),
            })

            # Print summary
            print(f"[epoch {epoch}] step={trainer.version} loss={avg_loss:.4f} grad_norm={grad_norm:.4f} "
                  f"reward={avg_reward:.2f} reward_all={rw_metrics.get('reward_overall', avg_reward):.2f}")

            # Sync weights to reference model
            if trainer.version % sync_ref_every == 0:
                await sync_titan_to_vllm(
                    trainer, reference,
                    abort_in_flight=abort_in_flight,
                    trainer_version=trainer.version,
                )

            # Sync weights to rollout model
            if trainer.version % sync_model_every == 0:
                await sync_titan_to_vllm(
                    trainer, rollout,
                    abort_in_flight=abort_in_flight,
                    trainer_version=trainer.version,
                )

            # Checkpoint
            if checkpoint_interval and trainer.version % checkpoint_interval == 0:
                ckpt_path = os.path.join(checkpoint_dir, f"{run_name}_step{trainer.version}")
                print(f"[step {trainer.version}] Saving checkpoint to {ckpt_path}")
                await trainer.export_to_hf(ckpt_path)

            # Reset accumulation
            accum_count = 0
            accum_loss = 0.0
            accum_ntokens = 0

            if max_steps and trainer.version >= max_steps:
                break

        # End of epoch cleanup
        producer.cancel()
        await asyncio.gather(producer, return_exceptions=True)
        buffer.reset()
        tracer.counter("epoch_complete", {"epoch": epoch, "steps": trainer.version})

    # =========================================================================
    # CLEANUP
    # =========================================================================
    print("\n=== Training complete ===")
    await rollout.stop(abort=True)

    # Save final checkpoint
    final_ckpt_path = os.path.join(checkpoint_dir, f"{run_name}_final")
    print(f"Saving final checkpoint to {final_ckpt_path}")
    await trainer.export_to_hf(final_ckpt_path)

    # Run summary
    run_elapsed = time.perf_counter() - run_start_time
    print(f"\n=== Run Summary ===")
    print(f"Total time: {run_elapsed:.1f}s")

    # Force exit to terminate any lingering background tasks (producer workers, etc.)
    # Without this, the process may hang for hours processing remaining prompts.
    import sys
    sys.exit(0)


if __name__ == "__main__":
    asyncio.run(main())
