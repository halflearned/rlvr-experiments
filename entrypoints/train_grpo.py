"""GRPO training with inline rollout logic."""

import asyncio
import argparse
import os

# Enable expandable segments to reduce CUDA memory fragmentation
# This helps prevent OOM errors that occur after a few training steps
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

import torch
from transformers import AutoTokenizer

from rlvr_experiments.algorithms.grpo import RolloutSample, TrainSample, make_batch, RewardStats
from rlvr_experiments.data import DataIterator, load_mbpp, load_humaneval, load_gsm8k, load_math, load_dummy
from rlvr_experiments.losses import GRPOLoss, compute_advantages
from rlvr_experiments.rollout_logger import log_rollout
from rlvr_experiments.runtime import Runtime
from rlvr_experiments.sample_logger import init_sample_logger, log_sample
from rlvr_experiments.syncing import sync_titan_to_vllm, sync_titan_to_titan
from rlvr_experiments.vllm_engine_actor import VLLMHandle
from rlvr_experiments.tracer import trace_span
from rlvr_experiments.verifiers import VerifierPool, MBPPVerifier, HumanEvalVerifier, MathVerifier

DATASETS = {
    "humaneval": (load_humaneval, HumanEvalVerifier),
    "mbpp": (load_mbpp, MBPPVerifier),
    "gsm8k": (load_gsm8k, MathVerifier),
    "math": (load_math, MathVerifier),
    "dummy": (load_dummy, MathVerifier),
}


async def main():
    import time
    run_start_time = time.perf_counter()

    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str)
    args = parser.parse_args()

    # --- Setup ---
    runtime = await Runtime.from_plan(args.config)
    plan = runtime.plan
    run_name = plan.run.get("name", "grpo_run")
    await runtime.start()

    trainer = runtime.roles["trainer"]
    reference = runtime.roles["reference"]
    rollout = runtime.roles["rollout"]
    buffer = runtime.buffer
    tracer = runtime.tracer

    tokenizer = AutoTokenizer.from_pretrained(**plan.tokenizer)
    pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    data_cfg = dict(plan.data)
    dataset_name = data_cfg.pop("dataset")
    load_fn, verifier_cls = DATASETS[dataset_name]

    verifier = VerifierPool(verifier_cls, **plan.verifier)
    data_iter = DataIterator(load_fn(**data_cfg), tokenizer=tokenizer, **plan.data_iter)
    loss_fn = GRPOLoss(**plan.loss)

    # Training config
    num_epochs = plan.training.get("num_epochs")
    max_steps = plan.training.get("max_steps")
    max_staleness = plan.training.get("max_staleness", 0)
    abort_in_flight = plan.training.get("abort_in_flight", True)
    checkpoint_interval = plan.training.get("checkpoint_interval", 50)  # Save HF checkpoint every N steps
    checkpoint_dir = os.environ.get("SM_MODEL_DIR", "/efs/rlvr-experiments/checkpoints")

    # Batching config - new style (prompts_per_*) with fallback to old style
    prompts_per_rollout_sync = plan.training.get("prompts_per_rollout_sync")
    prompts_per_reference_sync = plan.training.get("prompts_per_reference_sync")
    prompts_per_optim_step = plan.training.get("prompts_per_optim_step")
    prompts_per_forward_backward = plan.training.get("prompts_per_forward_backward")
    completions_per_micro_batch = plan.training.get("completions_per_micro_batch")  # For micro-batching within forward_backward
    completions_per_micro_batch_reference = plan.training.get("completions_per_micro_batch_reference")  # For reference logprobs

    assert prompts_per_optim_step % prompts_per_forward_backward == 0, \
        f"prompts_per_optim_step ({prompts_per_optim_step}) must be divisible by prompts_per_forward_backward ({prompts_per_forward_backward})"
    assert prompts_per_rollout_sync % prompts_per_optim_step == 0, \
        f"prompts_per_rollout_sync ({prompts_per_rollout_sync}) must be divisible by prompts_per_optim_step ({prompts_per_optim_step})"
    assert prompts_per_reference_sync % prompts_per_optim_step == 0, \
        f"prompts_per_reference_sync ({prompts_per_reference_sync}) must be divisible by prompts_per_optim_step ({prompts_per_optim_step})"

    accumulation_steps = prompts_per_optim_step // prompts_per_forward_backward
    sync_model_every = prompts_per_rollout_sync // prompts_per_optim_step
    sync_ref_every = prompts_per_reference_sync // prompts_per_optim_step

    print(f"[config] prompts_per_forward_backward={prompts_per_forward_backward}, "
          f"accumulation_steps={accumulation_steps}, sync_model_every={sync_model_every}, "
          f"sync_ref_every={sync_ref_every}")

    max_completion_len = plan.sampling.get("max_tokens", 512)
    completions_per_prompt = plan.sampling.get("n", 64)  # For grouped advantage normalization
    # Buckets for torch.compile cache efficiency. Use [max] for fixed size (no recompiles).
    # The last bucket is the max allowed length - samples exceeding it are filtered.
    seq_len_buckets = plan.training.get("seq_len_buckets") or [768]
    completion_len_buckets = plan.training.get("completion_len_buckets") or [max_completion_len]
    max_seq_len = seq_len_buckets[-1]
    sp = {**plan.sampling, "logprobs": 0}

    # Reward stats accumulator (tracks ALL samples including filtered)
    reward_stats = RewardStats()

    # --- Producer: generate -> {verify, ref_logprobs} concurrently -> buffer ---
    async def produce_epoch():
        async def process_one(item) -> str:
            """Process one item. Returns prompt_id for tracking."""
            prompt_id = item["problem"].get("prompt_id", "unknown")
            try:
                t0 = time.perf_counter()
                log_sample("generation_start", prompt_id=prompt_id)
                with trace_span("generate"):
                    response = await rollout.generate_single(item["template"], **sp)
                t_gen = time.perf_counter()
                # Capture generation_step AFTER generation - this reflects which trainer step's weights were used
                generated_at_step = rollout.generation_step
                n_completions = len(response.outputs)
                log_sample("generation_done", prompt_id=prompt_id, step=generated_at_step,
                           n_completions=n_completions, duration=t_gen - t0)

                completions = [out.text for out in response.outputs]
                rollout_sample = RolloutSample.from_vllm(response, pad_token_id)

                # Run verify and ref_logprobs concurrently (with tracing)
                async def traced_verify():
                    with trace_span("verify", args={"generated_at_step": generated_at_step}):
                        return await verifier.verify_completions(item["problem"], completions, version=generated_at_step)

                async def traced_ref_logprobs():
                    t_start = time.perf_counter()
                    with trace_span("ref_logprobs", args={"generated_at_step": generated_at_step}):
                        n = rollout_sample.input_ids.size(0)
                        mb = completions_per_micro_batch_reference or n
                        chunks = [await reference.compute_logprobs(
                            rollout_sample.input_ids[i:i+mb],
                            rollout_sample.completion_ids[i:i+mb],
                            torch.tensor([rollout_sample.prompt_len] * min(mb, n-i))
                        ) for i in range(0, n, mb)]
                        t_done = time.perf_counter()
                        print(f"[TIMING ref_logprobs] compute={t_done-t_start:.2f}s", flush=True)
                        return torch.cat(chunks, dim=0)

                rewards, ref_logprobs = await asyncio.gather(traced_verify(), traced_ref_logprobs())
                t_verify_ref = time.perf_counter()
                print(f"[TIMING process_one] gen={t_gen-t0:.2f}s verify+ref={t_verify_ref-t_gen:.2f}s total={t_verify_ref-t0:.2f}s", flush=True)

                # Log all rollouts (even filtered ones) for debugging
                log_rollout(prompt_id=prompt_id, prompt=item["prompt"],
                            completions=completions, rewards=rewards, generated_at_step=generated_at_step)

                # Filter - mark consumed even if filtered (don't retry)
                if torch.tensor(rewards, dtype=torch.float32).std() < 1e-6:
                    reward_stats.record(rewards, used=False)
                    tracer.counter("skipped", {"zero_variance_samples": 1})
                    buffer.stats.record_filtered(generated_at_step)
                    data_iter.mark_consumed(prompt_id)
                    log_sample("filtered", prompt_id=prompt_id, step=generated_at_step, reason="zero_variance")
                    return prompt_id
                if rollout_sample.input_ids.shape[1] > max_seq_len:
                    reward_stats.record(rewards, used=False)
                    tracer.counter("skipped", {"seq_too_long": 1})
                    buffer.stats.record_filtered(generated_at_step)
                    data_iter.mark_consumed(prompt_id)
                    log_sample("filtered", prompt_id=prompt_id, step=generated_at_step, reason="seq_too_long")
                    return prompt_id
                if rollout_sample.completion_ids.shape[1] > max_completion_len:
                    reward_stats.record(rewards, used=False)
                    tracer.counter("skipped", {"completion_too_long": 1})
                    buffer.stats.record_filtered(generated_at_step)
                    data_iter.mark_consumed(prompt_id)
                    log_sample("filtered", prompt_id=prompt_id, step=generated_at_step, reason="completion_too_long")
                    return prompt_id

                # Record stats for used sample
                reward_stats.record(rewards, used=True)
                sample = TrainSample(rollout_sample, rewards, ref_logprobs, item_id=prompt_id, generated_at_step=generated_at_step)
                await buffer.put(sample, generated_at_step, item_id=prompt_id)
                log_sample("buffered", prompt_id=prompt_id, step=generated_at_step)
                return prompt_id
            except Exception as e:
                # On error, mark pending for retry
                data_iter.mark_pending(prompt_id)
                raise

        # Limit concurrent tasks to avoid queueing thousands waiting for slots
        # 4 rollout replicas × 8 concurrent per replica = 32 active generations
        # Add 2x headroom for pipelining verify+ref after generation completes
        max_concurrent_tasks = 64
        tasks: set[asyncio.Task] = set()

        def spawn_tasks_up_to_limit():
            """Spawn tasks until we hit the limit or run out of pending items."""
            while len(tasks) < max_concurrent_tasks:
                item = data_iter.get_next()
                if item is None:
                    break
                tasks.add(asyncio.create_task(process_one(item)))

        # Initial spawn
        spawn_tasks_up_to_limit()

        # Process tasks, spawning new ones as slots free up
        while not data_iter.all_consumed():
            if not tasks:
                # All tasks done but not all consumed - items in buffer waiting
                await asyncio.sleep(0.01)
                spawn_tasks_up_to_limit()
                continue

            done, tasks = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                if task.exception():
                    print(f"[producer] task failed: {task.exception()}")

            # Spawn more tasks to replace completed ones
            spawn_tasks_up_to_limit()

        await buffer.mark_done()

    # --- Batch iterator: drain buffer and yield batches ---
    # Track current step for staleness and lag calculation
    current_step = 0

    def set_current_step(s):
        nonlocal current_step
        current_step = s

    async def batches(producer_task):
        samples = []
        while True:
            # Check if producer crashed
            if producer_task.done() and producer_task.exception():
                raise producer_task.exception()

            result = await buffer.pop()
            if result is None:
                return
            sample, item_id, generated_at_step = result

            # Consumer handles staleness check based on trainer steps
            min_step = current_step - max_staleness
            if generated_at_step < min_step:
                # Too stale - mark for retry
                data_iter.mark_pending(item_id)
                buffer.stats.record_wasted(generated_at_step)
                tracer.counter("retry", {"stale_evicted": 1, "generated_at_step": generated_at_step, "current_step": current_step})
                log_sample("evicted", prompt_id=item_id, step=current_step,
                           generated_at_step=generated_at_step, reason="stale")
                continue

            samples.append(sample)

            if len(samples) >= prompts_per_forward_backward:
                batch, stats = make_batch(
                    samples, pad_token_id,
                    seq_len_buckets=seq_len_buckets,
                    completion_len_buckets=completion_len_buckets,
                )
                # Mark all samples in this batch as consumed and log lag
                lags = []
                for s in samples:
                    data_iter.mark_consumed(s.item_id)
                    buffer.stats.record_used(s.generated_at_step)
                    lag = current_step - s.generated_at_step
                    lags.append(lag)
                    log_sample("trained", prompt_id=s.item_id, step=current_step,
                               generated_at_step=s.generated_at_step, lag=lag)
                # Log lag distribution for this batch
                tracer.counter("batch.lag", {
                    "lags": lags,
                    "mean_lag": sum(lags) / len(lags) if lags else 0,
                    "max_lag": max(lags) if lags else 0,
                    "consumed_at_step": current_step,
                })
                yield batch, stats
                samples = []

    # --- Logging helper ---
    cumulative_output_tokens = 0

    async def log_step(step, epoch, loss, grad_norm, batch, stats, ntokens, grpo_debug=None):
        nonlocal cumulative_output_tokens
        stats.trace(tracer)
        titan_metrics = await trainer.log_metrics(loss, grad_norm, ntokens=ntokens)
        if titan_metrics:
            tracer.counter("titan.metrics", titan_metrics)
        avg_reward = batch.rewards.mean().item()
        tracer.counter("metrics", {"loss": loss, "grad_norm": grad_norm, "avg_reward": avg_reward})

        # Log comprehensive reward stats (includes filtered samples)
        rw_metrics = reward_stats.get_metrics()
        if rw_metrics:
            tracer.counter("reward_stats", rw_metrics)

        vllm_metrics = await rollout.get_metrics()
        if vllm_metrics["calls"] > 0:
            tracer.counter("vllm.metrics", vllm_metrics)
            cumulative_output_tokens += vllm_metrics.get("output_tokens", 0)
        if grpo_debug:
            tracer.counter("grpo.debug", grpo_debug)

        # Enhanced print with overall stats
        rw_overall = rw_metrics.get("reward_overall", avg_reward)
        frac_correct = rw_metrics.get("frac_all_correct", 0)
        frac_wrong = rw_metrics.get("frac_all_wrong", 0)
        print(f"[epoch {epoch}] step={step} loss={loss:.4f} grad_norm={grad_norm:.4f} "
              f"reward={avg_reward:.2f} reward_all={rw_overall:.2f} "
              f"all_correct={frac_correct:.1%} all_wrong={frac_wrong:.1%}")

    # --- Main loop ---
    step = 0
    accum_count = 0
    accum_loss = 0.0
    accum_ntokens = 0  # Track tokens across accumulation for correct MFU
    last_batch = None
    last_stats = None

    for epoch in range(num_epochs or 999999):
        if max_steps and step >= max_steps:
            break

        data_iter.new_epoch(seed=epoch)
        tracer.counter("epoch", {"epoch": epoch})
        producer = asyncio.create_task(produce_epoch())

        async for batch, stats in batches(producer):
            accum_count += 1
            accum_ntokens += batch.input_ids.numel()  # Accumulate tokens for MFU
            last_batch = batch
            last_stats = stats

            # Compute advantages before forward/backward
            # This must be done on the full batch before any DDP sharding
            # Use group_size to normalize within each prompt's completions
            advantages = compute_advantages(batch.rewards, group_size=completions_per_prompt)

            # Forward/backward (gradients accumulate)
            t0 = time.perf_counter()
            with trace_span("forward_backward"):
                loss, grpo_debug = await trainer.forward_backward(
                    loss_fn, batch.input_ids,
                    loss_args=(batch.completion_ids, batch.ref_logprobs, batch.logprobs, advantages),
                    loss_kwargs={"padding_mask": batch.mask, "prompt_lens": batch.prompt_lens},
                    scale_loss=1.0 / accumulation_steps,
                    micro_batch_size=completions_per_micro_batch)
            t1 = time.perf_counter()
            accum_loss += loss
            del advantages  # Free memory immediately

            # Only step optimizer after accumulation_steps
            if accum_count < accumulation_steps:
                print(f"[accum {accum_count}/{accumulation_steps}] loss={loss:.4f} fwd_bwd={1000*(t1-t0):.0f}ms")
                continue

            step += 1
            set_current_step(step)  # Update for staleness checks and lag calculation

            # Optimizer step
            t2 = time.perf_counter()
            with trace_span("optim_step"):
                grad_norm = await trainer.optim_step()
            t3 = time.perf_counter()

            avg_loss = accum_loss / accumulation_steps
            print(f"[TIMING] forward_backward={1000*(t1-t0):.0f}ms, gap={1000*(t2-t1):.0f}ms, optim_step={1000*(t3-t2):.0f}ms")

            # Sync weights to reference model
            if step % sync_ref_every == 0:
                print(f"[step {step}] Starting sync_titan_to_vllm (trainer -> reference)")
                await sync_titan_to_vllm(trainer, reference, abort_in_flight=abort_in_flight, step=step)
                print(f"[step {step}] Finished sync_titan_to_vllm (reference)")
                
            if step % sync_model_every == 0:
                print(f"[step {step}] Starting sync_titan_to_vllm (trainer -> rollout, abort={abort_in_flight})")
                await sync_titan_to_vllm(trainer, rollout, abort_in_flight=abort_in_flight, step=step)
                print(f"[step {step}] Finished sync_titan_to_vllm")

            # Log
            await log_step(step, epoch, avg_loss, grad_norm, last_batch, last_stats, accum_ntokens, grpo_debug)

            # Save checkpoint every N steps
            if checkpoint_interval and step % checkpoint_interval == 0:
                ckpt_path = os.path.join(checkpoint_dir, f"{run_name}_step{step}")
                print(f"[step {step}] Saving checkpoint to {ckpt_path}")
                await trainer.export_to_hf(ckpt_path)
                print(f"[step {step}] Checkpoint saved")

            # Reset accumulation
            accum_count = 0
            accum_ntokens = 0
            accum_loss = 0.0

            if max_steps and step >= max_steps:
                break

        producer.cancel()
        await asyncio.gather(producer, return_exceptions=True)

        buffer.reset()  # Clear buffer for next epoch
        tracer.counter("epoch_complete", {"epoch": epoch, "steps_in_epoch": step})

    # --- Cleanup: abort any remaining in-flight requests ---
    print("\n=== Training complete, cleaning up ===")
    await rollout.stop(abort=True)

    # --- Final summary ---
    run_elapsed = time.perf_counter() - run_start_time
    # Get any remaining metrics not yet collected
    final_vllm_metrics = await rollout.get_metrics()
    cumulative_output_tokens += final_vllm_metrics.get("output_tokens", 0)

    # Sample fate statistics
    fates = buffer.stats.get_fates()
    total_used = sum(fates["used"].values())
    total_wasted = sum(fates["wasted"].values())
    total_filtered = sum(fates["filtered"].values())
    total_samples = total_used + total_wasted + total_filtered

    print(f"\n=== Run Summary ===")
    print(f"Total wall-clock time: {run_elapsed:.1f}s")
    print(f"Total output tokens: {cumulative_output_tokens}")
    print(f"Overall gen throughput: {cumulative_output_tokens / run_elapsed:.0f} tok/s")
    print(f"\n=== Sample Fate Statistics ===")
    print(f"Total samples: {total_samples}")
    print(f"  Used for training: {total_used} ({100*total_used/max(total_samples,1):.1f}%)")
    print(f"  Wasted (stale): {total_wasted} ({100*total_wasted/max(total_samples,1):.1f}%)")
    print(f"  Filtered: {total_filtered} ({100*total_filtered/max(total_samples,1):.1f}%)")
    if not abort_in_flight:
        print(f"  (abort_in_flight=False - staleness allowed)")

    tracer.counter("run_summary", {
        "total_time_s": run_elapsed,
        "total_output_tokens": cumulative_output_tokens,
        "overall_gen_tps": cumulative_output_tokens / run_elapsed if run_elapsed > 0 else 0,
        "samples_used": total_used,
        "samples_wasted": total_wasted,
        "samples_filtered": total_filtered,
        "abort_in_flight": abort_in_flight,
    })

    # --- Save final checkpoint ---
    final_ckpt_path = os.path.join(checkpoint_dir, f"{run_name}_final")
    print(f"=== Saving final checkpoint to {final_ckpt_path} ===")
    await trainer.export_to_hf(final_ckpt_path)


if __name__ == "__main__":
    asyncio.run(main())
