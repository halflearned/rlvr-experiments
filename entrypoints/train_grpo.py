"""GRPO training with inline rollout logic."""

import asyncio
import argparse

import torch
from transformers import AutoTokenizer

from rlvr_experiments.algorithms.grpo import RolloutSample, TrainSample, make_batch
from rlvr_experiments.data import DataIterator, load_mbpp, load_humaneval, load_gsm8k, load_dummy
from rlvr_experiments.losses import GRPOLoss
from rlvr_experiments.rollout_logger import log_rollout
from rlvr_experiments.runtime import Runtime
from rlvr_experiments.syncing import sync_titan_to_vllm, sync_titan_to_titan
from rlvr_experiments.tracer import trace_span
from rlvr_experiments.verifiers import VerifierPool, MBPPVerifier, HumanEvalVerifier, MathVerifier

DATASETS = {
    "humaneval": (load_humaneval, HumanEvalVerifier),
    "mbpp": (load_mbpp, MBPPVerifier),
    "gsm8k": (load_gsm8k, MathVerifier),
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
    prompts_per_batch = plan.training.get("prompts_per_batch", 4)
    sync_ref_every = plan.training.get("sync_ref_every", 10)
    sync_model_every = plan.training.get("sync_model_every", 5)
    max_staleness = plan.training.get("max_staleness", 0)

    max_completion_len = plan.sampling.get("max_tokens", 512)
    max_seq_len = plan.training.get("max_seq_len") or (max_completion_len + 256)
    sp = {**plan.sampling, "logprobs": 0}

    # --- Producer: generate -> {verify, ref_logprobs} concurrently -> buffer ---
    async def produce_epoch():
        async def process_one(item):
            with trace_span("generate"):
                response = await rollout.generate_single(item["template"], **sp)
            # Capture version AFTER generation - this reflects which weights were used
            version = rollout.model_version

            completions = [out.text for out in response.outputs]
            rollout_sample = RolloutSample.from_vllm(response, pad_token_id)

            # Run verify and ref_logprobs concurrently (with tracing)
            async def traced_verify():
                with trace_span("verify", args={"version": version}):
                    return await verifier.verify_completions(item["problem"], completions, version=version)

            async def traced_ref_logprobs():
                with trace_span("ref_logprobs", args={"version": version}):
                    return await reference.compute_logprobs(
                        rollout_sample.input_ids, rollout_sample.completion_ids,
                        torch.tensor([rollout_sample.prompt_len] * rollout_sample.input_ids.size(0)))

            rewards, ref_logprobs = await asyncio.gather(traced_verify(), traced_ref_logprobs())

            # Filter
            if torch.tensor(rewards, dtype=torch.float32).std() < 1e-6:
                tracer.counter("skipped", {"zero_variance_samples": 1})
                return
            if rollout_sample.input_ids.shape[1] > max_seq_len:
                tracer.counter("skipped", {"seq_too_long": 1})
                return
            if rollout_sample.completion_ids.shape[1] > max_completion_len:
                tracer.counter("skipped", {"completion_too_long": 1})
                return

            sample = TrainSample(rollout_sample, rewards, ref_logprobs)
            await buffer.put(sample, version)
            log_rollout(prompt_id=item["problem"].get("prompt_id", "unknown"), prompt=item["prompt"],
                        completions=completions, rewards=rewards, version=version)

        async with asyncio.TaskGroup() as tg:
            for item in data_iter:
                tg.create_task(process_one(item))

        await buffer.mark_done()

    # --- Batch iterator: drain buffer and yield batches ---
    async def batches(producer_task):
        samples = []
        while True:
            # Check if producer crashed
            if producer_task.done() and producer_task.exception():
                raise producer_task.exception()

            min_version = max(0, rollout.model_version - max_staleness)
            item = await buffer.pop(min_version)
            if item is None:
                return
            samples.append(item)
            if len(samples) >= prompts_per_batch:
                batch, stats = make_batch(samples, pad_token_id, max_seq_len, max_completion_len)
                yield batch, stats
                samples = []

    # --- Logging helper ---
    cumulative_output_tokens = 0

    async def log_step(step, epoch, loss, grad_norm, batch, stats):
        nonlocal cumulative_output_tokens
        stats.trace(tracer)
        titan_metrics = await trainer.log_metrics(loss, grad_norm, ntokens=batch.input_ids.numel())
        if titan_metrics:
            tracer.counter("titan.metrics", titan_metrics)
        avg_reward = batch.rewards.mean().item()
        tracer.counter("metrics", {"loss": loss, "grad_norm": grad_norm, "avg_reward": avg_reward})
        vllm_metrics = await rollout.get_metrics()
        if vllm_metrics["calls"] > 0:
            tracer.counter("vllm.metrics", vllm_metrics)
            cumulative_output_tokens += vllm_metrics.get("output_tokens", 0)
        print(f"[epoch {epoch}] step={step} loss={loss:.4f} grad_norm={grad_norm:.4f} reward={avg_reward:.2f}")

    # --- Main loop ---
    step = 0
    for epoch in range(num_epochs or 999999):
        if max_steps and step >= max_steps:
            break

        data_iter.new_epoch(seed=epoch)
        tracer.counter("epoch", {"epoch": epoch})
        producer = asyncio.create_task(produce_epoch())

        async for batch, stats in batches(producer):
            step += 1

            # Forward/backward
            with trace_span("forward_backward"):
                loss, grpo_debug = await trainer.forward_backward(
                    loss_fn, batch.input_ids,
                    loss_args=(batch.completion_ids, batch.ref_logprobs, batch.logprobs, batch.rewards),
                    loss_kwargs={"padding_mask": batch.mask, "prompt_lens": batch.prompt_lens})

            # Optimizer step
            with trace_span("optim_step"):
                grad_norm = await trainer.optim_step()

            # Sync weights
            if step % sync_ref_every == 0:
                await sync_titan_to_titan(trainer, reference)

            if step % sync_model_every == 0:
                await sync_titan_to_vllm(trainer, rollout, abort_in_flight=True)

            # Log
            await log_step(step, epoch, loss, grad_norm, batch, stats)
            if grpo_debug:
                tracer.counter("grpo.debug", grpo_debug)

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
    print(f"\n=== Run Summary ===")
    print(f"Total wall-clock time: {run_elapsed:.1f}s")
    print(f"Total output tokens: {cumulative_output_tokens}")
    print(f"Overall gen throughput: {cumulative_output_tokens / run_elapsed:.0f} tok/s")
    tracer.counter("run_summary", {
        "total_time_s": run_elapsed,
        "total_output_tokens": cumulative_output_tokens,
        "overall_gen_tps": cumulative_output_tokens / run_elapsed if run_elapsed > 0 else 0,
    })

    # --- Save checkpoint ---
    print("=== Saving checkpoint ===")
    await trainer.export_to_hf("/efs/rlvr-experiments/checkpoints/grpo_final")


if __name__ == "__main__":
    asyncio.run(main())
