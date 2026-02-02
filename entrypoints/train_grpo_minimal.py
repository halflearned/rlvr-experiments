"""Minimal GRPO training loop — for didactic purposes only.

This is a stripped-down version of train_grpo_sft.py that shows the core
algorithm without logging, tracing, SFT, error recovery, or defensive guards.
It runs, but you'd want all that stuff in practice.

The structure:
  1. Setup: load config, create runtime (trainer + vLLM + reference + verifier)
  2. Producer: async workers pull prompts, generate completions, verify, push to buffer
  3. Consumer: pull from buffer, batch, compute advantages, forward/backward, optim step
  4. Weight sync: after each optim step, sync trainer weights to vLLM engines via NCCL
"""

import argparse
import asyncio
import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from transformers import AutoTokenizer

from rlvr_experiments.algorithms.grpo import RolloutSample, TrainSample, make_batch, RewardStats
from rlvr_experiments.data import DataIterator, DATASET_LOADERS
from rlvr_experiments.losses import GRPOLoss, compute_grpo_advantages
from rlvr_experiments.runtime import Runtime
from rlvr_experiments.rollout_logger import log_rollout
from rlvr_experiments.sample_logger import log_sample
from rlvr_experiments.syncing import sync_titan_to_vllm
from rlvr_experiments.tracer import trace_span
from rlvr_experiments.verifiers import VerifierPool, MathVerifier


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str)
    args = parser.parse_args()

    # ── 1. Setup ────────────────────────────────────────────────────────
    runtime = await Runtime.from_plan(args.config)
    plan = runtime.plan
    training = plan.training
    sampling = plan.sampling
    seed = plan.run.get("seed", 42)

    await runtime.start()

    trainer = runtime.roles["trainer"]
    reference = runtime.roles["reference"]
    rollout = runtime.roles["rollout"]
    buffer = runtime.buffer
    tracer = runtime.tracer

    tokenizer = AutoTokenizer.from_pretrained(**plan.tokenizer)
    pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    # Data
    data_cfg = dict(plan.data)
    dataset_name = data_cfg.pop("dataset")
    data_iter = DataIterator(
        DATASET_LOADERS[dataset_name](**data_cfg),
        tokenizer=tokenizer, **plan.data_iter,
    )
    verify = VerifierPool(MathVerifier, **plan.verifier).verify_completions

    # Loss
    loss_cfg = dict(plan.loss)
    loss_cfg.pop("name", None)
    loss_fn = GRPOLoss(**loss_cfg)

    # Schedule
    accumulation_steps = training["prompts_per_optim_step"] // training["prompts_per_forward_backward"]
    sync_model_every = training["prompts_per_rollout_sync"] // training["prompts_per_optim_step"] or 1
    sync_ref_every = training["prompts_per_reference_sync"] // training["prompts_per_optim_step"] or 1
    max_staleness = training["max_staleness"]
    sampling_params = {**sampling, "logprobs": 0}
    temperature = sampling_params.get("temperature", 1.0)

    reward_stats = RewardStats()

    # ── Helpers ─────────────────────────────────────────────────────────
    async def compute_ref_logprobs(sample: RolloutSample) -> torch.Tensor:
        n = sample.input_ids.size(0)
        mb = training.get("completions_per_micro_batch_reference") or n
        chunks = []
        for i in range(0, n, mb):
            chunk = await reference.compute_logprobs(
                sample.input_ids[i:i+mb], sample.completion_ids[i:i+mb],
                torch.tensor([sample.prompt_len] * min(mb, n - i)),
                temperature=temperature,
            )
            chunks.append(chunk)
        return torch.cat(chunks, dim=0)

    def get_micro_batch_size(seq_len: int) -> int:
        cfg = training["completions_per_micro_batch"]
        if isinstance(cfg, dict):
            for bucket in sorted(cfg.keys()):
                if seq_len <= bucket:
                    return cfg[bucket]
            return cfg[max(cfg.keys())]
        return cfg

    # ── 2. Producer ─────────────────────────────────────────────────────
    async def produce():
        async def process_one(item):
            prompt_id = item["problem"]["prompt_id"]
            trainer_version = rollout.trainer_version
            log_sample("in_flight", prompt_id=prompt_id, version=trainer_version)

            with trace_span("generate"):
                response = await rollout.generate_single(item["template"], **sampling_params)
            completions = [out.text for out in response.outputs]
            rollout_sample = RolloutSample.from_vllm(response, pad_token_id, prompt_id=prompt_id)

            with trace_span("verify"):
                rewards = await verify(item["problem"], completions)
            log_rollout(prompt_id=prompt_id, prompt=item["template"], completions=completions,
                        rewards=rewards, trainer_version=trainer_version)
            if torch.tensor(rewards, dtype=torch.float32).std() < 1e-6:
                reward_stats.record(rewards, used=False)
                log_sample("filtered", prompt_id=prompt_id, version=trainer_version, reason="all_same_reward")
                data_iter.mark_done(prompt_id)  # no signal (all same reward) — skip
                return

            reward_stats.record(rewards, used=True)
            with trace_span("ref_logprobs"):
                ref_logprobs = await compute_ref_logprobs(rollout_sample)
            sample = TrainSample(rollout_sample, rewards, ref_logprobs,
                                 item_id=prompt_id, trainer_version=trainer_version)
            await buffer.put(sample, trainer_version, item_id=prompt_id)
            log_sample("buffered", prompt_id=prompt_id, version=trainer_version)

        async def worker():
            while True:
                item = await data_iter.get_next_async(wait_for_in_flight=False)
                if item is None:
                    return
                try:
                    await process_one(item)
                except asyncio.CancelledError:
                    raise
                except Exception as e:
                    print(f"[producer] failed {item['problem'].get('prompt_id')}: {e}")
                    log_sample("failed", prompt_id=item["problem"]["prompt_id"], version=trainer_version, error=str(e))
                    data_iter.mark_failed(item["problem"]["prompt_id"])

        try:
            while True:  # seamless epoch looping
                async with asyncio.TaskGroup() as tg:
                    for _ in range(training.get("max_concurrent_tasks", 64)):
                        tg.create_task(worker())
                data_iter.new_epoch(seed=seed + 1)
        finally:
            await buffer.mark_done()

    # ── 3. Consumer (batches generator) ─────────────────────────────────
    async def batches(producer_task):
        pending = []
        ppfb = training["prompts_per_forward_backward"]

        def evict_stale(samples):
            if not samples:
                return samples
            min_ver = trainer.version - max_staleness
            fresh, stale = [], []
            for s in samples:
                (stale if s.trainer_version < min_ver else fresh).append(s)
            for s in stale:
                log_sample("evicted", prompt_id=s.item_id, trained_at_step=trainer.version, trainer_version=s.trainer_version, reason="stale")
                data_iter.mark_pending(s.item_id)
            return fresh

        while True:
            if producer_task.done() and producer_task.exception():
                raise producer_task.exception()
            entry = await buffer.pop()
            if entry is None:
                if pending:
                    pending = evict_stale(pending)
                    if pending:
                        yield pending
                return
            pending.append(entry.item)
            if len(pending) >= ppfb:
                pending = evict_stale(pending)
                if len(pending) < ppfb:
                    continue
                yield pending
                pending = []

    # ── 4. Training loop ────────────────────────────────────────────────
    accum_count = 0
    accum_loss = 0.0
    accum_ntokens = 0
    data_iter.new_epoch(seed=seed)
    producer = asyncio.create_task(produce())

    async for sample_list in batches(producer):
        batch, stats = make_batch(sample_list, pad_token_id)
        group_sizes = [len(s.rewards) for s in sample_list]

        advantages = compute_grpo_advantages(batch.rewards, group_sizes=group_sizes)
        with trace_span("forward_backward"):
            loss, grpo_debug = await trainer.forward_backward(
                loss_fn, batch.input_ids,
                loss_args=(batch.completion_ids, batch.ref_logprobs, batch.logprobs, advantages),
                loss_kwargs={"padding_mask": batch.mask, "prompt_lens": batch.prompt_lens, "temperature": temperature},
                scale_loss=1.0 / accumulation_steps,
                micro_batch_size=get_micro_batch_size(stats.padded_seq_len),
            )
        del advantages
        for s in sample_list:
            log_sample("trained", prompt_id=s.item_id, trained_at_step=trainer.version, trainer_version=s.trainer_version)
            data_iter.mark_done(s.item_id)

        accum_count += 1
        accum_loss += float(loss)
        accum_ntokens += batch.input_ids.numel()
        if accum_count < accumulation_steps:
            continue

        # ── Optimizer step + metrics + weight sync ────────────────
        with trace_span("optim_step"):
            grad_norm = await trainer.optim_step()
        avg_loss = accum_loss / accumulation_steps
        avg_reward = batch.rewards.mean().item()
        rw = reward_stats.get_metrics()

        # Trace metrics (written to traces/trace.jsonl)
        stats.trace(tracer, step=trainer.version)
        tracer.counter("metrics", {"loss": avg_loss, "grad_norm": grad_norm, "avg_reward": avg_reward})
        tracer.counter("grpo.debug", grpo_debug)
        tracer.counter("reward_stats", rw)
        titan_metrics = await trainer.log_metrics(avg_loss, grad_norm, accum_ntokens)
        if titan_metrics:
            tracer.counter("titan.metrics", titan_metrics)
        rollout_metrics = await rollout.get_metrics()
        tracer.counter("rollout.metrics", rollout_metrics)
        ref_metrics = await reference.get_metrics()
        tracer.counter("reference.metrics", ref_metrics)

        print(f"step={trainer.version} loss={avg_loss:.4f} grad_norm={grad_norm:.4f}"
              f" reward={avg_reward:.2f} reward_all={rw.get('reward_overall', avg_reward):.2f}")

        accum_count = 0
        accum_loss = 0.0
        accum_ntokens = 0

        if trainer.version % sync_ref_every == 0:
            with trace_span("sync_reference"):
                await sync_titan_to_vllm(trainer, reference, trainer_version=trainer.version)
        if trainer.version % sync_model_every == 0:
            with trace_span("sync_rollout"):
                await sync_titan_to_vllm(trainer, rollout, abort_in_flight=training.get("abort_in_flight", False), trainer_version=trainer.version)

        if training.get("max_steps") and trainer.version >= training["max_steps"]:
            break

    producer.cancel()
    await asyncio.gather(producer, return_exceptions=True)
    print("Training complete.")
    await rollout.stop(abort=True)

    import sys
    sys.exit(0)


if __name__ == "__main__":
    asyncio.run(main())
