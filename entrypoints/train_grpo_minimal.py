"""Minimal GRPO training loop — for didactic purposes only.

This is a stripped-down version of train_grpo_sft.py that shows the core
algorithm without logging, tracing, SFT, error recovery, or defensive guards.
It would run, but you'd want all that stuff in practice.

The structure:
  1. Setup: load config, create runtime (trainer + vLLM + reference + verifier)
  2. Producer: 64 async workers pull prompts, generate completions, verify, push to buffer
  3. Consumer: pull from buffer, batch, compute advantages, forward/backward, optim step
  4. Weight sync: after each optim step, sync trainer weights to vLLM engines via NCCL
"""

import argparse
import asyncio
import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from transformers import AutoTokenizer

from rlvr_experiments.algorithms.grpo import RolloutSample, TrainSample, make_batch
from rlvr_experiments.data import DataIterator, load_gsm8k
from rlvr_experiments.losses import GRPOLoss, compute_grpo_advantages
from rlvr_experiments.runtime import Runtime
from rlvr_experiments.syncing import sync_titan_to_vllm
from rlvr_experiments.verifiers import VerifierPool, MathVerifier


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str)
    args = parser.parse_args()

    # ── Setup ──────────────────────────────────────────────────────────
    runtime = await Runtime.from_plan(args.config)
    plan = runtime.plan
    training = plan.training
    sampling = plan.sampling

    await runtime.start()

    trainer = runtime.roles["trainer"]
    reference = runtime.roles["reference"]
    rollout = runtime.roles["rollout"]
    buffer = runtime.buffer

    tokenizer = AutoTokenizer.from_pretrained(**plan.tokenizer)
    pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    data_iter = DataIterator(load_gsm8k(**plan.data), tokenizer=tokenizer, **plan.data_iter)
    verify = VerifierPool(MathVerifier, **plan.verifier).verify_completions
    loss_fn = GRPOLoss(**plan.loss)
    sampling_params = {**sampling, "logprobs": 0}
    temperature = sampling_params.get("temperature", 1.0)

    accumulation_steps = training["prompts_per_optim_step"] // training["prompts_per_forward_backward"]
    sync_model_every = training["prompts_per_rollout_sync"] // training["prompts_per_optim_step"] or 1
    sync_ref_every = training["prompts_per_reference_sync"] // training["prompts_per_optim_step"] or 1
    max_staleness = training["max_staleness"]  # how many steps old a sample can be before we toss it
    group_size = sampling_params["n"]  # completions per prompt
    micro_batch_size = training["completions_per_micro_batch"]  # max completions per forward pass

    # ── Ref logprobs helper ────────────────────────────────────────────
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

    # ── Producer ───────────────────────────────────────────────────────
    async def produce_epoch():
        async def worker():
            while True:
                item = await data_iter.get_next_async()
                if item is None:
                    return
                prompt_id = item["problem"]["prompt_id"]
                try:
                    # Generate completions from current policy
                    response = await rollout.generate_single(item["template"], **sampling_params)
                    completions = [out.text for out in response.outputs]
                    rollout_sample = RolloutSample.from_vllm(response, pad_token_id)

                    # Score with verifier
                    rewards = await verify(item["problem"], completions)

                    # Skip if all rewards identical (no signal)
                    if torch.tensor(rewards).std() < 1e-6:
                        data_iter.mark_done(prompt_id)
                        return

                    # Get reference model logprobs (for KL penalty)
                    ref_logprobs = await compute_ref_logprobs(rollout_sample)

                    # Push to buffer for consumer
                    # Tag with trainer_version so consumer can check staleness later
                    sample = TrainSample(rollout_sample, rewards, ref_logprobs,
                                         item_id=prompt_id, trainer_version=rollout.trainer_version)
                    await buffer.put(sample, rollout.trainer_version, item_id=prompt_id)
                except Exception:
                    data_iter.mark_failed(prompt_id)

        async with asyncio.TaskGroup() as tg:
            for _ in range(64):
                tg.create_task(worker())

    # ── Staleness eviction ─────────────────────────────────────────────
    def evict_stale(samples: list[TrainSample]) -> list[TrainSample]:
        """Drop samples whose trainer_version is too far behind current step."""
        if max_staleness < 0:  # -1 means no eviction
            return samples
        keep = []
        min_version = trainer.version - max_staleness
        for s in samples:
            if s.trainer_version < min_version:
                data_iter.mark_pending(s.item_id)  # requeue for fresh generation
            else:
                keep.append(s)
        return keep

    # ── Micro-batch size helper ────────────────────────────────────────
    def get_micro_batch_size(seq_len: int) -> int:
        """Shrink micro-batch for long sequences to avoid OOM."""
        for bucket in sorted(micro_batch_size.keys()):
            if seq_len <= bucket:
                return micro_batch_size[bucket]
        return micro_batch_size[max(micro_batch_size.keys())]

    # ── Consumer (training loop) ───────────────────────────────────────
    for epoch in range(training.get("num_epochs", 1)):
        data_iter.new_epoch(seed=42 + epoch)
        producer = asyncio.create_task(produce_epoch())

        pending = []
        accum_count = 0

        while True:
            entry = await buffer.pop()
            if entry is None:
                break

            pending.append(entry.item)
            if len(pending) < training["prompts_per_forward_backward"]:
                continue

            # ── Evict stale samples ────────────────────────────────────
            pending = evict_stale(pending)
            if len(pending) < training["prompts_per_forward_backward"]:
                continue  # not enough left after eviction, keep collecting

            # ── Make batch ─────────────────────────────────────────────
            batch, stats = make_batch(pending, pad_token_id)
            group_sizes = [group_size] * len(pending)
            item_ids = [s.item_id for s in pending]
            pending = []

            # ── Compute advantages (per-group reward normalization) ────
            advantages = compute_grpo_advantages(batch.rewards, group_sizes=group_sizes)

            # ── Forward + backward (with micro-batching) ───────────────
            loss, _ = await trainer.forward_backward(
                loss_fn,
                batch.input_ids,
                loss_args=(batch.completion_ids, batch.ref_logprobs, batch.logprobs, advantages),
                loss_kwargs={"padding_mask": batch.mask, "prompt_lens": batch.prompt_lens, "temperature": temperature},
                scale_loss=1.0 / accumulation_steps,
                micro_batch_size=get_micro_batch_size(stats.padded_seq_len),
            )

            # Mark items as consumed
            for item_id in item_ids:
                data_iter.mark_done(item_id)

            accum_count += 1
            if accum_count < accumulation_steps:
                continue

            # ── Optimizer step ─────────────────────────────────────────
            grad_norm = await trainer.optim_step()
            accum_count = 0

            print(f"step={trainer.version} loss={float(loss):.4f} grad_norm={grad_norm:.4f} reward={batch.rewards.mean():.2f}")

            # ── Weight sync ────────────────────────────────────────────
            if trainer.version % sync_ref_every == 0:
                await sync_titan_to_vllm(trainer, reference, trainer_version=trainer.version)
            if trainer.version % sync_model_every == 0:
                await sync_titan_to_vllm(trainer, rollout, trainer_version=trainer.version)

            if training.get("max_steps") and trainer.version >= training["max_steps"]:
                break

        producer.cancel()
        await asyncio.gather(producer, return_exceptions=True)
        buffer.reset()

    print("Training complete.")


if __name__ == "__main__":
    asyncio.run(main())
