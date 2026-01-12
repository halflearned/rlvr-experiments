#!/usr/bin/env python3
"""
Test if using smaller micro-batches with gradient accumulation is faster
than one large batch.

Goal: Compare total time for processing 48 samples:
1. One batch of 48 samples
2. Three batches of 16 samples with gradient accumulation
"""

import asyncio
import os
import sys
import time
import copy

import torch
import ray

sys.path.insert(0, "/efs/rlvr-experiments")

from rlvr_experiments.titan_actor import create_titan_group
from rlvr_experiments.losses import GRPOLoss


async def run_test():
    ray.init(ignore_reinit_error=True)

    config_path = "/efs/rlvr-experiments/configs/qwen3-06B-base.yaml"

    import yaml
    with open(config_path) as f:
        plan = yaml.safe_load(f)

    trainer_cfg = None
    for role in plan["roles"]:
        if role["name"] == "trainer":
            trainer_cfg = copy.deepcopy(role["config"])
            break

    trainer_cfg["trainable"] = True
    trainer_cfg["compile"] = {"enable": True, "components": ["model"]}

    print("Creating trainer...")
    trainer = create_titan_group(trainer_cfg, "trainer_microbatch", 1, 29505)
    print("Trainer created")

    loss_fn = GRPOLoss(beta=0.01, eps=0.2)

    seq_len = 768
    completion_len = 512

    # Warmup with batch=16
    print("\nWarming up...")
    for _ in range(3):
        input_ids = torch.randint(0, 150000, (16, seq_len))
        completion_ids = torch.randint(0, 150000, (16, completion_len))
        ref_logprobs = torch.randn(16, completion_len)
        rollout_logprobs = torch.randn(16, completion_len)
        rewards = torch.randn(16)
        mask = torch.ones(16, completion_len)
        prompt_lens = torch.full((16,), seq_len - completion_len, dtype=torch.long)

        await trainer.forward_backward(
            loss_fn,
            input_ids,
            loss_args=(completion_ids, ref_logprobs, rollout_logprobs, rewards),
            loss_kwargs={"padding_mask": mask, "prompt_lens": prompt_lens},
        )

    # Also warmup batch=48
    for _ in range(3):
        input_ids = torch.randint(0, 150000, (48, seq_len))
        completion_ids = torch.randint(0, 150000, (48, completion_len))
        ref_logprobs = torch.randn(48, completion_len)
        rollout_logprobs = torch.randn(48, completion_len)
        rewards = torch.randn(48)
        mask = torch.ones(48, completion_len)
        prompt_lens = torch.full((48,), seq_len - completion_len, dtype=torch.long)

        await trainer.forward_backward(
            loss_fn,
            input_ids,
            loss_args=(completion_ids, ref_logprobs, rollout_logprobs, rewards),
            loss_kwargs={"padding_mask": mask, "prompt_lens": prompt_lens},
        )

    print("\n" + "="*60)
    print("TEST 1: Single batch of 48 samples")
    print("="*60)

    times_single = []
    for trial in range(5):
        input_ids = torch.randint(0, 150000, (48, seq_len))
        completion_ids = torch.randint(0, 150000, (48, completion_len))
        ref_logprobs = torch.randn(48, completion_len)
        rollout_logprobs = torch.randn(48, completion_len)
        rewards = torch.randn(48)
        mask = torch.ones(48, completion_len)
        prompt_lens = torch.full((48,), seq_len - completion_len, dtype=torch.long)

        t0 = time.perf_counter()
        await trainer.forward_backward(
            loss_fn,
            input_ids,
            loss_args=(completion_ids, ref_logprobs, rollout_logprobs, rewards),
            loss_kwargs={"padding_mask": mask, "prompt_lens": prompt_lens},
        )
        await trainer.optim_step()
        t1 = time.perf_counter()
        times_single.append(t1 - t0)
        print(f"  Trial {trial+1}: {1000*(t1-t0):.0f}ms")

    avg_single = sum(times_single) / len(times_single)
    print(f"Average: {1000*avg_single:.0f}ms")

    print("\n" + "="*60)
    print("TEST 2: Three batches of 16 samples (simulated gradient accumulation)")
    print("="*60)
    print("(Note: This doesn't actually accumulate gradients, just measures time)")

    times_multi = []
    for trial in range(5):
        t0 = time.perf_counter()

        # Process 3 micro-batches
        for _ in range(3):
            input_ids = torch.randint(0, 150000, (16, seq_len))
            completion_ids = torch.randint(0, 150000, (16, completion_len))
            ref_logprobs = torch.randn(16, completion_len)
            rollout_logprobs = torch.randn(16, completion_len)
            rewards = torch.randn(16)
            mask = torch.ones(16, completion_len)
            prompt_lens = torch.full((16,), seq_len - completion_len, dtype=torch.long)

            await trainer.forward_backward(
                loss_fn,
                input_ids,
                loss_args=(completion_ids, ref_logprobs, rollout_logprobs, rewards),
                loss_kwargs={"padding_mask": mask, "prompt_lens": prompt_lens},
            )

        await trainer.optim_step()
        t1 = time.perf_counter()
        times_multi.append(t1 - t0)
        print(f"  Trial {trial+1}: {1000*(t1-t0):.0f}ms")

    avg_multi = sum(times_multi) / len(times_multi)
    print(f"Average: {1000*avg_multi:.0f}ms")

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Single batch (48):      {1000*avg_single:.0f}ms")
    print(f"Three micro-batches(16): {1000*avg_multi:.0f}ms")
    print(f"Speedup: {avg_single/avg_multi:.2f}x")

    ray.shutdown()


if __name__ == "__main__":
    asyncio.run(run_test())
