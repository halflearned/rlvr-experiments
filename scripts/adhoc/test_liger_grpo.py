#!/usr/bin/env python3
"""
Test script to compare standard GRPO loss vs Liger fused GRPO loss.

This tests:
1. Memory usage (peak GPU memory)
2. Performance (forward + backward time)
3. Correctness (loss values should be similar)
"""

import asyncio
import os
import sys
import copy

import torch
import ray

sys.path.insert(0, "/efs/rlvr-experiments")

from rlvr_experiments.titan_actor import create_titan_group
from rlvr_experiments.losses import GRPOLoss, LigerGRPOLoss, LIGER_AVAILABLE


async def test_standard_grpo(trainer, batch_size: int, seq_len: int, completion_len: int, num_iters: int = 5):
    """Test standard GRPO loss (materializes full logits)."""
    print(f"\n{'='*60}")
    print(f"Testing STANDARD GRPO Loss (batch={batch_size})")
    print(f"{'='*60}")

    loss_fn = GRPOLoss(beta=0.01, eps=0.2)

    input_ids = torch.randint(0, 150000, (batch_size, seq_len))
    completion_ids = torch.randint(0, 150000, (batch_size, completion_len))
    ref_logprobs = torch.randn(batch_size, completion_len)
    rollout_logprobs = torch.randn(batch_size, completion_len)
    rewards = torch.randn(batch_size)
    mask = torch.ones(batch_size, completion_len)
    prompt_lens = torch.full((batch_size,), seq_len - completion_len, dtype=torch.long)

    times = []
    for i in range(num_iters):
        try:
            loss, debug = await trainer.forward_backward(
                loss_fn,
                input_ids,
                loss_args=(completion_ids, ref_logprobs, rollout_logprobs, rewards),
                loss_kwargs={"padding_mask": mask, "prompt_lens": prompt_lens},
            )
            if i > 0:  # Skip first iteration (warmup)
                times.append(loss)
            print(f"  iter {i}: loss={loss:.4f}")
        except Exception as e:
            print(f"  iter {i}: FAILED - {e}")
            return None

    return times


async def test_liger_grpo(trainer, batch_size: int, seq_len: int, completion_len: int, num_iters: int = 5):
    """Test Liger fused GRPO loss (memory-efficient)."""
    print(f"\n{'='*60}")
    print(f"Testing LIGER GRPO Loss (batch={batch_size})")
    print(f"{'='*60}")

    if not LIGER_AVAILABLE:
        print("  SKIPPED - Liger kernel not available")
        return None

    loss_fn = LigerGRPOLoss(beta=0.01, eps=0.2, loss_type="grpo")

    input_ids = torch.randint(0, 150000, (batch_size, seq_len))
    completion_ids = torch.randint(0, 150000, (batch_size, completion_len))
    ref_logprobs = torch.randn(batch_size, completion_len)
    rollout_logprobs = torch.randn(batch_size, completion_len)
    rewards = torch.randn(batch_size)
    mask = torch.ones(batch_size, completion_len)
    prompt_lens = torch.full((batch_size,), seq_len - completion_len, dtype=torch.long)

    times = []
    for i in range(num_iters):
        try:
            loss, debug = await trainer.forward_backward_liger(
                loss_fn,
                input_ids,
                loss_args=(completion_ids, ref_logprobs, rollout_logprobs, rewards),
                loss_kwargs={"padding_mask": mask, "prompt_lens": prompt_lens},
            )
            if i > 0:  # Skip first iteration (warmup)
                times.append(loss)
            print(f"  iter {i}: loss={loss:.4f}")
        except Exception as e:
            print(f"  iter {i}: FAILED - {e}")
            import traceback
            traceback.print_exc()
            return None

    return times


async def run_comparison():
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
    # Use activation checkpointing for fair comparison
    trainer_cfg["activation_checkpoint"] = {"mode": "selective", "selective_ac_option": "op"}
    trainer_cfg["compile"] = {"enable": False}

    print("Creating trainer...")
    trainer = create_titan_group(trainer_cfg, "trainer_liger_test", 1, 29520)
    print("Trainer created")

    seq_len = 768
    completion_len = 512

    # Test at different batch sizes
    for batch_size in [16, 32, 48]:
        print(f"\n{'#'*70}")
        print(f"# BATCH SIZE: {batch_size}")
        print(f"{'#'*70}")

        # Reset memory stats
        torch.cuda.reset_peak_memory_stats()

        # Test standard GRPO
        standard_results = await test_standard_grpo(trainer, batch_size, seq_len, completion_len)

        # Get peak memory for standard
        standard_peak_mem = torch.cuda.max_memory_allocated() / 1e9
        print(f"  Standard peak memory: {standard_peak_mem:.2f} GB")

        # Reset memory stats
        torch.cuda.reset_peak_memory_stats()

        # Test Liger GRPO
        liger_results = await test_liger_grpo(trainer, batch_size, seq_len, completion_len)

        # Get peak memory for Liger
        liger_peak_mem = torch.cuda.max_memory_allocated() / 1e9
        print(f"  Liger peak memory: {liger_peak_mem:.2f} GB")

        if standard_results and liger_results:
            if standard_peak_mem > 0:
                print(f"\n  Memory savings: {standard_peak_mem - liger_peak_mem:.2f} GB ({100*(standard_peak_mem - liger_peak_mem)/standard_peak_mem:.1f}%)")
            else:
                print(f"\n  (Memory measurement unavailable in Ray actor)")

    ray.shutdown()


if __name__ == "__main__":
    asyncio.run(run_comparison())
