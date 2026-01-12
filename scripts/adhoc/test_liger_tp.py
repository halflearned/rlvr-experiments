#!/usr/bin/env python3
"""
Test Liger GRPO loss with Tensor Parallelism (TP>1).
"""

import asyncio
import os
import sys
import copy

import torch
import ray

sys.path.insert(0, "/efs/rlvr-experiments")

from rlvr_experiments.titan_actor import create_titan_group
from rlvr_experiments.losses import LigerGRPOLoss, LIGER_AVAILABLE


async def test_liger_with_tp(tp_degree: int = 2):
    """Test Liger GRPO loss with tensor parallelism."""
    print(f"\n{'='*60}")
    print(f"Testing Liger GRPO Loss with TP={tp_degree}")
    print(f"{'='*60}")

    if not LIGER_AVAILABLE:
        print("  SKIPPED - Liger kernel not available")
        return None

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

    # Enable TP
    trainer_cfg["trainable"] = True
    trainer_cfg["parallelism"]["data_parallel_shard_degree"] = 1
    trainer_cfg["parallelism"]["data_parallel_replicate_degree"] = 1
    trainer_cfg["parallelism"]["tensor_parallel_degree"] = tp_degree
    trainer_cfg["activation_checkpoint"] = {"mode": "selective", "selective_ac_option": "op"}
    trainer_cfg["compile"] = {"enable": False}

    print(f"Creating trainer with TP={tp_degree}...")
    trainer = create_titan_group(trainer_cfg, f"trainer_tp{tp_degree}", tp_degree, 29530)
    print("Trainer created")

    loss_fn = LigerGRPOLoss(beta=0.01, eps=0.2, loss_type="grpo")

    batch_size = 16
    seq_len = 768
    completion_len = 512

    input_ids = torch.randint(0, 150000, (batch_size, seq_len))
    completion_ids = torch.randint(0, 150000, (batch_size, completion_len))
    ref_logprobs = torch.randn(batch_size, completion_len)
    rollout_logprobs = torch.randn(batch_size, completion_len)
    rewards = torch.randn(batch_size)
    mask = torch.ones(batch_size, completion_len)
    prompt_lens = torch.full((batch_size,), seq_len - completion_len, dtype=torch.long)

    num_iters = 5
    for i in range(num_iters):
        try:
            loss, debug = await trainer.forward_backward_liger(
                loss_fn,
                input_ids,
                loss_args=(completion_ids, ref_logprobs, rollout_logprobs, rewards),
                loss_kwargs={"padding_mask": mask, "prompt_lens": prompt_lens},
            )
            print(f"  iter {i}: loss={loss:.4f} ✓")
        except Exception as e:
            print(f"  iter {i}: FAILED - {e}")
            import traceback
            traceback.print_exc()
            break

    ray.shutdown()


if __name__ == "__main__":
    asyncio.run(test_liger_with_tp(tp_degree=2))
