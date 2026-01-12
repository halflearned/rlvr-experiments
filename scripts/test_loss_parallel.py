#!/usr/bin/env python3
"""Test script to verify loss_parallel works with GRPOLoss.

This script tests whether we can use disable_loss_parallel=False to reduce
memory usage by keeping logits sharded across the vocab dimension.

Run with: torchrun --nproc_per_node=4 scripts/test_loss_parallel.py
"""

import os
import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor, Shard, Replicate, DeviceMesh
from torch.distributed.tensor.parallel import loss_parallel

# Must set before importing other modules
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def setup_distributed():
    """Initialize distributed training."""
    dist.init_process_group("nccl")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    return local_rank


def create_sharded_logits(batch_size, seq_len, vocab_size, mesh, device):
    """Create vocab-sharded logits tensor (simulating model output with TP)."""
    # Each rank creates its shard of the vocab
    tp_size = mesh.size()
    local_vocab = vocab_size // tp_size

    # Create local logits shard
    local_logits = torch.randn(
        batch_size, seq_len, local_vocab,
        device=device, dtype=torch.bfloat16, requires_grad=True
    )

    # Wrap as DTensor with Shard on vocab dimension
    sharded_logits = DTensor.from_local(
        local_logits,
        mesh,
        [Shard(2)],  # Shard on dim 2 (vocab)
    )

    return sharded_logits, local_logits


def test_cross_entropy_with_loss_parallel():
    """Test that F.cross_entropy works with vocab-sharded DTensor."""
    rank = setup_distributed()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{rank}")

    print(f"[Rank {rank}] Testing cross_entropy with loss_parallel, world_size={world_size}")

    # Create mesh for tensor parallelism
    mesh = DeviceMesh("cuda", list(range(world_size)))

    # Test parameters
    batch_size = 8
    seq_len = 10
    vocab_size = 1000 * world_size  # Must be divisible by world_size

    # Create sharded logits
    sharded_logits, local_logits = create_sharded_logits(
        batch_size, seq_len, vocab_size, mesh, device
    )

    # Create targets (same on all ranks) - must be replicated DTensor for backward to work
    torch.manual_seed(42)
    targets = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    # Wrap targets as replicated DTensor
    targets_dt = DTensor.from_local(targets, mesh, [Replicate()])

    # Test 1: Cross entropy with loss_parallel context
    print(f"[Rank {rank}] Test 1: cross_entropy with loss_parallel context")

    with loss_parallel():
        # Reshape for cross_entropy: [B*T, V] and [B*T]
        logits_flat = sharded_logits.reshape(-1, vocab_size)
        targets_flat = targets_dt.reshape(-1)

        loss = torch.nn.functional.cross_entropy(
            logits_flat, targets_flat, reduction="mean"
        )

    print(f"[Rank {rank}] Loss: {loss.item():.4f}")

    # Backward pass
    loss.backward()

    print(f"[Rank {rank}] Backward pass completed")
    print(f"[Rank {rank}] Local logits grad shape: {local_logits.grad.shape}")
    print(f"[Rank {rank}] Local logits grad norm: {local_logits.grad.norm().item():.4f}")

    # Test 2: Verify gradients are correct by comparing with full tensor version
    print(f"\n[Rank {rank}] Test 2: Comparing with full tensor computation")

    # Gather full logits on all ranks
    full_logits = sharded_logits.full_tensor().detach().clone().requires_grad_(True)

    # Compute loss on full tensor (no loss_parallel needed) - use regular tensor targets
    full_logits_flat = full_logits.reshape(-1, vocab_size)
    targets_regular = targets.reshape(-1)  # Use original non-DTensor targets
    loss_full = torch.nn.functional.cross_entropy(
        full_logits_flat, targets_regular, reduction="mean"
    )
    loss_full.backward()

    print(f"[Rank {rank}] Full tensor loss: {loss_full.item():.4f}")
    print(f"[Rank {rank}] Loss difference: {abs(loss.item() - loss_full.item()):.6f}")

    # Compare gradients for this rank's shard
    local_vocab = vocab_size // world_size
    start = rank * local_vocab
    end = start + local_vocab
    expected_grad = full_logits.grad[:, :, start:end]
    actual_grad = local_logits.grad

    grad_diff = (expected_grad - actual_grad).abs().max().item()
    print(f"[Rank {rank}] Max gradient difference: {grad_diff:.6f}")

    if grad_diff < 1e-4:
        print(f"[Rank {rank}] ✓ Gradients match!")
    else:
        print(f"[Rank {rank}] ✗ Gradient mismatch!")

    dist.destroy_process_group()
    return grad_diff < 1e-4


def test_compute_logprobs_with_dtensor():
    """Test that compute_logprobs works with vocab-sharded DTensor.

    Note: The current compute_logprobs uses gather() which doesn't work well with
    vocab-sharded DTensor. We test a workaround: do the slicing on local shards,
    then use F.cross_entropy with loss_parallel for the actual logprob computation.
    """
    rank = setup_distributed()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{rank}")

    print(f"\n[Rank {rank}] Testing compute_logprobs with DTensor")

    mesh = DeviceMesh("cuda", list(range(world_size)))

    batch_size = 4
    seq_len = 20
    completion_len = 10
    vocab_size = 1000 * world_size

    # Create sharded logits
    sharded_logits, local_logits = create_sharded_logits(
        batch_size, seq_len, vocab_size, mesh, device
    )

    # Create completion tokens (same on all ranks)
    torch.manual_seed(42)
    completion_ids = torch.randint(0, vocab_size, (batch_size, completion_len), device=device)
    prompt_len = seq_len - completion_len  # Same for all samples in this test

    print(f"[Rank {rank}] Sharded logits shape: {sharded_logits.shape}")
    print(f"[Rank {rank}] Local logits shape: {local_logits.shape}")

    # Alternative approach: slice on sequence dim first (works on local shards),
    # then use F.cross_entropy with loss_parallel for vocab-sharded cross entropy
    with loss_parallel():
        try:
            # Slice on sequence dimension - works because seq dim is replicated
            # We want positions [prompt_len-1, prompt_len-1+completion_len) to predict completion tokens
            sliced_logits = sharded_logits[:, prompt_len - 1 : prompt_len - 1 + completion_len, :]
            print(f"[Rank {rank}] Sliced logits shape: {sliced_logits.shape}")

            # Wrap completion_ids as replicated DTensor for cross_entropy
            completion_ids_dt = DTensor.from_local(completion_ids, mesh, [Replicate()])

            # Use cross_entropy to compute log probs (negated)
            logprobs = -torch.nn.functional.cross_entropy(
                sliced_logits.reshape(-1, vocab_size),
                completion_ids_dt.reshape(-1).long(),
                reduction="none",
            )
            logprobs = logprobs.reshape(batch_size, completion_len)

            print(f"[Rank {rank}] ✓ compute_logprobs succeeded")
            print(f"[Rank {rank}] Logprobs shape: {logprobs.shape}")
            print(f"[Rank {rank}] Logprobs is DTensor: {isinstance(logprobs, DTensor)}")

            # Get local tensor for backward
            logprobs_local = logprobs.to_local() if isinstance(logprobs, DTensor) else logprobs
            print(f"[Rank {rank}] Logprobs range: [{logprobs_local.min().item():.2f}, {logprobs_local.max().item():.2f}]")

            # Test backward
            logprobs_local.sum().backward()
            print(f"[Rank {rank}] ✓ Backward pass succeeded")
            print(f"[Rank {rank}] Grad norm: {local_logits.grad.norm().item():.4f}")
            success = True

        except Exception as e:
            print(f"[Rank {rank}] ✗ compute_logprobs failed: {e}")
            import traceback
            traceback.print_exc()
            success = False

    dist.destroy_process_group()
    return success


def get_prompt_groups(prompt_lens: torch.Tensor) -> list[tuple[int, int, int]]:
    """Find contiguous groups with same prompt length.

    Returns list of (start_idx, end_idx, prompt_len) tuples.
    """
    groups = []
    start = 0
    current_len = prompt_lens[0].item()
    for i in range(1, len(prompt_lens)):
        if prompt_lens[i].item() != current_len:
            groups.append((start, i, current_len))
            start = i
            current_len = prompt_lens[i].item()
    groups.append((start, len(prompt_lens), current_len))
    return groups


def compute_logprobs_dtensor(
    logits: DTensor,
    input_ids: torch.Tensor,
    prompt_lens: torch.Tensor,
    mesh: DeviceMesh,
) -> torch.Tensor:
    """DTensor-compatible compute_logprobs using group-based slicing.

    Instead of using gather (which doesn't work with vocab-sharded DTensor),
    we process each prompt group separately using simple slicing.
    """
    vocab_size = logits.shape[-1]
    target_len = input_ids.size(1)

    # Get groups of samples with same prompt length
    groups = get_prompt_groups(prompt_lens)

    # Process each group
    results = []
    for start, end, prompt_len in groups:
        # Simple slice - works with DTensor because we're slicing on batch and seq dims
        # which are replicated, not the vocab dim which is sharded
        group_logits = logits[start:end, prompt_len - 1 : prompt_len - 1 + target_len, :]
        group_ids = input_ids[start:end]

        # Wrap ids as replicated DTensor
        group_ids_dt = DTensor.from_local(group_ids, mesh, [Replicate()])

        # Compute logprobs via cross_entropy
        group_logprobs = -torch.nn.functional.cross_entropy(
            group_logits.reshape(-1, vocab_size),
            group_ids_dt.reshape(-1).long(),
            reduction="none",
        )
        group_logprobs = group_logprobs.reshape(end - start, target_len)
        results.append(group_logprobs)

    # Concatenate results
    logprobs = torch.cat(results, dim=0)
    return logprobs


def test_compute_logprobs_multi_prompt():
    """Test compute_logprobs with multiple prompts of different lengths."""
    rank = setup_distributed()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{rank}")

    print(f"\n[Rank {rank}] Testing compute_logprobs with multiple prompts")

    mesh = DeviceMesh("cuda", list(range(world_size)))

    # Simulate 2 prompts packed together:
    # - Prompt A: 2 samples, prompt_len=8
    # - Prompt B: 2 samples, prompt_len=12
    batch_size = 4
    seq_len = 20
    completion_len = 8  # Both prompts have same completion length for simplicity
    vocab_size = 1000 * world_size

    # Create sharded logits
    sharded_logits, local_logits = create_sharded_logits(
        batch_size, seq_len, vocab_size, mesh, device
    )

    # Create completion tokens and varying prompt_lens
    torch.manual_seed(42)
    completion_ids = torch.randint(0, vocab_size, (batch_size, completion_len), device=device)
    # Samples 0-1 have prompt_len=8, samples 2-3 have prompt_len=12
    prompt_lens = torch.tensor([8, 8, 12, 12], device=device, dtype=torch.long)

    print(f"[Rank {rank}] Sharded logits shape: {sharded_logits.shape}")
    print(f"[Rank {rank}] prompt_lens: {prompt_lens.tolist()}")
    print(f"[Rank {rank}] Groups: {get_prompt_groups(prompt_lens)}")

    with loss_parallel():
        try:
            logprobs = compute_logprobs_dtensor(
                sharded_logits,
                completion_ids,
                prompt_lens,
                mesh,
            )

            print(f"[Rank {rank}] ✓ compute_logprobs_dtensor succeeded")
            print(f"[Rank {rank}] Logprobs shape: {logprobs.shape}")
            print(f"[Rank {rank}] Logprobs is DTensor: {isinstance(logprobs, DTensor)}")

            # Get local tensor for stats
            logprobs_local = logprobs.to_local() if isinstance(logprobs, DTensor) else logprobs
            print(f"[Rank {rank}] Logprobs range: [{logprobs_local.min().item():.2f}, {logprobs_local.max().item():.2f}]")

            # Test backward
            logprobs_local.sum().backward()
            print(f"[Rank {rank}] ✓ Backward pass succeeded")
            print(f"[Rank {rank}] Grad norm: {local_logits.grad.norm().item():.4f}")
            success = True

        except Exception as e:
            print(f"[Rank {rank}] ✗ compute_logprobs_dtensor failed: {e}")
            import traceback
            traceback.print_exc()
            success = False

    dist.destroy_process_group()
    return success


def test_grpo_loss_with_dtensor():
    """Test that full GRPOLoss works with vocab-sharded DTensor."""
    rank = setup_distributed()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{rank}")

    print(f"\n[Rank {rank}] Testing GRPOLoss with DTensor")

    from rlvr_experiments.losses import GRPOLoss

    mesh = DeviceMesh("cuda", list(range(world_size)))

    batch_size = 4
    seq_len = 20
    completion_len = 10
    vocab_size = 1000 * world_size

    # Create sharded logits
    sharded_logits, local_logits = create_sharded_logits(
        batch_size, seq_len, vocab_size, mesh, device
    )

    # Create other inputs (same on all ranks)
    torch.manual_seed(42)
    completion_ids = torch.randint(0, vocab_size, (batch_size, completion_len), device=device)
    ref_logprobs = torch.randn(batch_size, completion_len, device=device)
    rollout_logprobs = torch.randn(batch_size, completion_len, device=device)
    rewards = torch.randn(batch_size, device=device)
    padding_mask = torch.ones(batch_size, completion_len, device=device)
    prompt_lens = torch.full((batch_size,), seq_len - completion_len, device=device, dtype=torch.long)

    loss_fn = GRPOLoss(beta=0.01, eps=0.2)

    # Test with loss_parallel context
    with loss_parallel():
        try:
            loss = loss_fn(
                sharded_logits,
                completion_ids,
                ref_logprobs,
                rollout_logprobs,
                rewards,
                padding_mask,
                prompt_lens=prompt_lens,
            )
            print(f"[Rank {rank}] ✓ GRPOLoss forward succeeded")
            print(f"[Rank {rank}] Loss: {loss.item():.4f}")

            # Test backward
            loss.backward()
            print(f"[Rank {rank}] ✓ Backward pass succeeded")
            print(f"[Rank {rank}] Grad norm: {local_logits.grad.norm().item():.4f}")
            success = True

        except Exception as e:
            print(f"[Rank {rank}] ✗ GRPOLoss failed: {e}")
            import traceback
            traceback.print_exc()
            success = False

    dist.destroy_process_group()
    return success


def test_memory_savings():
    """Measure memory savings with loss_parallel."""
    rank = setup_distributed()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{rank}")

    print(f"\n[Rank {rank}] Measuring memory usage")

    mesh = DeviceMesh("cuda", list(range(world_size)))

    # Realistic sizes
    batch_size = 64  # ppfb=1 with n=64
    seq_len = 768
    vocab_size = 151936  # Qwen3 vocab, round to nearest divisible
    vocab_size = (vocab_size // world_size) * world_size  # Make divisible

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()

    # Measure sharded logits memory
    sharded_logits, local_logits = create_sharded_logits(
        batch_size, seq_len, vocab_size, mesh, device
    )

    sharded_mem = torch.cuda.max_memory_allocated() / 1e9
    print(f"[Rank {rank}] Sharded logits memory: {sharded_mem:.2f} GB")
    print(f"[Rank {rank}] Local logits shape: {local_logits.shape}")

    # Compare with full tensor
    torch.cuda.reset_peak_memory_stats()
    del sharded_logits, local_logits
    torch.cuda.empty_cache()

    full_logits = torch.randn(
        batch_size, seq_len, vocab_size,
        device=device, dtype=torch.bfloat16
    )

    full_mem = torch.cuda.max_memory_allocated() / 1e9
    print(f"[Rank {rank}] Full logits memory: {full_mem:.2f} GB")
    print(f"[Rank {rank}] Full logits shape: {full_logits.shape}")

    print(f"[Rank {rank}] Memory savings: {full_mem - sharded_mem:.2f} GB ({(1 - sharded_mem/full_mem)*100:.1f}%)")

    dist.destroy_process_group()


def test_tp1_vs_tp4_equivalence():
    """Verify that TP=4 with loss_parallel produces same results as TP=1.

    This test compares the actual numerical values, not just that gradients match.
    We create the same logits on all ranks and verify the computed logprobs are identical.
    """
    rank = setup_distributed()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{rank}")

    print(f"\n[Rank {rank}] Testing TP=1 vs TP={world_size} equivalence")

    mesh = DeviceMesh("cuda", list(range(world_size)))

    batch_size = 4
    seq_len = 16
    completion_len = 8
    vocab_size = 1000 * world_size  # Must be divisible

    # Create IDENTICAL logits on all ranks using same seed
    torch.manual_seed(123)
    full_logits = torch.randn(
        batch_size, seq_len, vocab_size,
        device=device, dtype=torch.bfloat16
    )

    # Create completion tokens (same on all ranks)
    torch.manual_seed(456)
    completion_ids = torch.randint(0, vocab_size, (batch_size, completion_len), device=device)
    prompt_len = seq_len - completion_len

    # === TP=1 baseline: compute logprobs on full tensor ===
    sliced_full = full_logits[:, prompt_len - 1 : prompt_len - 1 + completion_len, :]
    logprobs_tp1 = -torch.nn.functional.cross_entropy(
        sliced_full.reshape(-1, vocab_size).float(),
        completion_ids.reshape(-1).long(),
        reduction="none",
    ).reshape(batch_size, completion_len)

    print(f"[Rank {rank}] TP=1 logprobs: mean={logprobs_tp1.mean().item():.6f}, "
          f"min={logprobs_tp1.min().item():.4f}, max={logprobs_tp1.max().item():.4f}")

    # === TP=4 with loss_parallel: shard logits across vocab ===
    # Each rank takes its portion of the vocab
    local_vocab = vocab_size // world_size
    start_v = rank * local_vocab
    end_v = start_v + local_vocab
    local_logits = full_logits[:, :, start_v:end_v].contiguous()

    # Wrap as DTensor with Shard on vocab dimension
    sharded_logits = DTensor.from_local(
        local_logits,
        mesh,
        [Shard(2)],  # Shard on dim 2 (vocab)
    )

    # Compute logprobs with loss_parallel
    with loss_parallel():
        sliced_sharded = sharded_logits[:, prompt_len - 1 : prompt_len - 1 + completion_len, :]
        completion_ids_dt = DTensor.from_local(completion_ids, mesh, [Replicate()])

        logprobs_tp4_dt = -torch.nn.functional.cross_entropy(
            sliced_sharded.reshape(-1, vocab_size).float(),
            completion_ids_dt.reshape(-1).long(),
            reduction="none",
        ).reshape(batch_size, completion_len)

    # Extract local tensor from DTensor
    logprobs_tp4 = logprobs_tp4_dt.to_local() if isinstance(logprobs_tp4_dt, DTensor) else logprobs_tp4_dt

    print(f"[Rank {rank}] TP={world_size} logprobs: mean={logprobs_tp4.mean().item():.6f}, "
          f"min={logprobs_tp4.min().item():.4f}, max={logprobs_tp4.max().item():.4f}")

    # === Compare ===
    diff = (logprobs_tp1 - logprobs_tp4).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()

    print(f"[Rank {rank}] Difference: max={max_diff:.8f}, mean={mean_diff:.8f}")

    if max_diff < 1e-4:
        print(f"[Rank {rank}] ✓ TP=1 and TP={world_size} produce equivalent results!")
        success = True
    else:
        print(f"[Rank {rank}] ✗ Results differ! max_diff={max_diff}")
        # Print some examples
        for i in range(min(3, batch_size)):
            print(f"[Rank {rank}]   Sample {i}: tp1={logprobs_tp1[i, 0].item():.6f}, "
                  f"tp4={logprobs_tp4[i, 0].item():.6f}")
        success = False

    dist.destroy_process_group()
    return success


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        test_name = sys.argv[1]
        if test_name == "cross_entropy":
            test_cross_entropy_with_loss_parallel()
        elif test_name == "compute_logprobs":
            test_compute_logprobs_with_dtensor()
        elif test_name == "multi_prompt":
            test_compute_logprobs_multi_prompt()
        elif test_name == "grpo_loss":
            test_grpo_loss_with_dtensor()
        elif test_name == "memory":
            test_memory_savings()
        elif test_name == "tp_equivalence":
            test_tp1_vs_tp4_equivalence()
        else:
            print(f"Unknown test: {test_name}")
            print("Available: cross_entropy, compute_logprobs, multi_prompt, grpo_loss, memory, tp_equivalence")
    else:
        # Run all tests
        print("=" * 60)
        print("Test 1: Cross-entropy with loss_parallel")
        print("=" * 60)
        test_cross_entropy_with_loss_parallel()
