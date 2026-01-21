#!/usr/bin/env python3
"""
Benchmark maximum batch size (B, T=768) for Qwen 1.7B with different parallelism configs.

Configurations to test:
(a) No parallelism (single GPU, no distributed)
(b) TP=1 (distributed but no sharding)
(c) TP=2
(d) TP=4
(e) FSDP=1 (distributed but no sharding)
(f) FSDP=2
(g) FSDP=4
(h) TP=2, FSDP=2

Usage:
    # Single GPU test (no parallelism)
    python scripts/benchmark_max_batch.py --mode single

    # Distributed tests
    torchrun --nproc_per_node=1 scripts/benchmark_max_batch.py --mode tp1
    torchrun --nproc_per_node=2 scripts/benchmark_max_batch.py --mode tp2
    torchrun --nproc_per_node=4 scripts/benchmark_max_batch.py --mode tp4
    torchrun --nproc_per_node=2 scripts/benchmark_max_batch.py --mode fsdp2
    torchrun --nproc_per_node=4 scripts/benchmark_max_batch.py --mode fsdp4
    torchrun --nproc_per_node=4 scripts/benchmark_max_batch.py --mode tp2_fsdp2

    # Run all tests
    python scripts/benchmark_max_batch.py --all
"""

import argparse
import gc
import os
import subprocess
import sys

import torch


def get_mem_gb():
    torch.cuda.synchronize()
    return torch.cuda.memory_allocated() / 1e9


def get_max_mem_gb():
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / 1e9


def reset_peak():
    torch.cuda.reset_peak_memory_stats()


def cleanup():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


def test_batch_size(model, T: int, B: int, device, loss_parallel_ctx=None):
    """Test if a batch size fits in memory. Returns (success, peak_mem_gb)."""
    cleanup()
    reset_peak()

    try:
        # Create input
        input_ids = torch.randint(0, 1000, (B, T), device=device)

        # Forward
        with torch.no_grad():
            outputs = model(input_ids)
            logits = outputs.logits

        # Simulate loss computation (this is where memory peaks)
        if loss_parallel_ctx:
            with loss_parallel_ctx():
                targets = torch.randint(0, logits.shape[-1], (B, T), device=device)
                # For DTensor, need to wrap targets
                if hasattr(logits, 'device_mesh'):
                    from torch.distributed.tensor import DTensor, Replicate
                    targets_dt = DTensor.from_local(targets, logits.device_mesh, [Replicate()])
                    loss = torch.nn.functional.cross_entropy(
                        logits.reshape(-1, logits.shape[-1]),
                        targets_dt.reshape(-1),
                    )
                else:
                    loss = torch.nn.functional.cross_entropy(
                        logits.reshape(-1, logits.shape[-1]),
                        targets.reshape(-1),
                    )
        else:
            vocab_size = logits.shape[-1]
            if hasattr(logits, 'full_tensor'):
                logits = logits.full_tensor()
            targets = torch.randint(0, vocab_size, (B, T), device=device)
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, vocab_size),
                targets.reshape(-1),
            )

        peak = get_max_mem_gb()
        del input_ids, outputs, logits, targets, loss
        cleanup()
        return True, peak

    except torch.cuda.OutOfMemoryError:
        cleanup()
        return False, -1


def find_max_batch(model, T: int, device, loss_parallel_ctx=None, max_mem_gb: float = 75.0):
    """Binary search for max batch size."""
    low, high = 1, 256
    best = 0
    best_mem = 0

    # First find upper bound
    while high <= 512:
        success, peak = test_batch_size(model, T, high, device, loss_parallel_ctx)
        if success and peak < max_mem_gb:
            low = high
            high *= 2
        else:
            break

    # Binary search
    while low <= high:
        mid = (low + high) // 2
        success, peak = test_batch_size(model, T, mid, device, loss_parallel_ctx)

        if success and peak < max_mem_gb:
            best = mid
            best_mem = peak
            low = mid + 1
        else:
            high = mid - 1

    return best, best_mem


def run_single_gpu_test(T: int = 768):
    """Test (a): No parallelism - single GPU."""
    print("\n" + "=" * 60)
    print("(a) No parallelism (single GPU)")
    print("=" * 60)

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    from transformers import AutoModelForCausalLM

    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-1.7B",
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    ).to(device)
    model.eval()

    model_mem = get_mem_gb()
    print(f"Model memory: {model_mem:.2f} GB")

    max_B, peak_mem = find_max_batch(model, T, device)
    print(f"Max batch size: B={max_B} (peak memory: {peak_mem:.2f} GB)")

    del model
    cleanup()
    return {"mode": "single", "max_B": max_B, "peak_mem": peak_mem, "model_mem": model_mem}


def run_distributed_test(mode: str, T: int = 768):
    """Run distributed test with specified parallelism."""
    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh

    # Parse mode
    tp_size = 1
    fsdp_size = 1

    if mode == "tp1":
        tp_size = 1
    elif mode == "tp2":
        tp_size = 2
    elif mode == "tp4":
        tp_size = 4
    elif mode == "fsdp1":
        fsdp_size = 1
    elif mode == "fsdp2":
        fsdp_size = 2
    elif mode == "fsdp4":
        fsdp_size = 4
    elif mode == "tp2_fsdp2":
        tp_size = 2
        fsdp_size = 2

    # Init distributed
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    if rank == 0:
        print("\n" + "=" * 60)
        mode_name = {
            "tp1": "(b) TP=1",
            "tp2": "(c) TP=2",
            "tp4": "(d) TP=4",
            "fsdp1": "(e) FSDP=1",
            "fsdp2": "(f) FSDP=2",
            "fsdp4": "(g) FSDP=4",
            "tp2_fsdp2": "(h) TP=2, FSDP=2",
        }.get(mode, mode)
        print(f"{mode_name} (world_size={world_size})")
        print("=" * 60)

    # Create device mesh
    if tp_size > 1 and fsdp_size > 1:
        mesh = init_device_mesh("cuda", (fsdp_size, tp_size), mesh_dim_names=("dp", "tp"))
        tp_mesh = mesh["tp"]
    elif tp_size > 1:
        mesh = init_device_mesh("cuda", (tp_size,), mesh_dim_names=("tp",))
        tp_mesh = mesh
    elif fsdp_size > 1:
        mesh = init_device_mesh("cuda", (fsdp_size,), mesh_dim_names=("dp",))
        tp_mesh = None
    else:
        mesh = init_device_mesh("cuda", (1,))
        tp_mesh = None

    # Load model with appropriate parallelism
    from transformers import AutoModelForCausalLM

    if rank == 0:
        print("Loading model...")

    if tp_size > 1:
        # Use tensor parallelism
        from torch.distributed.tensor.parallel import (
            ColwiseParallel,
            RowwiseParallel,
            parallelize_module,
            loss_parallel,
        )

        model = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen3-1.7B",
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
        ).to(device)

        # Parallelize layers
        for layer in model.model.layers:
            plan = {
                "self_attn.q_proj": ColwiseParallel(),
                "self_attn.k_proj": ColwiseParallel(),
                "self_attn.v_proj": ColwiseParallel(),
                "self_attn.o_proj": RowwiseParallel(),
                "mlp.gate_proj": ColwiseParallel(),
                "mlp.up_proj": ColwiseParallel(),
                "mlp.down_proj": RowwiseParallel(),
            }
            parallelize_module(layer, tp_mesh, plan)

        # Parallelize lm_head
        parallelize_module(model, tp_mesh, {"lm_head": ColwiseParallel()})

        loss_parallel_ctx = loss_parallel

    elif fsdp_size > 1:
        # Use FSDP
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        from torch.distributed.fsdp import MixedPrecision

        model = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen3-1.7B",
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
        ).to(device)

        mp_policy = MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16,
        )

        model = FSDP(model, mixed_precision=mp_policy)
        loss_parallel_ctx = None

    else:
        # No parallelism (but distributed)
        model = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen3-1.7B",
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
        ).to(device)
        loss_parallel_ctx = None

    model.eval()

    model_mem = get_mem_gb()
    if rank == 0:
        print(f"Model memory: {model_mem:.2f} GB")

    max_B, peak_mem = find_max_batch(model, T, device, loss_parallel_ctx)

    if rank == 0:
        print(f"Max batch size: B={max_B} (peak memory: {peak_mem:.2f} GB)")

    dist.destroy_process_group()

    return {"mode": mode, "max_B": max_B, "peak_mem": peak_mem, "model_mem": model_mem}


def run_all_tests():
    """Run all parallelism configurations."""
    results = []

    # (a) Single GPU
    print("\nRunning single GPU test...")
    result = subprocess.run(
        [sys.executable, __file__, "--mode", "single"],
        capture_output=True,
        text=True,
    )
    print(result.stdout)
    if result.returncode != 0:
        print(f"Error: {result.stderr}")

    # Distributed tests
    configs = [
        ("tp1", 1),
        ("tp2", 2),
        ("tp4", 4),
        ("fsdp2", 2),
        ("fsdp4", 4),
        ("tp2_fsdp2", 4),
    ]

    for mode, nproc in configs:
        print(f"\nRunning {mode} test with {nproc} GPUs...")
        result = subprocess.run(
            ["torchrun", f"--nproc_per_node={nproc}", __file__, "--mode", mode],
            capture_output=True,
            text=True,
        )
        print(result.stdout)
        if result.returncode != 0:
            print(f"Error: {result.stderr}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="single",
                        choices=["single", "tp1", "tp2", "tp4", "fsdp1", "fsdp2", "fsdp4", "tp2_fsdp2", "all"])
    parser.add_argument("--T", type=int, default=768, help="Sequence length")
    args = parser.parse_args()

    if args.mode == "all":
        run_all_tests()
    elif args.mode == "single":
        run_single_gpu_test(args.T)
    else:
        run_distributed_test(args.mode, args.T)


if __name__ == "__main__":
    main()
