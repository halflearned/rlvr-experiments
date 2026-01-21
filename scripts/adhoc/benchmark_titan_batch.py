#!/usr/bin/env python3
"""
Benchmark maximum batch size (B, T=768) for Qwen 1.7B using TitanModel.

Creates temporary TOML configs and runs torchrun with different parallelism settings.

Usage:
    python scripts/benchmark_titan_batch.py --all
"""

import argparse
import gc
import os
import subprocess
import sys
import tempfile
import time

import torch
import tomli_w as toml


def create_toml_config(tp: int, fsdp: int, disable_loss_parallel: bool) -> str:
    """Create a temporary TOML config file for TitanModel."""
    config = {
        "model": {
            "name": "qwen3",
            "flavor": "1.7B",
            "hf_assets_path": "/efs/rlvr-experiments/assets/hf/Qwen3-1.7B-Base",
        },
        "parallelism": {
            "tensor_parallel_degree": tp,
            "data_parallel_shard_degree": fsdp,
            "data_parallel_replicate_degree": 1,
            "context_parallel_degree": 1,
            "disable_loss_parallel": disable_loss_parallel,
        },
        "training": {
            "seq_len": 2048,
            "dtype": "bfloat16",
            "mixed_precision_param": "bfloat16",
            "mixed_precision_reduce": "float32",
        },
        "optimizer": {
            "name": "AdamW",
            "lr": 1e-5,
        },
        "lr_scheduler": {
            "warmup_steps": 1,
        },
        "checkpoint": {
            "enable": False,
            "folder": "",
            "initial_load_in_hf": True,
        },
        "metrics": {
            "log_freq": 1,
            "enable_tensorboard": False,
        },
        "activation_checkpoint": {
            "mode": "full",
            "selective_ac_option": "op",
        },
    }

    fd, path = tempfile.mkstemp(suffix=".toml")
    with os.fdopen(fd, "wb") as f:
        toml.dump(config, f)
    return path


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


def test_batch_size(model, T: int, B: int, device):
    """Test if a batch size fits in memory with forward+backward. Returns (success, peak_mem_gb)."""
    cleanup()
    reset_peak()

    try:
        # Create input
        input_ids = torch.randint(0, 1000, (B, T), device=device)

        # Forward is OUTSIDE train_context_mgr (just like titan_actor.py)
        logits = model.forward(input_ids)

        # Loss and backward are INSIDE train_context_mgr (for loss_parallel)
        with model.train_context_mgr(None):
            from torch.distributed.tensor import DTensor
            if isinstance(logits, DTensor):
                # With loss_parallel + DTensor:
                # - logits.shape returns GLOBAL shape (DTensor abstracts sharding)
                # - logits.shape[-1] = global vocab (151936)
                # - Internally, local tensor is [B, T, V/TP]
                vocab_size = logits.shape[-1]  # Global vocab

                # Flatten to [B*T, V] (DTensor handles sharding internally)
                logits_flat = logits.reshape(-1, vocab_size)

                # Targets in global vocab range, wrapped as replicated DTensor
                targets = torch.randint(0, vocab_size, (B * T,), device=device)
                from torch.distributed.tensor import Replicate
                targets_dt = DTensor.from_local(targets, logits.device_mesh, [Replicate()])

                # loss_parallel handles the distributed softmax
                loss = torch.nn.functional.cross_entropy(logits_flat, targets_dt)
            else:
                vocab_size = logits.shape[-1]
                targets = torch.randint(0, vocab_size, (B, T), device=device)
                loss = torch.nn.functional.cross_entropy(
                    logits.reshape(-1, vocab_size),
                    targets.reshape(-1),
                )

            # Backward inside train_context_mgr for loss_parallel
            loss.backward()

        peak = get_max_mem_gb()

        # Cleanup
        model.optimizers.zero_grad(set_to_none=True)
        del input_ids, logits, targets, loss
        cleanup()

        return True, peak

    except torch.cuda.OutOfMemoryError:
        cleanup()
        if hasattr(model, 'optimizers') and model.optimizers is not None:
            try:
                model.optimizers.zero_grad(set_to_none=True)
            except:
                pass
        return False, -1
    except Exception as e:
        rank = int(os.environ.get("RANK", 0))
        if rank == 0:
            print(f"Error at B={B}: {e}")
        cleanup()
        if hasattr(model, 'optimizers') and model.optimizers is not None:
            try:
                model.optimizers.zero_grad(set_to_none=True)
            except:
                pass
        return False, -1


def find_max_batch(model, T: int, device, max_mem_gb: float = 75.0):
    """Binary search for max batch size."""
    rank = int(os.environ.get("RANK", 0))

    low, high = 1, 256
    best = 0
    best_mem = 0

    # First, find if we can even do B=1
    success, peak = test_batch_size(model, T, 1, device)
    if not success:
        if rank == 0:
            print(f"  B=1: OOM - cannot run even minimum batch", flush=True)
        return 0, 0.0

    if rank == 0:
        print(f"  B=1: peak={peak:.2f} GB", flush=True)
    best = 1
    best_mem = peak

    # Find upper bound
    while high <= 512:
        success, peak = test_batch_size(model, T, high, device)
        if rank == 0:
            status = f"peak={peak:.2f} GB" if success else "OOM"
            print(f"  B={high:4d}: {status}", flush=True)
        if success and peak < max_mem_gb:
            best = high
            best_mem = peak
            low = high
            high *= 2
        else:
            break

    # Binary search
    while low <= high:
        mid = (low + high) // 2
        success, peak = test_batch_size(model, T, mid, device)

        if rank == 0:
            status = f"peak={peak:.2f} GB" if success else "OOM"
            print(f"  B={mid:4d}: {status}", flush=True)

        if success and peak < max_mem_gb:
            best = mid
            best_mem = peak
            low = mid + 1
        else:
            high = mid - 1

    return best, best_mem


def run_benchmark(config_path: str, tp: int, fsdp: int, T: int = 768):
    """Run benchmark with the given config."""
    from rlvr_experiments.model import TitanModel
    from torchtitan.config import ConfigManager

    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    if rank == 0:
        print(f"\nLoading model (TP={tp}, FSDP={fsdp})...", flush=True)

    # Parse config and create model
    job_config = ConfigManager().parse_args(["--job.config-file", config_path])
    model = TitanModel(job_config, trainable=True)

    model_mem = get_mem_gb()
    if rank == 0:
        print(f"Model memory: {model_mem:.2f} GB", flush=True)
        print(f"\nSearching for max batch size (T={T})...", flush=True)

    max_B, peak_mem = find_max_batch(model, T, device)

    if rank == 0:
        print(f"\n=> Max batch size: B={max_B} (peak memory: {peak_mem:.2f} GB)", flush=True)

    # Return result for collection
    return {"tp": tp, "fsdp": fsdp, "max_B": max_B, "peak_mem": peak_mem, "model_mem": model_mem}


def run_all_tests():
    """Run all parallelism configurations."""
    script = os.path.abspath(__file__)

    configs = [
        # (tp, fsdp, nproc, disable_loss_parallel, label)
        (1, 1, 1, True, "(a) TP=1, FSDP=1 (baseline)"),
        (2, 1, 2, True, "(b) TP=2, no loss_parallel"),
        (2, 1, 2, False, "(c) TP=2, WITH loss_parallel"),
        (4, 1, 4, True, "(d) TP=4, no loss_parallel"),
        (4, 1, 4, False, "(e) TP=4, WITH loss_parallel"),
        (1, 2, 2, True, "(f) FSDP=2"),
        (1, 4, 4, True, "(g) FSDP=4"),
        (2, 2, 4, True, "(h) TP=2, FSDP=2, no loss_parallel"),
        (2, 2, 4, False, "(i) TP=2, FSDP=2, WITH loss_parallel"),
    ]

    results = []

    for tp, fsdp, nproc, disable_lp, label in configs:
        print(f"\n{'='*60}")
        print(label)
        print(f"{'='*60}")

        # Create config
        config_path = create_toml_config(tp, fsdp, disable_lp)

        try:
            cmd = [
                "torchrun",
                f"--nproc_per_node={nproc}",
                script,
                "--config", config_path,
                f"--tp={tp}",
                f"--fsdp={fsdp}",
            ]

            result = subprocess.run(cmd, capture_output=False, text=True)
            if result.returncode != 0:
                print(f"FAILED (exit code {result.returncode})")
        finally:
            os.unlink(config_path)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="Config file path (for worker mode)")
    parser.add_argument("--tp", type=int, default=1, help="Tensor parallel degree")
    parser.add_argument("--fsdp", type=int, default=1, help="FSDP degree")
    parser.add_argument("--T", type=int, default=768, help="Sequence length")
    parser.add_argument("--all", action="store_true", help="Run all configurations")
    args = parser.parse_args()

    if args.all:
        run_all_tests()
    elif args.config:
        # Worker mode - actually run the benchmark
        run_benchmark(args.config, args.tp, args.fsdp, args.T)
    else:
        print("Usage: python benchmark_titan_batch.py --all")
        print("   or: torchrun --nproc_per_node=N benchmark_titan_batch.py --config <path> --tp X --fsdp Y")


if __name__ == "__main__":
    main()
