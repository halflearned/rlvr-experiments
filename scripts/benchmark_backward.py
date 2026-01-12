#!/usr/bin/env python
"""Benchmark forward/backward pass: TorchTitan vs HuggingFace."""

import time
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATH = "/efs/rlvr-experiments/assets/hf/Qwen3-0.6B"

def benchmark_hf(batch_size=3, n_completions=16, seq_len=768, warmup=3, iters=10):
    """Benchmark HuggingFace model forward/backward."""
    print(f"\n=== HuggingFace Benchmark ===")
    print(f"batch_size={batch_size}, n_completions={n_completions}, seq_len={seq_len}")
    print(f"Total sequences: {batch_size * n_completions}, Total tokens: {batch_size * n_completions * seq_len}")

    device = torch.device("cuda")

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
    )
    model.train()

    # Create dummy input
    total_batch = batch_size * n_completions
    input_ids = torch.randint(0, 32000, (total_batch, seq_len), device=device)
    labels = input_ids.clone()

    # Warmup
    print(f"Warming up ({warmup} iters)...")
    for _ in range(warmup):
        outputs = model(input_ids, labels=labels)
        loss = outputs.loss
        loss.backward()
        model.zero_grad()
        torch.cuda.synchronize()

    # Benchmark
    print(f"Benchmarking ({iters} iters)...")
    fwd_times = []
    bwd_times = []

    for i in range(iters):
        torch.cuda.synchronize()

        # Forward
        t0 = time.perf_counter()
        outputs = model(input_ids, labels=labels)
        loss = outputs.loss
        torch.cuda.synchronize()
        t1 = time.perf_counter()

        # Backward
        loss.backward()
        torch.cuda.synchronize()
        t2 = time.perf_counter()

        model.zero_grad()

        fwd_times.append(t1 - t0)
        bwd_times.append(t2 - t1)
        print(f"  iter {i}: fwd={fwd_times[-1]*1000:.1f}ms, bwd={bwd_times[-1]*1000:.1f}ms")

    avg_fwd = sum(fwd_times) / len(fwd_times)
    avg_bwd = sum(bwd_times) / len(bwd_times)
    print(f"\nHF Average: fwd={avg_fwd*1000:.1f}ms, bwd={avg_bwd*1000:.1f}ms, total={( avg_fwd+avg_bwd)*1000:.1f}ms")

    # Memory
    print(f"Peak memory: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")

    del model
    torch.cuda.empty_cache()

    return avg_fwd, avg_bwd


def benchmark_hf_with_gradient_checkpointing(batch_size=3, n_completions=16, seq_len=768, warmup=3, iters=10):
    """Benchmark HuggingFace model with gradient checkpointing."""
    print(f"\n=== HuggingFace + Gradient Checkpointing Benchmark ===")
    print(f"batch_size={batch_size}, n_completions={n_completions}, seq_len={seq_len}")

    device = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats()

    # Load model with gradient checkpointing
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
    )
    model.gradient_checkpointing_enable()
    model.train()

    # Create dummy input
    total_batch = batch_size * n_completions
    input_ids = torch.randint(0, 32000, (total_batch, seq_len), device=device)
    labels = input_ids.clone()

    # Warmup
    print(f"Warming up ({warmup} iters)...")
    for _ in range(warmup):
        outputs = model(input_ids, labels=labels)
        loss = outputs.loss
        loss.backward()
        model.zero_grad()
        torch.cuda.synchronize()

    # Benchmark
    print(f"Benchmarking ({iters} iters)...")
    fwd_times = []
    bwd_times = []

    for i in range(iters):
        torch.cuda.synchronize()

        # Forward
        t0 = time.perf_counter()
        outputs = model(input_ids, labels=labels)
        loss = outputs.loss
        torch.cuda.synchronize()
        t1 = time.perf_counter()

        # Backward
        loss.backward()
        torch.cuda.synchronize()
        t2 = time.perf_counter()

        model.zero_grad()

        fwd_times.append(t1 - t0)
        bwd_times.append(t2 - t1)
        print(f"  iter {i}: fwd={fwd_times[-1]*1000:.1f}ms, bwd={bwd_times[-1]*1000:.1f}ms")

    avg_fwd = sum(fwd_times) / len(fwd_times)
    avg_bwd = sum(bwd_times) / len(bwd_times)
    print(f"\nHF+GC Average: fwd={avg_fwd*1000:.1f}ms, bwd={avg_bwd*1000:.1f}ms, total={(avg_fwd+avg_bwd)*1000:.1f}ms")

    # Memory
    print(f"Peak memory: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")

    del model
    torch.cuda.empty_cache()

    return avg_fwd, avg_bwd


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=3)
    parser.add_argument("--n-completions", type=int, default=16)
    parser.add_argument("--seq-len", type=int, default=768)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=10)
    args = parser.parse_args()

    print("="*60)
    print("Forward/Backward Benchmark: HuggingFace vs Gradient Checkpointing")
    print("="*60)

    # Run HF without checkpointing first
    try:
        hf_fwd, hf_bwd = benchmark_hf(
            args.batch_size, args.n_completions, args.seq_len,
            args.warmup, args.iters
        )
    except torch.cuda.OutOfMemoryError:
        print("OOM without gradient checkpointing!")
        hf_fwd, hf_bwd = None, None

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    # Run HF with gradient checkpointing
    hf_gc_fwd, hf_gc_bwd = benchmark_hf_with_gradient_checkpointing(
        args.batch_size, args.n_completions, args.seq_len,
        args.warmup, args.iters
    )

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    if hf_fwd is not None:
        print(f"HF (no checkpointing): fwd={hf_fwd*1000:.1f}ms, bwd={hf_bwd*1000:.1f}ms, total={(hf_fwd+hf_bwd)*1000:.1f}ms")
    else:
        print(f"HF (no checkpointing): OOM")
    print(f"HF + grad checkpoint:  fwd={hf_gc_fwd*1000:.1f}ms, bwd={hf_gc_bwd*1000:.1f}ms, total={(hf_gc_fwd+hf_gc_bwd)*1000:.1f}ms")
