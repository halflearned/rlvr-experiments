#!/usr/bin/env python3
"""
Estimate maximum (B, T) batch size before OOM for GRPO training.

This script simulates the forward-backward memory usage for different batch sizes
to find the maximum we can use without OOM.

Usage:
    # Run on single GPU first to get baseline
    python scripts/estimate_max_batch.py --T 768 --tp 1

    # Run with tensor parallelism
    torchrun --nproc_per_node=4 scripts/estimate_max_batch.py --T 768 --tp 4
"""

import argparse
import gc
import os

import torch
import torch.distributed as dist


def get_device():
    if torch.cuda.is_available():
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        return torch.device(f"cuda:{local_rank}")
    return torch.device("cpu")


def get_mem_gb():
    """Get current GPU memory usage in GB."""
    if not torch.cuda.is_available():
        return 0.0
    torch.cuda.synchronize()
    return torch.cuda.memory_allocated() / 1e9


def get_mem_reserved_gb():
    """Get reserved GPU memory in GB."""
    if not torch.cuda.is_available():
        return 0.0
    torch.cuda.synchronize()
    return torch.cuda.memory_reserved() / 1e9


def get_max_mem_gb():
    """Get peak GPU memory in GB."""
    if not torch.cuda.is_available():
        return 0.0
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / 1e9


def reset_peak():
    """Reset peak memory stats."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def cleanup():
    """Force memory cleanup."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def simulate_grpo_fwd_bwd(
    B: int,
    T: int,
    vocab_size: int,
    hidden_size: int,
    tp: int = 1,
    use_loss_parallel: bool = True,
    device: torch.device = None,
):
    """
    Simulate GRPO forward-backward memory usage.

    In GRPO, we have:
    - logits: [B, seq_len, vocab] where seq_len = prompt + completion
    - But we only compute loss on completion tokens of length T

    For simplicity, we assume seq_len ≈ T (completion-only or prompt is small).

    With loss_parallel:
    - logits are sharded on vocab dimension: [B, T, vocab // tp]
    - No all-gather needed for cross_entropy

    Without loss_parallel:
    - Need full logits: [B, T, vocab]
    """
    if device is None:
        device = get_device()

    # Simulate sharded vocab for TP
    local_vocab = vocab_size // tp if use_loss_parallel else vocab_size

    # Hidden states (input to lm_head) - always replicated
    hidden = torch.randn(B, T, hidden_size, device=device, requires_grad=True)

    # Simulate lm_head weight (sharded for TP)
    lm_head = torch.nn.Linear(hidden_size, local_vocab, bias=False, device=device)

    # Forward: compute logits
    logits = lm_head(hidden)  # [B, T, local_vocab]

    # Simulate cross_entropy loss (this is where loss_parallel helps)
    targets = torch.randint(0, vocab_size, (B, T), device=device)

    if use_loss_parallel:
        # With loss_parallel, targets are modded to local vocab range
        # This is a simplification - actual loss_parallel does distributed softmax
        local_targets = targets % local_vocab
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, local_vocab),
            local_targets.reshape(-1),
        )
    else:
        # Without loss_parallel, need to all-gather logits
        # Simulate by using full vocab (this is the memory bottleneck)
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, local_vocab),
            (targets % local_vocab).reshape(-1),
        )

    # Backward
    loss.backward()

    return loss.item()


def find_max_batch_size(
    T: int,
    vocab_size: int,
    hidden_size: int,
    tp: int,
    use_loss_parallel: bool,
    max_mem_gb: float = 70.0,  # Leave some headroom from 80GB
    device: torch.device = None,
):
    """Binary search for maximum batch size."""
    if device is None:
        device = get_device()

    rank = int(os.environ.get("RANK", 0))

    # Start with a small batch and increase
    low, high = 1, 2048
    best = 0

    while low <= high:
        mid = (low + high) // 2

        cleanup()
        reset_peak()

        try:
            simulate_grpo_fwd_bwd(
                B=mid,
                T=T,
                vocab_size=vocab_size,
                hidden_size=hidden_size,
                tp=tp,
                use_loss_parallel=use_loss_parallel,
                device=device,
            )

            peak = get_max_mem_gb()

            if rank == 0:
                print(f"  B={mid:4d}: peak={peak:.2f} GB", flush=True)

            if peak < max_mem_gb:
                best = mid
                low = mid + 1
            else:
                high = mid - 1

        except torch.cuda.OutOfMemoryError:
            if rank == 0:
                print(f"  B={mid:4d}: OOM", flush=True)
            high = mid - 1
            cleanup()

        cleanup()

    return best


def estimate_memory_formula(
    B: int,
    T: int,
    vocab_size: int,
    hidden_size: int,
    tp: int,
    use_loss_parallel: bool,
) -> float:
    """
    Estimate peak memory in GB using analytical formula.

    Main memory consumers during GRPO forward-backward:
    1. Hidden states: B * T * hidden_size * 4 bytes (float32)
    2. Logits: B * T * (vocab / tp) * 4 bytes (with loss_parallel)
       OR: B * T * vocab * 4 bytes (without loss_parallel)
    3. Gradients: same as logits
    4. Intermediate activations: ~2x logits for backward

    This is a rough estimate - actual memory varies based on:
    - Activation checkpointing
    - Mixed precision
    - PyTorch memory fragmentation
    """
    bytes_per_elem = 4  # float32

    # Hidden states + gradients
    hidden_mem = 2 * B * T * hidden_size * bytes_per_elem

    # Logits
    local_vocab = vocab_size // tp if use_loss_parallel else vocab_size
    logits_mem = B * T * local_vocab * bytes_per_elem

    # Gradients for logits
    logits_grad_mem = logits_mem

    # Intermediate activations (rough estimate: 2x logits for softmax, etc.)
    activations_mem = 2 * logits_mem

    total_bytes = hidden_mem + logits_mem + logits_grad_mem + activations_mem
    return total_bytes / 1e9


def main():
    parser = argparse.ArgumentParser(description="Estimate max batch size for GRPO")
    parser.add_argument("--T", type=int, default=768, help="Completion length")
    parser.add_argument("--tp", type=int, default=1, help="Tensor parallelism degree")
    parser.add_argument("--vocab-size", type=int, default=151936, help="Vocabulary size")
    parser.add_argument("--hidden-size", type=int, default=2048, help="Hidden dimension")
    parser.add_argument("--max-mem-gb", type=float, default=70.0, help="Max memory to use (GB)")
    parser.add_argument("--test-B", type=int, nargs="+", help="Specific batch sizes to test")
    args = parser.parse_args()

    # Initialize distributed if running with torchrun
    if "RANK" in os.environ:
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
    else:
        rank = 0
        local_rank = 0

    device = get_device()

    if rank == 0:
        print(f"=" * 60)
        print(f"Estimating max batch size for GRPO")
        print(f"  T (completion length): {args.T}")
        print(f"  TP: {args.tp}")
        print(f"  Vocab size: {args.vocab_size}")
        print(f"  Hidden size: {args.hidden_size}")
        print(f"  Max memory target: {args.max_mem_gb} GB")
        print(f"=" * 60)

    # Test specific batch sizes if provided
    if args.test_B:
        if rank == 0:
            print(f"\nTesting specific batch sizes: {args.test_B}")

        for use_lp in [True, False]:
            if rank == 0:
                mode = "WITH loss_parallel" if use_lp else "WITHOUT loss_parallel"
                print(f"\n{mode}:")

            for B in args.test_B:
                cleanup()
                reset_peak()

                try:
                    simulate_grpo_fwd_bwd(
                        B=B,
                        T=args.T,
                        vocab_size=args.vocab_size,
                        hidden_size=args.hidden_size,
                        tp=args.tp,
                        use_loss_parallel=use_lp,
                        device=device,
                    )
                    peak = get_max_mem_gb()
                    est = estimate_memory_formula(
                        B=B,
                        T=args.T,
                        vocab_size=args.vocab_size,
                        hidden_size=args.hidden_size,
                        tp=args.tp,
                        use_loss_parallel=use_lp,
                    )
                    if rank == 0:
                        print(f"  B={B:4d}: peak={peak:.2f} GB (est={est:.2f} GB)")
                except torch.cuda.OutOfMemoryError:
                    if rank == 0:
                        print(f"  B={B:4d}: OOM")
                    cleanup()

                cleanup()
    else:
        # Binary search for max batch size
        for use_lp in [True, False]:
            if rank == 0:
                mode = "WITH loss_parallel" if use_lp else "WITHOUT loss_parallel"
                print(f"\n{mode}:")

            max_B = find_max_batch_size(
                T=args.T,
                vocab_size=args.vocab_size,
                hidden_size=args.hidden_size,
                tp=args.tp,
                use_loss_parallel=use_lp,
                max_mem_gb=args.max_mem_gb,
                device=device,
            )

            if rank == 0:
                print(f"  -> Max B = {max_B}")

                # Show formula estimate for comparison
                if max_B > 0:
                    est = estimate_memory_formula(
                        B=max_B,
                        T=args.T,
                        vocab_size=args.vocab_size,
                        hidden_size=args.hidden_size,
                        tp=args.tp,
                        use_loss_parallel=use_lp,
                    )
                    print(f"     Formula estimate: {est:.2f} GB")

    # Summary
    if rank == 0:
        print(f"\n" + "=" * 60)
        print("Memory formula (approximate):")
        print("  With loss_parallel:    ~4 * B * T * (V/TP) * 4 bytes")
        print("  Without loss_parallel: ~4 * B * T * V * 4 bytes")
        print(f"  V={args.vocab_size}, TP={args.tp}")
        local_vocab = args.vocab_size // args.tp
        print(f"  Local vocab = V/TP = {local_vocab}")
        print(f"\n  For B=64, T={args.T}:")
        with_lp = 4 * 64 * args.T * local_vocab * 4 / 1e9
        without_lp = 4 * 64 * args.T * args.vocab_size * 4 / 1e9
        print(f"    With loss_parallel:    ~{with_lp:.2f} GB")
        print(f"    Without loss_parallel: ~{without_lp:.2f} GB")
        print(f"    Savings: {without_lp - with_lp:.2f} GB ({100*(1 - with_lp/without_lp):.0f}%)")
        print(f"=" * 60)

    if "RANK" in os.environ:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
