#!/usr/bin/env python
"""Benchmark forward/backward pass with TitanModel to understand performance."""

import os
import sys
import time
import torch
import torch.nn.functional as F

# Force profiling on
os.environ["RLVR_PROFILE_TITAN"] = "1"

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=3, help="Number of prompts")
    parser.add_argument("--n-completions", type=int, default=16, help="Completions per prompt")
    parser.add_argument("--seq-len", type=int, default=768, help="Sequence length")
    parser.add_argument("--completion-len", type=int, default=512, help="Completion length")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--config", type=str, default="/efs/rlvr-experiments/.rlvr_titan_job_configs/tmp1bwpvbin.trainer.toml")
    parser.add_argument("--use-grpo-loss", action="store_true", help="Use GRPOLoss instead of simple CE loss")
    args = parser.parse_args()

    total_batch = args.batch_size * args.n_completions
    print(f"=== TitanModel Benchmark ===")
    print(f"batch_size={args.batch_size}, n_completions={args.n_completions}, seq_len={args.seq_len}")
    print(f"Total sequences: {total_batch}, Total tokens: {total_batch * args.seq_len}")
    print(f"Loss function: {'GRPOLoss' if args.use_grpo_loss else 'simple cross-entropy'}")

    # Setup distributed env for single GPU
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29501'
    os.environ['RANK'] = '0'
    os.environ['WORLD_SIZE'] = '1'
    os.environ['LOCAL_RANK'] = '0'

    from rlvr_experiments.model import TitanModel
    from torchtitan.config import ConfigManager

    print(f"\nLoading config from: {args.config}")
    job_config = ConfigManager().parse_args(["--job.config-file", args.config])

    print("Creating TitanModel...")
    model = TitanModel(job_config, trainable=True)
    print(f"Model loaded on device: {model.device}")

    # Create dummy input
    input_ids = torch.randint(0, 32000, (total_batch, args.seq_len), device=model.device)

    if args.use_grpo_loss:
        from rlvr_experiments.losses import GRPOLoss
        loss_fn = GRPOLoss(beta=0.01, eps=0.2)

        # Create dummy GRPO inputs
        prompt_len = args.seq_len - args.completion_len
        completion_ids = input_ids[:, prompt_len:].clone()
        ref_logprobs = torch.randn(total_batch, args.completion_len, device=model.device)
        rollout_logprobs = torch.randn(total_batch, args.completion_len, device=model.device)
        rewards = torch.randn(total_batch, device=model.device)
        padding_mask = torch.ones(total_batch, args.completion_len, device=model.device)
        prompt_lens = torch.full((total_batch,), prompt_len, dtype=torch.long, device=model.device)

        def grpo_loss_wrapper(logits):
            return loss_fn(logits, completion_ids, ref_logprobs, rollout_logprobs,
                          rewards, padding_mask, prompt_lens)

        loss_func = grpo_loss_wrapper
    else:
        # Simple cross-entropy loss function (similar to what GRPOLoss does internally)
        def simple_loss(logits):
            # Shift logits and labels for next-token prediction
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = input_ids[:, 1:].contiguous()
            # Flatten
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                reduction='mean'
            )
            return loss

        loss_func = simple_loss

    # Warmup
    print(f"\nWarming up ({args.warmup} iters)...")
    torch.cuda.reset_peak_memory_stats()
    for i in range(args.warmup):
        torch.cuda.synchronize()
        t0 = time.perf_counter()

        # Forward pass
        logits = model.forward(input_ids)
        torch.cuda.synchronize()
        t1 = time.perf_counter()

        # Loss
        loss = loss_func(logits)
        torch.cuda.synchronize()
        t2 = time.perf_counter()

        # Backward
        loss.backward()
        torch.cuda.synchronize()
        t3 = time.perf_counter()

        # Optimizer step
        model.optim_step()
        torch.cuda.synchronize()
        t4 = time.perf_counter()

        print(f"  warmup {i}: fwd={1000*(t1-t0):.0f}ms, loss={1000*(t2-t1):.0f}ms, bwd={1000*(t3-t2):.0f}ms, optim={1000*(t4-t3):.0f}ms")

    # Benchmark
    print(f"\nBenchmarking ({args.iters} iters)...")
    fwd_times = []
    loss_times = []
    bwd_times = []
    optim_times = []

    for i in range(args.iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()

        # Forward
        logits = model.forward(input_ids)
        torch.cuda.synchronize()
        t1 = time.perf_counter()

        # Loss computation
        loss = loss_func(logits)
        torch.cuda.synchronize()
        t2 = time.perf_counter()

        # Backward
        loss.backward()
        torch.cuda.synchronize()
        t3 = time.perf_counter()

        # Optimizer step
        model.optim_step()
        torch.cuda.synchronize()
        t4 = time.perf_counter()

        fwd_times.append(t1 - t0)
        loss_times.append(t2 - t1)
        bwd_times.append(t3 - t2)
        optim_times.append(t4 - t3)

        print(f"  iter {i}: fwd={fwd_times[-1]*1000:.0f}ms, loss={loss_times[-1]*1000:.0f}ms, bwd={bwd_times[-1]*1000:.0f}ms, optim={optim_times[-1]*1000:.0f}ms, total={(t4-t0)*1000:.0f}ms")

    avg_fwd = sum(fwd_times) / len(fwd_times)
    avg_loss = sum(loss_times) / len(loss_times)
    avg_bwd = sum(bwd_times) / len(bwd_times)
    avg_optim = sum(optim_times) / len(optim_times)

    print(f"\n=== Results ===")
    print(f"Average: fwd={avg_fwd*1000:.0f}ms, loss={avg_loss*1000:.0f}ms, bwd={avg_bwd*1000:.0f}ms, optim={avg_optim*1000:.0f}ms, total={(avg_fwd+avg_loss+avg_bwd+avg_optim)*1000:.0f}ms")
    print(f"Peak memory: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")

    # Calculate approximate MFU
    # For a 0.6B model with bfloat16:
    # - Forward FLOPs ≈ 2 * model_params * tokens = 2 * 0.6e9 * total_batch * seq_len
    # - Backward FLOPs ≈ 4 * model_params * tokens (2x forward for grads)
    model_params = 0.6e9  # 0.6B
    tokens = total_batch * args.seq_len
    fwd_flops = 2 * model_params * tokens
    bwd_flops = 4 * model_params * tokens
    total_flops = fwd_flops + bwd_flops

    total_time = avg_fwd + avg_loss + avg_bwd
    achieved_tflops = total_flops / total_time / 1e12

    # A100 80GB peak is ~312 TFLOPS for bfloat16
    peak_tflops = 312
    mfu = 100 * achieved_tflops / peak_tflops

    print(f"\nEstimated performance:")
    print(f"  Total FLOPs: {total_flops/1e12:.2f} TFLOPs")
    print(f"  Achieved: {achieved_tflops:.1f} TFLOPS")
    print(f"  MFU: {mfu:.1f}% (of A100 80GB peak {peak_tflops} TFLOPS)")


if __name__ == "__main__":
    main()
