#!/usr/bin/env python3
"""
Generate completions for DeepScaleR dataset and compute pass@k.

Usage:
    python scripts/adhoc/deepscaler_passk_eval.py --gpu 0 --start-idx 0 --end-idx 13

This script:
1. Downloads DeepScaleR-Preview-Dataset from HuggingFace
2. Samples 100 items randomly (seeded)
3. Generates 32 completions per item using vLLM
4. Verifies with MathVerifier
5. Computes pass@k for k in [1, 2, 4, 8, 16, 32]
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, required=True, help="GPU index to use")
    parser.add_argument("--start-idx", type=int, required=True, help="Start index in the 100 samples")
    parser.add_argument("--end-idx", type=int, required=True, help="End index (exclusive)")
    parser.add_argument("--model", type=str, default="/efs/rlvr-experiments/assets/hf/Qwen3-1.7B-Base")
    parser.add_argument("--num-samples", type=int, default=100, help="Total samples to draw from dataset")
    parser.add_argument("--num-completions", type=int, default=32, help="Completions per sample")
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="/efs/rlvr-experiments/experiments/deepscaler_passk")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    from datasets import load_dataset
    from vllm import LLM, SamplingParams
    from rlvr_experiments.verifiers.math import MathVerifier

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset
    print(f"Loading DeepScaleR dataset...")
    ds = load_dataset("agentica-org/DeepScaleR-Preview-Dataset", split="train")
    print(f"Dataset size: {len(ds)}")

    # Sample 100 items with fixed seed
    random.seed(args.seed)
    indices = random.sample(range(len(ds)), args.num_samples)
    indices.sort()  # Sort for reproducibility

    # Get the subset for this worker
    worker_indices = indices[args.start_idx:args.end_idx]
    print(f"Worker processing indices {args.start_idx} to {args.end_idx} ({len(worker_indices)} samples)")

    # Initialize vLLM
    print(f"Loading model from {args.model}...")
    llm = LLM(
        model=args.model,
        dtype="float16",
        tensor_parallel_size=1,
        gpu_memory_utilization=0.9,
        max_model_len=4096,
        seed=args.seed,
    )

    sampling_params = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        n=args.num_completions,
        seed=args.seed,
    )

    # Initialize verifier
    verifier = MathVerifier(timeout=10.0, max_workers=8, warmup=True)

    # Process samples
    results = []
    for i, ds_idx in enumerate(worker_indices):
        sample = ds[ds_idx]
        problem = sample["problem"]
        answer = sample["answer"]

        print(f"\n[{args.start_idx + i + 1}/{args.end_idx}] Processing dataset index {ds_idx}...")

        # Format prompt - use simple math prompt
        prompt = f"Solve the following math problem. Show your work and put your final answer in \\boxed{{}}.\n\nProblem: {problem}\n\nSolution:"

        # Generate completions
        outputs = llm.generate([prompt], sampling_params)
        completions = [o.text for o in outputs[0].outputs]

        # Verify each completion
        rewards = verifier.verify_batch_parallel(completions, answer)

        result = {
            "dataset_idx": ds_idx,
            "problem": problem,
            "answer": answer,
            "completions": completions,
            "rewards": rewards,
        }
        results.append(result)

        # Print progress
        num_correct = sum(rewards)
        print(f"  Correct: {num_correct}/{len(rewards)} ({100*num_correct/len(rewards):.1f}%)")

    # Save results for this worker
    output_file = output_dir / f"results_gpu{args.gpu}_idx{args.start_idx}-{args.end_idx}.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {output_file}")


if __name__ == "__main__":
    main()
