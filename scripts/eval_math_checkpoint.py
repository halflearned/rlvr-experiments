#!/usr/bin/env python3
"""
Evaluate a checkpoint on MATH test set using vLLM for generation and MathVerifier.

Usage:
    python scripts/eval_math_checkpoint.py <checkpoint_path> <output_dir> [--gpu GPU]

Example:
    python scripts/eval_math_checkpoint.py \
        results/qwen3-1.7B-math-lr5e6_20260125-101019/checkpoints/qwen3-1.7B-math-lr5e6_20260125-101019_final \
        results/qwen3-1.7B-math-lr5e6_20260125-101019/evals \
        --gpu 0
"""

import argparse
import asyncio
import json
import os
import time
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm
from vllm import LLM, SamplingParams

from rlvr_experiments.verifiers.math import MathVerifier


def load_math_test():
    """Load MATH test split from HuggingFace."""
    subjects = [
        "algebra",
        "counting_and_probability",
        "geometry",
        "intermediate_algebra",
        "number_theory",
        "prealgebra",
        "precalculus",
    ]

    all_rows = []
    for subject in subjects:
        ds = load_dataset("EleutherAI/hendrycks_math", subject, split="test")
        for row in ds:
            all_rows.append({
                "problem": row["problem"],
                "solution": row["solution"],
                "level": row["level"],
                "type": subject,
            })

    print(f"[load_math_test] Loaded {len(all_rows)} test problems")
    return all_rows


def extract_answer_from_solution(solution: str) -> str:
    """Extract the boxed answer from solution text."""
    # Look for \boxed{...}
    import re
    matches = re.findall(r'\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}', solution)
    if matches:
        return matches[-1]  # Return last match
    return solution.strip()


def main():
    parser = argparse.ArgumentParser(description="Evaluate checkpoint on MATH test set")
    parser.add_argument("checkpoint_path", type=str, help="Path to checkpoint")
    parser.add_argument("output_dir", type=str, help="Output directory for results")
    parser.add_argument("--gpu", type=int, default=0, help="GPU to use")
    parser.add_argument("--max-tokens", type=int, default=2048, help="Max tokens for generation")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for generation")
    args = parser.parse_args()

    # Set GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load test data
    print("[main] Loading MATH test set...")
    test_data = load_math_test()

    # Prepare prompts
    prompts = []
    problems = []
    for row in test_data:
        # Use simple prompt format (no chat template, matching training)
        prompt = f"Problem: {row['problem']}\n\nSolution:"
        prompts.append(prompt)
        problems.append({
            "prompt": prompt,
            "problem": row["problem"],
            "answer": extract_answer_from_solution(row["solution"]),
            "solution": row["solution"],
            "level": row["level"],
            "type": row["type"],
        })

    # Initialize vLLM with greedy sampling
    print(f"[main] Loading model from {args.checkpoint_path}...")
    llm = LLM(
        model=args.checkpoint_path,
        tensor_parallel_size=1,
        dtype="bfloat16",
        gpu_memory_utilization=0.9,
        max_model_len=4096,
        trust_remote_code=True,
    )

    sampling_params = SamplingParams(
        temperature=0,  # Greedy
        max_tokens=args.max_tokens,
    )

    # Generate completions
    print(f"[main] Generating completions for {len(prompts)} problems...")
    t0 = time.time()
    outputs = llm.generate(prompts, sampling_params)
    gen_time = time.time() - t0
    print(f"[main] Generation took {gen_time:.1f}s ({len(prompts)/gen_time:.1f} problems/s)")

    # Extract completions
    completions = [out.outputs[0].text for out in outputs]

    # Verify with MathVerifier
    print("[main] Verifying completions...")
    verifier = MathVerifier(timeout=5.0, max_workers=8)

    results = []
    correct = 0
    for i, (problem, completion) in enumerate(tqdm(zip(problems, completions), total=len(problems))):
        reward = verifier.verify(completion, problem["answer"])
        correct += int(reward > 0.5)
        results.append({
            "idx": i,
            "problem": problem["problem"],
            "answer": problem["answer"],
            "solution": problem["solution"],
            "level": problem["level"],
            "type": problem["type"],
            "completion": completion,
            "reward": reward,
        })

    accuracy = correct / len(results)
    print(f"[main] Accuracy: {correct}/{len(results)} = {accuracy:.2%}")

    # Compute per-level and per-type accuracy
    level_stats = {}
    type_stats = {}
    for r in results:
        level = r["level"]
        ptype = r["type"]

        if level not in level_stats:
            level_stats[level] = {"correct": 0, "total": 0}
        level_stats[level]["total"] += 1
        level_stats[level]["correct"] += int(r["reward"] > 0.5)

        if ptype not in type_stats:
            type_stats[ptype] = {"correct": 0, "total": 0}
        type_stats[ptype]["total"] += 1
        type_stats[ptype]["correct"] += int(r["reward"] > 0.5)

    # Save results
    completions_file = output_dir / "math_test_completions.jsonl"
    with open(completions_file, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"[main] Saved completions to {completions_file}")

    # Save summary
    summary = {
        "checkpoint": args.checkpoint_path,
        "total": len(results),
        "correct": correct,
        "accuracy": accuracy,
        "generation_time_s": gen_time,
        "by_level": {
            level: {
                "correct": stats["correct"],
                "total": stats["total"],
                "accuracy": stats["correct"] / stats["total"] if stats["total"] > 0 else 0,
            }
            for level, stats in sorted(level_stats.items())
        },
        "by_type": {
            ptype: {
                "correct": stats["correct"],
                "total": stats["total"],
                "accuracy": stats["correct"] / stats["total"] if stats["total"] > 0 else 0,
            }
            for ptype, stats in sorted(type_stats.items())
        },
    }

    summary_file = output_dir / "math_test_summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[main] Saved summary to {summary_file}")

    # Print summary table
    print("\n=== Results by Level ===")
    for level, stats in sorted(level_stats.items()):
        acc = stats["correct"] / stats["total"] if stats["total"] > 0 else 0
        print(f"  {level}: {stats['correct']}/{stats['total']} = {acc:.2%}")

    print("\n=== Results by Type ===")
    for ptype, stats in sorted(type_stats.items()):
        acc = stats["correct"] / stats["total"] if stats["total"] > 0 else 0
        print(f"  {ptype}: {stats['correct']}/{stats['total']} = {acc:.2%}")


if __name__ == "__main__":
    main()
