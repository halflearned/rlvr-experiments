#!/usr/bin/env python3
"""
Compute pass@k for arbitrary k values from saved evaluation results.

Usage:
    python scripts/compute_pass_at_k.py results.json --k 1,8,64,256,512
    python scripts/compute_pass_at_k.py results.json --k 1-512  # range
"""

import argparse
import json
from pathlib import Path


def compute_pass_at_k(correctness_masks: list[list[bool]], k: int) -> float:
    """
    Compute pass@k: fraction of prompts with at least one correct in first k completions.
    """
    if k <= 0:
        return 0.0

    passed = 0
    for mask in correctness_masks:
        # Check if any of the first k completions is correct
        if any(mask[:k]):
            passed += 1

    return passed / len(correctness_masks) if correctness_masks else 0.0


def parse_k_values(k_str: str) -> list[int]:
    """Parse k values from string like '1,8,64' or '1-512'."""
    k_values = []
    for part in k_str.split(","):
        part = part.strip()
        if "-" in part and not part.startswith("-"):
            start, end = part.split("-")
            # Generate powers of 2 in range
            k = 1
            while k <= int(end):
                if k >= int(start):
                    k_values.append(k)
                k *= 2
        else:
            k_values.append(int(part))
    return sorted(set(k_values))


def main():
    parser = argparse.ArgumentParser(description="Compute pass@k from evaluation results")
    parser.add_argument("results_file", type=Path, help="Path to JSON results file")
    parser.add_argument(
        "--k",
        type=str,
        default="1,8,64,256,512",
        help="Comma-separated k values or range like '1-512' (default: 1,8,64,256,512)",
    )
    parser.add_argument(
        "--by-level",
        action="store_true",
        help="Also compute pass@k by level (for MATH results)",
    )
    args = parser.parse_args()

    # Load results
    with open(args.results_file) as f:
        data = json.load(f)

    # Extract correctness masks
    per_prompt = data["per_prompt_results"]

    # Check if correctness_mask exists
    if not per_prompt or "correctness_mask" not in per_prompt[0]:
        print("Error: Results file doesn't contain correctness_mask data.")
        print("Re-run the evaluation with the updated scripts to generate this data.")
        return

    correctness_masks = [p["correctness_mask"] for p in per_prompt]
    n_completions = len(correctness_masks[0]) if correctness_masks else 0

    # Parse k values
    k_values = parse_k_values(args.k)
    k_values = [k for k in k_values if k <= n_completions]

    print(f"Results: {args.results_file.name}")
    print(f"Dataset: {data['metadata'].get('dataset', 'unknown')}")
    print(f"Prompts: {len(correctness_masks)}")
    print(f"Completions per prompt: {n_completions}")
    print()

    # Compute pass@k for each k
    print("Pass@k results:")
    print("-" * 40)
    for k in k_values:
        pass_k = compute_pass_at_k(correctness_masks, k)
        num_passed = sum(1 for m in correctness_masks if any(m[:k]))
        print(f"  Pass@{k:>4}: {pass_k*100:5.1f}%  ({num_passed}/{len(correctness_masks)})")

    # By level (for MATH)
    if args.by_level and "level" in per_prompt[0]:
        print()
        print("Pass@k by level:")
        print("-" * 40)

        levels = sorted(set(p["level"] for p in per_prompt))
        for level in levels:
            level_masks = [p["correctness_mask"] for p in per_prompt if p["level"] == level]
            print(f"\n{level} ({len(level_masks)} problems):")
            for k in k_values:
                pass_k = compute_pass_at_k(level_masks, k)
                num_passed = sum(1 for m in level_masks if any(m[:k]))
                print(f"  Pass@{k:>4}: {pass_k*100:5.1f}%  ({num_passed}/{len(level_masks)})")


if __name__ == "__main__":
    main()
