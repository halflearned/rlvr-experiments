#!/usr/bin/env python3
"""Generate CSV with prompt IDs and pass@k values from IFEval results."""

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path


def comb(n: int, k: int) -> int:
    """Compute binomial coefficient C(n, k)."""
    if k < 0 or k > n:
        return 0
    if k == 0 or k == n:
        return 1
    k = min(k, n - k)
    result = 1
    for i in range(k):
        result = result * (n - i) // (i + 1)
    return result


def pass_at_k_prob(n: int, c: int, k: int) -> float:
    """Probability that at least one of k samples is correct."""
    if c == 0:
        return 0.0
    if k >= n:
        return 1.0 if c > 0 else 0.0
    if n - c < k:
        return 1.0
    return 1.0 - comb(n - c, k) / comb(n, k)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    seed_files = sorted(input_dir.glob("seed*.jsonl"))

    if not seed_files:
        print("No seed files found!")
        return

    print(f"Found {len(seed_files)} seed files")

    # Aggregate results by prompt index
    prompt_data = defaultdict(lambda: {"constraint_type": None, "scores": []})

    for seed_file in seed_files:
        with open(seed_file) as f:
            for line in f:
                data = json.loads(line)
                idx = data["idx"]
                prompt_data[idx]["constraint_type"] = data["constraint_type"]
                prompt_data[idx]["scores"].extend(data["scores"])

    print(f"Total prompts: {len(prompt_data)}")

    # Determine valid k values (powers of 2 up to min completions)
    min_completions = min(len(d["scores"]) for d in prompt_data.values())
    print(f"Completions per prompt: {min_completions}")

    k_values = [k for k in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512] if k <= min_completions]
    print(f"K values: {k_values}")

    # Write CSV
    with open(args.output, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["idx", "constraint_type", "n", "correct"] + [f"pass@{k}" for k in k_values]
        writer.writerow(header)

        for idx, data in sorted(prompt_data.items()):
            scores = data["scores"]
            n = len(scores)
            c = sum(1 for s in scores if s > 0.5)

            row = [idx, data["constraint_type"], n, c]
            for k in k_values:
                row.append(f"{pass_at_k_prob(n, c, k):.6f}")
            writer.writerow(row)

    print(f"Wrote {len(prompt_data)} rows to {args.output}")


if __name__ == "__main__":
    main()
