#!/usr/bin/env python3
"""Merge pass@512 shard results into a single merged_summary.json."""
import json
import math
import sys
from pathlib import Path


def pass_at_k_estimator(n, c, k):
    """Unbiased pass@k estimator."""
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def merge(output_dir: str, benchmark: str):
    output_dir = Path(output_dir)
    all_results = []

    for shard_dir in sorted(output_dir.glob("shard_*")):
        if not shard_dir.is_dir():
            continue
        results_file = shard_dir / "verification_results.jsonl"
        if results_file.exists():
            with open(results_file) as f:
                for line in f:
                    if line.strip():
                        all_results.append(json.loads(line))

    if not all_results:
        print(f"No results found in {output_dir}!")
        return

    n = all_results[0].get("num_completions", len(all_results[0].get("scores", [])))
    pass_at_k = {}
    for k in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]:
        if k > n:
            break
        estimates = []
        for r in all_results:
            c = r.get("num_correct", sum(1 for s in r.get("scores", []) if s > 0))
            estimates.append(pass_at_k_estimator(n, c, k))
        pass_at_k[f"pass@{k}"] = sum(estimates) / len(estimates)

    total_correct = sum(
        r.get("num_correct", sum(1 for s in r.get("scores", []) if s > 0))
        for r in all_results
    )
    total_completions = sum(
        r.get("num_completions", len(r.get("scores", [])))
        for r in all_results
    )

    summary = {
        "benchmark": benchmark,
        "num_prompts": len(all_results),
        "num_completions_per_prompt": n,
        "num_completions": total_completions,
        "num_correct": total_correct,
        "overall_pass_rate": total_correct / total_completions if total_completions else 0,
        "avg_per_prompt_pass_rate": sum(
            r.get("pass_rate", r.get("num_correct", 0) / max(r.get("num_completions", 1), 1))
            for r in all_results
        ) / len(all_results),
        "pass_at_k": pass_at_k,
    }

    # Write merged verification_results
    merged_vr = output_dir / "verification_results.jsonl"
    with open(merged_vr, "w") as f:
        for r in all_results:
            f.write(json.dumps(r) + "\n")

    # Write summary
    summary_file = output_dir / "merged_summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Merged {len(all_results)} prompts for {benchmark}")
    print(f"Pass@k:")
    for k, rate in pass_at_k.items():
        print(f"  {k}: {rate*100:.2f}%")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <output_dir> <benchmark>")
        sys.exit(1)
    merge(sys.argv[1], sys.argv[2])
