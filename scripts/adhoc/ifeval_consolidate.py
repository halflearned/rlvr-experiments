#!/usr/bin/env python3
"""
Consolidate batched IFEval results and compute pass@k.

Reads all seed*.jsonl files from a directory and computes pass@k for any k.

Usage:
    python scripts/ifeval_consolidate.py --input-dir experiments/ifeval-curriculum
    python scripts/ifeval_consolidate.py --input-dir experiments/ifeval-curriculum --k 1,8,16,32,64,128,256,512
"""

import argparse
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
    """
    Probability that at least one of k samples is correct,
    given c correct out of n total samples.

    Uses the formula: 1 - C(n-c, k) / C(n, k)
    """
    if c == 0:
        return 0.0
    if k >= n:
        return 1.0 if c > 0 else 0.0
    if n - c < k:
        return 1.0  # Must get at least one correct

    # P(all wrong) = C(n-c, k) / C(n, k)
    return 1.0 - comb(n - c, k) / comb(n, k)


def main():
    parser = argparse.ArgumentParser(description="Consolidate IFEval results")
    parser.add_argument("--input-dir", type=str, required=True, help="Directory with seed*.jsonl files")
    parser.add_argument("--k", type=str, default="1,2,4,8,16,32,64,128,256,512",
                        help="Comma-separated k values for pass@k")
    parser.add_argument("--output", type=str, help="Output JSON file for consolidated results")
    parser.add_argument("--by-constraint", action="store_true",
                        help="Also compute pass@k by constraint type")
    args = parser.parse_args()

    k_values = [int(k.strip()) for k in args.k.split(",")]
    input_dir = Path(args.input_dir)

    print("=" * 60)
    print("IFEval Results Consolidation")
    print("=" * 60)
    print(f"Input directory: {input_dir}")
    print(f"K values: {k_values}")
    print()

    # Find all seed files
    seed_files = sorted(input_dir.glob("seed*.jsonl"))
    if not seed_files:
        print("ERROR: No seed*.jsonl files found!")
        return

    print(f"Found {len(seed_files)} seed files:")
    for f in seed_files:
        print(f"  {f.name}")
    print()

    # Aggregate results by prompt index
    # prompt_data[idx] = {"constraint_type": str, "scores": [list of all scores]}
    prompt_data = defaultdict(lambda: {"constraint_type": None, "scores": []})

    for seed_file in seed_files:
        with open(seed_file) as f:
            for line in f:
                data = json.loads(line)
                idx = data["idx"]
                prompt_data[idx]["constraint_type"] = data["constraint_type"]
                prompt_data[idx]["scores"].extend(data["scores"])

    print(f"Total prompts with results: {len(prompt_data)}")

    # Check completions per prompt
    completions_counts = [len(d["scores"]) for d in prompt_data.values()]
    min_completions = min(completions_counts)
    max_completions = max(completions_counts)
    avg_completions = sum(completions_counts) / len(completions_counts)

    print(f"Completions per prompt: min={min_completions}, max={max_completions}, avg={avg_completions:.1f}")

    # Warn if k > min_completions
    valid_k_values = [k for k in k_values if k <= min_completions]
    if len(valid_k_values) < len(k_values):
        skipped = [k for k in k_values if k > min_completions]
        print(f"\nWARNING: Skipping k values {skipped} (need more completions)")
        k_values = valid_k_values

    if not k_values:
        print("ERROR: No valid k values!")
        return

    # Compute pass@k for each prompt
    results = []
    for idx, data in sorted(prompt_data.items()):
        scores = data["scores"]
        n = len(scores)
        c = sum(1 for s in scores if s > 0.5)

        result = {
            "idx": idx,
            "constraint_type": data["constraint_type"],
            "n": n,
            "correct": c,
            "pass_rate": c / n if n > 0 else 0,
        }

        for k in k_values:
            result[f"pass@{k}"] = pass_at_k_prob(n, c, k)

        results.append(result)

    # Compute aggregate pass@k
    print("\n" + "=" * 60)
    print("PASS@K RESULTS")
    print("=" * 60)

    print(f"\nDataset: IFEval (RLVR-IFeval)")
    print(f"Prompts: {len(results)}")
    print(f"Total completions: {sum(completions_counts):,}")
    print()

    print("Pass@k (averaged across prompts):")
    aggregate_pass_at_k = {}
    for k in k_values:
        probs = [r[f"pass@{k}"] for r in results]
        pass_k = sum(probs) / len(probs)
        aggregate_pass_at_k[k] = pass_k
        print(f"  pass@{k}: {pass_k*100:.2f}%")

    # Per-completion pass rate
    total_correct = sum(r["correct"] for r in results)
    total_n = sum(r["n"] for r in results)
    avg_pass_rate = total_correct / total_n
    print(f"\nAverage per-completion pass rate: {avg_pass_rate*100:.2f}%")

    # By constraint type
    if args.by_constraint:
        print("\n" + "-" * 40)
        print("PASS@K BY CONSTRAINT TYPE")
        print("-" * 40)

        by_constraint = defaultdict(list)
        for r in results:
            by_constraint[r["constraint_type"]].append(r)

        for constraint_type in sorted(by_constraint.keys()):
            ct_results = by_constraint[constraint_type]
            print(f"\n{constraint_type} ({len(ct_results)} prompts):")
            for k in k_values[:5]:  # Just show first 5 k values
                probs = [r[f"pass@{k}"] for r in ct_results]
                pass_k = sum(probs) / len(probs)
                print(f"  pass@{k}: {pass_k*100:.1f}%")

    # Difficulty ranking (by pass@1)
    print("\n" + "-" * 40)
    print("PROMPT DIFFICULTY RANKING (by pass@1)")
    print("-" * 40)

    sorted_by_difficulty = sorted(results, key=lambda r: r.get("pass@1", r["pass_rate"]))

    print("\nHardest prompts (pass@1 = 0):")
    hard_prompts = [r for r in sorted_by_difficulty if r.get("pass@1", r["pass_rate"]) == 0]
    for r in hard_prompts[:10]:
        print(f"  idx={r['idx']}: {r['constraint_type']} (0/{r['n']} correct)")

    print(f"\nTotal prompts with pass@1=0: {len(hard_prompts)}")

    print("\nEasiest prompts:")
    for r in sorted_by_difficulty[-5:]:
        p1 = r.get("pass@1", r["pass_rate"])
        print(f"  idx={r['idx']}: {r['constraint_type']} ({r['correct']}/{r['n']} correct, pass@1={p1*100:.1f}%)")

    # Curriculum bins
    if 1 in k_values:
        print("\n" + "-" * 40)
        print("CURRICULUM DIFFICULTY BINS (by pass@1)")
        print("-" * 40)

        bins = [
            (0.0, 0.0, "impossible (0%)"),
            (0.001, 0.05, "very hard (0-5%)"),
            (0.05, 0.2, "hard (5-20%)"),
            (0.2, 0.5, "medium (20-50%)"),
            (0.5, 0.8, "easy (50-80%)"),
            (0.8, 1.0, "very easy (80-100%)"),
        ]

        for low, high, label in bins:
            if low == 0.0 and high == 0.0:
                count = len([r for r in results if r.get("pass@1", 0) == 0])
            else:
                count = len([r for r in results if low < r.get("pass@1", 0) <= high])
            print(f"  {label}: {count} prompts")

    # Save output
    if args.output:
        output_data = {
            "metadata": {
                "input_dir": str(input_dir),
                "num_prompts": len(results),
                "total_completions": sum(completions_counts),
                "completions_per_prompt_range": [min_completions, max_completions],
            },
            "pass_at_k": aggregate_pass_at_k,
            "per_prompt_results": results,
        }

        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
