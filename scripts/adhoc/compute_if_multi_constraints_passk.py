#!/usr/bin/env python3
"""
Aggregate results from if_multi_constraints_passk_eval.py and compute pass@k.

Usage:
    python scripts/adhoc/compute_if_multi_constraints_passk.py --results-dir /efs/rlvr-experiments/experiments/if_multi_constraints_passk
"""

import argparse
import json
from pathlib import Path
from collections import defaultdict


def pass_at_k(n: int, c: int, k: int) -> float:
    """
    Compute pass@k given n total samples and c correct samples.

    Uses the unbiased estimator from the Codex paper:
    pass@k = 1 - C(n-c, k) / C(n, k)
    """
    if n - c < k:
        return 1.0
    result = 1.0
    for i in range(k):
        result *= (n - c - i) / (n - i)
    return 1.0 - result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=str, default="/efs/rlvr-experiments/experiments/if_multi_constraints_passk")
    parser.add_argument("--output-file", type=str, default=None, help="Output file for aggregated results")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)

    # Load all result files
    all_results = []
    result_files = sorted(results_dir.glob("results_gpu*.json"))

    if not result_files:
        print(f"No result files found in {results_dir}")
        return

    print(f"Found {len(result_files)} result files:")
    for f in result_files:
        print(f"  - {f.name}")
        with open(f) as fp:
            data = json.load(fp)
            all_results.extend(data)

    print(f"\nTotal samples: {len(all_results)}")

    # Compute pass@k for each sample
    k_values = [1, 2, 4, 8, 16, 32]

    # Per-sample pass@k
    sample_passk = {k: [] for k in k_values}

    for result in all_results:
        rewards = result["rewards"]
        n = len(rewards)
        c = sum(rewards)  # Number of correct completions

        for k in k_values:
            if k <= n:
                pk = pass_at_k(n, c, k)
                sample_passk[k].append(pk)

    # Average pass@k across all samples
    print("\n" + "="*50)
    print("Pass@k Results")
    print("="*50)

    passk_results = {}
    for k in k_values:
        if sample_passk[k]:
            avg_pk = sum(sample_passk[k]) / len(sample_passk[k])
            passk_results[f"pass@{k}"] = avg_pk
            print(f"pass@{k}: {avg_pk:.4f} ({avg_pk*100:.2f}%)")

    # Additional statistics
    print("\n" + "="*50)
    print("Additional Statistics")
    print("="*50)

    # Per-sample accuracy (fraction of correct completions)
    accuracies = [sum(r["rewards"]) / len(r["rewards"]) for r in all_results]
    avg_accuracy = sum(accuracies) / len(accuracies)
    print(f"Average per-sample accuracy: {avg_accuracy:.4f} ({avg_accuracy*100:.2f}%)")

    # Problems with at least one correct answer
    any_correct = sum(1 for r in all_results if sum(r["rewards"]) > 0)
    print(f"Problems with >=1 correct: {any_correct}/{len(all_results)} ({100*any_correct/len(all_results):.2f}%)")

    # Problems with all correct
    all_correct = sum(1 for r in all_results if sum(r["rewards"]) == len(r["rewards"]))
    print(f"Problems with all correct: {all_correct}/{len(all_results)} ({100*all_correct/len(all_results):.2f}%)")

    # Breakdown by constraint type
    print("\n" + "="*50)
    print("Breakdown by Constraint Type")
    print("="*50)

    constraint_stats = defaultdict(lambda: {"total": 0, "correct": 0, "any_correct": 0})
    for r in all_results:
        ct = r.get("constraint_type", "unknown")
        n_correct = sum(r["rewards"])
        constraint_stats[ct]["total"] += len(r["rewards"])
        constraint_stats[ct]["correct"] += n_correct
        if n_correct > 0:
            constraint_stats[ct]["any_correct"] += 1

    for ct in sorted(constraint_stats.keys()):
        stats = constraint_stats[ct]
        pct = 100 * stats["correct"] / stats["total"] if stats["total"] > 0 else 0
        print(f"  {ct}: {stats['correct']}/{stats['total']} ({pct:.1f}%)")

    # Save aggregated results
    output_file = args.output_file or (results_dir / "aggregated_results.json")
    aggregated = {
        "num_samples": len(all_results),
        "num_completions_per_sample": len(all_results[0]["rewards"]) if all_results else 0,
        "pass_at_k": passk_results,
        "average_accuracy": avg_accuracy,
        "problems_with_any_correct": any_correct,
        "problems_with_all_correct": all_correct,
        "constraint_type_breakdown": dict(constraint_stats),
        "results": all_results,
    }

    with open(output_file, "w") as f:
        json.dump(aggregated, f, indent=2)
    print(f"\nSaved aggregated results to {output_file}")


if __name__ == "__main__":
    main()
