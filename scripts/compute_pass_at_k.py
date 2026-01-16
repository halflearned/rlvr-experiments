#!/usr/bin/env python3
"""
Compute pass@k for arbitrary k values from saved evaluation results.

Usage:
    python scripts/compute_pass_at_k.py results.json --k 1,8,64,256,512
    python scripts/compute_pass_at_k.py results.json --k 1-512  # range
    python scripts/compute_pass_at_k.py results.json --lookup math_algebra_42
    python scripts/compute_pass_at_k.py results.json --lookup math_algebra_42 --k 1,8,32
"""

import argparse
import json
from pathlib import Path


def load_results(results_file: str | Path) -> dict:
    """Load results from JSON file."""
    with open(results_file) as f:
        return json.load(f)


def get_results_by_id(data: dict) -> dict[str, dict]:
    """Build a lookup dict from prompt_id to result."""
    return {p["prompt_id"]: p for p in data["per_prompt_results"]}


def lookup_prompt(data: dict, prompt_id: str) -> dict | None:
    """Look up a specific prompt by ID."""
    by_id = get_results_by_id(data)
    return by_id.get(prompt_id)


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


def compute_prompt_stats(result: dict, k_values: list[int] = None) -> dict:
    """Compute statistics for a single prompt result."""
    mask = result.get("correctness_mask", [])
    details = result.get("completion_details", [])
    n = len(mask)

    stats = {
        "prompt_id": result["prompt_id"],
        "prompt": result["prompt"],
        "target_answer": result["target_answer"],
        "num_completions": result["num_completions"],
        "num_correct": result["num_correct"],
        "pass_rate": result["pass_rate"],
    }

    # Add optional fields
    if result.get("level"):
        stats["level"] = result["level"]
    if result.get("subject"):
        stats["subject"] = result["subject"]

    # Compute pass@k for requested values
    if k_values:
        stats["pass_at_k"] = {}
        for k in k_values:
            if k <= n:
                stats["pass_at_k"][k] = any(mask[:k])

    # Compute length stats if completion_details available
    if details:
        lengths = [d["length"] for d in details]
        correct_lengths = [d["length"] for d in details if d["correct"]]
        incorrect_lengths = [d["length"] for d in details if not d["correct"]]

        stats["length_stats"] = {
            "all": {"avg": sum(lengths)/len(lengths) if lengths else 0, "min": min(lengths) if lengths else 0, "max": max(lengths) if lengths else 0},
            "correct": {"avg": sum(correct_lengths)/len(correct_lengths) if correct_lengths else 0, "count": len(correct_lengths)},
            "incorrect": {"avg": sum(incorrect_lengths)/len(incorrect_lengths) if incorrect_lengths else 0, "count": len(incorrect_lengths)},
        }

        # Finish reason breakdown
        stop_count = sum(1 for d in details if d["finish_reason"] == "stop")
        length_count = sum(1 for d in details if d["finish_reason"] == "length")
        stats["finish_reasons"] = {"stop": stop_count, "length": length_count}

    return stats


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
    parser.add_argument(
        "--lookup",
        type=str,
        default=None,
        help="Look up a specific prompt by prompt_id",
    )
    parser.add_argument(
        "--list-ids",
        action="store_true",
        help="List all prompt IDs in the results file",
    )
    args = parser.parse_args()

    # Load results
    with open(args.results_file) as f:
        data = json.load(f)

    per_prompt = data["per_prompt_results"]

    # Handle --list-ids
    if args.list_ids:
        print(f"Prompt IDs in {args.results_file.name}:")
        print("-" * 40)
        for p in per_prompt:
            extra = ""
            if p.get("level"):
                extra = f" [{p['level']}] {p.get('subject', '')}"
            print(f"  {p['prompt_id']}{extra} - {p['pass_rate']*100:.1f}% pass rate")
        return

    # Handle --lookup
    if args.lookup:
        result = lookup_prompt(data, args.lookup)
        if result is None:
            print(f"Error: prompt_id '{args.lookup}' not found.")
            print(f"Use --list-ids to see available prompt IDs.")
            return

        k_values = parse_k_values(args.k)
        stats = compute_prompt_stats(result, k_values)

        print(f"Prompt: {stats['prompt_id']}")
        print("=" * 60)
        if stats.get("level"):
            print(f"Level: {stats['level']}")
        if stats.get("subject"):
            print(f"Subject: {stats['subject']}")
        print(f"Target answer: {stats['target_answer']}")
        print(f"\nPrompt text:\n{stats['prompt'][:500]}{'...' if len(stats['prompt']) > 500 else ''}")
        print()
        print(f"Completions: {stats['num_completions']}")
        print(f"Correct: {stats['num_correct']} ({stats['pass_rate']*100:.1f}%)")
        print()

        if stats.get("pass_at_k"):
            print("Pass@k:")
            for k, passed in stats["pass_at_k"].items():
                print(f"  Pass@{k}: {'Yes' if passed else 'No'}")

        if stats.get("length_stats"):
            ls = stats["length_stats"]
            print(f"\nCompletion lengths:")
            print(f"  All: avg={ls['all']['avg']:.0f}, min={ls['all']['min']}, max={ls['all']['max']}")
            print(f"  Correct ({ls['correct']['count']}): avg={ls['correct']['avg']:.0f}")
            print(f"  Incorrect ({ls['incorrect']['count']}): avg={ls['incorrect']['avg']:.0f}")

        if stats.get("finish_reasons"):
            fr = stats["finish_reasons"]
            print(f"\nFinish reasons: stop={fr['stop']}, length={fr['length']}")

        return

    # Default: compute pass@k for all prompts

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
