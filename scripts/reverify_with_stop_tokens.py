#!/usr/bin/env python3
"""Re-verify pass@k results after truncating completions at stop tokens.

Usage:
    python -u scripts/reverify_with_stop_tokens.py <input_jsonl> <output_dir>

Streams input line-by-line to avoid loading entire JSONL into memory.
Truncates each completion at stop tokens, re-runs MathVerifier, writes
new results + summary.
"""

import json
import sys
import os
import math

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

STOP_TOKENS = ["[Question]", "Question:", "Q:", "\n\n\n", "\n\n"]


def truncate_at_stop(text: str) -> str:
    """Truncate text at the earliest stop token."""
    earliest_pos = len(text)
    for stop in STOP_TOKENS:
        pos = text.find(stop)
        if pos != -1 and pos < earliest_pos:
            earliest_pos = pos
    return text[:earliest_pos].rstrip()


def pass_at_k(n, c, k):
    """Compute pass@k from n total samples with c correct."""
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def main():
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <input_jsonl> <output_dir>")
        sys.exit(1)

    input_path = sys.argv[1]
    output_dir = sys.argv[2]
    os.makedirs(output_dir, exist_ok=True)

    # Load gold answers first (small)
    print("Loading GSM8K dataset for target answers...")
    from datasets import load_dataset
    answer_map = {}
    for split in ["train", "test"]:
        ds = load_dataset("openai/gsm8k", "main", split=split)
        offset = 0 if split == "train" else len(load_dataset("openai/gsm8k", "main", split="train"))
        for idx, row in enumerate(ds):
            pid = f"gsm8k_{offset + idx}"
            answer_map[pid] = row["answer"].split("####")[-1].strip()
    print(f"Loaded {len(answer_map)} gold answers")

    # Import verifier
    from rlvr_experiments.verifiers.math import MathVerifier
    verifier = MathVerifier(timeout=5.0, max_workers=32, warmup=True)

    # Stream through JSONL, process one record at a time
    output_jsonl = os.path.join(output_dir, "all_verification_results.jsonl")
    outf = open(output_jsonl, "w")

    total_prompts = 0
    total_correct = 0
    total_completions = 0
    truncated_count = 0
    unchanged_count = 0
    # For pass@k aggregation
    per_prompt_stats = []  # list of (n, c, pass_rate)

    print(f"Streaming {input_path}, truncating & re-verifying...")
    with open(input_path) as f:
        for line_num, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            total_prompts += 1

            pid = rec["prompt_id"]
            target = answer_map.get(pid)
            if target is None:
                print(f"  WARNING: No answer found for {pid}, skipping")
                continue

            # Truncate and re-verify each completion
            scores = []
            for comp in rec["completions"]:
                original = comp["text"]
                truncated = truncate_at_stop(original)
                if len(truncated) < len(original):
                    truncated_count += 1
                else:
                    unchanged_count += 1
                comp["text"] = truncated
                score = verifier.verify(truncated, target)
                scores.append(score)
                comp["score"] = score

            num_correct = sum(1 for s in scores if s > 0)
            n = len(scores)
            pr = num_correct / n if n else 0.0
            rec["scores"] = scores
            rec["num_correct"] = num_correct
            rec["pass_rate"] = pr

            total_correct += num_correct
            total_completions += n
            per_prompt_stats.append((n, num_correct, pr))

            # Write immediately (don't accumulate)
            outf.write(json.dumps(rec) + "\n")

            # Don't keep rec in memory - let GC collect completions
            del rec

            if total_prompts % 100 == 0 or total_prompts == 1:
                print(f"  [{total_prompts}] correct so far: {total_correct}/{total_completions} "
                      f"({100*total_correct/total_completions:.2f}%), "
                      f"truncated: {truncated_count}/{truncated_count+unchanged_count}")

    outf.close()

    # Compute summary
    num_prompts = len(per_prompt_stats)
    overall_pass_rate = total_correct / total_completions if total_completions else 0
    avg_pass_rate = sum(pr for _, _, pr in per_prompt_stats) / num_prompts if num_prompts else 0

    k_values = [1, 2, 4, 8, 16, 32, 64, 128]
    pass_at_k_results = {}
    for k in k_values:
        vals = []
        for n, c, _ in per_prompt_stats:
            if n >= k:
                vals.append(pass_at_k(n, c, k))
        if vals:
            pass_at_k_results[f"pass@{k}"] = sum(vals) / len(vals)

    truncation_stats = {"truncated": truncated_count, "unchanged": unchanged_count}
    summary = {
        "dataset": "gsm8k",
        "model": "Qwen3-1.7B-Base",
        "note": "Re-verified after truncating at stop tokens",
        "stop_tokens": STOP_TOKENS,
        "num_prompts": num_prompts,
        "num_completions": total_completions,
        "num_correct": total_correct,
        "overall_pass_rate": overall_pass_rate,
        "avg_per_prompt_pass_rate": avg_pass_rate,
        "pass_at_k": pass_at_k_results,
        "truncation_stats": truncation_stats,
    }

    summary_path = os.path.join(output_dir, "merged_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print(f"RESULTS (with stop token truncation)")
    print(f"{'='*60}")
    print(f"Prompts: {num_prompts}")
    print(f"Total completions: {total_completions}")
    print(f"Total correct: {total_correct}")
    print(f"Truncated: {truncated_count}/{truncated_count+unchanged_count} "
          f"({100*truncated_count/(truncated_count+unchanged_count):.1f}%)")
    print(f"Overall pass rate: {overall_pass_rate:.4f}")
    print(f"Avg per-prompt pass rate: {avg_pass_rate:.4f}")
    print(f"\nPass@k:")
    for k, v in pass_at_k_results.items():
        print(f"  {k}: {v:.4f}")

    # Compare with original
    orig_summary_path = os.path.join(os.path.dirname(input_path), "merged_summary.json")
    if os.path.exists(orig_summary_path):
        with open(orig_summary_path) as f:
            orig = json.load(f)
        print(f"\nComparison with original:")
        print(f"  Overall pass rate: {orig['overall_pass_rate']:.4f} -> {overall_pass_rate:.4f} "
              f"({'+' if overall_pass_rate > orig['overall_pass_rate'] else ''}"
              f"{overall_pass_rate - orig['overall_pass_rate']:.4f})")
        for k in k_values:
            key = f"pass@{k}"
            if key in orig.get("pass_at_k", {}) and key in pass_at_k_results:
                old = orig["pass_at_k"][key]
                new = pass_at_k_results[key]
                print(f"  {key}: {old:.4f} -> {new:.4f} "
                      f"({'+' if new > old else ''}{new - old:.4f})")

    print(f"\nResults written to {output_dir}")


if __name__ == "__main__":
    main()
