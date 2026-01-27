#!/usr/bin/env python3
"""
Verify lm_eval samples using MathVerifier and save results with rewards.

Takes lm_eval sample JSONL files and runs MathVerifier on each completion,
saving full results with rewards.

Usage:
    python scripts/adhoc/verify_lm_eval_samples.py \
        <input_dir> \
        <output_dir> \
        [--workers N]
"""

import argparse
import json
import sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from rlvr_experiments.verifiers.math import MathVerifier


def load_samples(sample_file: Path) -> list[dict]:
    """Load samples from a JSONL file."""
    samples = []
    with open(sample_file, "r") as f:
        for line in f:
            samples.append(json.loads(line))
    return samples


def extract_completion_and_answer(sample: dict) -> tuple[str, str]:
    """Extract completion and gold answer from a sample."""
    # Get completion - use filtered_resps if available, else resps
    if "filtered_resps" in sample and sample["filtered_resps"]:
        completion = sample["filtered_resps"][0]
    elif "resps" in sample and sample["resps"]:
        completion = sample["resps"][0][0] if isinstance(sample["resps"][0], list) else sample["resps"][0]
    else:
        completion = ""

    # Get gold answer - try target first, then doc.answer
    if "target" in sample:
        answer = sample["target"]
    elif "doc" in sample and "answer" in sample["doc"]:
        answer = sample["doc"]["answer"]
    else:
        answer = ""

    return completion, answer


def verify_batch(items: list[tuple[int, str, str]], timeout: float = 5.0) -> list[tuple[int, float]]:
    """Verify a batch of (index, completion, answer) tuples."""
    verifier = MathVerifier(timeout=timeout, max_workers=1)
    results = []
    for idx, completion, answer in items:
        reward = verifier.verify(completion, answer)
        results.append((idx, reward))
    return results


def main():
    parser = argparse.ArgumentParser(description="Verify lm_eval samples with MathVerifier")
    parser.add_argument("input_dir", type=str, help="Directory containing sample JSONL files")
    parser.add_argument("output_dir", type=str, help="Output directory for verified results")
    parser.add_argument("--workers", type=int, default=8, help="Number of parallel workers")
    parser.add_argument("--batch-size", type=int, default=50, help="Batch size per worker")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all sample files
    sample_files = list(input_dir.glob("samples_*.jsonl"))
    if not sample_files:
        # Check subdirectories (lm_eval creates nested dirs)
        sample_files = list(input_dir.glob("*/samples_*.jsonl"))

    if not sample_files:
        print(f"No sample files found in {input_dir}")
        return

    print(f"Found {len(sample_files)} sample files")

    verifier = MathVerifier(timeout=5.0, max_workers=args.workers)

    all_results = []
    total_correct = 0
    total_samples = 0

    for sample_file in sorted(sample_files):
        task_name = sample_file.stem.replace("samples_", "").rsplit("_", 1)[0]  # Remove timestamp
        print(f"\nProcessing {task_name}...")

        samples = load_samples(sample_file)
        print(f"  Loaded {len(samples)} samples")

        # Extract completions and answers
        items = []
        for i, sample in enumerate(samples):
            completion, answer = extract_completion_and_answer(sample)
            items.append((i, completion, answer, sample))

        # Verify all samples
        task_results = []
        task_correct = 0

        for i, completion, answer, sample in tqdm(items, desc=f"  Verifying {task_name}"):
            reward = verifier.verify(completion, answer)

            result = {
                "task": task_name,
                "doc_id": sample.get("doc_id", i),
                "prompt": sample.get("arguments", {}).get("gen_args_0", {}).get("arg_0", ""),
                "completion": completion,
                "gold_answer": answer,
                "reward": reward,
                "correct": reward > 0,
                "exact_match": sample.get("exact_match", 0),
                "problem": sample.get("doc", {}).get("problem", ""),
                "level": sample.get("doc", {}).get("level", ""),
                "type": sample.get("doc", {}).get("type", ""),
            }
            task_results.append(result)

            if reward > 0:
                task_correct += 1

        task_accuracy = task_correct / len(items) if items else 0
        print(f"  {task_name}: {task_correct}/{len(items)} = {task_accuracy:.2%}")

        all_results.extend(task_results)
        total_correct += task_correct
        total_samples += len(items)

    # Save all results
    output_file = output_dir / "verified_samples.jsonl"
    with open(output_file, "w") as f:
        for result in all_results:
            f.write(json.dumps(result) + "\n")
    print(f"\nSaved {len(all_results)} verified samples to {output_file}")

    # Compute and save summary
    overall_accuracy = total_correct / total_samples if total_samples else 0

    # Group by task
    by_task = {}
    for r in all_results:
        task = r["task"]
        if task not in by_task:
            by_task[task] = {"correct": 0, "total": 0, "exact_match": 0}
        by_task[task]["total"] += 1
        if r["correct"]:
            by_task[task]["correct"] += 1
        by_task[task]["exact_match"] += r.get("exact_match", 0)

    # Group by level (for hendrycks_math)
    by_level = {}
    for r in all_results:
        level = r.get("level", "unknown")
        if level not in by_level:
            by_level[level] = {"correct": 0, "total": 0}
        by_level[level]["total"] += 1
        if r["correct"]:
            by_level[level]["correct"] += 1

    summary = {
        "total_samples": total_samples,
        "total_correct": total_correct,
        "overall_accuracy": overall_accuracy,
        "by_task": {
            task: {
                "total": stats["total"],
                "correct": stats["correct"],
                "accuracy": stats["correct"] / stats["total"] if stats["total"] else 0,
                "exact_match_count": stats["exact_match"],
                "exact_match_acc": stats["exact_match"] / stats["total"] if stats["total"] else 0,
            }
            for task, stats in sorted(by_task.items())
        },
        "by_level": {
            level: {
                "total": stats["total"],
                "correct": stats["correct"],
                "accuracy": stats["correct"] / stats["total"] if stats["total"] else 0,
            }
            for level, stats in sorted(by_level.items())
        },
    }

    summary_file = output_dir / "summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary to {summary_file}")

    # Print summary
    print(f"\n{'='*60}")
    print(f"OVERALL: {total_correct}/{total_samples} = {overall_accuracy:.2%}")
    print(f"{'='*60}")
    print(f"\nBy Task:")
    for task, stats in sorted(by_task.items()):
        acc = stats["correct"] / stats["total"] if stats["total"] else 0
        em_acc = stats["exact_match"] / stats["total"] if stats["total"] else 0
        print(f"  {task}: {stats['correct']}/{stats['total']} = {acc:.2%} (exact_match: {em_acc:.2%})")

    print(f"\nBy Level:")
    for level, stats in sorted(by_level.items()):
        if stats["total"] > 0:
            acc = stats["correct"] / stats["total"]
            print(f"  {level}: {stats['correct']}/{stats['total']} = {acc:.2%}")


if __name__ == "__main__":
    main()
