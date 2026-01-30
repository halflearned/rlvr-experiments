#!/usr/bin/env python3
"""
Build an SFT dataset from pass-at-k results.

For each prompt with pass_rate < threshold, selects the prompt and one random
correct completion. Output is a JSONL file compatible with load_sft_jsonl().

Usage:
    python scripts/build_sft_from_pass_at_k.py \
        --subset both \
        --max-pass-rate 0.5 \
        --output data/sft_hard_problems.jsonl

    # GSM8K only
    python scripts/build_sft_from_pass_at_k.py \
        --subset gsm8k \
        --output data/sft_gsm8k_hard.jsonl

    # MATH only
    python scripts/build_sft_from_pass_at_k.py \
        --subset math \
        --output data/sft_math_hard.jsonl
"""

import argparse
import json
import os
import random
from pathlib import Path


# Pass-at-k result paths
PASS_AT_K_PATHS = {
    "gsm8k": "results/qwen3-1.7B-base/evals/gsm8k/pass-at-k",
    "math": "results/qwen3-1.7B-base/evals/math/pass-at-k",
}

# Prompt reconstruction: we pull prompts from shard completions.jsonl files
# which contain (prompt_id, prompt, problem, completions)


def load_prompts_from_shards(pass_at_k_dir: str) -> dict[str, dict]:
    """Load prompt_id -> {prompt, problem} from shard completions files."""
    prompts = {}
    shard_idx = 0
    while True:
        shard_dir = os.path.join(pass_at_k_dir, f"shard_{shard_idx}")
        completions_file = os.path.join(shard_dir, "completions.jsonl")
        if not os.path.exists(completions_file):
            break
        with open(completions_file) as f:
            for line in f:
                if not line.strip():
                    continue
                item = json.loads(line)
                prompts[item["prompt_id"]] = {
                    "prompt": item["prompt"],
                    "problem": item.get("problem", {}),
                }
        shard_idx += 1
    return prompts


def load_verification_results(pass_at_k_dir: str) -> list[dict]:
    """Load all verification results from the merged file."""
    results_file = os.path.join(pass_at_k_dir, "all_verification_results.jsonl")
    results = []
    with open(results_file) as f:
        for line in f:
            if not line.strip():
                continue
            results.append(json.loads(line))
    return results


def build_sft_examples(
    dataset_name: str,
    pass_at_k_dir: str,
    max_pass_rate: float,
    min_pass_rate: float,
    rng: random.Random,
) -> list[dict]:
    """Build SFT examples for a single dataset."""
    abs_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        pass_at_k_dir,
    )

    print(f"[{dataset_name}] Loading prompts from shards...")
    prompts = load_prompts_from_shards(abs_dir)
    print(f"[{dataset_name}] Loaded {len(prompts)} prompts")

    print(f"[{dataset_name}] Loading verification results...")
    results = load_verification_results(abs_dir)
    print(f"[{dataset_name}] Loaded {len(results)} results")

    examples = []
    skipped_no_correct = 0
    skipped_no_prompt = 0

    for item in results:
        pass_rate = item["pass_rate"]
        if pass_rate < min_pass_rate or pass_rate >= max_pass_rate:
            continue

        prompt_id = item["prompt_id"]
        if prompt_id not in prompts:
            skipped_no_prompt += 1
            continue

        # Find correct completions
        correct_completions = [
            c for c, s in zip(item["completions"], item["scores"])
            if s > 0
        ]

        if not correct_completions:
            skipped_no_correct += 1
            continue

        # Pick one random correct completion
        chosen = rng.choice(correct_completions)

        prompt_info = prompts[prompt_id]

        # Build problem dict matching the dataset's verifier format
        problem = dict(prompt_info.get("problem", {}))
        problem.setdefault("prompt_id", prompt_id)
        problem.setdefault("dataset_name", dataset_name)

        examples.append({
            "prompt_id": prompt_id,
            "prompt": prompt_info["prompt"],
            "problem": problem,
            "completion": chosen["text"],
            "pass_rate": pass_rate,
            "source": dataset_name,
        })

    print(f"[{dataset_name}] Selected {len(examples)} examples "
          f"(skipped {skipped_no_correct} with no correct completion, "
          f"{skipped_no_prompt} with no prompt)")
    return examples


def main():
    parser = argparse.ArgumentParser(
        description="Build SFT dataset from pass-at-k results"
    )
    parser.add_argument(
        "--subset",
        type=str,
        default="both",
        choices=["gsm8k", "math", "both"],
        help="Which dataset(s) to include (default: both)",
    )
    parser.add_argument(
        "--max-pass-rate",
        type=float,
        default=0.5,
        help="Include problems with pass_rate < this value (default: 0.5)",
    )
    parser.add_argument(
        "--min-pass-rate",
        type=float,
        default=0.0,
        help="Exclude problems with pass_rate < this value, i.e. too hard (default: 0.0, keeps all including 0%%)",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output JSONL file path",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for completion selection (default: 42)",
    )
    args = parser.parse_args()

    rng = random.Random(args.seed)

    datasets_to_process = []
    if args.subset in ("gsm8k", "both"):
        datasets_to_process.append("gsm8k")
    if args.subset in ("math", "both"):
        datasets_to_process.append("math")

    all_examples = []
    for dataset_name in datasets_to_process:
        examples = build_sft_examples(
            dataset_name=dataset_name,
            pass_at_k_dir=PASS_AT_K_PATHS[dataset_name],
            max_pass_rate=args.max_pass_rate,
            min_pass_rate=args.min_pass_rate,
            rng=rng,
        )
        all_examples.extend(examples)

    # Shuffle
    rng.shuffle(all_examples)

    # Write output
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        for ex in all_examples:
            f.write(json.dumps(ex) + "\n")

    # Print summary
    print(f"\nSummary:")
    print(f"  Total examples: {len(all_examples)}")
    for ds in datasets_to_process:
        count = sum(1 for ex in all_examples if ex["source"] == ds)
        print(f"  {ds}: {count}")
    print(f"  Pass rate range: [{args.min_pass_rate}, {args.max_pass_rate})")
    print(f"  Output: {args.output}")


if __name__ == "__main__":
    main()
