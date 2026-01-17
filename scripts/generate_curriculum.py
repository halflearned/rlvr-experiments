#!/usr/bin/env python3
"""Generate curriculum files from pass@k CSVs.

Now that data.py uses simple numeric indices (gsm8k_0, math_0, ifeval_0, etc.),
curriculum generation is straightforward - just read the CSV and sort by pass@k.

For IFEval, the CSV was generated with shuffle_seed=42, so we need to map
shuffled indices back to original dataset indices.

Output: Text files with one prompt_id per line, sorted by pass@k (ascending by default).
"""

import argparse
import csv
import random
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def build_ifeval_shuffle_map(shuffle_seed: int = 42, num_prompts: int = 3200) -> dict[int, int]:
    """Build mapping from shuffled index to original dataset index.

    The IFEval batched generation was run with --shuffle-seed 42 --num-prompts 3200,
    which shuffles the dataset deterministically and takes the first 3200.

    Returns: dict mapping shuffled_idx -> original_idx
    """
    from datasets import load_dataset

    print(f"[ifeval] Building shuffle map (seed={shuffle_seed}, n={num_prompts})...")
    hf_dataset = load_dataset("allenai/RLVR-IFeval", split="train")

    # Create list of original indices
    orig_indices = list(range(len(hf_dataset)))

    # Shuffle with same seed as batched_gen.py
    random.seed(shuffle_seed)
    random.shuffle(orig_indices)

    # Take first num_prompts
    selected = orig_indices[:num_prompts]

    # Build map: shuffled_idx -> original_idx
    shuffle_map = {i: orig_idx for i, orig_idx in enumerate(selected)}
    print(f"[ifeval] Shuffle map built: {len(shuffle_map)} entries")
    return shuffle_map


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True,
                        choices=["gsm8k", "math", "mbpp", "ifeval", "all"])
    parser.add_argument("--pass-rate-dir", type=str,
                        default="experiments/qwen3-1.7b-base-pass-rate")
    parser.add_argument("--ifeval-dir", type=str,
                        default="experiments/ifeval-curriculum",
                        help="Directory with IFEval pass_at_k.csv")
    parser.add_argument("--output-dir", type=str,
                        default="experiments/curricula")
    parser.add_argument("--sort-key", type=str, default="pass@16",
                        help="Column to sort by (ascending)")
    parser.add_argument("--descending", action="store_true",
                        help="Sort in descending order (easy to hard)")
    parser.add_argument("--ifeval-shuffle-seed", type=int, default=42,
                        help="Shuffle seed used for IFEval batched generation")
    parser.add_argument("--ifeval-num-prompts", type=int, default=3200,
                        help="Number of prompts used in IFEval batched generation")
    args = parser.parse_args()

    pass_rate_dir = Path(args.pass_rate_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    datasets = ["gsm8k", "math", "mbpp", "ifeval"] if args.dataset == "all" else [args.dataset]

    # Build IFEval shuffle map if needed
    ifeval_shuffle_map = None
    if "ifeval" in datasets:
        ifeval_shuffle_map = build_ifeval_shuffle_map(
            args.ifeval_shuffle_seed,
            args.ifeval_num_prompts
        )

    for dataset in datasets:
        print(f"\n{'='*60}")
        print(f"Processing {dataset}")
        print(f"{'='*60}")

        # Load CSV
        if dataset == "ifeval":
            csv_path = Path(args.ifeval_dir) / "ifeval_pass_at_k.csv"
        else:
            csv_path = pass_rate_dir / f"{dataset}_pass_at_k.csv"

        if not csv_path.exists():
            print(f"[{dataset}] CSV not found: {csv_path}")
            continue

        print(f"[{dataset}] Reading CSV from {csv_path}")

        rows = []
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)

        print(f"[{dataset}] CSV has {len(rows)} rows")

        # Build curriculum list
        curriculum = []

        for row in rows:
            csv_idx = row["idx"]

            if dataset == "ifeval":
                # IFEval CSV uses shuffled indices, map back to original
                try:
                    shuffled_idx = int(csv_idx)
                except ValueError:
                    print(f"[{dataset}] Warning: couldn't parse idx '{csv_idx}'")
                    continue

                if shuffled_idx not in ifeval_shuffle_map:
                    print(f"[{dataset}] Warning: shuffled idx {shuffled_idx} not in map")
                    continue

                orig_idx = ifeval_shuffle_map[shuffled_idx]
                prompt_id = f"ifeval_{orig_idx}"

            elif dataset == "mbpp":
                # MBPP uses task_id directly (e.g., "mbpp_601")
                prompt_id = csv_idx

            else:
                # GSM8K, MATH use format "dataset_N" where N is the index
                # The CSV idx column already has this format
                prompt_id = csv_idx

            # Get the sort value
            sort_val = float(row.get(args.sort_key, row.get("pass@16", "0")))
            curriculum.append((prompt_id, sort_val))

        print(f"[{dataset}] Mapped {len(curriculum)} prompts")

        # Sort by pass@k (ascending = hard to easy, descending = easy to hard)
        curriculum.sort(key=lambda x: x[1], reverse=args.descending)

        # Write curriculum file
        output_path = output_dir / f"{dataset}_curriculum.txt"
        with open(output_path, "w") as f:
            for prompt_id, _ in curriculum:
                f.write(f"{prompt_id}\n")

        print(f"[{dataset}] Wrote {len(curriculum)} prompt_ids to {output_path}")

        # Show sample
        print(f"[{dataset}] First 5 entries (hardest):")
        for pid, val in curriculum[:5]:
            print(f"  {pid}: {args.sort_key}={val:.4f}")
        print(f"[{dataset}] Last 5 entries (easiest):")
        for pid, val in curriculum[-5:]:
            print(f"  {pid}: {args.sort_key}={val:.4f}")


if __name__ == "__main__":
    main()
