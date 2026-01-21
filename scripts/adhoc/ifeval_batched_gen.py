#!/usr/bin/env python3
"""
Batched IFEval generation with incremental saves.

Generates completions in smaller batches (n=16 or 32) with different seeds.
Run multiple times with different seeds to build up to pass@512.

Usage:
    # Generate 16 completions for 50 prompts with seed 0
    python scripts/ifeval_batched_gen.py --seed 0 --n 16 --num-prompts 50 --gpu 0

    # Run in parallel on 8 GPUs with different seed ranges
    for i in {0..7}; do
        python scripts/ifeval_batched_gen.py --seed $i --n 16 --num-prompts 50 --gpu $i &
    done

To get 512 completions per prompt, run 32 times with seeds 0-31 (32 * 16 = 512).
"""

import argparse
import json
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def main():
    parser = argparse.ArgumentParser(description="Batched IFEval generation")
    parser.add_argument("--seed", type=int, required=True, help="Random seed for sampling")
    parser.add_argument("--n", type=int, default=16, help="Completions per prompt per batch")
    parser.add_argument("--num-prompts", type=int, default=50, help="Number of prompts (0=all)")
    parser.add_argument("--gpu", type=int, default=0, help="GPU index to use")
    parser.add_argument("--max-tokens", type=int, default=2048, help="Max tokens per completion")
    parser.add_argument("--output-dir", type=str, default="experiments/ifeval-curriculum",
                        help="Output directory")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    parser.add_argument("--model", type=str,
                        default="/efs/rlvr-experiments/assets/hf/Qwen3-1.7B-Base",
                        help="Model path")
    parser.add_argument("--prompt-offset", type=int, default=0,
                        help="Start from this prompt index (for parallel runs)")
    parser.add_argument("--shuffle-seed", type=int, default=None,
                        help="Seed for random prompt selection (deterministic shuffle)")
    parser.add_argument("--save-completions", action="store_true",
                        help="Save full completion texts (larger files)")
    args = parser.parse_args()

    # Set GPU visibility
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    # Import after setting CUDA_VISIBLE_DEVICES
    from datasets import load_dataset
    from vllm import LLM, SamplingParams
    from rlvr_experiments.verifiers.ifeval import IFEvalVerifier

    print("=" * 60)
    print(f"IFEval Batched Generation - Seed {args.seed}")
    print("=" * 60)
    print(f"GPU: {args.gpu}")
    print(f"Completions per prompt: {args.n}")
    print(f"Max tokens: {args.max_tokens}")
    print(f"Temperature: {args.temperature}")
    print()

    # Load dataset
    print("Loading RLVR-IFeval dataset...")
    hf_dataset = load_dataset("allenai/RLVR-IFeval", split="train")

    # Build prompt list
    all_rows = []
    for i, row in enumerate(hf_dataset):
        user_content = row["messages"][0]["content"] if row["messages"] else ""
        all_rows.append({
            "idx": i,
            "prompt": user_content,
            "ground_truth": row["ground_truth"],
            "constraint_type": row["constraint_type"],
            "constraint": row["constraint"],
        })

    print(f"Total prompts in dataset: {len(all_rows)}")

    # Shuffle if requested (deterministic based on shuffle_seed)
    if args.shuffle_seed is not None:
        random.seed(args.shuffle_seed)
        random.shuffle(all_rows)
        print(f"Shuffled prompts with seed {args.shuffle_seed}")

    # Select prompts
    if args.num_prompts > 0:
        # Use consistent ordering based on prompt_offset
        start_idx = args.prompt_offset
        end_idx = min(start_idx + args.num_prompts, len(all_rows))
        rows = all_rows[start_idx:end_idx]
        print(f"Using prompts {start_idx} to {end_idx-1} ({len(rows)} prompts)")
    else:
        rows = all_rows
        print(f"Using all {len(rows)} prompts")

    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / f"seed{args.seed:03d}_n{args.n}_prompts{len(rows)}.jsonl"
    print(f"Output file: {output_file}")

    # Check if partially completed
    completed_indices = set()
    if output_file.exists():
        with open(output_file) as f:
            for line in f:
                data = json.loads(line)
                completed_indices.add(data["idx"])
        print(f"Resuming: {len(completed_indices)} prompts already completed")

    remaining_rows = [r for r in rows if r["idx"] not in completed_indices]
    if not remaining_rows:
        print("All prompts already completed!")
        return

    print(f"Remaining prompts: {len(remaining_rows)}")

    # Initialize vLLM
    print("\nInitializing vLLM...")
    llm = LLM(
        model=args.model,
        tensor_parallel_size=1,
        max_model_len=8192,  # Some IFEval prompts are long
        gpu_memory_utilization=0.90,
        dtype="bfloat16",
        enable_prefix_caching=True,
        trust_remote_code=True,
        seed=args.seed,
    )

    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=1.0,
        top_k=-1,
        max_tokens=args.max_tokens,
        n=args.n,
        seed=args.seed,
    )

    # Initialize verifier
    verifier = IFEvalVerifier()

    # Process in batches to allow incremental saves
    batch_size = 10  # Process 10 prompts at a time
    total_completions = 0
    start_time = time.time()

    print(f"\nGenerating completions...")

    for batch_start in range(0, len(remaining_rows), batch_size):
        batch_rows = remaining_rows[batch_start:batch_start + batch_size]
        prompts = [r["prompt"] for r in batch_rows]

        # Generate
        outputs = llm.generate(prompts, sampling_params)

        # Verify and save
        with open(output_file, "a") as f:
            for row, output in zip(batch_rows, outputs):
                completions = [o.text for o in output.outputs]
                scores = [verifier.verify(c, row["ground_truth"]) for c in completions]

                result = {
                    "idx": row["idx"],
                    "constraint_type": row["constraint_type"],
                    "n": len(completions),
                    "correct": sum(1 for s in scores if s > 0.5),
                    "scores": scores,
                }

                if args.save_completions:
                    result["completions"] = completions

                f.write(json.dumps(result) + "\n")
                total_completions += len(completions)

        elapsed = time.time() - start_time
        done = batch_start + len(batch_rows)
        print(f"  Processed {done}/{len(remaining_rows)} prompts "
              f"({total_completions:,} completions, {elapsed:.1f}s)")

    # Summary
    elapsed = time.time() - start_time
    print("\n" + "=" * 60)
    print("COMPLETE")
    print("=" * 60)
    print(f"Seed: {args.seed}")
    print(f"Prompts processed: {len(remaining_rows)}")
    print(f"Completions generated: {total_completions:,}")
    print(f"Time: {elapsed:.1f}s ({total_completions/elapsed:.1f} completions/sec)")
    print(f"Output: {output_file}")


if __name__ == "__main__":
    main()
