#!/usr/bin/env python3
"""
Evaluate a checkpoint on IFEval and IFBench test sets using vLLM for generation.

Usage:
    python scripts/eval_ifeval_checkpoint.py <checkpoint_path> <output_dir> [--gpu GPU]

Example:
    python scripts/eval_ifeval_checkpoint.py \
        results/qwen3-1.7B-ifeval-lr5e6-beta1e3_20260125-083856/checkpoints/step100 \
        results/qwen3-1.7B-ifeval-lr5e6-beta1e3_20260125-083856/evals \
        --gpu 0
"""

import argparse
import json
import os
import time
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm
from vllm import LLM, SamplingParams


def load_ifeval():
    """Load IFEval test set from HuggingFace."""
    ds = load_dataset("google/IFEval", split="train")  # IFEval only has train split

    rows = []
    for i, row in enumerate(ds):
        rows.append({
            "id": f"ifeval_{i}",
            "prompt": row["prompt"],
            "instruction_id_list": row.get("instruction_id_list", []),
            "kwargs": row.get("kwargs", []),
        })

    print(f"[load_ifeval] Loaded {len(rows)} IFEval prompts")
    return rows


def load_ifbench():
    """Load IFBench test set from HuggingFace."""
    ds = load_dataset("allenai/IFBench_test", split="train")  # IFBench_test only has train split

    rows = []
    for i, row in enumerate(ds):
        rows.append({
            "id": f"ifbench_{i}",
            "prompt": row["prompt"],
            "instruction_id_list": row.get("instruction_id_list", []),
            "kwargs": row.get("kwargs", []),
        })

    print(f"[load_ifbench] Loaded {len(rows)} IFBench prompts")
    return rows


def main():
    parser = argparse.ArgumentParser(description="Evaluate checkpoint on IFEval/IFBench")
    parser.add_argument("checkpoint_path", type=str, help="Path to checkpoint")
    parser.add_argument("output_dir", type=str, help="Output directory for results")
    parser.add_argument("--gpu", type=int, default=0, help="GPU to use")
    parser.add_argument("--max-tokens", type=int, default=2048, help="Max tokens for generation")
    parser.add_argument("--datasets", type=str, default="ifeval,ifbench",
                        help="Comma-separated list of datasets to evaluate")
    args = parser.parse_args()

    # Set GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load datasets
    datasets_to_eval = args.datasets.split(",")
    all_data = []

    if "ifeval" in datasets_to_eval:
        ifeval_data = load_ifeval()
        all_data.extend([("ifeval", d) for d in ifeval_data])

    if "ifbench" in datasets_to_eval:
        ifbench_data = load_ifbench()
        all_data.extend([("ifbench", d) for d in ifbench_data])

    # Prepare prompts - use simple format for base model
    prompts = []
    metadata = []
    for dataset_name, row in all_data:
        # Simple prompt format (no chat template for base model)
        prompt = row["prompt"]
        prompts.append(prompt)
        metadata.append({
            "id": row["id"],
            "dataset": dataset_name,
            "prompt": prompt,
            "instruction_id_list": row.get("instruction_id_list", []),
            "kwargs": row.get("kwargs", []),
        })

    # Initialize vLLM with greedy sampling
    print(f"[main] Loading model from {args.checkpoint_path}...")
    llm = LLM(
        model=args.checkpoint_path,
        tensor_parallel_size=1,
        dtype="bfloat16",
        gpu_memory_utilization=0.9,
        max_model_len=4096,
        trust_remote_code=True,
    )

    sampling_params = SamplingParams(
        temperature=0,  # Greedy
        max_tokens=args.max_tokens,
    )

    # Generate completions
    print(f"[main] Generating completions for {len(prompts)} prompts...")
    t0 = time.time()
    outputs = llm.generate(prompts, sampling_params)
    gen_time = time.time() - t0
    print(f"[main] Generation took {gen_time:.1f}s ({len(prompts)/gen_time:.1f} prompts/s)")

    # Extract completions
    completions = [out.outputs[0].text for out in outputs]

    # Save results by dataset
    results_by_dataset = {}
    for i, (meta, completion) in enumerate(zip(metadata, completions)):
        dataset_name = meta["dataset"]
        if dataset_name not in results_by_dataset:
            results_by_dataset[dataset_name] = []

        results_by_dataset[dataset_name].append({
            "id": meta["id"],
            "prompt": meta["prompt"],
            "completion": completion,
            "instruction_id_list": meta["instruction_id_list"],
            "kwargs": meta["kwargs"],
        })

    # Save completions for each dataset
    for dataset_name, results in results_by_dataset.items():
        completions_file = output_dir / f"{dataset_name}_completions.jsonl"
        with open(completions_file, "w") as f:
            for r in results:
                f.write(json.dumps(r) + "\n")
        print(f"[main] Saved {len(results)} completions to {completions_file}")

    # Save summary
    summary = {
        "checkpoint": args.checkpoint_path,
        "generation_time_s": gen_time,
        "datasets": {
            name: {"count": len(results)}
            for name, results in results_by_dataset.items()
        },
    }

    summary_file = output_dir / "ifeval_summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[main] Saved summary to {summary_file}")


if __name__ == "__main__":
    main()
