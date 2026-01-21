#!/usr/bin/env python3
"""Direct IFEval evaluation without multiprocessing wrapper."""

import json
import random
import time
from datetime import datetime
from pathlib import Path

from datasets import load_dataset
from vllm import LLM, SamplingParams

# Import IFEval verifier
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from rlvr_experiments.verifiers.ifeval import IFEvalVerifier


def compute_pass_at_k(n: int, c: int, k: int) -> float:
    """Compute pass@k metric."""
    if n - c < k:
        return 1.0
    return 1.0 - (
        (n - c) * (n - c - 1) * ... * (n - c - k + 1)
        / (n * (n - 1) * ... * (n - k + 1))
    )


def comb(n, k):
    """Compute binomial coefficient."""
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
    """Probability that at least one of k samples is correct, given c correct out of n."""
    if c == 0:
        return 0.0
    if k >= n:
        return 1.0 if c > 0 else 0.0
    # P(at least 1 correct in k) = 1 - P(all wrong in k)
    # P(all wrong) = C(n-c, k) / C(n, k)
    return 1.0 - comb(n - c, k) / comb(n, k)


def main():
    # Config
    num_prompts = 100
    n_completions = 512
    max_tokens = 512
    max_model_len = 4096  # Need longer for some IFEval prompts
    seed = 42

    print("=" * 60)
    print("IFEval Direct Evaluation")
    print("=" * 60)

    # Load dataset
    print("\nLoading RLVR-IFeval dataset...")
    hf_dataset = load_dataset("allenai/RLVR-IFeval", split="train")

    # Sample prompts
    all_rows = []
    for i, row in enumerate(hf_dataset):
        user_content = row["messages"][0]["content"] if row["messages"] else ""
        all_rows.append({
            "prompt": user_content,
            "problem": {
                "prompt_id": f"ifeval_{i}",
                "ground_truth": row["ground_truth"],
                "constraint_type": row["constraint_type"],
                "constraint": row["constraint"],
            },
        })

    print(f"Total prompts: {len(all_rows)}")

    random.seed(seed)
    rows = random.sample(all_rows, min(num_prompts, len(all_rows)))
    print(f"Sampled {len(rows)} prompts")

    # Show constraint distribution
    constraint_counts = {}
    for r in rows:
        ct = r["problem"]["constraint_type"]
        constraint_counts[ct] = constraint_counts.get(ct, 0) + 1
    print(f"Constraint types in sample: {len(constraint_counts)} unique")

    # Format prompts (raw text for base model)
    prompts = [r["prompt"] for r in rows]

    print(f"\nExample prompt:\n{'-'*40}")
    print(prompts[0][:500])
    print(f"{'-'*40}")

    # Initialize vLLM
    print("\nInitializing vLLM...")
    llm = LLM(
        model="/efs/rlvr-experiments/assets/hf/Qwen3-1.7B-Base",
        tensor_parallel_size=1,
        max_model_len=max_model_len,
        gpu_memory_utilization=0.90,
        dtype="bfloat16",
        enable_prefix_caching=True,
        trust_remote_code=True,
    )

    sampling_params = SamplingParams(
        temperature=1.0,
        top_p=1.0,
        top_k=-1,
        max_tokens=max_tokens,
        n=n_completions,
    )

    # Generate
    print(f"\nGenerating {n_completions} completions for {len(prompts)} prompts...")
    print(f"Total: {n_completions * len(prompts):,} completions")

    start_time = time.time()
    outputs = llm.generate(prompts, sampling_params)
    gen_time = time.time() - start_time

    print(f"Generation completed in {gen_time:.1f}s")
    total_completions = sum(len(o.outputs) for o in outputs)
    print(f"Generated {total_completions:,} completions")
    print(f"Throughput: {total_completions / gen_time:.1f} completions/sec")

    # Verify
    print("\nVerifying completions...")
    verifier = IFEvalVerifier()

    results = []
    for i, (row, output) in enumerate(zip(rows, outputs)):
        completions = [o.text for o in output.outputs]
        scores = [verifier.verify(c, row["problem"]["ground_truth"]) for c in completions]

        correct_count = sum(1 for s in scores if s > 0.5)
        results.append({
            "prompt_id": row["problem"]["prompt_id"],
            "constraint_type": row["problem"]["constraint_type"],
            "n": len(scores),
            "correct": correct_count,
            "pass_rate": correct_count / len(scores) if scores else 0,
        })

        if (i + 1) % 20 == 0:
            print(f"  Verified {i + 1}/{len(rows)} prompts")

    # Compute pass@k
    k_values = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
    pass_at_k = {}
    for k in k_values:
        if k > n_completions:
            continue
        probs = [pass_at_k_prob(r["n"], r["correct"], k) for r in results]
        pass_at_k[k] = sum(probs) / len(probs)

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Dataset: IFEval (RLVR-IFeval)")
    print(f"Prompts: {len(results)}")
    print(f"Completions per prompt: {n_completions}")
    print()

    print("Pass@k:")
    for k, v in pass_at_k.items():
        print(f"  pass@{k}: {v*100:.2f}%")

    avg_pass_rate = sum(r["pass_rate"] for r in results) / len(results)
    print(f"\nAverage pass rate (per-completion): {avg_pass_rate*100:.2f}%")

    # Save results
    output_dir = Path("experiments/qwen3-1.7b-base-pass-rate")
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"ifeval_Qwen3-1.7B-Base_{timestamp}.json"

    output_data = {
        "metadata": {
            "dataset": "ifeval",
            "split": "train",
            "model_path": "/efs/rlvr-experiments/assets/hf/Qwen3-1.7B-Base",
            "num_prompts": len(results),
            "completions_per_prompt": n_completions,
            "max_tokens": max_tokens,
            "max_model_len": max_model_len,
            "temperature": 1.0,
            "top_p": 1.0,
            "seed": seed,
            "generation_time_seconds": gen_time,
            "timestamp": timestamp,
        },
        "pass_at_k": pass_at_k,
        "per_prompt_results": results,
    }

    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    main()
