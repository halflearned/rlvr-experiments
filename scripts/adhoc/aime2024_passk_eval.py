#!/usr/bin/env python3
"""
Generate completions for AIME 2024 and compute pass@k.

Usage:
    python scripts/adhoc/aime2024_passk_eval.py --gpu 0

This runs all 30 AIME 2024 problems on a single GPU since it's a small dataset.
"""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0, help="GPU index to use")
    parser.add_argument("--model", type=str, default="/efs/rlvr-experiments/assets/hf/Qwen3-1.7B-Base")
    parser.add_argument("--num-completions", type=int, default=32, help="Completions per problem")
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="/efs/rlvr-experiments/experiments/aime2024_passk")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    from datasets import load_dataset
    from vllm import LLM, SamplingParams
    from rlvr_experiments.verifiers.math import MathVerifier

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load AIME dataset and filter for 2024
    print("Loading AIME dataset...")
    ds = load_dataset("AI-MO/aimo-validation-aime", split="train")
    aime_2024 = [x for x in ds if "2024" in x['url']]
    print(f"AIME 2024 problems: {len(aime_2024)}")

    # Initialize vLLM
    print(f"Loading model from {args.model}...")
    llm = LLM(
        model=args.model,
        dtype="float16",
        tensor_parallel_size=1,
        gpu_memory_utilization=0.9,
        max_model_len=8192,  # Need longer context for AIME
        seed=args.seed,
    )

    sampling_params = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        n=args.num_completions,
        seed=args.seed,
    )

    # Initialize verifier
    verifier = MathVerifier(timeout=10.0, max_workers=8, warmup=True)

    # Process all problems
    results = []
    for i, sample in enumerate(aime_2024):
        problem = sample["problem"]
        answer = sample["answer"]
        url = sample["url"]

        print(f"\n[{i+1}/{len(aime_2024)}] {url.split('/')[-1]}")

        # Format prompt
        prompt = f"Solve the following math problem. Show your work step by step and put your final answer in \\boxed{{}}.\n\nProblem: {problem}\n\nSolution:"

        # Generate completions
        outputs = llm.generate([prompt], sampling_params)
        completions = [o.text for o in outputs[0].outputs]

        # Verify each completion
        rewards = verifier.verify_batch_parallel(completions, answer)

        result = {
            "problem_id": url.split("/")[-1],
            "url": url,
            "problem": problem,
            "answer": answer,
            "completions": completions,
            "rewards": rewards,
        }
        results.append(result)

        num_correct = sum(rewards)
        print(f"  Answer: {answer}, Correct: {num_correct}/{len(rewards)} ({100*num_correct/len(rewards):.1f}%)")

    # Save results
    output_file = output_dir / "results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {output_file}")

    # Compute pass@k
    def pass_at_k(n, c, k):
        if n - c < k:
            return 1.0
        result = 1.0
        for i in range(k):
            result *= (n - c - i) / (n - i)
        return 1.0 - result

    k_values = [1, 2, 4, 8, 16, 32]
    print("\n" + "="*50)
    print("AIME 2024 Pass@k Results")
    print("="*50)

    for k in k_values:
        pks = []
        for r in results:
            n = len(r["rewards"])
            c = sum(r["rewards"])
            pks.append(pass_at_k(n, c, k))
        avg_pk = sum(pks) / len(pks)
        print(f"pass@{k}: {avg_pk:.4f} ({avg_pk*100:.2f}%)")

    # Additional stats
    any_correct = sum(1 for r in results if sum(r["rewards"]) > 0)
    print(f"\nProblems with ≥1 correct: {any_correct}/{len(results)} ({100*any_correct/len(results):.1f}%)")


if __name__ == "__main__":
    main()
