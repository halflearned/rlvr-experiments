#!/usr/bin/env python3
"""
Evaluate IFBench performance comparing base model vs DR-GRPO trained checkpoint.
Generates completions and computes rewards for both models using the IF_multi_constraints verifier.
"""

import argparse
import json
import os
import random
from datetime import datetime

import torch
from vllm import LLM, SamplingParams
from datasets import load_dataset


def load_ifbench_prompts(num_samples: int = None, seed: int = 42):
    """Load IFBench prompts from allenai/IFBench_test."""
    ds = load_dataset("allenai/IFBench_test", split="train")

    items = []
    for row in ds:
        items.append({
            "prompt_id": row["key"],
            "prompt": row["prompt"],
            "instruction_id_list": row["instruction_id_list"],
            "kwargs": row["kwargs"],
        })

    if num_samples and num_samples < len(items):
        random.seed(seed)
        items = random.sample(items, num_samples)

    return items


def evaluate_model(
    model_path: str,
    prompts: list,
    output_dir: str,
    model_name: str,
    gpu: int = 0,
    n_completions: int = 1,
    max_tokens: int = 2048,
    temperature: float = 0.0,
):
    """Generate completions and evaluate with IFEval verifier."""
    from rlvr_experiments.verifiers.if_multi_constraints import IFMultiConstraintsVerifier

    os.makedirs(output_dir, exist_ok=True)

    # Initialize vLLM
    print(f"[{model_name}] Loading model from {model_path}...")
    llm = LLM(
        model=model_path,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.85,
        max_model_len=4096,
        dtype="float16",
        seed=42,
    )

    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=max_tokens,
        n=n_completions,
    )

    # Generate completions
    print(f"[{model_name}] Generating completions for {len(prompts)} prompts...")
    prompt_texts = [p["prompt"] for p in prompts]
    outputs = llm.generate(prompt_texts, sampling_params)

    # Initialize verifier
    verifier = IFMultiConstraintsVerifier()

    # Process results
    results = []
    total_reward = 0

    for i, (prompt_data, output) in enumerate(zip(prompts, outputs)):
        completions = [o.text for o in output.outputs]

        # Build ground_truth for verifier
        # Note: verifier expects "instruction_id" not "instruction_id_list"
        ground_truth = json.dumps({
            "instruction_id": prompt_data["instruction_id_list"],
            "kwargs": prompt_data["kwargs"],
        })

        # Compute rewards for each completion
        rewards = []
        for completion in completions:
            reward = verifier.verify(completion, ground_truth)
            rewards.append(reward)

        avg_reward = sum(rewards) / len(rewards) if rewards else 0.0
        total_reward += avg_reward

        results.append({
            "prompt_id": prompt_data["prompt_id"],
            "prompt": prompt_data["prompt"],
            "completions": completions,
            "rewards": rewards,
            "avg_reward": avg_reward,
            "instruction_id_list": prompt_data["instruction_id_list"],
        })

        if (i + 1) % 50 == 0:
            print(f"[{model_name}] Processed {i + 1}/{len(prompts)}, running avg reward: {total_reward / (i + 1):.4f}")

    # Save results
    output_file = os.path.join(output_dir, f"{model_name}_results.json")
    with open(output_file, "w") as f:
        json.dump({
            "model_path": model_path,
            "model_name": model_name,
            "num_prompts": len(prompts),
            "n_completions": n_completions,
            "temperature": temperature,
            "avg_reward": total_reward / len(prompts),
            "results": results,
        }, f, indent=2)

    print(f"[{model_name}] Average reward: {total_reward / len(prompts):.4f}")
    print(f"[{model_name}] Results saved to {output_file}")

    # Cleanup
    del llm
    torch.cuda.empty_cache()

    return total_reward / len(prompts)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", default="/efs/rlvr-experiments/assets/hf/Qwen3-1.7B-Base")
    parser.add_argument("--trained-model", default="/efs/rlvr-experiments/checkpoints/qwen3_17b_20260124_112132_step100")
    parser.add_argument("--output-dir", default="/efs/rlvr-experiments/experiments/ifbench_comparison")
    parser.add_argument("--num-samples", type=int, default=None, help="Number of samples (default: all)")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--n-completions", type=int, default=1, help="Number of completions per prompt")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--only-base", action="store_true", help="Only evaluate base model")
    parser.add_argument("--only-trained", action="store_true", help="Only evaluate trained model")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    # Create timestamped output dir
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, timestamp)
    os.makedirs(output_dir, exist_ok=True)

    # Load prompts
    print(f"Loading IFBench prompts...")
    prompts = load_ifbench_prompts(num_samples=args.num_samples)
    print(f"Loaded {len(prompts)} prompts")

    # Save prompts for reference
    with open(os.path.join(output_dir, "prompts.json"), "w") as f:
        json.dump(prompts, f, indent=2)

    results = {}

    # Evaluate base model
    if not args.only_trained:
        print("\n" + "="*60)
        print("Evaluating BASE model")
        print("="*60)
        results["base"] = evaluate_model(
            model_path=args.base_model,
            prompts=prompts,
            output_dir=output_dir,
            model_name="base",
            gpu=args.gpu,
            n_completions=args.n_completions,
            temperature=args.temperature,
        )

    # Evaluate trained model
    if not args.only_base:
        print("\n" + "="*60)
        print("Evaluating TRAINED model (DR-GRPO C=50 step 100)")
        print("="*60)
        results["trained"] = evaluate_model(
            model_path=args.trained_model,
            prompts=prompts,
            output_dir=output_dir,
            model_name="trained_step100",
            gpu=args.gpu,
            n_completions=args.n_completions,
            temperature=args.temperature,
        )

    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for name, reward in results.items():
        print(f"{name}: {reward:.4f}")

    if "base" in results and "trained" in results:
        improvement = results["trained"] - results["base"]
        print(f"Improvement: {improvement:+.4f} ({improvement / results['base'] * 100:+.1f}%)")

    # Save summary
    with open(os.path.join(output_dir, "summary.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()
