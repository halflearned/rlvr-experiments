#!/usr/bin/env python3
"""Generate completions for GSM8K test set and verify with MathVerifier.

Usage:
    python scripts/eval_checkpoint_gsm8k.py <checkpoint_path> <output_dir> [--gpu GPU]

Generates completions using vLLM with greedy sampling (temperature=0), then
runs MathVerifier on each completion.

Outputs:
    <output_dir>/completions.jsonl  - Raw completions with prompts
    <output_dir>/results.jsonl      - Completions with rewards
    <output_dir>/summary.json       - Aggregate metrics
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def load_gsm8k_test():
    """Load GSM8K test set."""
    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k", "main", split="test")
    return ds


def format_prompt(question: str) -> str:
    """Format question as prompt (base model style)."""
    return f"Question: {question}\nAnswer:"


def generate_completions(checkpoint_path: str, prompts: list[str], gpu: int = 0):
    """Generate completions using vLLM."""
    from vllm import LLM, SamplingParams

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)

    llm = LLM(
        model=checkpoint_path,
        tensor_parallel_size=1,
        dtype="bfloat16",
        gpu_memory_utilization=0.9,
        max_model_len=2048,
        trust_remote_code=True,
    )

    sampling_params = SamplingParams(
        temperature=0,  # greedy
        max_tokens=1024,
        stop=["Question:", "\n\nQuestion:"],
    )

    outputs = llm.generate(prompts, sampling_params)

    completions = []
    for output in outputs:
        text = output.outputs[0].text
        completions.append(text)

    return completions


def verify_completions(completions: list[str], answers: list[str]):
    """Verify completions against gold answers using MathVerifier."""
    from rlvr_experiments.verifiers.math import MathVerifier

    verifier = MathVerifier(timeout=5.0, max_workers=8)

    rewards = []
    for completion, answer in zip(completions, answers):
        reward = verifier.verify(completion, answer)
        rewards.append(reward)

    return rewards


def main():
    parser = argparse.ArgumentParser(description="Evaluate checkpoint on GSM8K test")
    parser.add_argument("checkpoint_path", help="Path to checkpoint")
    parser.add_argument("output_dir", help="Output directory for results")
    parser.add_argument("--gpu", type=int, default=0, help="GPU to use")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load test set
    print("Loading GSM8K test set...")
    ds = load_gsm8k_test()
    print(f"Loaded {len(ds)} examples")

    # Format prompts
    prompts = [format_prompt(ex["question"]) for ex in ds]
    answers = [ex["answer"] for ex in ds]

    # Generate completions
    print(f"Generating completions with checkpoint: {args.checkpoint_path}")
    print(f"Using GPU {args.gpu}")
    completions = generate_completions(args.checkpoint_path, prompts, args.gpu)

    # Save raw completions
    completions_path = output_dir / "completions.jsonl"
    with open(completions_path, "w") as f:
        for i, (prompt, completion, answer) in enumerate(zip(prompts, completions, answers)):
            f.write(json.dumps({
                "id": i,
                "prompt": prompt,
                "completion": completion,
                "gold_answer": answer,
            }) + "\n")
    print(f"Saved completions to {completions_path}")

    # Verify completions
    print("Verifying completions...")
    rewards = verify_completions(completions, answers)

    # Save results with rewards
    results_path = output_dir / "results.jsonl"
    with open(results_path, "w") as f:
        for i, (prompt, completion, answer, reward) in enumerate(zip(prompts, completions, answers, rewards)):
            f.write(json.dumps({
                "id": i,
                "prompt": prompt,
                "completion": completion,
                "gold_answer": answer,
                "reward": reward,
            }) + "\n")
    print(f"Saved results to {results_path}")

    # Compute summary
    accuracy = sum(rewards) / len(rewards)
    summary = {
        "checkpoint": args.checkpoint_path,
        "dataset": "gsm8k",
        "split": "test",
        "n_examples": len(rewards),
        "n_correct": sum(rewards),
        "accuracy": accuracy,
    }

    summary_path = output_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary to {summary_path}")

    print(f"\n=== Results ===")
    print(f"Accuracy: {accuracy:.4f} ({sum(rewards)}/{len(rewards)})")


if __name__ == "__main__":
    main()
