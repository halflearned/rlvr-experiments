#!/usr/bin/env python3
"""
Evaluate IFBench using the official AllenAI evaluator.
Generates completions for both base and trained models, then runs IFBench evaluation.
"""

import argparse
import json
import os
import sys
from datetime import datetime

import torch
from vllm import LLM, SamplingParams
from datasets import load_dataset


def load_ifbench_prompts():
    """Load IFBench prompts from allenai/IFBench_test."""
    ds = load_dataset("allenai/IFBench_test", split="train")
    return list(ds)


def generate_completions(
    model_path: str,
    prompts: list,
    output_file: str,
    max_tokens: int = 2048,
    temperature: float = 0.0,
):
    """Generate completions and save in IFBench format."""
    print(f"Loading model from {model_path}...")
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
        n=1,
    )

    # Generate completions
    print(f"Generating completions for {len(prompts)} prompts...")
    prompt_texts = [p["prompt"] for p in prompts]
    outputs = llm.generate(prompt_texts, sampling_params)

    # Save in IFBench format (jsonl with "prompt" and "response" fields)
    results = []
    for prompt_data, output in zip(prompts, outputs):
        completion = output.outputs[0].text
        results.append({
            "prompt": prompt_data["prompt"],
            "response": completion,
        })

    with open(output_file, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    print(f"Saved {len(results)} completions to {output_file}")

    # Cleanup
    del llm
    torch.cuda.empty_cache()


def run_ifbench_eval(input_data: str, response_data: str, output_dir: str):
    """Run official IFBench evaluation."""
    # Add IFBench to path
    sys.path.insert(0, "/tmp/IFBench")
    import evaluation_lib

    inputs = evaluation_lib.read_prompt_list(input_data)
    prompt_to_response = evaluation_lib.read_prompt_to_response_dict(response_data)

    results = {}
    for mode, func in [
        ("strict", evaluation_lib.test_instruction_following_strict),
        ("loose", evaluation_lib.test_instruction_following_loose),
    ]:
        outputs = []
        for inp in inputs:
            outputs.append(func(inp, prompt_to_response))

        follow_all = [o.follow_all_instructions for o in outputs]
        accuracy = sum(follow_all) / len(outputs)
        results[f"{mode}_accuracy"] = accuracy

        # Count per-instruction accuracy
        all_follow = []
        for o in outputs:
            all_follow.extend(o.follow_instruction_list)
        inst_accuracy = sum(all_follow) / len(all_follow) if all_follow else 0
        results[f"{mode}_inst_accuracy"] = inst_accuracy

        print(f"{mode.upper()} Accuracy: {accuracy:.4f} (prompt-level), {inst_accuracy:.4f} (instruction-level)")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", default="/efs/rlvr-experiments/assets/hf/Qwen3-1.7B-Base")
    parser.add_argument("--trained-model", default="/efs/rlvr-experiments/checkpoints/qwen3_17b_20260124_112132_step100")
    parser.add_argument("--output-dir", default="/efs/rlvr-experiments/experiments/ifbench_official")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--only-base", action="store_true", help="Only evaluate base model")
    parser.add_argument("--only-trained", action="store_true", help="Only evaluate trained model")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    # Create timestamped output dir
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, timestamp)
    os.makedirs(output_dir, exist_ok=True)

    # Load prompts
    print("Loading IFBench prompts...")
    prompts = load_ifbench_prompts()
    print(f"Loaded {len(prompts)} prompts")

    # Save IFBench test data in jsonl format for evaluator
    ifbench_data_file = os.path.join(output_dir, "ifbench_test.jsonl")
    with open(ifbench_data_file, "w") as f:
        for p in prompts:
            f.write(json.dumps(dict(p)) + "\n")

    results = {}

    # Evaluate base model
    if not args.only_trained:
        print("\n" + "="*60)
        print("Evaluating BASE model")
        print("="*60)
        base_response_file = os.path.join(output_dir, "base-responses.jsonl")
        generate_completions(
            model_path=args.base_model,
            prompts=prompts,
            output_file=base_response_file,
        )
        print("\nRunning IFBench evaluation on base model...")
        results["base"] = run_ifbench_eval(ifbench_data_file, base_response_file, output_dir)

    # Evaluate trained model
    if not args.only_base:
        print("\n" + "="*60)
        print("Evaluating TRAINED model (DR-GRPO step 100)")
        print("="*60)
        trained_response_file = os.path.join(output_dir, "trained-responses.jsonl")
        generate_completions(
            model_path=args.trained_model,
            prompts=prompts,
            output_file=trained_response_file,
        )
        print("\nRunning IFBench evaluation on trained model...")
        results["trained"] = run_ifbench_eval(ifbench_data_file, trained_response_file, output_dir)

    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for name, res in results.items():
        print(f"\n{name}:")
        for k, v in res.items():
            print(f"  {k}: {v:.4f}")

    if "base" in results and "trained" in results:
        print("\nImprovement:")
        for metric in ["strict_accuracy", "loose_accuracy", "strict_inst_accuracy", "loose_inst_accuracy"]:
            base_val = results["base"][metric]
            trained_val = results["trained"][metric]
            diff = trained_val - base_val
            rel = (diff / base_val * 100) if base_val > 0 else 0
            print(f"  {metric}: {diff:+.4f} ({rel:+.1f}%)")

    # Save summary
    with open(os.path.join(output_dir, "summary.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()
