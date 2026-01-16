#!/usr/bin/env python3
"""
Test script to evaluate Qwen 1.7B Base on GSM8k using vLLM.

Loads 100 randomly selected prompts from GSM8k, generates 512 completions
per prompt using vLLM, and computes the pass rate using MathVerifier.

Supports data parallelism across multiple GPUs for faster generation.

Usage:
    python scripts/test_gsm8k_pass_rate.py [--num-prompts 100] [--n 512] [--seed 42]
"""

import argparse
import json
import os
import random
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path

from datasets import load_dataset
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from rlvr_experiments.verifiers.math import MathVerifier


@dataclass
class CompletionDetail:
    """Details for a single completion."""
    length: int  # Token count
    finish_reason: str  # "stop" or "length"
    correct: bool


@dataclass
class PromptResult:
    """Results for a single prompt."""
    prompt_id: str
    prompt: str
    target_answer: str
    num_completions: int
    num_correct: int
    pass_rate: float
    correctness_mask: list[bool]  # Per-completion correctness for pass@k computation
    completion_details: list[dict]  # Per-completion length, finish_reason, correct
    sample_correct_completion: str | None = None
    sample_incorrect_completion: str | None = None


def parse_args():
    parser = argparse.ArgumentParser(description="Test Qwen 1.7B Base on GSM8k")
    parser.add_argument(
        "--model-path",
        type=str,
        default="/efs/rlvr-experiments/assets/hf/Qwen3-1.7B-Base",
        help="Path to the model",
    )
    parser.add_argument(
        "--num-prompts",
        type=int,
        default=100,
        help="Number of prompts to sample from GSM8k (default: 100)",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=512,
        help="Number of completions per prompt (default: 512)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="Max tokens to generate per completion (default: 512)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.6,
        help="Sampling temperature (default: 0.6)",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.95,
        help="Top-p sampling (default: 0.95)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="Top-k sampling (default: 20)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for prompt selection (default: 42)",
    )
    parser.add_argument(
        "--data-parallel-size",
        type=int,
        default=8,
        help="Number of GPUs for data parallelism (default: 8)",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="Tensor parallel size per replica (default: 1)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "test"],
        help="GSM8k split to use (default: test)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed results for each prompt",
    )
    return parser.parse_args()


def load_and_sample_prompts(split: str, num_prompts: int, seed: int) -> list[dict]:
    """Load GSM8k and randomly sample prompts."""
    print(f"Loading GSM8k {split} split...")

    hf_dataset = load_dataset("openai/gsm8k", "main", split=split)

    # Convert to list of dicts
    all_rows = []
    for i, row in enumerate(hf_dataset):
        question = f"\n\nProblem:{row['question'].strip()}"
        answer = row["answer"].split("####")[-1].strip()
        prompt_id = f"gsm8k_{i}"
        all_rows.append({
            "prompt": question,
            "problem": {"answer": answer, "prompt_id": prompt_id},
        })

    print(f"Loaded {len(all_rows)} prompts from GSM8k {split}")

    # Sample randomly
    random.seed(seed)
    sampled = random.sample(all_rows, min(num_prompts, len(all_rows)))
    print(f"Sampled {len(sampled)} prompts (seed={seed})")

    return sampled


def format_prompts(
    rows: list[dict],
    tokenizer,
    system_prompt: str = "Solve the following math problem and provide the final answer inside \\boxed{}",
    assistant_prefix: str = "Let's think step by step.",
) -> list[str]:
    """Apply chat template to prompts."""
    formatted = []
    for row in rows:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": row["prompt"]},
        ]
        content = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        formatted.append(content + assistant_prefix)
    return formatted


def worker(gpu_id, prompts, model_path, sampling_params_dict, tp_size, result_queue):
    """Worker function for each GPU."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    sp = SamplingParams(**sampling_params_dict)

    llm = LLM(
        model=model_path,
        tensor_parallel_size=tp_size,
        max_model_len=1024,
        gpu_memory_utilization=0.90,
        dtype="bfloat16",
        enable_prefix_caching=True,
        trust_remote_code=True,
    )

    outputs = llm.generate(prompts, sp)

    # Convert outputs to serializable format (text, finish_reason, num_tokens)
    results = []
    for output in outputs:
        completions = [(o.text, o.finish_reason, len(o.token_ids)) for o in output.outputs]
        results.append(completions)

    result_queue.put((gpu_id, results))


def main():
    args = parse_args()

    print("=" * 60)
    print("GSM8k Pass Rate Test")
    print("=" * 60)
    print(f"Model: {args.model_path}")
    print(f"Prompts: {args.num_prompts}")
    print(f"Completions per prompt: {args.n}")
    print(f"Max tokens: {args.max_tokens}")
    print(f"Temperature: {args.temperature}")
    print(f"Top-p: {args.top_p}")
    print(f"Top-k: {args.top_k}")
    print(f"Split: {args.split}")
    print(f"Seed: {args.seed}")
    print(f"Data parallel size: {args.data_parallel_size}")
    print(f"Tensor parallel size: {args.tensor_parallel_size}")
    print("=" * 60)

    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen3-1.7B-Base",
        use_fast=False,
    )

    # Load and sample prompts
    rows = load_and_sample_prompts(args.split, args.num_prompts, args.seed)

    # Format prompts with chat template
    print("\nFormatting prompts with chat template...")
    formatted_prompts = format_prompts(rows, tokenizer)

    # Set up sampling parameters
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_tokens=args.max_tokens,
        n=args.n,
    )

    # Split prompts across GPUs
    dp_size = args.data_parallel_size
    tp_size = args.tensor_parallel_size

    # Each DP replica uses tp_size GPUs
    num_replicas = dp_size // tp_size

    print(f"\nUsing {num_replicas} vLLM replicas across {dp_size} GPUs")

    # Distribute prompts across replicas
    prompts_per_replica = [[] for _ in range(num_replicas)]
    rows_per_replica = [[] for _ in range(num_replicas)]
    for i, (prompt, row) in enumerate(zip(formatted_prompts, rows)):
        replica_idx = i % num_replicas
        prompts_per_replica[replica_idx].append(prompt)
        rows_per_replica[replica_idx].append(row)

    for i, prompts in enumerate(prompts_per_replica):
        print(f"  Replica {i}: {len(prompts)} prompts")

    # Generate completions in parallel using multiprocessing
    print(f"\nGenerating {args.n} completions for each of {len(formatted_prompts)} prompts...")
    print(f"Total completions to generate: {args.n * len(formatted_prompts):,}")

    start_time = time.time()

    # Use multiprocessing to run on different GPUs
    import torch.multiprocessing as mp
    mp.set_start_method('spawn', force=True)

    # Create result queue and processes
    result_queue = mp.Queue()
    processes = []

    sampling_params_dict = {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "max_tokens": args.max_tokens,
        "n": args.n,
    }

    for replica_idx in range(num_replicas):
        gpu_id = replica_idx * tp_size  # Starting GPU for this replica
        if tp_size == 1:
            gpu_ids = str(gpu_id)
        else:
            gpu_ids = ",".join(str(gpu_id + j) for j in range(tp_size))

        p = mp.Process(
            target=worker,
            args=(gpu_ids, prompts_per_replica[replica_idx], args.model_path,
                  sampling_params_dict, tp_size, result_queue)
        )
        processes.append(p)
        p.start()

    # Collect results
    all_completions = [None] * num_replicas
    for _ in range(num_replicas):
        gpu_id, results = result_queue.get()
        # Parse gpu_id back to replica index
        if isinstance(gpu_id, str):
            replica_idx = int(gpu_id.split(",")[0]) // tp_size
        else:
            replica_idx = gpu_id // tp_size
        all_completions[replica_idx] = results
        print(f"  Replica {replica_idx} completed: {len(results)} prompts")

    for p in processes:
        p.join()

    gen_time = time.time() - start_time

    # Reassemble results in original order
    outputs_by_idx = {}
    for replica_idx in range(num_replicas):
        for local_idx, completions in enumerate(all_completions[replica_idx]):
            global_idx = local_idx * num_replicas + replica_idx
            outputs_by_idx[global_idx] = completions

    print(f"\nGeneration completed in {gen_time:.1f}s")
    total_completions = sum(len(c) for c in outputs_by_idx.values())
    print(f"Generated {total_completions:,} completions")
    print(f"Throughput: {total_completions / gen_time:.1f} completions/sec")

    # Verify completions
    print("\nVerifying completions...")
    verifier = MathVerifier()

    results: list[PromptResult] = []
    total_correct = 0
    total_completions_verified = 0

    # Track finish reasons and correctness
    finish_stats = {
        "stop_correct": 0,
        "stop_incorrect": 0,
        "length_correct": 0,
        "length_incorrect": 0,
    }

    # Track completion lengths
    length_stats = {
        "correct_lengths": [],
        "incorrect_lengths": [],
        "stop_lengths": [],
        "length_lengths": [],  # truncated completions
    }

    for i, row in enumerate(rows):
        problem = row["problem"]
        target = problem["answer"]
        prompt_id = problem["prompt_id"]

        completions_with_reason = outputs_by_idx.get(i, [])

        # Verify each completion
        correct_count = 0
        sample_correct = None
        sample_incorrect = None
        correctness_mask = []
        completion_details = []

        for completion, finish_reason, num_tokens in completions_with_reason:
            score = verifier.verify(completion, target)
            is_correct = score > 0
            correctness_mask.append(is_correct)
            completion_details.append({
                "length": num_tokens,
                "finish_reason": finish_reason,
                "correct": is_correct,
            })

            # Track finish reason stats
            if finish_reason == "stop":
                finish_stats["stop_correct" if is_correct else "stop_incorrect"] += 1
                length_stats["stop_lengths"].append(num_tokens)
            else:  # "length"
                finish_stats["length_correct" if is_correct else "length_incorrect"] += 1
                length_stats["length_lengths"].append(num_tokens)

            # Track lengths by correctness
            if is_correct:
                length_stats["correct_lengths"].append(num_tokens)
                correct_count += 1
                if sample_correct is None:
                    sample_correct = completion
            else:
                length_stats["incorrect_lengths"].append(num_tokens)
                if sample_incorrect is None:
                    sample_incorrect = completion

        pass_rate = correct_count / len(completions_with_reason) if completions_with_reason else 0.0

        result = PromptResult(
            prompt_id=prompt_id,
            prompt=row["prompt"],
            target_answer=target,
            num_completions=len(completions_with_reason),
            num_correct=correct_count,
            pass_rate=pass_rate,
            correctness_mask=correctness_mask,
            completion_details=completion_details,
            sample_correct_completion=sample_correct,
            sample_incorrect_completion=sample_incorrect,
        )
        results.append(result)

        total_correct += correct_count
        total_completions_verified += len(completions_with_reason)

        if args.verbose:
            print(f"\n[{i+1}/{len(rows)}] {prompt_id}")
            print(f"  Target: {target}")
            print(f"  Correct: {correct_count}/{len(completions_with_reason)} ({pass_rate*100:.1f}%)")

    # Compute aggregate statistics
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    overall_pass_rate = total_correct / total_completions_verified if total_completions_verified else 0
    avg_per_prompt_pass_rate = sum(r.pass_rate for r in results) / len(results) if results else 0

    # Pass@k metrics
    prompts_with_any_correct = sum(1 for r in results if r.num_correct > 0)
    pass_at_1 = prompts_with_any_correct / len(results) if results else 0

    # Distribution of pass rates
    pass_rate_buckets = {
        "0%": 0,
        "1-10%": 0,
        "11-25%": 0,
        "26-50%": 0,
        "51-75%": 0,
        "76-99%": 0,
        "100%": 0,
    }

    for r in results:
        pr = r.pass_rate * 100
        if pr == 0:
            pass_rate_buckets["0%"] += 1
        elif pr <= 10:
            pass_rate_buckets["1-10%"] += 1
        elif pr <= 25:
            pass_rate_buckets["11-25%"] += 1
        elif pr <= 50:
            pass_rate_buckets["26-50%"] += 1
        elif pr <= 75:
            pass_rate_buckets["51-75%"] += 1
        elif pr < 100:
            pass_rate_buckets["76-99%"] += 1
        else:
            pass_rate_buckets["100%"] += 1

    print(f"\nTotal prompts: {len(results)}")
    print(f"Total completions: {total_completions_verified:,}")
    print(f"Total correct: {total_correct:,}")
    print(f"\nOverall pass rate: {overall_pass_rate*100:.2f}%")
    print(f"Average per-prompt pass rate: {avg_per_prompt_pass_rate*100:.2f}%")
    print(f"\nPass@{args.n} (at least 1 correct): {prompts_with_any_correct}/{len(results)} ({pass_at_1*100:.1f}%)")

    print("\nPass rate distribution:")
    for bucket, count in pass_rate_buckets.items():
        pct = count / len(results) * 100 if results else 0
        bar = "█" * int(pct / 2)
        print(f"  {bucket:>8}: {count:3d} ({pct:5.1f}%) {bar}")

    # Finish reason breakdown
    total_stopped = finish_stats["stop_correct"] + finish_stats["stop_incorrect"]
    total_truncated = finish_stats["length_correct"] + finish_stats["length_incorrect"]
    print(f"\nFinish reason breakdown:")
    print(f"  Completed (stop): {total_stopped:,} ({total_stopped/total_completions_verified*100:.1f}%)")
    print(f"    - Correct: {finish_stats['stop_correct']:,}")
    print(f"    - Incorrect: {finish_stats['stop_incorrect']:,}")
    print(f"  Truncated (length): {total_truncated:,} ({total_truncated/total_completions_verified*100:.1f}%)")
    print(f"    - Correct: {finish_stats['length_correct']:,}")
    print(f"    - Incorrect: {finish_stats['length_incorrect']:,}")

    # Completion length stats
    def avg(lst):
        return sum(lst) / len(lst) if lst else 0
    def median(lst):
        if not lst:
            return 0
        s = sorted(lst)
        n = len(s)
        return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2

    print(f"\nCompletion length (tokens):")
    print(f"  All completions: avg={avg(length_stats['correct_lengths'] + length_stats['incorrect_lengths']):.0f}, median={median(length_stats['correct_lengths'] + length_stats['incorrect_lengths']):.0f}")
    print(f"  Correct: avg={avg(length_stats['correct_lengths']):.0f}, median={median(length_stats['correct_lengths']):.0f}")
    print(f"  Incorrect: avg={avg(length_stats['incorrect_lengths']):.0f}, median={median(length_stats['incorrect_lengths']):.0f}")
    print(f"  Completed (stop): avg={avg(length_stats['stop_lengths']):.0f}, median={median(length_stats['stop_lengths']):.0f}")
    print(f"  Truncated (length): avg={avg(length_stats['length_lengths']):.0f}, median={median(length_stats['length_lengths']):.0f}")

    # Show some examples
    print("\n" + "=" * 60)
    print("SAMPLE RESULTS")
    print("=" * 60)

    # Best performing
    sorted_by_pass_rate = sorted(results, key=lambda r: r.pass_rate, reverse=True)
    print("\nTop 3 highest pass rates:")
    for r in sorted_by_pass_rate[:3]:
        print(f"\n  Prompt: {r.prompt[:80]}...")
        print(f"  Target: {r.target_answer}")
        print(f"  Pass rate: {r.pass_rate*100:.1f}% ({r.num_correct}/{r.num_completions})")
        if r.sample_correct_completion:
            sample = r.sample_correct_completion[:200].replace("\n", " ")
            print(f"  Sample correct: {sample}...")

    # Worst performing
    print("\nBottom 3 lowest pass rates:")
    for r in sorted_by_pass_rate[-3:]:
        print(f"\n  Prompt: {r.prompt[:80]}...")
        print(f"  Target: {r.target_answer}")
        print(f"  Pass rate: {r.pass_rate*100:.1f}% ({r.num_correct}/{r.num_completions})")
        if r.sample_incorrect_completion:
            sample = r.sample_incorrect_completion[:200].replace("\n", " ")
            print(f"  Sample incorrect: {sample}...")

    print("\n" + "=" * 60)
    print(f"Generation time: {gen_time:.1f}s")
    print(f"Throughput: {total_completions_verified / gen_time:.1f} completions/sec")
    print("=" * 60)

    # Compute length summary stats for JSON
    length_summary = {
        "correct": {
            "count": len(length_stats["correct_lengths"]),
            "avg": avg(length_stats["correct_lengths"]),
            "median": median(length_stats["correct_lengths"]),
        },
        "incorrect": {
            "count": len(length_stats["incorrect_lengths"]),
            "avg": avg(length_stats["incorrect_lengths"]),
            "median": median(length_stats["incorrect_lengths"]),
        },
        "completed_stop": {
            "count": len(length_stats["stop_lengths"]),
            "avg": avg(length_stats["stop_lengths"]),
            "median": median(length_stats["stop_lengths"]),
        },
        "truncated_length": {
            "count": len(length_stats["length_lengths"]),
            "avg": avg(length_stats["length_lengths"]),
            "median": median(length_stats["length_lengths"]),
        },
    }

    # Save results to JSON
    output_dir = Path("experiments/qwen3-1.7B-base-pass-rate")
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = Path(args.model_path).name
    output_file = output_dir / f"gsm8k_{model_name}_{timestamp}.json"

    output_data = {
        "metadata": {
            "dataset": "gsm8k",
            "split": args.split,
            "model_path": args.model_path,
            "num_prompts": args.num_prompts,
            "completions_per_prompt": args.n,
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "seed": args.seed,
            "timestamp": timestamp,
            "generation_time_seconds": gen_time,
        },
        "summary": {
            "total_prompts": len(results),
            "total_completions": total_completions_verified,
            "total_correct": total_correct,
            "overall_pass_rate": overall_pass_rate,
            "pass_at_k": pass_at_1,
            "pass_rate_distribution": pass_rate_buckets,
            "finish_reason_breakdown": finish_stats,
            "completion_length": length_summary,
        },
        "per_prompt_results": [asdict(r) for r in results],
    }

    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"\nResults saved to: {output_file}")

    return overall_pass_rate


if __name__ == "__main__":
    main()
