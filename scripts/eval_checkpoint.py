#!/usr/bin/env python3
"""
Evaluate a checkpoint on supported benchmarks using vLLM for generation.

Supported benchmarks:
- gsm8k: GSM8K test set (math word problems)
- ifeval: Google IFEval (instruction following)
- ifbench: AllenAI IFBench test set (instruction following)

Usage:
    python scripts/eval_checkpoint.py <checkpoint_path> <output_dir> --benchmark <benchmark> [--gpu GPU]

Examples:
    # Evaluate on GSM8K
    python scripts/eval_checkpoint.py \\
        results/my_run/checkpoints/step100 \\
        results/my_run/evals/gsm8k \\
        --benchmark gsm8k --gpu 0

    # Evaluate on IFEval
    python scripts/eval_checkpoint.py \\
        results/my_run/checkpoints/step100 \\
        results/my_run/evals/ifeval \\
        --benchmark ifeval --gpu 0

    # Evaluate on IFBench
    python scripts/eval_checkpoint.py \\
        results/my_run/checkpoints/step100 \\
        results/my_run/evals/ifbench \\
        --benchmark ifbench --gpu 0

Outputs:
    <output_dir>/completions.jsonl  - Raw completions with prompts
    <output_dir>/results.jsonl      - Completions with verification results
    <output_dir>/summary.json       - Aggregate metrics

IMPORTANT: Always specify --gpu N to select which GPU to use!
vLLM does not respect CUDA_VISIBLE_DEVICES reliably.
"""

import argparse
import os
import subprocess
import sys

# =============================================================================
# GPU Selection - MUST happen before any other imports that touch CUDA
# =============================================================================

def find_free_gpu():
    """Find a GPU with minimal memory usage."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True
        )
        gpus = []
        for line in result.stdout.strip().split("\n"):
            idx, mem = line.split(",")
            gpus.append((int(idx.strip()), int(mem.strip())))
        # Find GPU with least memory used
        gpus.sort(key=lambda x: x[1])
        if gpus and gpus[0][1] < 1000:  # Less than 1GB used
            return gpus[0][0]
        return None
    except Exception:
        return None

def setup_gpu():
    """Parse --gpu argument early and set CUDA_VISIBLE_DEVICES before any CUDA imports."""
    # If CUDA_VISIBLE_DEVICES is already set externally, respect it
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        gpu_str = os.environ["CUDA_VISIBLE_DEVICES"]
        print(f"[eval] Using CUDA_VISIBLE_DEVICES={gpu_str} (set externally)")
        return int(gpu_str.split(",")[0])

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--gpu", type=int, default=None)
    args, _ = parser.parse_known_args()

    if args.gpu is None:
        # Try to find a free GPU
        free_gpu = find_free_gpu()
        if free_gpu is not None:
            print(f"[eval] Auto-selected GPU {free_gpu} (least memory used)")
            args.gpu = free_gpu
        else:
            print("[eval] ERROR: No --gpu specified and no free GPU found (<1GB memory).")
            print("[eval] Please specify --gpu N where N is a free GPU index.")
            print("[eval] Check available GPUs with: nvidia-smi --query-gpu=index,memory.used --format=csv")
            sys.exit(1)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    print(f"[eval] Using GPU {args.gpu}")
    return args.gpu

# Set GPU before any other imports
_SELECTED_GPU = setup_gpu()

import json
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


# =============================================================================
# Benchmark Configurations
# =============================================================================

BENCHMARK_CONFIGS = {
    "gsm8k": {
        "max_tokens": 1024,
        "max_model_len": 2048,
        # Include "Question:" which the model often outputs instead of "Q:"
        "stop_sequences": ["[Question]", "Question:", "Q:", "\n\n\n"],
    },
    "math": {
        "max_tokens": 1024,
        "max_model_len": 2048,
        "stop_sequences": ["Problem:", "\n\n\n"],
    },
    "ifeval": {
        "max_tokens": 2048,
        "max_model_len": 4096,
        "stop_sequences": None,
    },
    "ifbench": {
        "max_tokens": 2048,
        "max_model_len": 4096,
        "stop_sequences": None,
    },
    "mmlu": {
        "max_tokens": 32,
        "max_model_len": 2048,
        "stop_sequences": ["\n", "Question:", "Q:"],
    },
}


# =============================================================================
# Dataset Loaders
# =============================================================================

def load_gsm8k_test():
    """Load GSM8K test set."""
    from datasets import load_dataset

    ds = load_dataset("openai/gsm8k", "main", split="test")

    rows = []
    for i, item in enumerate(ds):
        # Format: "Q: ... A:" to match training format
        prompt = f"Q: {item['question'].strip()}\nA:"
        answer = item["answer"].split("####")[-1].strip()
        rows.append({
            "id": f"gsm8k_{i}",
            "prompt": prompt,
            "gold_answer": answer,
        })

    print(f"[load_gsm8k_test] Loaded {len(rows)} examples")
    return rows


def load_math_test():
    """Load MATH test set (Hendrycks et al.)."""
    from datasets import load_dataset

    # Load all subjects from EleutherAI/hendrycks_math
    subjects = [
        "algebra", "counting_and_probability", "geometry",
        "intermediate_algebra", "number_theory", "prealgebra", "precalculus"
    ]

    rows = []
    for subject in subjects:
        ds = load_dataset("EleutherAI/hendrycks_math", subject, split="test")
        for i, item in enumerate(ds):
            # Format: "Problem:\n{problem}\n\nSolution:" to match training format
            prompt = f"Problem:\n{item['problem'].strip()}\n\nSolution:"
            # Extract answer from solution (boxed answer)
            solution = item["solution"]
            rows.append({
                "id": f"math_{subject}_{i}",
                "prompt": prompt,
                "gold_answer": solution,  # Full solution for MathVerifier
                "level": item.get("level", ""),
                "type": subject,
            })

    print(f"[load_math_test] Loaded {len(rows)} examples")
    return rows


def load_ifeval():
    """Load Google IFEval test set."""
    from datasets import load_dataset

    ds = load_dataset("google/IFEval", split="train")  # IFEval only has "train" split

    rows = []
    for i, item in enumerate(ds):
        rows.append({
            "id": f"ifeval_{i}",
            "prompt": item["prompt"],
            "instruction_id_list": item.get("instruction_id_list", []),
            "kwargs": item.get("kwargs", []),
        })

    print(f"[load_ifeval] Loaded {len(rows)} examples")
    return rows


def load_ifbench():
    """Load AllenAI IFBench test set."""
    from datasets import load_dataset

    ds = load_dataset("allenai/IFBench_test", split="train")  # IFBench_test only has "train" split

    rows = []
    for i, item in enumerate(ds):
        rows.append({
            "id": f"ifbench_{i}",
            "prompt": item["prompt"],
            "instruction_id_list": item.get("instruction_id_list", []),
            "kwargs": item.get("kwargs", []),
        })

    print(f"[load_ifbench] Loaded {len(rows)} examples")
    return rows


def load_mmlu():
    """Load CAIS MMLU test set (zero-shot multiple choice)."""
    from datasets import load_dataset

    ds = load_dataset("cais/mmlu", "all", split="test")

    LETTERS = ["A", "B", "C", "D"]
    rows = []
    for i, item in enumerate(ds):
        choices_str = "\n".join(f"({LETTERS[j]}) {c}" for j, c in enumerate(item["choices"]))
        prompt = f"Question: {item['question'].strip()}\nChoices:\n{choices_str}\nAnswer: ("
        gold_letter = LETTERS[item["answer"]]
        rows.append({
            "id": f"mmlu_{i}",
            "prompt": prompt,
            "gold_answer": gold_letter,
            "subject": item.get("subject", "unknown"),
        })

    print(f"[load_mmlu] Loaded {len(rows)} examples")
    return rows


LOADERS = {
    "gsm8k": load_gsm8k_test,
    "math": load_math_test,
    "ifeval": load_ifeval,
    "ifbench": load_ifbench,
    "mmlu": load_mmlu,
}


# =============================================================================
# Verifiers
# =============================================================================

def verify_gsm8k(completions: list[str], rows: list[dict]) -> list[dict]:
    """Verify GSM8K completions using MathVerifier."""
    from rlvr_experiments.verifiers.math import MathVerifier

    verifier = MathVerifier(timeout=5.0, max_workers=8)

    results = []
    for completion, row in zip(completions, rows):
        reward = verifier.verify(completion, row["gold_answer"])
        results.append({
            "reward": reward,
            "correct": reward > 0,
        })

    return results


def verify_math(completions: list[str], rows: list[dict]) -> list[dict]:
    """Verify MATH completions using MathVerifier."""
    from rlvr_experiments.verifiers.math import MathVerifier

    verifier = MathVerifier(timeout=5.0, max_workers=8)

    results = []
    for completion, row in zip(completions, rows):
        reward = verifier.verify(completion, row["gold_answer"])
        results.append({
            "reward": reward,
            "correct": reward > 0,
            "level": row.get("level", ""),
            "type": row.get("type", ""),
        })

    return results


def verify_ifeval(completions: list[str], rows: list[dict]) -> list[dict]:
    """Verify IFEval completions using IFMultiConstraintsVerifier."""
    from rlvr_experiments.verifiers.if_multi_constraints import (
        INSTRUCTION_FUNCTIONS,
        _remove_thinking_section,
    )

    results = []
    for completion, row in zip(completions, rows):
        instruction_id_list = row.get("instruction_id_list", [])
        kwargs_list = row.get("kwargs", [])

        if not instruction_id_list:
            results.append({
                "all_pass": False,
                "pass_count": 0,
                "total_count": 0,
                "score": 0.0,
                "per_instruction": [],
            })
            continue

        answer = _remove_thinking_section(completion)
        if not answer:
            results.append({
                "all_pass": False,
                "pass_count": 0,
                "total_count": len(instruction_id_list),
                "score": 0.0,
                "per_instruction": [(iid, False) for iid in instruction_id_list],
            })
            continue

        per_instruction = []
        pass_count = 0

        for i, instruction_id in enumerate(instruction_id_list):
            kwargs = kwargs_list[i] if i < len(kwargs_list) else {}
            if kwargs is None:
                kwargs = {}

            func = INSTRUCTION_FUNCTIONS.get(instruction_id)
            if func is None:
                per_instruction.append((instruction_id, False))
                continue

            kwargs = {k: v for k, v in kwargs.items() if v is not None}
            try:
                ok = func(answer, **kwargs) if kwargs else func(answer)
            except Exception:
                ok = False

            per_instruction.append((instruction_id, ok))
            if ok:
                pass_count += 1

        total = len(instruction_id_list)
        results.append({
            "all_pass": pass_count == total,
            "pass_count": pass_count,
            "total_count": total,
            "score": pass_count / total if total > 0 else 0.0,
            "per_instruction": per_instruction,
        })

    return results


def verify_ifbench(completions: list[str], rows: list[dict]) -> list[dict]:
    """Verify IFBench completions using IFBenchVerifier."""
    from rlvr_experiments.verifiers.ifbench import IFBenchVerifier

    verifier = IFBenchVerifier()

    results = []
    for completion, row in zip(completions, rows):
        instruction_id_list = row.get("instruction_id_list", [])
        kwargs_list = row.get("kwargs", [])

        result = verifier.verify(completion, instruction_id_list, kwargs_list)
        results.append({
            "all_pass": result["all_pass"],
            "pass_count": result["pass_count"],
            "total_count": result["total_count"],
            "score": result["pass_count"] / result["total_count"] if result["total_count"] > 0 else 0.0,
            "per_instruction": result["per_instruction"],
        })

    return results


def verify_mmlu(completions: list[str], rows: list[dict]) -> list[dict]:
    """Verify MMLU completions by extracting A/B/C/D answer."""
    import re

    results = []
    for completion, row in zip(completions, rows):
        gold = row["gold_answer"]
        text = completion.strip()
        # Try to extract first A-D letter
        extracted = None
        match = re.search(r'\(?([A-D])\)?', text)
        if match:
            extracted = match.group(1)
        correct = extracted == gold if extracted else False
        results.append({
            "correct": correct,
            "extracted": extracted,
            "gold_answer": gold,
            "subject": row.get("subject", "unknown"),
        })

    return results


VERIFIERS = {
    "gsm8k": verify_gsm8k,
    "math": verify_math,
    "ifeval": verify_ifeval,
    "ifbench": verify_ifbench,
    "mmlu": verify_mmlu,
}


# =============================================================================
# Summary Computation
# =============================================================================

def compute_summary_gsm8k(results: list[dict], rows: list[dict]) -> dict:
    """Compute summary metrics for GSM8K."""
    n_correct = sum(1 for r in results if r["correct"])
    n_total = len(results)
    accuracy = n_correct / n_total if n_total > 0 else 0.0

    return {
        "n_examples": n_total,
        "n_correct": n_correct,
        "accuracy": accuracy,
    }


def compute_summary_ifeval(results: list[dict], rows: list[dict]) -> dict:
    """Compute summary metrics for IFEval/IFBench."""
    n_prompts = len(results)
    prompt_pass = sum(1 for r in results if r["all_pass"])
    inst_pass = sum(r["pass_count"] for r in results)
    inst_total = sum(r["total_count"] for r in results)

    prompt_level_acc = prompt_pass / n_prompts if n_prompts > 0 else 0.0
    inst_level_acc = inst_pass / inst_total if inst_total > 0 else 0.0

    return {
        "n_prompts": n_prompts,
        "prompt_pass": prompt_pass,
        "prompt_level_strict_acc": prompt_level_acc,
        "inst_pass": inst_pass,
        "inst_total": inst_total,
        "inst_level_acc": inst_level_acc,
    }


def compute_summary_math(results: list[dict], rows: list[dict]) -> dict:
    """Compute summary metrics for MATH, including per-level and per-subject breakdowns."""
    n_correct = sum(1 for r in results if r["correct"])
    n_total = len(results)
    accuracy = n_correct / n_total if n_total > 0 else 0.0

    # Per-level breakdown
    level_stats = {}
    for r in results:
        level = r.get("level", "unknown")
        if level not in level_stats:
            level_stats[level] = {"correct": 0, "total": 0}
        level_stats[level]["total"] += 1
        if r["correct"]:
            level_stats[level]["correct"] += 1

    level_accuracy = {
        level: stats["correct"] / stats["total"] if stats["total"] > 0 else 0.0
        for level, stats in sorted(level_stats.items())
    }

    # Per-subject breakdown
    subject_stats = {}
    for r in results:
        subject = r.get("type", "unknown")
        if subject not in subject_stats:
            subject_stats[subject] = {"correct": 0, "total": 0}
        subject_stats[subject]["total"] += 1
        if r["correct"]:
            subject_stats[subject]["correct"] += 1

    subject_accuracy = {
        subject: stats["correct"] / stats["total"] if stats["total"] > 0 else 0.0
        for subject, stats in sorted(subject_stats.items())
    }

    return {
        "n_examples": n_total,
        "n_correct": n_correct,
        "accuracy": accuracy,
        "level_accuracy": level_accuracy,
        "subject_accuracy": subject_accuracy,
    }


def compute_summary_mmlu(results: list[dict], rows: list[dict]) -> dict:
    """Compute summary metrics for MMLU, including per-subject breakdown."""
    n_correct = sum(1 for r in results if r["correct"])
    n_total = len(results)
    accuracy = n_correct / n_total if n_total > 0 else 0.0

    subject_stats = {}
    for r in results:
        subject = r.get("subject", "unknown")
        if subject not in subject_stats:
            subject_stats[subject] = {"correct": 0, "total": 0}
        subject_stats[subject]["total"] += 1
        if r["correct"]:
            subject_stats[subject]["correct"] += 1

    subject_accuracy = {
        subject: stats["correct"] / stats["total"] if stats["total"] > 0 else 0.0
        for subject, stats in sorted(subject_stats.items())
    }

    return {
        "n_examples": n_total,
        "n_correct": n_correct,
        "accuracy": accuracy,
        "subject_accuracy": subject_accuracy,
    }


SUMMARY_FUNCTIONS = {
    "gsm8k": compute_summary_gsm8k,
    "math": compute_summary_math,
    "ifeval": compute_summary_ifeval,
    "ifbench": compute_summary_ifeval,  # Same metrics as IFEval
    "mmlu": compute_summary_mmlu,
}


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Evaluate checkpoint on a benchmark")
    parser.add_argument("checkpoint_path", type=str, help="Path to checkpoint")
    parser.add_argument("output_dir", type=str, help="Output directory for results")
    parser.add_argument("--benchmark", type=str, required=True,
                        choices=["gsm8k", "math", "ifeval", "ifbench", "mmlu"],
                        help="Benchmark to evaluate on")
    parser.add_argument("--gpu", type=int, default=0, help="GPU to use")
    parser.add_argument("--max-tokens", type=int, default=None,
                        help="Override max tokens for generation")
    args = parser.parse_args()

    # Get benchmark config
    config = BENCHMARK_CONFIGS[args.benchmark]
    max_tokens = args.max_tokens or config["max_tokens"]
    max_model_len = config["max_model_len"]
    stop_sequences = config["stop_sequences"]

    # GPU already set at module load time via setup_gpu()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset
    print(f"[eval] Loading {args.benchmark} benchmark...")
    loader = LOADERS[args.benchmark]
    rows = loader()

    # Extract prompts
    prompts = [row["prompt"] for row in rows]

    # Initialize vLLM
    from vllm import LLM, SamplingParams

    print(f"[eval] Loading model from {args.checkpoint_path}...")
    llm = LLM(
        model=args.checkpoint_path,
        tensor_parallel_size=1,
        dtype="bfloat16",
        gpu_memory_utilization=0.9,
        max_model_len=max_model_len,
        trust_remote_code=True,
    )

    sampling_params = SamplingParams(
        temperature=0,  # Greedy
        max_tokens=max_tokens,
        stop=stop_sequences,
    )

    # Generate completions
    print(f"[eval] Generating completions for {len(prompts)} prompts...")
    t0 = time.time()
    outputs = llm.generate(prompts, sampling_params)
    gen_time = time.time() - t0
    print(f"[eval] Generation took {gen_time:.1f}s ({len(prompts)/gen_time:.1f} prompts/s)")

    # Extract completions
    completions = [out.outputs[0].text for out in outputs]

    # Save raw completions
    completions_path = output_dir / "completions.jsonl"
    with open(completions_path, "w") as f:
        for row, completion in zip(rows, completions):
            f.write(json.dumps({
                **row,
                "completion": completion,
            }) + "\n")
    print(f"[eval] Saved completions to {completions_path}")

    # Verify completions
    print(f"[eval] Verifying completions...")
    t0 = time.time()
    verifier = VERIFIERS[args.benchmark]
    results = verifier(completions, rows)
    verify_time = time.time() - t0
    print(f"[eval] Verification took {verify_time:.1f}s")

    # Save results with verification
    results_path = output_dir / "results.jsonl"
    with open(results_path, "w") as f:
        for row, completion, result in zip(rows, completions, results):
            f.write(json.dumps({
                **row,
                "completion": completion,
                "verification": result,
            }) + "\n")
    print(f"[eval] Saved results to {results_path}")

    # Compute and save summary
    summary_fn = SUMMARY_FUNCTIONS[args.benchmark]
    metrics = summary_fn(results, rows)

    summary = {
        "checkpoint": args.checkpoint_path,
        "benchmark": args.benchmark,
        "generation_time_s": gen_time,
        "verification_time_s": verify_time,
        **metrics,
    }

    summary_path = output_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[eval] Saved summary to {summary_path}")

    # Print summary
    print(f"\n{'='*50}")
    print(f"RESULTS: {args.benchmark}")
    print(f"{'='*50}")

    if args.benchmark == "gsm8k":
        print(f"Accuracy: {metrics['accuracy']:.4f} ({metrics['n_correct']}/{metrics['n_examples']})")
    elif args.benchmark == "math":
        print(f"Overall Accuracy: {metrics['accuracy']:.4f} ({metrics['n_correct']}/{metrics['n_examples']})")
        print(f"\nPer-level accuracy:")
        for level, acc in metrics["level_accuracy"].items():
            print(f"  {level}: {acc:.4f}")
        print(f"\nPer-subject accuracy:")
        for subject, acc in metrics["subject_accuracy"].items():
            print(f"  {subject}: {acc:.4f}")
    elif args.benchmark == "mmlu":
        print(f"Overall Accuracy: {metrics['accuracy']:.4f} ({metrics['n_correct']}/{metrics['n_examples']})")
        print(f"\nPer-subject accuracy (showing top/bottom 5):")
        sorted_subjects = sorted(metrics["subject_accuracy"].items(), key=lambda x: x[1], reverse=True)
        for subject, acc in sorted_subjects[:5]:
            print(f"  {subject}: {acc:.4f}")
        print(f"  ...")
        for subject, acc in sorted_subjects[-5:]:
            print(f"  {subject}: {acc:.4f}")
    else:
        print(f"Prompt-level strict accuracy: {metrics['prompt_pass']}/{metrics['n_prompts']} = {metrics['prompt_level_strict_acc']:.2%}")
        print(f"Instruction-level accuracy:   {metrics['inst_pass']}/{metrics['inst_total']} = {metrics['inst_level_acc']:.2%}")

    print(f"{'='*50}")


if __name__ == "__main__":
    main()
