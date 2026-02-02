#!/usr/bin/env python3
"""
Generate completions and compute pass@k for any dataset with a load_xxx function and verifier.

Generates N completions per prompt using vLLM (temperature=1 by default for diversity),
verifies each completion using the dataset's verifier, and streams results to disk.
Supports resumption from a specific prompt index if the run is interrupted.

Usage:
    # Basic: Generate 128 completions per prompt for gsm8k training set
    python scripts/eval_pass_at_k.py gsm8k --split train --n 128 --output-dir results/qwen3-1.7B-base/evals/gsm8k

    # Resume from prompt index 100
    python scripts/eval_pass_at_k.py gsm8k --split train --n 128 --output-dir results/... --resume-from 100

    # Use specific GPUs
    python scripts/eval_pass_at_k.py gsm8k --split train --n 128 --gpus 0,1,2,3

    # Different batch size (prompts processed at once)
    python scripts/eval_pass_at_k.py gsm8k --split train --n 128 --batch-size 32
"""

import argparse
import asyncio
import json
import os
import sys
import time
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from vllm import LLM, SamplingParams


# =============================================================================
# Dataset and Verifier Registry
# =============================================================================

DATASET_REGISTRY = {
    "gsm8k": {
        "loader": "load_gsm8k",
        "verifier": "MathVerifier",
        "verifier_module": "rlvr_experiments.verifiers.math",
        "answer_key": "answer",  # Key in problem dict for ground truth
        "stop": ["[Question]", "Question:", "Q:", "\n\n\n"],
    },
    "math": {
        "loader": "load_math",
        "verifier": "MathVerifier",
        "verifier_module": "rlvr_experiments.verifiers.math",
        "answer_key": "answer",
        "stop": ["Problem:", "\n\n\n"],
    },
    "aime": {
        "loader": "load_aime",
        "verifier": "MathVerifier",
        "verifier_module": "rlvr_experiments.verifiers.math",
        "answer_key": "answer",
        "stop": ["Problem:", "\n\n\n"],
    },
    "beyondaime": {
        "loader": "load_beyondaime",
        "verifier": "MathVerifier",
        "verifier_module": "rlvr_experiments.verifiers.math",
        "answer_key": "answer",
        "stop": ["Problem:", "\n\n\n"],
    },
    "deepscaler": {
        "loader": "load_deepscaler",
        "verifier": "MathVerifier",
        "verifier_module": "rlvr_experiments.verifiers.math",
        "answer_key": "answer",
    },
    "mbpp": {
        "loader": "load_mbpp",
        "verifier": "MBPPVerifier",
        "verifier_module": "rlvr_experiments.verifiers.code",
        "answer_key": None,  # Uses problem dict directly
    },
    "humaneval": {
        "loader": "load_humaneval",
        "verifier": "HumanEvalVerifier",
        "verifier_module": "rlvr_experiments.verifiers.code",
        "answer_key": None,
    },
    "ifeval": {
        "loader": "load_ifeval",
        "verifier": "IFMultiConstraintsVerifier",
        "verifier_module": "rlvr_experiments.verifiers.if_multi_constraints",
        "answer_key": "ground_truth",
    },
    "if_multi_constraints": {
        "loader": "load_if_multi_constraints",
        "verifier": "IFMultiConstraintsVerifier",
        "verifier_module": "rlvr_experiments.verifiers.if_multi_constraints",
        "answer_key": "ground_truth",
    },
    "allenai_rlvr": {
        "loader": "load_allenai_rlvr",
        "verifier": "MultiVerifier",
        "verifier_module": "rlvr_experiments.verifiers.multi",
        "answer_key": None,  # Uses verifier_type dispatch
    },
    "ifbench": {
        "loader": "load_ifbench",
        "verifier": "IFBenchVerifier",
        "verifier_module": "rlvr_experiments.verifiers.ifbench",
        "answer_key": None,
    },
}


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class CompletionResult:
    """Result for a single completion."""
    text: str
    score: float
    finish_reason: str
    num_tokens: int


@dataclass
class PromptResult:
    """Results for a single prompt."""
    prompt_id: str
    prompt: str
    formatted_prompt: str
    problem: dict
    completions: list[CompletionResult] = field(default_factory=list)
    num_correct: int = 0
    pass_rate: float = 0.0
    generation_time_s: float = 0.0
    verification_time_s: float = 0.0


# =============================================================================
# Helpers
# =============================================================================

def load_dataset_items(dataset_name: str, split: str, **loader_kwargs) -> list[dict]:
    """Load dataset without Ray to avoid CPU contention in multi-shard runs.

    Instead of calling data.py loaders (which return ray.data.Dataset), we replicate
    the HuggingFace loading + preprocessing logic directly here.
    """
    items = _load_dataset_items_no_ray(dataset_name, split, **loader_kwargs)
    print(f"Loaded {len(items)} prompts from {dataset_name} ({split})")
    return items


def _load_hf_gsm8k(split: str) -> list[dict]:
    """Load GSM8k without Ray."""
    from datasets import load_dataset as hf_load_dataset
    import os

    local_cache = f"/tmp/gsm8k_{split}_cache"
    if os.path.exists(local_cache) and os.path.exists(os.path.join(local_cache, "dataset_info.json")):
        from datasets import Dataset
        hf_dataset = Dataset.load_from_disk(local_cache)
    else:
        hf_dataset = hf_load_dataset("openai/gsm8k", "main", split=split)

    from rlvr_experiments.data import GSM8K_SYSTEM_PROMPT, GSM8K_ASSISTANT_PREFIX, GSM8K_MAX_COMPLETION_LEN
    results = []
    for i, row in enumerate(hf_dataset):
        question = f"Q: {row['question'].strip()}\nA:"
        answer = row["answer"].split("####")[-1].strip()
        results.append({
            "prompt": question,
            "problem": {
                "answer": answer,
                "prompt_id": f"gsm8k_{i}",
                "verifier_type": "gsm8k",
                "dataset_name": "gsm8k",
                "system_prompt": GSM8K_SYSTEM_PROMPT,
                "assistant_prefix": GSM8K_ASSISTANT_PREFIX,
                "max_completion_len": GSM8K_MAX_COMPLETION_LEN,
            },
        })
    return results


def _load_hf_math(split: str, **kwargs) -> list[dict]:
    """Load MATH without Ray, matching data.py's load_math exactly."""
    from datasets import load_dataset as hf_load_dataset
    import os
    from rlvr_experiments.data import MATH_SYSTEM_PROMPT, MATH_ASSISTANT_PREFIX, MATH_MAX_COMPLETION_LEN

    level_filter = kwargs.get("level", None)

    # Try local cache first
    local_cache = f"/tmp/math_{split}_cache"
    if os.path.exists(local_cache) and os.path.exists(os.path.join(local_cache, "dataset_info.json")):
        from datasets import Dataset
        hf_dataset = Dataset.load_from_disk(local_cache)
        all_rows = list(hf_dataset)
    else:
        subjects = [
            "algebra", "counting_and_probability", "geometry",
            "intermediate_algebra", "number_theory", "prealgebra", "precalculus",
        ]
        all_rows = []
        for subject in subjects:
            hf_dataset = hf_load_dataset("EleutherAI/hendrycks_math", subject, split=split)
            all_rows.extend(list(hf_dataset))

    # Filter by level if specified
    if level_filter is not None:
        level_strs = {f"Level {l}" for l in level_filter}
        all_rows = [row for row in all_rows if row["level"] in level_strs]

    # Build subject-based prompt IDs (matching data.py)
    subject_counters = {}
    results = []
    for row in all_rows:
        subject = row["type"].lower()
        if subject not in subject_counters:
            subject_counters[subject] = 0
        subject_idx = subject_counters[subject]
        subject_counters[subject] += 1

        prompt = f"Problem:\n{row['problem'].strip()}\n\nSolution:"
        prompt_id = f"math_{subject}_{subject_idx}"
        results.append({
            "prompt": prompt,
            "problem": {
                "answer": row["solution"],
                "prompt_id": prompt_id,
                "verifier_type": "math",
                "dataset_name": "math",
                "system_prompt": MATH_SYSTEM_PROMPT,
                "assistant_prefix": MATH_ASSISTANT_PREFIX,
                "max_completion_len": MATH_MAX_COMPLETION_LEN,
            },
        })
    return results


def _load_hf_aime(split: str) -> list[dict]:
    """Load AIME without Ray, matching data.py's load_aime exactly."""
    from datasets import load_dataset as hf_load_dataset
    from rlvr_experiments.data import MATH_SYSTEM_PROMPT, MATH_ASSISTANT_PREFIX, MATH_MAX_COMPLETION_LEN

    # AIME always uses train split
    hf_dataset = hf_load_dataset("AI-MO/aimo-validation-aime", split="train")
    results = []
    for i, row in enumerate(hf_dataset):
        prompt = f"Problem:\n{row['problem'].strip()}\n\nSolution:"
        results.append({
            "prompt": prompt,
            "problem": {
                "answer": row["answer"],
                "prompt_id": f"aime_{i}",
                "verifier_type": "math",
                "dataset_name": "aime",
                "system_prompt": MATH_SYSTEM_PROMPT,
                "assistant_prefix": MATH_ASSISTANT_PREFIX,
                "max_completion_len": MATH_MAX_COMPLETION_LEN,
            },
        })
    return results


def _load_hf_beyondaime(split: str) -> list[dict]:
    """Load BeyondAIME without Ray, matching data.py's load_beyondaime exactly."""
    from datasets import load_dataset as hf_load_dataset
    from rlvr_experiments.data import MATH_SYSTEM_PROMPT, MATH_ASSISTANT_PREFIX, MATH_MAX_COMPLETION_LEN

    # BeyondAIME always uses test split
    hf_dataset = hf_load_dataset("ByteDance-Seed/BeyondAIME", split="test")
    results = []
    for i, row in enumerate(hf_dataset):
        prompt = f"Problem:\n{row['problem'].strip()}\n\nSolution:"
        results.append({
            "prompt": prompt,
            "problem": {
                "answer": str(row["answer"]),
                "prompt_id": f"beyondaime_{i}",
                "verifier_type": "math",
                "dataset_name": "beyondaime",
                "system_prompt": MATH_SYSTEM_PROMPT,
                "assistant_prefix": MATH_ASSISTANT_PREFIX,
                "max_completion_len": MATH_MAX_COMPLETION_LEN,
            },
        })
    return results


def _load_hf_ifeval(split: str) -> list[dict]:
    """Load Google IFEval (541 prompts) without Ray.

    Uses google/IFEval (the eval benchmark), NOT allenai/RLVR-IFeval (14973 training prompts).
    """
    from datasets import load_dataset as hf_load_dataset
    import json as _json

    hf_dataset = hf_load_dataset("google/IFEval", split="train")
    items = list(hf_dataset)
    print(f"[load_ifeval] Loaded {len(items)} prompts from google/IFEval")

    results = []
    for i, row in enumerate(items):
        prompt = row["prompt"]
        # Parse instruction_id_list and kwargs from the row
        instruction_id_list = row.get("instruction_id_list", [])
        kwargs_raw = row.get("kwargs", [])
        # kwargs may be JSON strings or dicts
        kwargs = []
        for k in kwargs_raw:
            if isinstance(k, str):
                kwargs.append(_json.loads(k))
            else:
                kwargs.append(k)

        prompt_id = f"ifeval_{i}"
        # Build ground_truth in the format IFMultiConstraintsVerifier expects
        ground_truth = [{"instruction_id": instruction_id_list, "kwargs": kwargs}]
        results.append({
            "prompt": prompt,
            "problem": {
                "prompt_id": prompt_id,
                "instruction_id_list": instruction_id_list,
                "kwargs": kwargs,
                "ground_truth": ground_truth,
                "verifier_type": "ifeval",
                "dataset_name": "ifeval",
            },
        })
    return results


def _load_hf_ifbench(split: str) -> list[dict]:
    """Load IFBench without Ray."""
    from datasets import load_dataset as hf_load_dataset

    hf_dataset = hf_load_dataset("allenai/IFBench_test", split="train")
    items = list(hf_dataset)

    results = []
    for i, row in enumerate(items):
        prompt_id = f"ifbench_{i}"
        results.append({
            "prompt": row["prompt"],
            "problem": {
                "prompt_id": prompt_id,
                "instruction_id_list": row.get("instruction_id_list", []),
                "kwargs": row.get("kwargs", []),
                "verifier_type": "ifbench",
                "dataset_name": "ifbench",
            },
        })
    return results


def _load_dataset_items_no_ray(dataset_name: str, split: str, **loader_kwargs) -> list[dict]:
    """Dispatch to Ray-free loaders for known datasets, fall back to Ray for others."""
    if dataset_name == "gsm8k":
        return _load_hf_gsm8k(split)
    elif dataset_name == "math":
        return _load_hf_math(split, **loader_kwargs)
    elif dataset_name == "aime":
        return _load_hf_aime(split)
    elif dataset_name == "beyondaime":
        return _load_hf_beyondaime(split)
    elif dataset_name == "ifeval":
        return _load_hf_ifeval(split)
    elif dataset_name == "ifbench":
        return _load_hf_ifbench(split)
    else:
        # Fall back to Ray-based loader for other datasets
        from rlvr_experiments import data
        loader_name = DATASET_REGISTRY[dataset_name]["loader"]
        loader_fn = getattr(data, loader_name)
        if dataset_name == "humaneval":
            ray_ds = loader_fn(**loader_kwargs)
        else:
            ray_ds = loader_fn(split=split, **loader_kwargs)
        return list(ray_ds.iter_rows())


def get_verifier(dataset_name: str, **verifier_kwargs):
    """Get the appropriate verifier for a dataset."""
    import importlib

    config = DATASET_REGISTRY[dataset_name]
    module = importlib.import_module(config["verifier_module"])
    verifier_cls = getattr(module, config["verifier"])
    return verifier_cls(**verifier_kwargs)


def format_prompt_for_generation(item: dict, dataset_name: str) -> str:
    """Format a dataset item into a prompt string for the model.

    Uses the raw prompt from the dataset (no chat template for base models).
    """
    # The data loaders already format the prompt appropriately
    return item["prompt"]


async def verify_completions_batch(
    verifier,
    dataset_name: str,
    problem: dict,
    completions: list[str],
) -> list[float]:
    """Verify a batch of completions for a single problem."""
    config = DATASET_REGISTRY[dataset_name]

    # Use the verifier's verify_completions method if available
    if hasattr(verifier, "verify_completions"):
        result = await verifier.verify_completions(problem, completions)
        # CodeVerifier returns (scores, durations_ms) tuple; extract just scores
        if isinstance(result, tuple):
            result = result[0]
        # For ifeval/ifbench: binarize scores (1.0 only if ALL constraints pass)
        if dataset_name in ("ifeval", "ifbench"):
            result = [1.0 if s >= 1.0 else 0.0 for s in result]
        return result

    # Fall back to individual verification
    answer_key = config.get("answer_key")
    if answer_key:
        target = problem.get(answer_key, "")
        scores = []
        for completion in completions:
            score = verifier.verify(completion, target)
            scores.append(score)
        return scores

    # For code verifiers, verify each completion
    scores = []
    for completion in completions:
        try:
            if hasattr(verifier, "verify"):
                result = await verifier.verify(problem, completion)
                scores.append(1.0 if result.all_passed else 0.0)
            else:
                scores.append(0.0)
        except Exception as e:
            print(f"Verification error: {e}")
            scores.append(0.0)
    return scores


# =============================================================================
# Main Generation and Verification
# =============================================================================

def run_generation_and_verification(
    items: list[dict],
    dataset_name: str,
    model_path: str,
    n_completions: int,
    batch_size: int,
    max_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    max_model_len: int,
    gpus: str,
    output_dir: Path,
    resume_from: int,
    verifier_kwargs: dict,
):
    """Run generation and verification, streaming results to disk."""

    # Set CUDA devices
    os.environ["CUDA_VISIBLE_DEVICES"] = gpus
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    # Initialize vLLM
    print(f"\nInitializing vLLM with model: {model_path}")
    print(f"  GPUs: {gpus}")
    print(f"  Max model len: {max_model_len}")

    num_gpus = len(gpus.split(","))
    llm = LLM(
        model=model_path,
        tensor_parallel_size=1,  # Always TP=1 as per CLAUDE.md
        max_model_len=max_model_len,
        gpu_memory_utilization=float(os.environ.get("VLLM_GPU_MEMORY_UTILIZATION", "0.90")),
        dtype="bfloat16",
        enable_prefix_caching=True,
        trust_remote_code=True,
    )

    # Get dataset-specific stop tokens
    stop_tokens = DATASET_REGISTRY[dataset_name].get("stop", [])
    if stop_tokens:
        print(f"  Stop tokens: {stop_tokens}")

    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        max_tokens=max_tokens,
        n=n_completions,
        stop=stop_tokens if stop_tokens else None,
    )

    # Initialize verifier
    print(f"\nInitializing verifier: {DATASET_REGISTRY[dataset_name]['verifier']}")
    verifier = get_verifier(dataset_name, **verifier_kwargs)

    # Output files
    output_dir.mkdir(parents=True, exist_ok=True)
    completions_file = output_dir / "completions.jsonl"
    results_file = output_dir / "verification_results.jsonl"
    summary_file = output_dir / "summary.json"
    metadata_file = output_dir / "metadata.json"

    # Write metadata
    metadata = {
        "dataset": dataset_name,
        "model_path": model_path,
        "n_completions": n_completions,
        "batch_size": batch_size,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "max_model_len": max_model_len,
        "gpus": gpus,
        "num_prompts": len(items),
        "started_at": datetime.now().isoformat(),
    }
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2)

    # Resume handling
    processed_ids = set()
    if resume_from > 0:
        print(f"\nResuming from prompt index {resume_from}")
        # Read already processed prompt IDs from results file
        if results_file.exists():
            with open(results_file) as f:
                for line in f:
                    if line.strip():
                        result = json.loads(line)
                        processed_ids.add(result["prompt_id"])
            print(f"  Found {len(processed_ids)} already processed prompts")

    # Process in batches
    total_prompts = len(items)
    total_completions = 0
    total_correct = 0
    all_pass_rates = []

    start_time = time.time()
    batch_idx = 0

    # Open output files in append mode
    completions_f = open(completions_file, "a")
    results_f = open(results_file, "a")

    try:
        for batch_start in range(0, total_prompts, batch_size):
            batch_end = min(batch_start + batch_size, total_prompts)
            batch_items = items[batch_start:batch_end]

            # Filter out already processed items
            batch_items = [
                item for item in batch_items
                if item["problem"]["prompt_id"] not in processed_ids
            ]

            if not batch_items:
                print(f"Batch {batch_idx}: all {batch_end - batch_start} prompts already processed, skipping")
                batch_idx += 1
                continue

            print(f"\nBatch {batch_idx}: processing {len(batch_items)} prompts ({batch_start+1}-{batch_end}/{total_prompts})")

            # Format prompts
            formatted_prompts = [
                format_prompt_for_generation(item, dataset_name)
                for item in batch_items
            ]

            # Generate completions
            gen_start = time.time()
            outputs = llm.generate(formatted_prompts, sampling_params)
            gen_time = time.time() - gen_start

            print(f"  Generation: {gen_time:.1f}s for {len(batch_items) * n_completions} completions")

            # Verify and save results
            verify_start = time.time()

            for item_idx, (item, output) in enumerate(zip(batch_items, outputs)):
                problem = item["problem"]
                prompt_id = problem["prompt_id"]

                # Extract completions
                completions = [
                    {
                        "text": o.text,
                        "finish_reason": o.finish_reason,
                        "num_tokens": len(o.token_ids),
                    }
                    for o in output.outputs
                ]

                # Write raw completions
                completion_record = {
                    "prompt_id": prompt_id,
                    "prompt": item["prompt"],
                    "problem": problem,
                    "completions": completions,
                }
                completions_f.write(json.dumps(completion_record) + "\n")
                completions_f.flush()

                # Verify completions
                completion_texts = [c["text"] for c in completions]

                # Use asyncio to run verification
                scores = asyncio.run(
                    verify_completions_batch(verifier, dataset_name, problem, completion_texts)
                )

                # Combine completion data with scores
                for comp, score in zip(completions, scores):
                    comp["score"] = score

                num_correct = sum(1 for s in scores if s > 0)
                pass_rate = num_correct / len(scores) if scores else 0.0

                # Write verification result
                result_record = {
                    "prompt_id": prompt_id,
                    "num_completions": len(scores),
                    "num_correct": num_correct,
                    "pass_rate": pass_rate,
                    "scores": scores,
                    "completions": completions,  # Include full completion data with scores
                }
                results_f.write(json.dumps(result_record) + "\n")
                results_f.flush()

                total_completions += len(scores)
                total_correct += num_correct
                all_pass_rates.append(pass_rate)
                processed_ids.add(prompt_id)

                if (item_idx + 1) % 10 == 0 or item_idx == len(batch_items) - 1:
                    elapsed = time.time() - start_time
                    prompts_done = len(processed_ids)
                    rate = prompts_done / elapsed if elapsed > 0 else 0
                    print(f"  [{prompts_done}/{total_prompts}] {prompt_id}: {num_correct}/{len(scores)} correct ({pass_rate*100:.1f}%)")

            verify_time = time.time() - verify_start
            print(f"  Verification: {verify_time:.1f}s")

            batch_idx += 1

    finally:
        completions_f.close()
        results_f.close()

    # Compute final statistics
    elapsed = time.time() - start_time
    overall_pass_rate = total_correct / total_completions if total_completions else 0
    avg_per_prompt_pass_rate = sum(all_pass_rates) / len(all_pass_rates) if all_pass_rates else 0

    # Compute pass@k statistics
    pass_at_k = {}
    if results_file.exists():
        # Reload all results to compute pass@k
        all_results = []
        with open(results_file) as f:
            for line in f:
                if line.strip():
                    all_results.append(json.loads(line))

        for k in [1, 2, 4, 8, 16, 32, 64, 128]:
            if k > n_completions:
                break
            # pass@k: fraction of prompts where at least one of first k completions is correct
            passed = sum(
                1 for r in all_results
                if any(s > 0 for s in r["scores"][:k])
            )
            pass_at_k[f"pass@{k}"] = passed / len(all_results) if all_results else 0

    # Write summary
    summary = {
        "dataset": dataset_name,
        "model_path": model_path,
        "n_completions": n_completions,
        "total_prompts": len(processed_ids),
        "total_completions": total_completions,
        "total_correct": total_correct,
        "overall_pass_rate": overall_pass_rate,
        "avg_per_prompt_pass_rate": avg_per_prompt_pass_rate,
        "pass_at_k": pass_at_k,
        "elapsed_seconds": elapsed,
        "throughput_completions_per_sec": total_completions / elapsed if elapsed > 0 else 0,
        "completed_at": datetime.now().isoformat(),
    }

    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Dataset: {dataset_name}")
    print(f"Model: {model_path}")
    print(f"Prompts: {len(processed_ids)}")
    print(f"Completions per prompt: {n_completions}")
    print(f"Total completions: {total_completions:,}")
    print(f"Total correct: {total_correct:,}")
    print(f"\nOverall pass rate: {overall_pass_rate*100:.2f}%")
    print(f"Avg per-prompt pass rate: {avg_per_prompt_pass_rate*100:.2f}%")
    print(f"\nPass@k:")
    for k, rate in pass_at_k.items():
        print(f"  {k}: {rate*100:.2f}%")
    print(f"\nElapsed: {elapsed:.1f}s")
    print(f"Throughput: {total_completions / elapsed:.1f} completions/sec")
    print(f"\nResults saved to: {output_dir}")
    print("=" * 60)

    return summary


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate completions and compute pass@k",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # GSM8K training set with 128 completions per prompt
    python scripts/eval_pass_at_k.py gsm8k --split train --n 128

    # MATH Level 3-5 test set
    python scripts/eval_pass_at_k.py math --split test --level 3,4,5 --n 64

    # MBPP test set with code execution
    python scripts/eval_pass_at_k.py mbpp --split test --n 32

    # Resume from prompt 100
    python scripts/eval_pass_at_k.py gsm8k --split train --n 128 --resume-from 100
        """,
    )

    # Required
    parser.add_argument(
        "dataset",
        type=str,
        choices=list(DATASET_REGISTRY.keys()),
        help="Dataset to evaluate",
    )

    # Dataset options
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Dataset split (default: train)",
    )
    parser.add_argument(
        "--level",
        type=str,
        default=None,
        help="MATH difficulty levels (comma-separated, e.g., '3,4,5')",
    )

    # Model
    parser.add_argument(
        "--model-path",
        type=str,
        default="/efs/rlvr-experiments/assets/hf/Qwen3-1.7B-Base",
        help="Path to model (default: Qwen3-1.7B-Base)",
    )

    # Generation
    parser.add_argument(
        "--n",
        type=int,
        default=128,
        help="Number of completions per prompt (default: 128)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Number of prompts to process at once (default: 16)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=1024,
        help="Max tokens per completion (default: 1024)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature (default: 1.0 for diversity)",
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
        default=50,
        help="Top-k sampling (default: 50)",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=2048,
        help="Max model context length (default: 2048)",
    )
    parser.add_argument(
        "--max-prompt-len",
        type=int,
        default=None,
        help="Filter out prompts longer than this (in tokens). Default: max_model_len - max_tokens",
    )

    # Hardware
    parser.add_argument(
        "--gpus",
        type=str,
        default="0",
        help="Comma-separated GPU IDs (default: 0)",
    )
    parser.add_argument(
        "--gpu-index",
        type=int,
        default=None,
        help="GPU index within --gpus list. If set, only process shard gpu-index/num-gpus of the data.",
    )
    parser.add_argument(
        "--num-shards",
        type=int,
        default=None,
        help="Total number of shards (for distributed runs). Defaults to number of GPUs.",
    )

    # Output
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: results/<model>/<dataset>/pass-at-k)",
    )

    # Resume
    parser.add_argument(
        "--resume-from",
        type=int,
        default=0,
        help="Resume from prompt index (default: 0, start fresh)",
    )

    # Shuffle
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle the dataset before processing (useful for partial runs)",
    )
    parser.add_argument(
        "--shuffle-seed",
        type=int,
        default=42,
        help="Seed for shuffling (default: 42)",
    )
    parser.add_argument(
        "--max-prompts",
        type=int,
        default=None,
        help="Maximum number of prompts to evaluate (default: all)",
    )

    # Verifier options
    parser.add_argument(
        "--verifier-workers",
        type=int,
        default=32,
        help="Number of verifier workers for parallel verification (default: 32)",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print(f"Pass@k Evaluation: {args.dataset}")
    print("=" * 60)
    print(f"Model: {args.model_path}")
    print(f"Dataset: {args.dataset} ({args.split})")
    print(f"Completions per prompt: {args.n}")
    print(f"Batch size: {args.batch_size}")
    print(f"Max tokens: {args.max_tokens}")
    print(f"Temperature: {args.temperature}")
    print(f"GPUs: {args.gpus}")
    print("=" * 60)

    # Build loader kwargs
    loader_kwargs = {}
    if args.level and args.dataset == "math":
        loader_kwargs["level"] = [int(l.strip()) for l in args.level.split(",")]

    # Load dataset
    items = load_dataset_items(args.dataset, args.split, **loader_kwargs)

    # Shuffle if requested (useful for partial runs)
    if args.shuffle:
        import random
        random.seed(args.shuffle_seed)
        random.shuffle(items)
        print(f"Shuffled dataset with seed {args.shuffle_seed}")

    # Limit number of prompts if requested
    if args.max_prompts is not None and args.max_prompts < len(items):
        items = items[:args.max_prompts]
        print(f"Limited to {args.max_prompts} prompts")

    # Filter prompts that are too long
    max_prompt_len = args.max_prompt_len
    if max_prompt_len is None:
        max_prompt_len = args.max_model_len - args.max_tokens
    if max_prompt_len > 0:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        original_len = len(items)
        filtered_items = []
        for item in items:
            prompt_text = item["prompt"]
            tokens = tokenizer.encode(prompt_text)
            if len(tokens) <= max_prompt_len:
                filtered_items.append(item)
        items = filtered_items
        if len(items) < original_len:
            print(f"Filtered {original_len - len(items)} prompts exceeding {max_prompt_len} tokens ({len(items)} remaining)")

    # Handle sharding for multi-GPU runs
    num_gpus = len(args.gpus.split(","))
    num_shards = args.num_shards if args.num_shards else num_gpus

    if args.gpu_index is not None:
        # Shard the dataset
        shard_idx = args.gpu_index
        shard_size = (len(items) + num_shards - 1) // num_shards
        shard_start = shard_idx * shard_size
        shard_end = min(shard_start + shard_size, len(items))
        items = items[shard_start:shard_end]
        print(f"Shard {shard_idx}/{num_shards}: processing prompts {shard_start}-{shard_end} ({len(items)} prompts)")

        # Use the specific GPU from the list
        gpu_list = args.gpus.split(",")
        args.gpus = gpu_list[shard_idx] if shard_idx < len(gpu_list) else gpu_list[0]

    # Set output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        model_name = Path(args.model_path).name.lower()
        output_dir = Path(f"results/{model_name}/evals/{args.dataset}/pass-at-k")

    # For sharded runs, use shard-specific subdirectory
    if args.gpu_index is not None:
        output_dir = output_dir / f"shard_{args.gpu_index}"

    print(f"\nOutput directory: {output_dir}")

    # Build verifier kwargs
    verifier_kwargs = {}
    if args.dataset in ("gsm8k", "math", "deepscaler", "aime", "beyondaime"):
        verifier_kwargs["max_workers"] = args.verifier_workers
        verifier_kwargs["warmup"] = True

    # Run
    summary = run_generation_and_verification(
        items=items,
        dataset_name=args.dataset,
        model_path=args.model_path,
        n_completions=args.n,
        batch_size=args.batch_size,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_model_len=args.max_model_len,
        gpus=args.gpus,
        output_dir=output_dir,
        resume_from=args.resume_from,
        verifier_kwargs=verifier_kwargs,
    )

    return summary


if __name__ == "__main__":
    main()
