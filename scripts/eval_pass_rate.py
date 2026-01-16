#!/usr/bin/env python3
"""
Unified pass rate evaluation script for math and code datasets.

Supports multiple datasets (GSM8k, MATH, MBPP, etc.) through a registry pattern.
Generates multiple completions per prompt using vLLM with data parallelism,
and computes pass rates using appropriate verifiers.

Usage:
    python scripts/eval_pass_rate.py --dataset gsm8k --num-prompts 100 --n 512
    python scripts/eval_pass_rate.py --dataset math --levels 3,4,5 --num-prompts 100 --n 512
    python scripts/eval_pass_rate.py --dataset mbpp --num-prompts 100 --n 64
"""

import argparse
import asyncio
import json
import os
import queue
import random
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path

from datasets import load_dataset
from vllm import LLM, SamplingParams

from rlvr_experiments.verifiers.math import MathVerifier
from rlvr_experiments.verifiers.code import MBPPVerifier
from rlvr_experiments.verifiers.code_executor import CodeExecutor, ExecutorConfig
from rlvr_experiments.verifiers.ifeval import IFEvalVerifier


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class PromptResult:
    """Results for a single prompt."""
    prompt_id: str
    prompt: str
    target_answer: str
    num_completions: int
    num_correct: int
    pass_rate: float
    correctness_mask: list[bool]
    completion_details: list[dict]  # Per-completion: length, finish_reason, correct
    sample_correct_completion: str | None = None
    sample_incorrect_completion: str | None = None
    # Optional metadata fields (dataset-specific)
    level: str | None = None
    subject: str | None = None


# =============================================================================
# Dataset Registry
# =============================================================================

class DatasetLoader(ABC):
    """Abstract base class for dataset loaders."""

    @abstractmethod
    def load_and_sample(self, num_prompts: int, seed: int, **kwargs) -> list[dict]:
        """Load dataset and sample prompts. Returns list of row dicts."""
        pass

    @abstractmethod
    def format_prompt(self, row: dict) -> str:
        """Format a row into a prompt for the model (plain text, no chat template).

        Args:
            row: Full row dict containing 'prompt' and 'problem' fields.
        """
        pass

    @abstractmethod
    def get_output_prefix(self) -> str:
        """Return prefix for output filename."""
        pass

    def get_extra_fields(self, row: dict) -> dict:
        """Return extra fields to include in PromptResult."""
        return {}

    def is_code_dataset(self) -> bool:
        """Return True if this is a code dataset requiring execution-based verification."""
        return False


class GSM8kLoader(DatasetLoader):
    """Loader for GSM8k dataset."""

    def __init__(self, split: str = "test"):
        self.split = split

    def load_and_sample(self, num_prompts: int, seed: int, **kwargs) -> list[dict]:
        print(f"Loading GSM8k {self.split} split...")
        hf_dataset = load_dataset("openai/gsm8k", "main", split=self.split)

        all_rows = []
        for i, row in enumerate(hf_dataset):
            question = row['question'].strip()
            answer = row["answer"].split("####")[-1].strip()
            prompt_id = f"gsm8k_{i}"
            all_rows.append({
                "prompt": question,
                "problem": {"answer": answer, "prompt_id": prompt_id},
            })

        print(f"Loaded {len(all_rows)} prompts from GSM8k {self.split}")

        # num_prompts <= 0 means use all prompts
        if num_prompts <= 0:
            print(f"Using all {len(all_rows)} prompts")
            return all_rows

        random.seed(seed)
        sampled = random.sample(all_rows, min(num_prompts, len(all_rows)))
        print(f"Sampled {len(sampled)} prompts (seed={seed})")

        return sampled

    def format_prompt(self, row: dict) -> str:
        """Format GSM8k problem as plain text prompt."""
        problem_text = row["prompt"]
        return f"""Solve the following math problem. Put the final answer in \\boxed{{}}.

Problem: {problem_text}

Let's think step by step."""

    def get_output_prefix(self) -> str:
        return "gsm8k"


class MATHLoader(DatasetLoader):
    """Loader for MATH dataset."""

    def __init__(self, split: str = "test", levels: list[int] = None):
        self.split = split
        self.levels = levels or [3, 4, 5]

    def _extract_boxed_answer(self, solution: str) -> str | None:
        """Extract the answer from \\boxed{} in the solution."""
        match = re.search(r"\\boxed\{", solution)
        if not match:
            return None
        start = match.end()
        depth = 1
        for i, ch in enumerate(solution[start:], start):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return solution[start:i].strip()
        return None

    def load_and_sample(self, num_prompts: int, seed: int, **kwargs) -> list[dict]:
        print(f"Loading MATH {self.split} split (levels {self.levels})...")

        subjects = ['algebra', 'counting_and_probability', 'geometry',
                    'intermediate_algebra', 'number_theory', 'prealgebra', 'precalculus']

        all_rows = []
        level_strs = [f"Level {l}" for l in self.levels]

        for subj in subjects:
            ds = load_dataset('EleutherAI/hendrycks_math', subj, split=self.split)
            for i, row in enumerate(ds):
                if row['level'] in level_strs:
                    answer = self._extract_boxed_answer(row['solution'])
                    if answer is None:
                        continue

                    prompt_id = f"math_{subj}_{i}"
                    all_rows.append({
                        "prompt": row['problem'],
                        "problem": {"answer": answer, "prompt_id": prompt_id},
                        "level": row['level'],
                        "subject": row['type'],
                    })
            print(f"  {subj}: loaded {sum(1 for r in all_rows if subj in r['problem']['prompt_id'])} problems")

        print(f"Total: {len(all_rows)} problems from levels {self.levels}")

        # num_prompts <= 0 means use all prompts
        if num_prompts <= 0:
            sampled = all_rows
            print(f"Using all {len(sampled)} prompts")
        else:
            random.seed(seed)
            sampled = random.sample(all_rows, min(num_prompts, len(all_rows)))
            print(f"Sampled {len(sampled)} prompts (seed={seed})")

        # Show level distribution
        level_counts = {}
        for r in sampled:
            level_counts[r['level']] = level_counts.get(r['level'], 0) + 1
        print("Level distribution:", level_counts)

        return sampled

    def format_prompt(self, row: dict) -> str:
        """Format MATH problem as plain text prompt."""
        problem_text = row["prompt"]
        return f"""Solve the following math problem. Put the final answer in \\boxed{{}}.

Problem: {problem_text}

Let's think step by step."""

    def get_output_prefix(self) -> str:
        levels_str = "-".join(str(l) for l in self.levels)
        return f"math_L{levels_str}"

    def get_extra_fields(self, row: dict) -> dict:
        return {
            "level": row.get("level"),
            "subject": row.get("subject"),
        }


class MBPPLoader(DatasetLoader):
    """Loader for MBPP dataset."""

    def __init__(self, split: str = "test"):
        self.split = split

    def load_and_sample(self, num_prompts: int, seed: int, **kwargs) -> list[dict]:
        print(f"Loading MBPP {self.split} split...")
        # MBPP has 'train', 'test', 'validation', and 'prompt' splits
        # The 'test' split has 500 problems
        hf_dataset = load_dataset("google-research-datasets/mbpp", "full", split=self.split)

        all_rows = []
        for i, row in enumerate(hf_dataset):
            prompt_id = f"mbpp_{row['task_id']}"
            all_rows.append({
                "prompt": row['text'],  # The problem description
                "problem": {
                    "prompt_id": prompt_id,
                    "task_id": row['task_id'],
                    "code": row['code'],  # Reference solution
                    "test_list": row['test_list'],  # List of assert statements
                    "test_setup_code": row.get('test_setup_code', ''),
                    "challenge_test_list": row.get('challenge_test_list', []),
                },
            })

        print(f"Loaded {len(all_rows)} prompts from MBPP {self.split}")

        # num_prompts <= 0 means use all prompts
        if num_prompts <= 0:
            print(f"Using all {len(all_rows)} prompts")
            return all_rows

        random.seed(seed)
        sampled = random.sample(all_rows, min(num_prompts, len(all_rows)))
        print(f"Sampled {len(sampled)} prompts (seed={seed})")

        return sampled

    def format_prompt(self, row: dict) -> str:
        """Format MBPP problem as plain text prompt.

        Following the bigcode-evaluation-harness / EvalPlus standard,
        we include the first test case in the prompt. This is critical because
        MBPP uses non-pythonic function names (e.g., count_Substring_With_Equal_Ends)
        that the model cannot infer from the problem description alone.
        """
        problem_text = row["prompt"]
        test_list = row["problem"]["test_list"]
        first_test = test_list[0] if test_list else ""

        return f"""Write a Python function to solve the following problem.

{problem_text}

{first_test}

```python
"""

    def get_output_prefix(self) -> str:
        return "mbpp"

    def is_code_dataset(self) -> bool:
        return True


class APPSLoader(DatasetLoader):
    """Loader for APPS (Automated Programming Progress Standard) dataset."""

    def __init__(self, split: str = "test", difficulty: str = None):
        self.split = split
        self.difficulty = difficulty  # None means all, or "introductory", "interview", "competition"

    def load_and_sample(self, num_prompts: int, seed: int, **kwargs) -> list[dict]:
        print(f"Loading APPS {self.split} split (using parquet revision)...")
        # Load from parquet revision to avoid script-based loader issues
        hf_dataset = load_dataset(
            "codeparrot/apps",
            split=self.split,
            revision="refs/convert/parquet"
        )

        all_rows = []
        for i, row in enumerate(hf_dataset):
            # Filter by difficulty if specified
            if self.difficulty and row['difficulty'] != self.difficulty:
                continue

            # Parse JSON strings
            try:
                input_output = json.loads(row['input_output']) if row['input_output'] else {}
                solutions = json.loads(row['solutions']) if row['solutions'] else []
            except json.JSONDecodeError:
                continue

            prompt_id = f"apps_{row['problem_id']}"

            all_rows.append({
                "prompt": row['question'],
                "problem": {
                    "prompt_id": prompt_id,
                    "problem_id": row['problem_id'],
                    "input_output": input_output,
                    "solutions": solutions,
                    "difficulty": row['difficulty'],
                    "starter_code": row.get('starter_code', ''),
                    "url": row.get('url', ''),
                },
            })

        if self.difficulty:
            print(f"Loaded {len(all_rows)} {self.difficulty} problems from APPS {self.split}")
        else:
            print(f"Loaded {len(all_rows)} problems from APPS {self.split}")

        # Count by difficulty
        difficulty_counts = {}
        for r in all_rows:
            d = r['problem']['difficulty']
            difficulty_counts[d] = difficulty_counts.get(d, 0) + 1
        print("Difficulty distribution:", difficulty_counts)

        # num_prompts <= 0 means use all prompts
        if num_prompts <= 0:
            print(f"Using all {len(all_rows)} prompts")
            return all_rows

        random.seed(seed)
        sampled = random.sample(all_rows, min(num_prompts, len(all_rows)))
        print(f"Sampled {len(sampled)} prompts (seed={seed})")

        return sampled

    def format_prompt(self, row: dict) -> str:
        """Format APPS problem as plain text prompt.

        APPS problems are more complex than MBPP - they involve stdin/stdout
        rather than function definitions. We format the prompt to ask for
        a complete program.
        """
        question = row["prompt"]
        starter_code = row["problem"].get("starter_code", "")

        if starter_code:
            return f"""Write a Python program to solve the following problem. The program should read from stdin and write to stdout.

Problem:
{question}

You must use this starter code:
{starter_code}

```python
{starter_code}"""
        else:
            return f"""Write a Python program to solve the following problem. The program should read from stdin and write to stdout.

Problem:
{question}

```python
"""

    def get_output_prefix(self) -> str:
        if self.difficulty:
            return f"apps_{self.difficulty}"
        return "apps"

    def is_code_dataset(self) -> bool:
        return True


class IFEvalLoader(DatasetLoader):
    """Loader for RLVR-IFeval instruction-following dataset."""

    def __init__(self, split: str = "train"):
        self.split = split

    def load_and_sample(self, num_prompts: int, seed: int, **kwargs) -> list[dict]:
        print(f"Loading RLVR-IFeval {self.split} split...")
        hf_dataset = load_dataset("allenai/RLVR-IFeval", split=self.split)

        all_rows = []
        for i, row in enumerate(hf_dataset):
            # Extract user message from chat format
            user_content = row["messages"][0]["content"] if row["messages"] else ""
            prompt_id = f"ifeval_{i}"

            all_rows.append({
                "prompt": user_content,
                "problem": {
                    "prompt_id": prompt_id,
                    "ground_truth": row["ground_truth"],
                    "constraint_type": row["constraint_type"],
                    "constraint": row["constraint"],
                },
            })

        print(f"Loaded {len(all_rows)} prompts from RLVR-IFeval {self.split}")

        # Show constraint type distribution
        constraint_counts = {}
        for r in all_rows:
            ct = r["problem"]["constraint_type"]
            constraint_counts[ct] = constraint_counts.get(ct, 0) + 1
        print(f"Constraint types: {len(constraint_counts)} unique")

        # num_prompts <= 0 means use all prompts
        if num_prompts <= 0:
            print(f"Using all {len(all_rows)} prompts")
            return all_rows

        random.seed(seed)
        sampled = random.sample(all_rows, min(num_prompts, len(all_rows)))
        print(f"Sampled {len(sampled)} prompts (seed={seed})")

        return sampled

    def format_prompt(self, row: dict) -> str:
        """Format IFeval prompt - the constraint is already embedded in the user message."""
        return row["prompt"]

    def get_output_prefix(self) -> str:
        return "ifeval"

    def is_ifeval_dataset(self) -> bool:
        return True


# Registry of available datasets
DATASET_REGISTRY = {
    "gsm8k": GSM8kLoader,
    "math": MATHLoader,
    "mbpp": MBPPLoader,
    "apps": APPSLoader,
    "ifeval": IFEvalLoader,
}


# =============================================================================
# vLLM Worker
# =============================================================================

def generate_with_vllm(
    prompts: list[str],
    model_path: str,
    sampling_params_dict: dict,
    tp_size: int,
    max_model_len: int,
    cuda_visible_devices: str | None = None,
) -> list[list[tuple[str, str, int]]]:
    """Generate completions with vLLM and return a JSON-serializable structure."""
    if cuda_visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
    # vLLM spawns its own EngineCore subprocesses; forcing spawn avoids
    # fork-related hangs in multi-threaded contexts.
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    # Avoid HuggingFace tokenizers threadpool + fork interactions.
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    sp = SamplingParams(**sampling_params_dict)
    llm = LLM(
        model=model_path,
        tensor_parallel_size=tp_size,
        max_model_len=max_model_len,
        gpu_memory_utilization=0.90,
        dtype="bfloat16",
        enable_prefix_caching=True,
        trust_remote_code=True,
    )

    outputs = llm.generate(prompts, sp)

    results: list[list[tuple[str, str, int]]] = []
    for output in outputs:
        completions = [(o.text, o.finish_reason, len(o.token_ids)) for o in output.outputs]
        results.append(completions)
    return results


def worker(gpu_id, prompts, model_path, sampling_params_dict, tp_size, max_model_len, result_queue):
    """Worker function for each GPU."""
    results = generate_with_vllm(
        prompts,
        model_path=model_path,
        sampling_params_dict=sampling_params_dict,
        tp_size=tp_size,
        max_model_len=max_model_len,
        cuda_visible_devices=str(gpu_id),
    )
    result_queue.put((gpu_id, results))


# =============================================================================
# Main Script
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate pass rate on math datasets")

    # Dataset selection
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=list(DATASET_REGISTRY.keys()),
        help=f"Dataset to evaluate on: {list(DATASET_REGISTRY.keys())}",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "test"],
        help="Dataset split to use (default: test)",
    )

    # MATH-specific options
    parser.add_argument(
        "--levels",
        type=str,
        default="3,4,5",
        help="Comma-separated list of MATH levels (default: 3,4,5)",
    )

    # Model and generation
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
        help="Number of prompts to sample (default: 100, 0 or -1 for all prompts)",
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
        default=None,
        help="Max tokens per completion (default: 512 for gsm8k, 2048 for math)",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=None,
        help="Max model context length (default: 1024 for gsm8k, 4096 for math)",
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

    # Parallelism
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
        "--generation-backend",
        type=str,
        default="multiprocess",
        choices=["multiprocess", "direct"],
        help="Generation backend: 'multiprocess' starts one worker process per replica; "
             "'direct' runs generation in the main process (requires data_parallel_size == tensor_parallel_size).",
    )
    parser.add_argument(
        "--generation-timeout-seconds",
        type=float,
        default=None,
        help="Optional wall-clock timeout for generation (multiprocess backend only).",
    )

    # Output
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: experiments/<model-name>-pass-rate)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed results for each prompt",
    )
    parser.add_argument(
        "--completions-file",
        type=str,
        default=None,
        help="Load completions from file instead of generating (for recovery)",
    )

    return parser.parse_args()


def format_prompts(rows: list[dict], loader: DatasetLoader) -> list[str]:
    """Format prompts using plain text (no chat template for base models)."""
    formatted = []
    for row in rows:
        prompt = loader.format_prompt(row)
        formatted.append(prompt)
    return formatted


def main():
    args = parse_args()

    # Set dataset-specific defaults
    if args.max_tokens is None:
        if args.dataset == "math":
            args.max_tokens = 2048
        elif args.dataset == "mbpp":
            args.max_tokens = 512  # Code completions are usually shorter
        else:
            args.max_tokens = 512
    if args.max_model_len is None:
        if args.dataset == "math":
            args.max_model_len = 4096
        elif args.dataset == "mbpp":
            args.max_model_len = 2048
        else:
            args.max_model_len = 1024

    # Create dataset loader
    if args.dataset == "math":
        levels = [int(l.strip()) for l in args.levels.split(",")]
        loader = MATHLoader(split=args.split, levels=levels)
    else:
        loader = DATASET_REGISTRY[args.dataset](split=args.split)

    # Print config
    print("=" * 60)
    print(f"{args.dataset.upper()} Pass Rate Evaluation")
    print("=" * 60)
    print(f"Model: {args.model_path}")
    print(f"Dataset: {args.dataset} ({args.split})")
    if args.dataset == "math":
        print(f"Levels: {levels}")
    print(f"Prompts: {args.num_prompts if args.num_prompts > 0 else 'all'}")
    print(f"Completions per prompt: {args.n}")
    print(f"Max tokens: {args.max_tokens}")
    print(f"Max model len: {args.max_model_len}")
    print(f"Temperature: {args.temperature}")
    print(f"Top-p: {args.top_p}")
    print(f"Top-k: {args.top_k}")
    print(f"Seed: {args.seed}")
    print(f"Data parallel size: {args.data_parallel_size}")
    print(f"Tensor parallel size: {args.tensor_parallel_size}")
    print("=" * 60)

    # Load and sample prompts
    rows = loader.load_and_sample(args.num_prompts, args.seed)

    # Either load completions from file or generate new ones
    if args.completions_file:
        # Load pre-generated completions
        print(f"\nLoading completions from {args.completions_file}...")
        with open(args.completions_file) as f:
            completions_data = json.load(f)

        # Build lookup by prompt_id for alignment
        completions_by_prompt_id = {}
        for prompt_data in completions_data["prompts"]:
            completions_by_prompt_id[prompt_data["prompt_id"]] = [
                (c["text"], c["finish_reason"], c["num_tokens"])
                for c in prompt_data["completions"]
            ]

        # Build outputs_by_idx aligned with rows (which were sampled from dataset)
        outputs_by_idx = {}
        missing_prompts = []
        for i, row in enumerate(rows):
            prompt_id = row["problem"]["prompt_id"]
            if prompt_id in completions_by_prompt_id:
                outputs_by_idx[i] = completions_by_prompt_id[prompt_id]
            else:
                missing_prompts.append(prompt_id)
                outputs_by_idx[i] = []

        if missing_prompts:
            print(f"WARNING: {len(missing_prompts)} prompts not found in completions file")
            print(f"  Missing: {missing_prompts[:5]}{'...' if len(missing_prompts) > 5 else ''}")

        gen_time = completions_data["metadata"].get("generation_time_seconds", 0)
        total_completions = sum(len(c) for c in outputs_by_idx.values())
        print(f"Loaded {total_completions:,} completions from disk (aligned by prompt_id)")

    else:
        # Generate completions
        print("\nFormatting prompts...")
        formatted_prompts = format_prompts(rows, loader)

        # Show example formatted prompt
        print(f"\nExample formatted prompt:\n{'-'*40}")
        print(formatted_prompts[0])
        print(f"{'-'*40}")

        # Set up parallelism
        dp_size = args.data_parallel_size
        tp_size = args.tensor_parallel_size
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

        print(f"\nGenerating {args.n} completions for each of {len(formatted_prompts)} prompts...")
        print(f"Total completions to generate: {args.n * len(formatted_prompts):,}")

        start_time = time.time()

        sampling_params_dict = {
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "max_tokens": args.max_tokens,
            "n": args.n,
        }

        if args.generation_backend == "direct":
            if num_replicas != 1:
                raise ValueError(
                    "--generation-backend direct requires data_parallel_size == tensor_parallel_size "
                    f"(got data_parallel_size={dp_size}, tensor_parallel_size={tp_size})"
                )
            # Run vLLM in the current process (no outer multiprocessing wrapper).
            all_completions = [
                generate_with_vllm(
                    prompts_per_replica[0],
                    model_path=args.model_path,
                    sampling_params_dict=sampling_params_dict,
                    tp_size=tp_size,
                    max_model_len=args.max_model_len,
                    cuda_visible_devices=None,
                )
            ]
            gen_time = time.time() - start_time
        else:
            import torch.multiprocessing as mp

            ctx = mp.get_context("spawn")
            result_queue = ctx.Queue()
            processes = []

            def _kill_worker_tree(p) -> None:
                if p.pid is None:
                    return
                try:
                    from vllm.utils.system_utils import kill_process_tree
                    kill_process_tree(p.pid)
                except Exception:
                    try:
                        p.kill()
                    except Exception:
                        pass

            def _cleanup_workers() -> None:
                for proc in processes:
                    if proc.is_alive() or proc.exitcode is None:
                        _kill_worker_tree(proc)
                for proc in processes:
                    proc.join(timeout=5)

            for replica_idx in range(num_replicas):
                gpu_id = replica_idx * tp_size
                if tp_size == 1:
                    gpu_ids = str(gpu_id)
                else:
                    gpu_ids = ",".join(str(gpu_id + j) for j in range(tp_size))

                p = ctx.Process(
                    target=worker,
                    args=(
                        gpu_ids,
                        prompts_per_replica[replica_idx],
                        args.model_path,
                        sampling_params_dict,
                        tp_size,
                        args.max_model_len,
                        result_queue,
                    ),
                )
                processes.append(p)
                p.start()

            generation_deadline = (
                time.monotonic() + args.generation_timeout_seconds
                if args.generation_timeout_seconds is not None
                else None
            )

            # Collect results
            all_completions = [None] * num_replicas
            received = 0
            try:
                while received < num_replicas:
                    timeout_s = 10.0
                    if generation_deadline is not None:
                        remaining = generation_deadline - time.monotonic()
                        if remaining <= 0:
                            raise TimeoutError(
                                f"Timed out waiting for vLLM workers after {args.generation_timeout_seconds:.0f}s"
                            )
                        timeout_s = min(timeout_s, remaining)

                    try:
                        gpu_id, results = result_queue.get(timeout=timeout_s)
                    except queue.Empty:
                        # Avoid hanging forever if a worker died before putting results.
                        finished_without_result = [
                            idx
                            for idx, proc in enumerate(processes)
                            if proc.exitcode is not None and all_completions[idx] is None
                        ]
                        if finished_without_result:
                            raise RuntimeError(
                                "One or more vLLM worker processes exited unexpectedly: "
                                + ", ".join(
                                    f"replica={idx} exitcode={processes[idx].exitcode}"
                                    for idx in finished_without_result
                                )
                            )
                        continue

                    if isinstance(gpu_id, str):
                        replica_idx = int(gpu_id.split(",")[0]) // tp_size
                    else:
                        replica_idx = gpu_id // tp_size
                    all_completions[replica_idx] = results
                    received += 1
                    print(f"  Replica {replica_idx} completed: {len(results)} prompts")
            except Exception:
                _cleanup_workers()
                raise

            for proc in processes:
                proc.join()
            for idx, proc in enumerate(processes):
                if proc.exitcode != 0:
                    raise RuntimeError(
                        f"vLLM worker replica {idx} exited with code {proc.exitcode}"
                    )

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

        # Save raw completions to disk (for recovery if verification fails)
        if args.output_dir:
            completions_dir = Path(args.output_dir)
        else:
            model_name = Path(args.model_path).name.lower()
            completions_dir = Path(f"experiments/{model_name}-pass-rate")
        completions_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        completions_file = completions_dir / f"{loader.get_output_prefix()}_{Path(args.model_path).name}_completions_{timestamp}.json"

        print(f"\nSaving completions to {completions_file}...")
        completions_data = {
            "metadata": {
                "dataset": args.dataset,
                "split": args.split,
                "model_path": args.model_path,
                "num_prompts": len(rows),
                "completions_per_prompt": args.n,
                "max_tokens": args.max_tokens,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "top_k": args.top_k,
                "seed": args.seed,
                "generation_time_seconds": gen_time,
                "timestamp": timestamp,
            },
            "prompts": [
                {
                    "prompt_id": row["problem"]["prompt_id"],
                    "prompt": row["prompt"],
                    "target": row["problem"].get("answer", str(row["problem"].get("test_list", []))),
                    "completions": [
                        {"text": c[0], "finish_reason": c[1], "num_tokens": c[2]}
                        for c in outputs_by_idx.get(i, [])
                    ],
                }
                for i, row in enumerate(rows)
            ],
        }
        with open(completions_file, "w") as f:
            json.dump(completions_data, f)
        print(f"Saved {total_completions:,} completions to disk")

    # Verify completions
    print("\nVerifying completions...")

    is_code = loader.is_code_dataset()
    is_ifeval = hasattr(loader, 'is_ifeval_dataset') and loader.is_ifeval_dataset()
    if is_ifeval:
        ifeval_verifier = IFEvalVerifier()
    elif not is_code:
        # Use many workers for parallel math verification (192 vCPUs available)
        math_verifier = MathVerifier(max_workers=128, warmup=True)

    results: list[PromptResult] = []
    total_correct = 0
    total_completions_verified = 0

    finish_stats = {
        "stop_correct": 0,
        "stop_incorrect": 0,
        "length_correct": 0,
        "length_incorrect": 0,
    }

    length_stats = {
        "correct_lengths": [],
        "incorrect_lengths": [],
        "stop_lengths": [],
        "length_lengths": [],
    }

    # Checkpointing setup
    if args.output_dir:
        checkpoint_dir = Path(args.output_dir)
    else:
        model_name = Path(args.model_path).name.lower()
        checkpoint_dir = Path(f"experiments/{model_name}-pass-rate")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_file = checkpoint_dir / f"{loader.get_output_prefix()}_{Path(args.model_path).name}_checkpoint.json"

    # Try to resume from checkpoint
    start_idx = 0
    if checkpoint_file.exists():
        print(f"Found checkpoint file: {checkpoint_file}")
        try:
            with open(checkpoint_file) as f:
                checkpoint = json.load(f)
            results = [PromptResult(**r) for r in checkpoint["results"]]
            finish_stats = checkpoint["finish_stats"]
            length_stats = checkpoint["length_stats"]
            total_correct = checkpoint["total_correct"]
            total_completions_verified = checkpoint["total_completions_verified"]
            start_idx = checkpoint["next_idx"]
            print(f"Resuming from prompt {start_idx}/{len(rows)} ({start_idx/len(rows)*100:.1f}%)")
        except Exception as e:
            print(f"Failed to load checkpoint: {e}, starting fresh")
            start_idx = 0
            results = []

    verify_start = time.time()

    for i, row in enumerate(rows):
        # Skip already processed prompts
        if i < start_idx:
            continue
        problem = row["problem"]
        prompt_id = problem["prompt_id"]

        # For math datasets, target is the answer string
        # For code datasets, target is the problem dict with test_list
        # For ifeval datasets, target is the constraint description
        if is_code:
            target = str(problem.get("test_list", []))  # Store for reference
        elif is_ifeval:
            target = problem.get("constraint", "")  # Store constraint description
        else:
            target = problem["answer"]

        completions_with_reason = outputs_by_idx.get(i, [])

        correct_count = 0
        sample_correct = None
        sample_incorrect = None
        correctness_mask = []
        completion_details = []

        if is_code:
            # Verify all completions for this problem in parallel using async
            # NOTE: Create a new executor for each prompt to avoid asyncio.Semaphore issues
            # when calling asyncio.run() multiple times (each call creates/destroys an event loop)
            executor = CodeExecutor(ExecutorConfig(timeout=10.0), max_concurrent=32)
            code_verifier = MBPPVerifier(executor=executor)
            completions_only = [c[0] for c in completions_with_reason]

            async def verify_all():
                tasks = [code_verifier.verify(problem, c) for c in completions_only]
                return await asyncio.gather(*tasks)

            test_results = asyncio.run(verify_all())

            for idx, (completion, finish_reason, num_tokens) in enumerate(completions_with_reason):
                test_result = test_results[idx]
                is_correct = test_result.all_passed
                correctness_mask.append(is_correct)
                completion_details.append({
                    "length": num_tokens,
                    "finish_reason": finish_reason,
                    "correct": is_correct,
                })

                if finish_reason == "stop":
                    finish_stats["stop_correct" if is_correct else "stop_incorrect"] += 1
                    length_stats["stop_lengths"].append(num_tokens)
                else:
                    finish_stats["length_correct" if is_correct else "length_incorrect"] += 1
                    length_stats["length_lengths"].append(num_tokens)

                if is_correct:
                    length_stats["correct_lengths"].append(num_tokens)
                    correct_count += 1
                    if sample_correct is None:
                        sample_correct = completion
                else:
                    length_stats["incorrect_lengths"].append(num_tokens)
                    if sample_incorrect is None:
                        sample_incorrect = completion
        elif is_ifeval:
            # IFEval verification - check if completions satisfy constraints
            completions_only = [c[0] for c in completions_with_reason]
            ground_truth = problem.get("ground_truth", "")
            scores = [ifeval_verifier.verify(c, ground_truth) for c in completions_only]

            for idx, (completion, finish_reason, num_tokens) in enumerate(completions_with_reason):
                is_correct = scores[idx] > 0
                correctness_mask.append(is_correct)
                completion_details.append({
                    "length": num_tokens,
                    "finish_reason": finish_reason,
                    "correct": is_correct,
                })

                if finish_reason == "stop":
                    finish_stats["stop_correct" if is_correct else "stop_incorrect"] += 1
                    length_stats["stop_lengths"].append(num_tokens)
                else:
                    finish_stats["length_correct" if is_correct else "length_incorrect"] += 1
                    length_stats["length_lengths"].append(num_tokens)

                if is_correct:
                    length_stats["correct_lengths"].append(num_tokens)
                    correct_count += 1
                    if sample_correct is None:
                        sample_correct = completion
                else:
                    length_stats["incorrect_lengths"].append(num_tokens)
                    if sample_incorrect is None:
                        sample_incorrect = completion
        else:
            # Math verification (parallel) - submit all completions at once
            completions_only = [c[0] for c in completions_with_reason]
            scores = math_verifier.verify_batch_parallel(completions_only, target)

            for idx, (completion, finish_reason, num_tokens) in enumerate(completions_with_reason):
                is_correct = scores[idx] > 0
                correctness_mask.append(is_correct)
                completion_details.append({
                    "length": num_tokens,
                    "finish_reason": finish_reason,
                    "correct": is_correct,
                })

                if finish_reason == "stop":
                    finish_stats["stop_correct" if is_correct else "stop_incorrect"] += 1
                    length_stats["stop_lengths"].append(num_tokens)
                else:
                    finish_stats["length_correct" if is_correct else "length_incorrect"] += 1
                    length_stats["length_lengths"].append(num_tokens)

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

        extra_fields = loader.get_extra_fields(row)

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
            **extra_fields,
        )
        results.append(result)

        total_correct += correct_count
        total_completions_verified += len(completions_with_reason)

        # Progress display
        if (i + 1) % 100 == 0 or (i + 1) == len(rows):
            elapsed = time.time() - verify_start
            rate = (i + 1 - start_idx) / elapsed if elapsed > 0 else 0
            remaining = (len(rows) - i - 1) / rate if rate > 0 else 0
            print(f"[{i+1}/{len(rows)}] Verified {i+1-start_idx} prompts in {elapsed:.0f}s ({rate:.1f} prompts/s, ~{remaining/60:.0f}m remaining)")

        # Save checkpoint every 50 prompts
        if (i + 1) % 50 == 0:
            checkpoint_data = {
                "next_idx": i + 1,
                "results": [asdict(r) for r in results],
                "finish_stats": finish_stats,
                "length_stats": length_stats,
                "total_correct": total_correct,
                "total_completions_verified": total_completions_verified,
            }
            with open(checkpoint_file, "w") as f:
                json.dump(checkpoint_data, f)

        if args.verbose or (is_code and (i + 1) % 10 == 0):
            print(f"[{i+1}/{len(rows)}] {prompt_id}: {correct_count}/{len(completions_with_reason)} correct ({pass_rate*100:.1f}%)")

    verify_time = time.time() - verify_start
    print(f"\nVerification completed in {verify_time:.1f}s")

    # Remove checkpoint file on successful completion
    if checkpoint_file.exists():
        checkpoint_file.unlink()
        print("Checkpoint file removed.")

    if is_code:
        print(f"Verification throughput: {total_completions_verified / verify_time:.1f} completions/sec")

    # Compute statistics
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    overall_pass_rate = total_correct / total_completions_verified if total_completions_verified else 0
    avg_per_prompt_pass_rate = sum(r.pass_rate for r in results) / len(results) if results else 0
    prompts_with_any_correct = sum(1 for r in results if r.num_correct > 0)
    pass_at_k = prompts_with_any_correct / len(results) if results else 0

    # Level breakdown (for MATH)
    if args.dataset == "math":
        print("\nResults by level:")
        level_breakdown = {}
        for level in sorted(set(r.level for r in results if r.level)):
            level_results = [r for r in results if r.level == level]
            level_correct = sum(r.num_correct for r in level_results)
            level_total = sum(r.num_completions for r in level_results)
            level_pass_rate = level_correct / level_total if level_total else 0
            level_pass_at_k = sum(1 for r in level_results if r.num_correct > 0) / len(level_results) if level_results else 0
            level_breakdown[level] = {
                "pass_rate": level_pass_rate,
                "pass_at_k": level_pass_at_k,
                "num_problems": len(level_results),
            }
            print(f"  {level}: {level_pass_rate*100:.1f}% pass rate, {level_pass_at_k*100:.0f}% Pass@{args.n} ({len(level_results)} problems)")

        print("\nResults by subject:")
        subject_breakdown = {}
        for subject in sorted(set(r.subject for r in results if r.subject)):
            subj_results = [r for r in results if r.subject == subject]
            subj_correct = sum(r.num_correct for r in subj_results)
            subj_total = sum(r.num_completions for r in subj_results)
            subj_pass_rate = subj_correct / subj_total if subj_total else 0
            subject_breakdown[subject] = {
                "pass_rate": subj_pass_rate,
                "num_problems": len(subj_results),
            }
            print(f"  {subject}: {subj_pass_rate*100:.1f}% ({len(subj_results)} problems)")

    # Pass rate distribution
    pass_rate_buckets = {"0%": 0, "1-10%": 0, "11-25%": 0, "26-50%": 0, "51-75%": 0, "76-99%": 0, "100%": 0}
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
    print(f"\nPass@{args.n} (at least 1 correct): {prompts_with_any_correct}/{len(results)} ({pass_at_k*100:.1f}%)")

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

    # Length stats
    def avg(lst):
        return sum(lst) / len(lst) if lst else 0
    def median(lst):
        if not lst:
            return 0
        s = sorted(lst)
        n = len(s)
        return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2

    print(f"\nCompletion length (tokens):")
    all_lengths = length_stats['correct_lengths'] + length_stats['incorrect_lengths']
    print(f"  All completions: avg={avg(all_lengths):.0f}, median={median(all_lengths):.0f}")
    print(f"  Correct: avg={avg(length_stats['correct_lengths']):.0f}, median={median(length_stats['correct_lengths']):.0f}")
    print(f"  Incorrect: avg={avg(length_stats['incorrect_lengths']):.0f}, median={median(length_stats['incorrect_lengths']):.0f}")
    print(f"  Completed (stop): avg={avg(length_stats['stop_lengths']):.0f}, median={median(length_stats['stop_lengths']):.0f}")
    print(f"  Truncated (length): avg={avg(length_stats['length_lengths']):.0f}, median={median(length_stats['length_lengths']):.0f}")

    # Sample results
    print("\n" + "=" * 60)
    print("SAMPLE RESULTS")
    print("=" * 60)

    sorted_by_pass_rate = sorted(results, key=lambda r: r.pass_rate, reverse=True)
    print("\nTop 3 highest pass rates:")
    for r in sorted_by_pass_rate[:3]:
        extra = f" [{r.level}] {r.subject}" if r.level else ""
        print(f"\n {extra}")
        print(f"  Prompt: {r.prompt[:80]}...")
        print(f"  Target: {r.target_answer}")
        print(f"  Pass rate: {r.pass_rate*100:.1f}% ({r.num_correct}/{r.num_completions})")
        if r.sample_correct_completion:
            sample = r.sample_correct_completion[:200].replace("\n", " ")
            print(f"  Sample correct: {sample}...")

    print("\nBottom 3 lowest pass rates:")
    for r in sorted_by_pass_rate[-3:]:
        extra = f" [{r.level}] {r.subject}" if r.level else ""
        print(f"\n {extra}")
        print(f"  Prompt: {r.prompt[:80]}...")
        print(f"  Target: {r.target_answer}")
        print(f"  Pass rate: {r.pass_rate*100:.1f}% ({r.num_correct}/{r.num_completions})")
        if r.sample_incorrect_completion:
            sample = r.sample_incorrect_completion[:200].replace("\n", " ")
            print(f"  Sample incorrect: {sample}...")

    print("\n" + "=" * 60)
    print(f"Generation time: {gen_time:.1f}s")
    print(f"Throughput: {total_completions_verified / gen_time:.1f} completions/sec")
    print("=" * 60)

    # Prepare output
    length_summary = {
        "correct": {"count": len(length_stats["correct_lengths"]), "avg": avg(length_stats["correct_lengths"]), "median": median(length_stats["correct_lengths"])},
        "incorrect": {"count": len(length_stats["incorrect_lengths"]), "avg": avg(length_stats["incorrect_lengths"]), "median": median(length_stats["incorrect_lengths"])},
        "completed_stop": {"count": len(length_stats["stop_lengths"]), "avg": avg(length_stats["stop_lengths"]), "median": median(length_stats["stop_lengths"])},
        "truncated_length": {"count": len(length_stats["length_lengths"]), "avg": avg(length_stats["length_lengths"]), "median": median(length_stats["length_lengths"])},
    }

    # Determine output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        model_name = Path(args.model_path).name.lower()
        output_dir = Path(f"experiments/{model_name}-pass-rate")
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = Path(args.model_path).name
    output_file = output_dir / f"{loader.get_output_prefix()}_{model_name}_{timestamp}.json"

    output_data = {
        "metadata": {
            "dataset": args.dataset,
            "split": args.split,
            "model_path": args.model_path,
            "num_prompts": args.num_prompts,
            "completions_per_prompt": args.n,
            "max_tokens": args.max_tokens,
            "max_model_len": args.max_model_len,
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
            "pass_at_k": pass_at_k,
            "pass_rate_distribution": pass_rate_buckets,
            "finish_reason_breakdown": finish_stats,
            "completion_length": length_summary,
        },
        "per_prompt_results": [asdict(r) for r in results],
    }

    # Add dataset-specific breakdowns
    if args.dataset == "math":
        output_data["metadata"]["levels"] = levels
        output_data["summary"]["by_level"] = level_breakdown
        output_data["summary"]["by_subject"] = subject_breakdown

    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"\nResults saved to: {output_file}")

    return overall_pass_rate


if __name__ == "__main__":
    main()
