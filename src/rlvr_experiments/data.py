from __future__ import annotations

import asyncio
import hashlib
import ray.data
from datasets import load_dataset

from .sample_logger import log_sample


def _hash_prompt(prompt: str) -> str:
    """Generate a short hash for prompts without explicit IDs."""
    return hashlib.sha256(prompt.encode()).hexdigest()[:12]


# --- Default config per dataset type ---
# These can be overridden by config, but provide sensible defaults for mixed training.

# GSM8K: Grade school math, shorter reasoning chains
# Format: "Q: ... A:" to match lm-evaluation-harness (no system prompt, no \boxed{})
# Model should output reasoning then "The answer is X."
GSM8K_SYSTEM_PROMPT = ""
GSM8K_ASSISTANT_PREFIX = ""
GSM8K_MAX_COMPLETION_LEN = 512

# MATH: Competition math, longer reasoning needed
# Format: "Problem:\n{problem}\n\nSolution:" to match minerva_math eval format
# This ensures training and eval use the same prompt format
MATH_SYSTEM_PROMPT = ""
MATH_ASSISTANT_PREFIX = ""
MATH_MAX_COMPLETION_LEN = 1024

# Code datasets
CODE_SYSTEM_PROMPT = ""
CODE_ASSISTANT_PREFIX = ""
CODE_MAX_COMPLETION_LEN = 512

# IFEval: Instruction following
IFEVAL_SYSTEM_PROMPT = ""
IFEVAL_ASSISTANT_PREFIX = ""
IFEVAL_MAX_COMPLETION_LEN = 2048


def load_gsm8k(split: str = "train") -> ray.data.Dataset:
    """
    Load GSM8k dataset as a Ray Dataset.

    Returns dataset with columns: "prompt", "problem"
    where "problem" is a dict with "answer" and "prompt_id" for use with MathVerifier.
    """
    import os
    import subprocess

    # Check for S3 cache first (for SageMaker VPC environments without internet)
    s3_cache = f"s3://sagemaker-us-west-2-503561457547/rlvr-experiments/datasets/gsm8k_{split}/"
    local_cache = f"/tmp/gsm8k_{split}_cache"

    use_local_cache = False
    if os.path.exists(local_cache):
        print(f"[load_gsm8k] Loading from local cache: {local_cache}")
        use_local_cache = True
    else:
        # Try S3 first, fall back to HuggingFace Hub
        try:
            print(f"[load_gsm8k] Trying S3 cache: {s3_cache}")
            os.makedirs(local_cache, exist_ok=True)
            subprocess.run(
                ["aws", "s3", "sync", s3_cache, local_cache, "--quiet"],
                check=True, capture_output=True
            )
            print(f"[load_gsm8k] Loaded from S3 cache")
            use_local_cache = True
        except Exception as e:
            print(f"[load_gsm8k] S3 cache not available ({e}), loading from HuggingFace Hub")

    if use_local_cache:
        from datasets import Dataset
        hf_dataset = Dataset.load_from_disk(local_cache)
        items = list(hf_dataset)
    else:
        hf_dataset = load_dataset("openai/gsm8k", "main", split=split)
        items = list(hf_dataset)

    # Add index to each item for deterministic prompt_id
    indexed_items = [{"_idx": i, **item} for i, item in enumerate(items)]
    ds = ray.data.from_items(indexed_items)

    def preprocess(row):
        # Format: "Q: ... A:" to match lm-evaluation-harness GSM8K format
        question = f"Q: {row['question'].strip()}\nA:"
        answer = row["answer"].split("####")[-1].strip()
        # Use numeric index for prompt_id
        prompt_id = f"gsm8k_{row['_idx']}"
        return {
            "prompt": question,
            "problem": {
                "answer": answer,
                "prompt_id": prompt_id,
                "verifier_type": "gsm8k",
                "dataset_name": "gsm8k",
                "system_prompt": GSM8K_SYSTEM_PROMPT,
                "assistant_prefix": GSM8K_ASSISTANT_PREFIX,
                "max_completion_len": GSM8K_MAX_COMPLETION_LEN,
            },
        }

    return ds.map(preprocess)


def load_humaneval(**kwargs) -> ray.data.Dataset:
    """
    Load HumanEval dataset as a Ray Dataset.

    HumanEval has 164 Python programming problems. Each row contains:
    - prompt: Function signature + docstring (given to model)
    - test: Unit tests in "def check(candidate): ..." format
    - entry_point: Function name to test
    - task_id: Unique identifier (e.g., "HumanEval/0")

    The model should generate the function body to complete the prompt.

    Note: HumanEval only has a "test" split (164 problems), so the split
    parameter is ignored. kwargs are accepted for config compatibility.

    Returns dataset with columns: "prompt", "problem"
    where "problem" is a dict containing test, entry_point, task_id, prompt_id
    for use with HumanEvalVerifier.
    """
    # HumanEval only has "test" split - ignore any split param from config
    hf_dataset = load_dataset("openai/openai_humaneval", split="test")
    ds = ray.data.from_huggingface(hf_dataset)

    def preprocess(row):
        # The prompt is the function signature + docstring
        # Model needs to generate the function body
        # Use task_id as prompt_id (e.g., "HumanEval/0")
        return {
            "prompt": row["prompt"],
            "problem": {
                "prompt": row["prompt"],
                "test": row["test"],
                "entry_point": row["entry_point"],
                "task_id": row["task_id"],
                "prompt_id": row["task_id"],
                "verifier_type": "humaneval",
                "dataset_name": "humaneval",
                "system_prompt": CODE_SYSTEM_PROMPT,
                "assistant_prefix": CODE_ASSISTANT_PREFIX,
                "max_completion_len": CODE_MAX_COMPLETION_LEN,
            },
        }

    return ds.map(preprocess)


def load_mbpp(split: str = "train") -> ray.data.Dataset:
    """
    Load MBPP (Mostly Basic Python Problems) dataset as a Ray Dataset.

    MBPP has 974 Python programming problems across splits:
    - train: 374 problems (for RL training)
    - test: 500 problems (for evaluation)
    - validation: 90 problems
    - prompt: 10 problems (few-shot examples)

    Each row contains:
    - text: Problem description in natural language
    - code: Canonical solution
    - test_list: List of assert statements for verification
    - task_id: Unique identifier

    Returns dataset with columns: "prompt", "problem"
    where "problem" is a dict for use with MBPPVerifier.
    """
    import os
    import subprocess
    import tempfile

    # Check for S3 cache first (for SageMaker VPC environments without internet)
    s3_cache = f"s3://sagemaker-us-west-2-503561457547/rlvr-experiments/datasets/mbpp_{split}/"
    local_cache = f"/tmp/mbpp_{split}_cache"

    use_local_cache = False
    if os.path.exists(local_cache):
        # Already cached locally
        print(f"[load_mbpp] Loading from local cache: {local_cache}")
        use_local_cache = True
    else:
        # Try S3 first, fall back to HuggingFace Hub
        try:
            print(f"[load_mbpp] Trying S3 cache: {s3_cache}")
            os.makedirs(local_cache, exist_ok=True)
            subprocess.run(
                ["aws", "s3", "sync", s3_cache, local_cache, "--quiet"],
                check=True, capture_output=True
            )
            print(f"[load_mbpp] Loaded from S3 cache")
            use_local_cache = True
        except Exception as e:
            print(f"[load_mbpp] S3 cache not available ({e}), loading from HuggingFace Hub")

    if use_local_cache:
        # Load directly from disk using Arrow - avoids HuggingFace Hub entirely
        from datasets import Dataset
        hf_dataset = Dataset.load_from_disk(local_cache)
        # Convert to list of dicts and create Ray dataset directly (avoids HF metadata fetches)
        ds = ray.data.from_items(list(hf_dataset))
    else:
        hf_dataset = load_dataset("google-research-datasets/mbpp", "full", split=split)
        ds = ray.data.from_huggingface(hf_dataset)

    def preprocess(row):
        # Use lm_eval MBPP prompt format for consistency with standard evals
        # Model needs to generate a complete function ending with [DONE]
        prompt_id = f"mbpp_{row['task_id']}"
        test_list = row["test_list"]
        # Format: task description + test cases + [BEGIN] marker
        prompt = (
            f"You are an expert Python programmer, and here is your task: {row['text']} "
            f"Your code should pass these tests:\n\n"
            f"{test_list[0]}\n{test_list[1]}\n{test_list[2]}\n[BEGIN]\n"
        )
        return {
            "prompt": prompt,
            "problem": {
                "text": row["text"],
                "test_list": test_list,
                "task_id": row["task_id"],
                "prompt_id": prompt_id,
                "verifier_type": "mbpp",
                "dataset_name": "mbpp",
                "system_prompt": CODE_SYSTEM_PROMPT,
                "assistant_prefix": CODE_ASSISTANT_PREFIX,
                "max_completion_len": CODE_MAX_COMPLETION_LEN,
            },
        }

    return ds.map(preprocess)


def load_math(
    split: str = "train",
    level: list[int] | None = None,
) -> ray.data.Dataset:
    """
    Load MATH dataset (Hendrycks et al.) as a Ray Dataset.

    The MATH dataset contains 12,500 challenging competition mathematics problems
    with step-by-step solutions. Problems span 7 subjects and 5 difficulty levels.

    Args:
        split: Dataset split ("train" or "test")
        level: Filter by difficulty level(s). Can be:
            - None: Include all levels (1-5)
            - list[int]: Specific levels (e.g., [1, 2, 3] for Levels 1-3)

    Returns dataset with columns: "prompt", "problem"
    where "problem" is a dict with "answer" and "prompt_id" for use with MathVerifier.
    """
    import os
    import subprocess

    # Check for S3 cache first (for SageMaker VPC environments without internet)
    s3_cache = f"s3://sagemaker-us-west-2-503561457547/rlvr-experiments/datasets/math_{split}/"
    local_cache = f"/tmp/math_{split}_cache"

    use_local_cache = False
    if os.path.exists(local_cache):
        print(f"[load_math] Loading from local cache: {local_cache}")
        use_local_cache = True
    else:
        # Try S3 first, fall back to HuggingFace Hub
        try:
            print(f"[load_math] Trying S3 cache: {s3_cache}")
            os.makedirs(local_cache, exist_ok=True)
            subprocess.run(
                ["aws", "s3", "sync", s3_cache, local_cache, "--quiet"],
                check=True, capture_output=True
            )
            print(f"[load_math] Loaded from S3 cache")
            use_local_cache = True
        except Exception as e:
            print(f"[load_math] S3 cache not available ({e}), loading from HuggingFace Hub")

    if use_local_cache:
        from datasets import Dataset
        hf_dataset = Dataset.load_from_disk(local_cache)
        all_rows = list(hf_dataset)
    else:
        # Load all subjects and concatenate
        subjects = [
            "algebra",
            "counting_and_probability",
            "geometry",
            "intermediate_algebra",
            "number_theory",
            "prealgebra",
            "precalculus",
        ]

        all_rows = []
        for subject in subjects:
            hf_dataset = load_dataset("EleutherAI/hendrycks_math", subject, split=split)
            all_rows.extend(list(hf_dataset))

    # Filter by level if specified
    if level is not None:
        level_strs = {f"Level {l}" for l in level}
        all_rows = [row for row in all_rows if row["level"] in level_strs]

    # Add index to each item for deterministic prompt_id
    # Group by subject for more interpretable IDs
    subject_counters = {}
    indexed_rows = []
    for row in all_rows:
        subject = row["type"].lower()
        if subject not in subject_counters:
            subject_counters[subject] = 0
        row["_subject"] = subject
        row["_subject_idx"] = subject_counters[subject]
        subject_counters[subject] += 1
        indexed_rows.append(row)

    ds = ray.data.from_items(indexed_rows)

    def preprocess(row):
        # Format: "Problem:\n{problem}\n\nSolution:" to match minerva_math eval format
        # Model should output reasoning with final answer in \boxed{}
        question = f"Problem:\n{row['problem'].strip()}\n\nSolution:"
        # Use subject + index for prompt_id (e.g., math_algebra_0)
        prompt_id = f"math_{row['_subject']}_{row['_subject_idx']}"
        return {
            "prompt": question,
            "problem": {
                "answer": row["solution"],
                "prompt_id": prompt_id,
                "verifier_type": "minerva_math",
                "dataset_name": "math",
                "system_prompt": MATH_SYSTEM_PROMPT,
                "assistant_prefix": MATH_ASSISTANT_PREFIX,
                "max_completion_len": MATH_MAX_COMPLETION_LEN,
            },
        }

    return ds.map(preprocess)


def load_apps(
    split: str = "train",
    difficulty: list[str] | None = None,
) -> ray.data.Dataset:
    """
    Load APPS (Automated Programming Progress Standard) dataset as a Ray Dataset.

    APPS has 5000 training and 5000 test problems from competitive programming
    sites like Codeforces, with three difficulty levels.

    Args:
        split: Dataset split ("train" or "test")
        difficulty: Filter by difficulty level(s). Can be:
            - None: Include all levels
            - list[str]: Specific levels (e.g., ["introductory", "interview"])
            Valid values: "introductory", "interview", "competition"

    Returns dataset with columns: "prompt", "problem"
    where "problem" is a dict for use with APPSVerifier.
    """
    import json
    from huggingface_hub import hf_hub_download

    # Download JSONL file from HuggingFace
    filename = f"{split}.jsonl"
    path = hf_hub_download(
        repo_id="codeparrot/apps",
        filename=filename,
        repo_type="dataset"
    )
    print(f"[load_apps] Loaded {split} from: {path}")

    # Parse JSONL
    all_rows = []
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            # Filter by difficulty if specified
            if difficulty is not None and row.get("difficulty") not in difficulty:
                continue
            all_rows.append(row)

    print(f"[load_apps] Loaded {len(all_rows)} problems (difficulty filter: {difficulty})")
    ds = ray.data.from_items(all_rows)

    def preprocess(row):
        prompt_id = f"apps_{row['id']}"
        question = row["question"]

        # Parse input_output - it's a JSON string
        io_data = row.get("input_output", "{}")
        if isinstance(io_data, str):
            io_data = json.loads(io_data) if io_data else {}

        # Ensure inputs/outputs are always lists (for consistent Arrow schema)
        inputs = io_data.get("inputs", [])
        outputs = io_data.get("outputs", [])
        if not isinstance(inputs, list):
            inputs = []
        if not isinstance(outputs, list):
            outputs = []

        # Parse solutions - it's a JSON string containing a list
        solutions = row.get("solutions", "[]")
        if isinstance(solutions, str):
            solutions = json.loads(solutions) if solutions else []
        if not isinstance(solutions, list):
            solutions = []

        # Format prompt - include starter code if present
        starter = row.get("starter_code", "")
        if starter is None:
            starter = ""
        starter = starter.strip()
        if starter:
            prompt = f"{question}\n\nStarter code:\n```python\n{starter}\n```\n"
        else:
            prompt = question

        return {
            "prompt": prompt,
            "problem": {
                "question": question,
                "inputs": inputs,
                "outputs": outputs,
                "num_solutions": len(solutions),  # Store count instead of list to avoid schema issues
                "starter_code": starter,
                "difficulty": row.get("difficulty", "unknown"),
                "task_id": row["id"],
                "prompt_id": prompt_id,
                "verifier_type": "apps",
                "dataset_name": "apps",
                "system_prompt": CODE_SYSTEM_PROMPT,
                "assistant_prefix": CODE_ASSISTANT_PREFIX,
                "max_completion_len": CODE_MAX_COMPLETION_LEN,
            },
        }

    return ds.map(preprocess)


def load_ifeval(split: str = "train") -> ray.data.Dataset:
    """
    Load RLVR-IFeval dataset as a Ray Dataset.

    RLVR-IFeval contains 14,973 instruction-following prompts with verifiable
    constraints from IFEval. Each prompt includes a constraint that can be
    automatically verified (e.g., "all lowercase", "include keyword X times").

    Args:
        split: Dataset split ("train")

    Returns dataset with columns: "prompt", "problem"
    where "problem" is a dict containing ground_truth JSON for use with IFEvalVerifier.
    """
    import os
    import subprocess

    # Check for S3 cache first (for SageMaker VPC environments without internet)
    s3_cache = f"s3://sagemaker-us-west-2-503561457547/rlvr-experiments/datasets/ifeval_{split}/"
    local_cache = f"/tmp/ifeval_{split}_cache"

    use_local_cache = False
    if os.path.exists(local_cache):
        print(f"[load_ifeval] Loading from local cache: {local_cache}")
        use_local_cache = True
    else:
        # Try S3 first, fall back to HuggingFace Hub
        try:
            print(f"[load_ifeval] Trying S3 cache: {s3_cache}")
            os.makedirs(local_cache, exist_ok=True)
            subprocess.run(
                ["aws", "s3", "sync", s3_cache, local_cache, "--quiet"],
                check=True, capture_output=True
            )
            print(f"[load_ifeval] Loaded from S3 cache")
            use_local_cache = True
        except Exception as e:
            print(f"[load_ifeval] S3 cache not available ({e}), loading from HuggingFace Hub")

    if use_local_cache:
        from datasets import Dataset
        hf_dataset = Dataset.load_from_disk(local_cache)
        items = list(hf_dataset)
    else:
        hf_dataset = load_dataset("allenai/RLVR-IFeval", split=split)
        items = list(hf_dataset)

    # Add index to each item for deterministic prompt_id
    indexed_items = [{"_idx": i, **item} for i, item in enumerate(items)]
    ds = ray.data.from_items(indexed_items)

    def preprocess(row):
        # Extract user message content from messages list
        messages = row["messages"]
        user_content = ""
        for msg in messages:
            if msg["role"] == "user":
                user_content = msg["content"]
                break

        # Use numeric index for prompt_id
        prompt_id = f"ifeval_{row['_idx']}"

        return {
            "prompt": user_content,
            "problem": {
                "ground_truth": row["ground_truth"],
                "constraint_type": row["constraint_type"],
                "constraint": row["constraint"],
                "prompt_id": prompt_id,
                "verifier_type": "ifeval",
                "dataset_name": "ifeval",
                "system_prompt": IFEVAL_SYSTEM_PROMPT,
                "assistant_prefix": IFEVAL_ASSISTANT_PREFIX,
                "max_completion_len": IFEVAL_MAX_COMPLETION_LEN,
            },
        }

    return ds.map(preprocess)


def load_dummy(split: str = "train", num_samples: int = 64) -> ray.data.Dataset:
    """
    Load a dummy dataset with a single question repeated N times.

    Useful for testing that rewards are increasing on a single problem.
    Returns dataset with columns: "prompt", "problem"
    where "problem" is a dict with "answer" and "prompt_id" for use with MathVerifier.
    """
    prompt = "\n\nProblem:What is ((7/12) + (5/18)) / (31/36)?"
    rows = [
        {
            "prompt": prompt,
            "problem": {
                "answer": "1",
                "prompt_id": f"dummy_{i}",
                "verifier_type": "math",
                "dataset_name": "dummy",
                "system_prompt": GSM8K_SYSTEM_PROMPT,
                "assistant_prefix": GSM8K_ASSISTANT_PREFIX,
                "max_completion_len": GSM8K_MAX_COMPLETION_LEN,
            },
        }
        for i in range(num_samples)
    ]
    return ray.data.from_items(rows)


# --- AllenAI RLVR Dataset ---
# AllenAI's RLVR-GSM-MATH-IF-Mixed-Constraints dataset has CoT examples baked into prompts:
# - GSM8K: 8-shot CoT, "So the answer is X." format
# - MATH: 4-shot CoT, \boxed{} format
# - IFEval: Constraint-based instruction following with verification metadata

ALLENAI_GSM8K_MAX_COMPLETION_LEN = 512


def _transform_gsm8k_to_lmeval_format(prompt: str, num_shots: int = 4) -> str:
    """
    Transform AllenAI GSM8K prompt to match lm_eval gsm8k_cot format:
    - Replace "Question:" -> "Q:" and "Answer:" -> "A:"
    - Keep only first num_shots examples (AllenAI has 8, lm_eval 4-shot uses 4)
    - Change "So the answer is" -> "The answer is" to match lm_eval format

    Returns transformed prompt (without trailing "A:" - that's added as assistant_prefix)
    """
    import re

    # Replace Question:/Answer: with Q:/A:
    prompt = prompt.replace("Question:", "Q:")
    prompt = prompt.replace("Answer:", "A:")

    # Change "So the answer is" to "The answer is" in fewshot examples
    prompt = prompt.replace("So the answer is", "The answer is")

    # Split into Q&A pairs and keep only first num_shots + the final question
    # Pattern: Q: ... A: ... (repeated), then final Q: ...
    # We need to find where each Q: starts
    parts = re.split(r'(?=Q:)', prompt)
    parts = [p for p in parts if p.strip()]  # Remove empty parts

    if len(parts) > num_shots + 1:
        # Keep first num_shots complete Q&A pairs + final question
        # Each complete pair is "Q: question\nA: answer\n\n"
        # Final question is just "Q: question"
        kept_parts = parts[:num_shots] + [parts[-1]]
        prompt = "".join(kept_parts)

    return prompt
ALLENAI_MATH_MAX_COMPLETION_LEN = 1024
ALLENAI_IFEVAL_MAX_COMPLETION_LEN = 2048


def load_allenai_rlvr(
    split: str = "train",
    datasets: list[str] | None = None,
) -> ray.data.Dataset:
    """
    Load AllenAI's RLVR-GSM-MATH-IF-Mixed-Constraints dataset.

    This dataset has CoT examples baked into the prompts, matching the format
    used in the Tulu-3 paper. Key features:
    - GSM8K: 8-shot CoT with "So the answer is X." format
    - MATH: 4-shot CoT with \\boxed{} format
    - IFEval: Constraint verification metadata in ground_truth field

    Args:
        split: Dataset split (only "train" available)
        datasets: List of datasets to include. Options: ["gsm8k", "MATH", "ifeval"]
                  If None, includes all datasets.

    Returns:
        Ray Dataset with columns: "prompt", "problem"
        where "problem" contains verifier metadata.
    """
    import os
    import subprocess

    # Check for S3 cache first (for SageMaker VPC environments)
    s3_cache = "s3://sagemaker-us-west-2-503561457547/rlvr-experiments/datasets/allenai_rlvr_train/"
    local_cache = "/tmp/allenai_rlvr_train_cache"

    use_local_cache = False
    # Check for valid local cache (must have dataset files, not just empty dir)
    if os.path.exists(local_cache) and os.path.exists(os.path.join(local_cache, "dataset_info.json")):
        print(f"[load_allenai_rlvr] Loading from local cache: {local_cache}")
        use_local_cache = True
    else:
        # Try S3 first, fall back to HuggingFace Hub
        try:
            print(f"[load_allenai_rlvr] Trying S3 cache: {s3_cache}")
            os.makedirs(local_cache, exist_ok=True)
            result = subprocess.run(
                ["aws", "s3", "sync", s3_cache, local_cache, "--quiet"],
                check=True, capture_output=True
            )
            # Check if we actually got files
            if os.path.exists(os.path.join(local_cache, "dataset_info.json")):
                print(f"[load_allenai_rlvr] Loaded from S3 cache")
                use_local_cache = True
            else:
                print(f"[load_allenai_rlvr] S3 cache empty, loading from HuggingFace Hub")
                # Clean up empty directory
                import shutil
                shutil.rmtree(local_cache, ignore_errors=True)
        except Exception as e:
            print(f"[load_allenai_rlvr] S3 cache not available ({e}), loading from HuggingFace Hub")
            # Clean up any partial download
            import shutil
            shutil.rmtree(local_cache, ignore_errors=True)

    if use_local_cache:
        from datasets import Dataset
        hf_dataset = Dataset.load_from_disk(local_cache)
        items = list(hf_dataset)
    else:
        hf_dataset = load_dataset("allenai/RLVR-GSM-MATH-IF-Mixed-Constraints", split=split)
        items = list(hf_dataset)

    # Filter by dataset if specified
    if datasets is not None:
        dataset_set = set(d.lower() for d in datasets)
        items = [item for item in items if item["dataset"].lower() in dataset_set]
        print(f"[load_allenai_rlvr] Filtered to datasets {datasets}: {len(items)} samples")

    # Add index for deterministic prompt_ids
    indexed_items = [{"_idx": i, **item} for i, item in enumerate(items)]
    ds = ray.data.from_items(indexed_items)

    def preprocess(row):
        # Extract user message content from messages list
        messages = row["messages"]
        user_content = ""
        for msg in messages:
            if msg["role"] == "user":
                user_content = msg["content"]
                break

        dataset_name = row["dataset"].lower()
        prompt_id = f"allenai_{dataset_name}_{row['_idx']}"

        # Determine verifier type and settings based on source dataset
        assistant_prefix = ""
        if dataset_name == "gsm8k":
            verifier_type = "allenai_gsm8k"
            max_completion_len = ALLENAI_GSM8K_MAX_COMPLETION_LEN
            # Transform to lm_eval format: Q:/A:, 4-shot, "The answer is"
            user_content = _transform_gsm8k_to_lmeval_format(user_content, num_shots=4)
            assistant_prefix = "A:"
        elif dataset_name == "math":
            verifier_type = "allenai_math"
            max_completion_len = ALLENAI_MATH_MAX_COMPLETION_LEN
        elif dataset_name == "ifeval":
            verifier_type = "ifeval"  # Reuse existing IFEval verifier
            max_completion_len = ALLENAI_IFEVAL_MAX_COMPLETION_LEN
        else:
            verifier_type = "unknown"
            max_completion_len = 1024

        # Convert None to empty string to avoid Arrow schema mismatch
        # (GSM8K/MATH have None for constraint fields, IFEval has strings)
        constraint_type = row["constraint_type"] if row["constraint_type"] is not None else ""
        constraint = row["constraint"] if row["constraint"] is not None else ""

        return {
            "prompt": user_content,
            "problem": {
                "ground_truth": row["ground_truth"],
                "constraint_type": constraint_type,
                "constraint": constraint,
                "source_dataset": row["dataset"],
                "messages": messages,
                "prompt_id": prompt_id,
                "verifier_type": verifier_type,
                "dataset_name": f"allenai_{dataset_name}",
                "system_prompt": "",  # AllenAI doesn't use system prompts
                "assistant_prefix": assistant_prefix,
                "max_completion_len": max_completion_len,
            },
        }

    return ds.map(preprocess)


def load_allenai_gsm8k_mini(
    n_samples: int = 10,
    seed: int = 42,
) -> ray.data.Dataset:
    """
    Load a fixed mini subset of GSM8K from AllenAI RLVR for overfitting tests.

    This is useful for verifying the training harness is working correctly
    by checking if the model can overfit on a small fixed set of samples.

    Args:
        n_samples: Number of samples to select (default: 10)
        seed: Random seed for sample selection (default: 42)

    Returns:
        Ray Dataset with n_samples GSM8K problems
    """
    import random

    # Load full GSM8K subset from AllenAI RLVR
    full_ds = load_allenai_rlvr(datasets=["gsm8k"])

    # Materialize and select fixed subset
    all_rows = list(full_ds.iter_rows())

    # Use deterministic random selection
    rng = random.Random(seed)
    selected_rows = rng.sample(all_rows, min(n_samples, len(all_rows)))

    print(f"[load_allenai_gsm8k_mini] Selected {len(selected_rows)} samples from {len(all_rows)} GSM8K problems (seed={seed})")

    # Print selected prompt_ids for reproducibility
    for row in selected_rows:
        print(f"  - {row['problem']['prompt_id']}")

    return ray.data.from_items(selected_rows)


# --- Dataset registry for load_mixed ---

DATASET_LOADERS = {
    "gsm8k": load_gsm8k,
    "math": load_math,
    "humaneval": load_humaneval,
    "mbpp": load_mbpp,
    "apps": load_apps,
    "ifeval": load_ifeval,
    "dummy": load_dummy,
    "allenai_rlvr": load_allenai_rlvr,
    "allenai_gsm8k_mini": load_allenai_gsm8k_mini,
}


def load_mixed(
    datasets: list[dict],
    mode: str = "weighted",
    seed: int = 42,
) -> tuple[ray.data.Dataset, list[str]]:
    """
    Load and combine multiple datasets with configurable ordering.

    Supports two modes:
    - "weighted": Probabilistic interleaving based on weights (default)
    - "sequential": Concatenate datasets in config order with optional count limits

    Each dataset config can specify:
    - name: Dataset name (required, must be in DATASET_LOADERS)
    - split: Split to load (default: "train")
    - weight: Sampling probability weight for weighted mode (default: 1.0)
    - count: Number of samples to take for sequential mode (default: all)
    - order: Path to file with line-separated prompt_ids for priority ordering
    - **kwargs: Additional args passed to the loader (e.g., level for MATH)

    Args:
        datasets: List of dataset configs
        mode: "weighted" for probabilistic interleaving, "sequential" for concatenation
        seed: Random seed for shuffling and weighted sampling

    Returns:
        Tuple of (combined_dataset, order_list) where:
        - combined_dataset: Ray Dataset with all samples
        - order_list: List of prompt_ids in the final order

    Note: Each dataset type has its own namespaced prompt_ids (gsm8k_, math_, etc.),
    so mixing different datasets won't cause ID collisions. If the same dataset is
    used twice, duplicate prompt_ids may appear in the order list. DataIterator
    handles duplicates by tracking status per prompt_id, so each unique prompt_id
    is only processed once.

    Example weighted config:
        data:
          dataset: mixed
          mode: weighted  # default
          datasets:
            - name: gsm8k
              weight: 0.7
            - name: math
              weight: 0.3

    Example sequential config:
        data:
          dataset: mixed
          mode: sequential
          datasets:
            - name: gsm8k
              count: 500
            - name: math
              count: 500
            - name: mbpp
              count: 200
    """
    import os
    import random

    if not datasets:
        raise ValueError("datasets list cannot be empty")

    if mode not in ("weighted", "sequential"):
        raise ValueError(f"mode must be 'weighted' or 'sequential', got '{mode}'")

    rng = random.Random(seed)

    # Step 1: Load all datasets and build per-dataset ordered prompt_id lists
    all_rows = []
    per_dataset_queues: list[tuple[str, list[str]]] = []  # preserve config order for sequential
    prompt_id_to_row: dict[str, dict] = {}
    weights: dict[str, float] = {}
    counts: dict[str, int | None] = {}

    for ds_config in datasets:
        name = ds_config.get("name")
        if not name:
            raise ValueError("Each dataset config must have a 'name' key")

        if name not in DATASET_LOADERS:
            raise ValueError(
                f"Unknown dataset: {name}. Available: {list(DATASET_LOADERS.keys())}"
            )

        # Extract loader kwargs (everything except our special keys)
        special_keys = ("name", "weight", "count", "order")
        loader_kwargs = {k: v for k, v in ds_config.items() if k not in special_keys}

        # Load the dataset
        loader = DATASET_LOADERS[name]
        ds = loader(**loader_kwargs)

        # Materialize to list
        rows = list(ds.iter_rows())
        row_by_id = {row["problem"]["prompt_id"]: row for row in rows}

        # Build ordered prompt_id list for this dataset
        order = ds_config.get("order")
        if order:
            # Use explicit order from file, filtering to valid IDs
            # Handle S3 paths
            if order.startswith("s3://"):
                import subprocess
                import tempfile
                with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as tmp:
                    tmp_path = tmp.name
                subprocess.run(["aws", "s3", "cp", order, tmp_path, "--quiet"], check=True)
                with open(tmp_path, "r") as f:
                    file_order = [line.strip() for line in f if line.strip()]
                os.remove(tmp_path)
            else:
                with open(order, "r") as f:
                    file_order = [line.strip() for line in f if line.strip()]
            ordered_ids = [pid for pid in file_order if pid in row_by_id]
            if not ordered_ids:
                raise ValueError(f"Order file {order} has no valid prompt_ids for {name}")
            print(f"[load_mixed] {name}: using order file with {len(ordered_ids)}/{len(file_order)} valid IDs")
        else:
            # No order file -> shuffle
            ordered_ids = list(row_by_id.keys())
            rng.shuffle(ordered_ids)

        # Store config values
        weights[name] = ds_config.get("weight", 1.0)
        counts[name] = ds_config.get("count")  # None means all

        # Apply count limit for sequential mode
        if mode == "sequential" and counts[name] is not None:
            ordered_ids = ordered_ids[:counts[name]]

        per_dataset_queues.append((name, ordered_ids))

        # Add rows and index (only those in ordered_ids)
        ordered_ids_set = set(ordered_ids)
        for row in rows:
            pid = row["problem"]["prompt_id"]
            if pid in ordered_ids_set:
                prompt_id_to_row[pid] = row
                all_rows.append(row)

        count_info = f"count={counts[name]}" if counts[name] else "all"
        weight_info = f"weight={weights[name]}"
        print(f"[load_mixed] Loaded {len(ordered_ids)} samples from {name} ({weight_info if mode == 'weighted' else count_info})")

    # Step 2: Build final order based on mode
    final_order: list[str] = []

    if mode == "sequential":
        # Sequential: just concatenate in config order
        for name, ordered_ids in per_dataset_queues:
            final_order.extend(ordered_ids)
        print(f"[load_mixed] Total: {len(final_order)} samples, sequential from {len(datasets)} datasets")

    else:  # weighted
        # Convert to dict for weighted sampling
        queues_dict = {name: list(ids) for name, ids in per_dataset_queues}

        def get_active_datasets() -> list[str]:
            return [n for n, q in queues_dict.items() if q]

        def sample_dataset() -> str | None:
            active = get_active_datasets()
            if not active:
                return None
            # Normalize weights for active datasets
            active_weights = [weights[n] for n in active]
            total = sum(active_weights)
            probs = [w / total for w in active_weights]
            # Weighted random choice
            r = rng.random()
            cumsum = 0.0
            for n, p in zip(active, probs):
                cumsum += p
                if r <= cumsum:
                    return n
            return active[-1]  # Fallback

        while True:
            ds_name = sample_dataset()
            if ds_name is None:
                break
            # Pop next item from this dataset's queue
            prompt_id = queues_dict[ds_name].pop(0)
            final_order.append(prompt_id)

        print(f"[load_mixed] Total: {len(final_order)} samples, interleaved from {len(datasets)} datasets")

    return ray.data.from_items(all_rows), final_order


class DataIterator:
    """
    Iterator over a Ray Dataset with status tracking for each prompt.
    Applies chat template to prompts for direct use with vLLM.

    All datasets should have columns "prompt" and "problem", where
    "problem" is a dict containing whatever the verifier needs.

    Tracks status per prompt_id: "pending" | "in_flight" | "done" | "failed"
    - pending: not yet processed or needs retry
    - in_flight: currently being processed (generated, or waiting to be trained)
    - done: finished (trained or intentionally skipped/filtered)
    - failed: permanently failed (will not be retried)

    Supports priority ordering:
    - Pass `order` to __init__ or new_epoch() to specify initial prompt order
    - Retried prompts (via mark_pending) are moved to the front of the queue

    Usage:
        data_iter = DataIterator(ds, tokenizer=tokenizer)

        for epoch in range(num_epochs):
            data_iter.new_epoch(seed=epoch)

            while not data_iter.all_done():
                item = data_iter.get_next()
                if item is None:
                    # No pending items, wait for in-flight to complete
                    continue
                # process item...
                # on success: data_iter.mark_done(prompt_id)
                # on waste: data_iter.mark_pending(prompt_id)
                # on permanent failure: data_iter.mark_failed(prompt_id)
    """

    def __init__(
        self,
        ds: ray.data.Dataset,
        tokenizer,
        system_prompt: str = "",
        assistant_prefix: str = "",
        order: list[str] | str | None = None,
        skip_chat_template: bool = False,
    ):
        self.ds = ds
        self.tokenizer = tokenizer
        self.assistant_prefix = assistant_prefix
        self.system_prompt = system_prompt
        self.skip_chat_template = skip_chat_template
        # Build index: prompt_id -> row data
        self._prompt_id_index: dict[str, dict] = {}
        self._build_index()
        self._epoch_started = False
        # Status tracking: prompt_id -> "pending" | "in_flight" | "done" | "failed"
        self._status: dict[str, str] = {}

        # Load order from file if it's a string path
        if isinstance(order, str):
            print(f"[DataIterator] Loading prompt order from: {order}")
            with open(order, "r") as f:
                order = [line.strip() for line in f if line.strip()]
            print(f"[DataIterator] Loaded {len(order)} prompt_ids from order file")

        # Ordered list of prompt_ids for iteration (determines priority)
        # Items at the front are processed first
        # Items not in order are ignored (not processed)
        if order is not None:
            # Filter to only valid prompt_ids, ignore unknown ones silently
            valid_ids = set(self._prompt_id_index.keys())
            self._order: list[str] = [pid for pid in order if pid in valid_ids]
            if not self._order:
                raise ValueError("order contains no valid prompt_ids")
            # Store explicit order to preserve across new_epoch calls
            self._explicit_order: list[str] | None = list(self._order)
            skipped = len(order) - len(self._order)
            if skipped > 0:
                print(f"[DataIterator] Note: {skipped} prompt_ids from order not found in dataset")
        else:
            self._order: list[str] = list(self._prompt_id_index.keys())
            self._explicit_order = None

    def _build_index(self) -> None:
        """Build prompt_id -> row index."""
        for batch in self.ds.iter_batches(batch_size=256):
            prompts = list(batch["prompt"])
            problems = list(batch["problem"])
            for p, prob in zip(prompts, problems):
                prompt_id = prob.get("prompt_id")
                if prompt_id:
                    self._prompt_id_index[prompt_id] = {"prompt": p, "problem": prob}

    def _apply_template(self, prompt: str, problem: dict) -> str:
        """Apply chat template to a single prompt.

        Uses per-row system_prompt/assistant_prefix from problem dict if present,
        otherwise falls back to global defaults.

        If skip_chat_template=True, returns the raw prompt + assistant_prefix without
        applying any chat formatting. Use this for base models that don't understand
        chat special tokens like <|im_start|>.
        """
        # Per-row overrides take precedence over global config
        system_prompt = problem.get("system_prompt") or self.system_prompt
        assistant_prefix = problem.get("assistant_prefix") or self.assistant_prefix

        if self.skip_chat_template:
            # For base models: just return raw prompt + prefix
            # Optionally prepend system prompt as regular text if provided
            if system_prompt:
                return system_prompt + "\n\n" + prompt + assistant_prefix
            return prompt + assistant_prefix

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        content = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,  # TODO: make configurable
        ) + assistant_prefix
        return content

    def _get_item(self, prompt_id: str) -> dict:
        """Get formatted item for a prompt_id."""
        row = self._prompt_id_index[prompt_id]
        return {
            "template": self._apply_template(row["prompt"], row["problem"]),
            "prompt": row["prompt"],
            "problem": row["problem"],
        }

    # --- Status management ---

    def new_epoch(self, seed: int | None = None, order: list[str] | None = None) -> None:
        """Reset all non-failed statuses to pending for a new epoch.

        Args:
            seed: Random seed for shuffling. If None and order is None, keeps current order.
                  If an explicit order was provided at init time, seed is ignored to preserve
                  the intended ordering (e.g., sequential dataset mode).
            order: Explicit ordering of prompt_ids. If provided, overrides seed-based shuffling.
                   Items not in order are appended at the end.
        """
        import random

        self._epoch_started = True

        if order is not None:
            # Use provided order, filter to valid ids only
            # Items not in order are ignored (not processed this epoch)
            valid_ids = set(self._prompt_id_index.keys())
            self._order = [pid for pid in order if pid in valid_ids]
            if not self._order:
                raise ValueError("order contains no valid prompt_ids")
        elif self._explicit_order is not None:
            # Preserve explicit order from init (e.g., sequential dataset mode)
            # Don't shuffle even if seed is provided
            self._order = list(self._explicit_order)
        elif seed is not None:
            rng = random.Random(seed)
            rng.shuffle(self._order)

        for pid in self._order:
            if self._status.get(pid) == "failed":
                continue
            self._status[pid] = "pending"

    def mark_done(self, prompt_id: str) -> None:
        """Mark a prompt as finished (trained or intentionally skipped)."""
        if self._status.get(prompt_id) == "failed":
            return
        self._status[prompt_id] = "done"

    def mark_failed(self, prompt_id: str) -> None:
        """Mark a prompt as permanently failed (will not be retried)."""
        self._status[prompt_id] = "failed"

    def mark_pending(self, prompt_id: str) -> None:
        """Mark a prompt as pending (for retry) and move it to front of queue.

        Retried items get priority over fresh items to minimize staleness waste.
        """
        if self._status.get(prompt_id) == "failed":
            return
        self._status[prompt_id] = "pending"
        # Move to front of order list for priority processing
        if prompt_id in self._order:
            self._order.remove(prompt_id)
        self._order.insert(0, prompt_id)
        log_sample("pending", prompt_id=prompt_id)

    def all_done(self) -> bool:
        """Check if all prompts are finished (done or failed)."""
        return all(s in {"done", "failed"} for s in self._status.values())

    def pending_count(self) -> int:
        """Count of pending prompts."""
        return sum(1 for s in self._status.values() if s == "pending")

    def in_flight_count(self) -> int:
        """Count of in-flight prompts."""
        return sum(1 for s in self._status.values() if s == "in_flight")

    def done_count(self) -> int:
        """Count of done prompts."""
        return sum(1 for s in self._status.values() if s == "done")

    def failed_count(self) -> int:
        """Count of failed prompts."""
        return sum(1 for s in self._status.values() if s == "failed")

    def _require_epoch(self) -> None:
        if not self._epoch_started:
            raise RuntimeError("DataIterator.new_epoch() must be called before iterating.")

    def get_next(self) -> dict | None:
        """Get next pending item and mark it in_flight.

        Returns the next available item (either a retry or new item), or None
        if all items are either in_flight, done, or failed. This is the primary
        interface for getting work items.

        Items are processed in order of self._order (front = highest priority).
        """
        self._require_epoch()
        for pid in self._order:
            if self._status.get(pid) == "pending":
                self._status[pid] = "in_flight"
                log_sample("in_flight", prompt_id=pid)
                return self._get_item(pid)
        return None

    # --- Async iteration interface ---

    async def get_next_async(self, poll_interval: float = 0.01) -> dict | None:
        """Await the next pending item, or return None when the epoch is exhausted.

        This is a simple async wrapper around `get_next()` that waits when there
        are no pending items but some are still in-flight (e.g. waiting to be
        marked done, retried, or failed).
        """
        while True:
            item = self.get_next()
            if item is not None:
                return item
            if self.all_done():
                return None
            await asyncio.sleep(poll_interval)

    async def async_items(self, poll_interval: float = 0.01):
        """Async generator yielding items until all are finished (done or failed).

        Yields items as they become available (either new or retried).
        Waits when items are in-flight but not yet resolved.

        Usage:
            async for item in data_iter.async_items():
                await process(item)
        """
        while True:
            item = await self.get_next_async(poll_interval=poll_interval)
            if item is None:
                return
            yield item
