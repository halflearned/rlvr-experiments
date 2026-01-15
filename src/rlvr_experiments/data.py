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
GSM8K_SYSTEM_PROMPT = "Solve the following math problem and provide the final answer inside \\boxed{}"
GSM8K_ASSISTANT_PREFIX = "Let's think step by step."
GSM8K_MAX_COMPLETION_LEN = 512

# MATH: Competition math, longer reasoning needed
MATH_SYSTEM_PROMPT = "Solve the following math problem and provide the final answer inside \\boxed{}"
MATH_ASSISTANT_PREFIX = "Let's think step by step."
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
    hf_dataset = load_dataset("openai/gsm8k", "main", split=split)
    ds = ray.data.from_huggingface(hf_dataset)

    def preprocess(row):
        question = f"\n\nProblem:{row['question'].strip()}"
        answer = row["answer"].split("####")[-1].strip()
        # GSM8K doesn't have task_id, so use hash of question
        prompt_id = f"gsm8k_{_hash_prompt(question)}"
        return {
            "prompt": question,
            "problem": {
                "answer": answer,
                "prompt_id": prompt_id,
                "verifier_type": "math",
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

    ds = ray.data.from_items(all_rows)

    def preprocess(row):
        question = f"\n\nProblem:{row['problem'].strip()}"
        # Store full solution - math_verify extracts from \boxed{} automatically
        prompt_id = f"math_{row['type']}_{_hash_prompt(question)}"
        return {
            "prompt": question,
            "problem": {
                "answer": row["solution"],
                "prompt_id": prompt_id,
                "verifier_type": "math",
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
    hf_dataset = load_dataset("allenai/RLVR-IFeval", split=split)
    ds = ray.data.from_huggingface(hf_dataset)

    def preprocess(row):
        # Extract user message content from messages list
        messages = row["messages"]
        user_content = ""
        for msg in messages:
            if msg["role"] == "user":
                user_content = msg["content"]
                break

        # Use hash of prompt as ID since dataset doesn't have explicit IDs
        prompt_id = f"ifeval_{_hash_prompt(user_content)}"

        return {
            "prompt": user_content,
            "problem": {
                "ground_truth": row["ground_truth"],
                "constraint_type": row["constraint_type"],
                "constraint": row["constraint"],
                "prompt_id": prompt_id,
                "verifier_type": "ifeval",
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
                "prompt_id": "dummy_0",
                "verifier_type": "math",
                "system_prompt": GSM8K_SYSTEM_PROMPT,
                "assistant_prefix": GSM8K_ASSISTANT_PREFIX,
                "max_completion_len": GSM8K_MAX_COMPLETION_LEN,
            },
        }
    ] * num_samples
    return ray.data.from_items(rows)


# --- Dataset registry for load_mixed ---

DATASET_LOADERS = {
    "gsm8k": load_gsm8k,
    "math": load_math,
    "humaneval": load_humaneval,
    "mbpp": load_mbpp,
    "apps": load_apps,
    "ifeval": load_ifeval,
    "dummy": load_dummy,
}


def load_mixed(
    datasets: list[dict],
    shuffle: bool = True,
    seed: int = 42,
) -> ray.data.Dataset:
    """
    Load and combine multiple datasets for mixed training.

    Each dataset config can specify:
    - name: Dataset name (required, must be in DATASET_LOADERS)
    - split: Split to load (default: "train")
    - weight: Sampling weight / max samples (optional)
    - **kwargs: Additional args passed to the loader (e.g., level for MATH)

    Args:
        datasets: List of dataset configs, e.g.:
            [
                {"name": "gsm8k", "split": "train"},
                {"name": "math", "split": "train", "level": [1, 2, 3]},
                {"name": "ifeval"},
                {"name": "mbpp", "split": "train"},
            ]
        shuffle: Whether to shuffle the combined dataset
        seed: Random seed for shuffling

    Returns:
        Combined Ray Dataset with all samples from specified datasets.
        Each sample has verifier_type, system_prompt, assistant_prefix in problem dict.

    Example config:
        data:
          datasets:
            - name: gsm8k
              split: train
            - name: math
              split: train
              level: [1, 2, 3]
            - name: ifeval
            - name: mbpp
              split: train
    """
    import random

    if not datasets:
        raise ValueError("datasets list cannot be empty")

    all_rows = []

    for ds_config in datasets:
        name = ds_config.get("name")
        if not name:
            raise ValueError("Each dataset config must have a 'name' key")

        if name not in DATASET_LOADERS:
            raise ValueError(
                f"Unknown dataset: {name}. Available: {list(DATASET_LOADERS.keys())}"
            )

        # Extract loader kwargs (everything except 'name' and 'weight')
        loader_kwargs = {k: v for k, v in ds_config.items() if k not in ("name", "weight")}

        # Load the dataset
        loader = DATASET_LOADERS[name]
        ds = loader(**loader_kwargs)

        # Materialize to list for combining
        rows = list(ds.iter_rows())

        # Apply weight/sampling if specified
        weight = ds_config.get("weight")
        if weight is not None and isinstance(weight, int) and weight < len(rows):
            rng = random.Random(seed)
            rows = rng.sample(rows, weight)

        print(f"[load_mixed] Loaded {len(rows)} samples from {name}")
        all_rows.extend(rows)

    print(f"[load_mixed] Total: {len(all_rows)} samples from {len(datasets)} datasets")

    # Shuffle if requested
    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(all_rows)

    return ray.data.from_items(all_rows)


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
    ):
        self.ds = ds
        self.tokenizer = tokenizer
        self.assistant_prefix = assistant_prefix
        self.system_prompt = system_prompt
        # Build index: prompt_id -> row data
        self._prompt_id_index: dict[str, dict] = {}
        self._build_index()
        self._epoch_started = False
        # Status tracking: prompt_id -> "pending" | "in_flight" | "done" | "failed"
        self._status: dict[str, str] = {}
        # Ordered list of prompt_ids for iteration
        self._prompt_ids: list[str] = list(self._prompt_id_index.keys())

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
        """
        # Per-row overrides take precedence over global config
        system_prompt = problem.get("system_prompt") or self.system_prompt
        assistant_prefix = problem.get("assistant_prefix") or self.assistant_prefix

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

    def new_epoch(self, seed: int | None = None) -> None:
        """Reset all non-failed statuses to pending for a new epoch and shuffle order."""
        import random

        self._epoch_started = True
        if seed is not None:
            rng = random.Random(seed)
            rng.shuffle(self._prompt_ids)
        for pid in self._prompt_ids:
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
        """Mark a prompt as pending (for retry)."""
        if self._status.get(prompt_id) == "failed":
            return
        self._status[prompt_id] = "pending"
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
        """
        self._require_epoch()
        for pid in self._prompt_ids:
            if self._status[pid] == "pending":
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
