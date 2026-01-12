from __future__ import annotations

import hashlib
from typing import Iterator
import ray.data
from datasets import load_dataset

from .sample_logger import log_sample


def _hash_prompt(prompt: str) -> str:
    """Generate a short hash for prompts without explicit IDs."""
    return hashlib.sha256(prompt.encode()).hexdigest()[:12]



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
            "problem": {"answer": answer, "prompt_id": prompt_id},
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
    hf_dataset = load_dataset("google-research-datasets/mbpp", "full", split=split)
    ds = ray.data.from_huggingface(hf_dataset)

    def preprocess(row):
        # The prompt is the problem description
        # Model needs to generate a complete function
        # Use task_id as prompt_id (e.g., "mbpp_123")
        prompt_id = f"mbpp_{row['task_id']}"
        return {
            "prompt": row["text"],
            "problem": {
                "text": row["text"],
                "test_list": row["test_list"],
                "task_id": row["task_id"],
                "prompt_id": prompt_id,
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
            "problem": {"answer": row["solution"], "prompt_id": prompt_id},
        }

    return ds.map(preprocess)


def load_dummy(split: str = "train", num_samples = 64) -> ray.data.Dataset:
    """
    Load a dummy dataset with a single question repeated 64 times.

    Useful for testing that rewards are increasing on a single problem.
    Returns dataset with columns: "prompt", "problem"
    where "problem" is a dict with "answer" and "prompt_id" for use with MathVerifier.
    """
    prompt = "\n\nProblem:What is ((7/12) + (5/18)) / (31/36)?"
    rows = [
        {
            "prompt": prompt,
            "problem": {"answer": "1", "prompt_id": "dummy_0"},
        }
    ] * num_samples
    return ray.data.from_items(rows)


class DataIterator:
    """
    Iterator over a Ray Dataset with status tracking for each prompt.
    Applies chat template to prompts for direct use with vLLM.

    All datasets should have columns "prompt" and "problem", where
    "problem" is a dict containing whatever the verifier needs.

    Tracks status per prompt_id: "pending" | "in_flight" | "consumed"
    - pending: not yet processed or needs retry
    - in_flight: currently being processed
    - consumed: successfully trained on

    Usage:
        data_iter = DataIterator(ds, tokenizer=tokenizer)

        for epoch in range(num_epochs):
            data_iter.new_epoch(seed=epoch)

            while not data_iter.all_consumed():
                item = data_iter.next_pending()
                if item is None:
                    # No pending items, wait for in-flight to complete
                    continue
                # process item...
                # on success: data_iter.mark_consumed(prompt_id)
                # on waste: data_iter.mark_pending(prompt_id)
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
        # Status tracking: prompt_id -> "pending" | "in_flight" | "consumed"
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

    def _apply_template(self, prompt: str) -> str:
        """Apply chat template to a single prompt."""
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": prompt})
        content = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,  # TODO: make configurable
        ) + self.assistant_prefix
        return content

    def _get_item(self, prompt_id: str) -> dict:
        """Get formatted item for a prompt_id."""
        row = self._prompt_id_index[prompt_id]
        return {
            "template": self._apply_template(row["prompt"]),
            "prompt": row["prompt"],
            "problem": row["problem"],
        }

    # --- Status management ---

    def new_epoch(self, seed: int | None = None) -> None:
        """Reset all statuses to pending for a new epoch and shuffle order."""
        import random

        if seed is not None:
            rng = random.Random(seed)
            rng.shuffle(self._prompt_ids)
        for pid in self._prompt_ids:
            self._status[pid] = "pending"

    def mark_in_flight(self, prompt_id: str) -> None:
        """Mark a prompt as currently being processed."""
        self._status[prompt_id] = "in_flight"

    def mark_consumed(self, prompt_id: str) -> None:
        """Mark a prompt as successfully consumed (trained on)."""
        self._status[prompt_id] = "consumed"

    def mark_pending(self, prompt_id: str) -> None:
        """Mark a prompt as pending (for retry)."""
        self._status[prompt_id] = "pending"
        log_sample("pending", prompt_id=prompt_id)

    def all_consumed(self) -> bool:
        """Check if all prompts have been consumed."""
        return all(s == "consumed" for s in self._status.values())

    def pending_count(self) -> int:
        """Count of pending prompts."""
        return sum(1 for s in self._status.values() if s == "pending")

    def in_flight_count(self) -> int:
        """Count of in-flight prompts."""
        return sum(1 for s in self._status.values() if s == "in_flight")

    def consumed_count(self) -> int:
        """Count of consumed prompts."""
        return sum(1 for s in self._status.values() if s == "consumed")

    def get_next(self) -> dict | None:
        """Get next pending item and mark it in_flight.

        Returns the next available item (either a retry or new item), or None
        if all items are either in_flight or consumed. This is the primary
        interface for getting work items.
        """
        for pid in self._prompt_ids:
            if self._status.get(pid) == "pending":
                self._status[pid] = "in_flight"
                log_sample("in_flight", prompt_id=pid)
                return self._get_item(pid)
        return None

    # Legacy alias
    next_pending = get_next

    # --- Legacy iteration interface (for compatibility) ---

    def __iter__(self):
        return self

    def __next__(self) -> dict:
        """Legacy interface: get next pending item or raise StopIteration."""
        item = self.get_next()
        if item is None:
            raise StopIteration
        return item
