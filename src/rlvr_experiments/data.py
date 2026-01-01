from __future__ import annotations

import asyncio
import threading
from typing import Iterator
import ray.data
from datasets import load_dataset



def load_gsm8k(split: str = "train") -> ray.data.Dataset:
    """
    Load GSM8k dataset as a Ray Dataset.

    Returns dataset with columns: "prompt", "answer"
    """
    hf_dataset = load_dataset("openai/gsm8k", "main", split=split)
    ds = ray.data.from_huggingface(hf_dataset)

    def preprocess(row):
        question = f"\n\nProblem:{row['question'].strip()}"
        answer = row["answer"].split("####")[-1].strip()
        return {
            "prompt": question,
            "answer": answer,
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
    where "problem" is a dict containing test, entry_point, task_id
    for use with HumanEvalVerifier.
    """
    # HumanEval only has "test" split - ignore any split param from config
    hf_dataset = load_dataset("openai/openai_humaneval", split="test")
    ds = ray.data.from_huggingface(hf_dataset)

    def preprocess(row):
        # The prompt is the function signature + docstring
        # Model needs to generate the function body
        return {
            "prompt": row["prompt"],
            "problem": {
                "prompt": row["prompt"],
                "test": row["test"],
                "entry_point": row["entry_point"],
                "task_id": row["task_id"],
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
        return {
            "prompt": row["text"],
            "problem": {
                "text": row["text"],
                "test_list": row["test_list"],
                "task_id": row["task_id"],
            },
        }

    return ds.map(preprocess)


def load_dummy(split: str = "train") -> ray.data.Dataset:
    """
    Load a dummy dataset with a single question repeated 64 times.

    Useful for testing that rewards are increasing on a single problem.
    Returns dataset with columns: "prompt", "answer"
    """
    rows = [
        {
            "prompt": "\n\nProblem:What is ((7/12) + (5/18)) / (31/36)?",
            "answer": "1",
        }
    ] * 64
    return ray.data.from_items(rows)


class DataIterator:
    """
    Iterator over a Ray Dataset with epoch and batch support.
    Applies chat template to prompts for direct use with vLLM.

    Supports two dataset formats:
    - Math datasets: columns "prompt", "answer" (e.g., GSM8K)
    - Code datasets: columns "prompt", "problem" (e.g., HumanEval)

    Usage:
        data_iter = DataIterator(ds, batch_size=16, tokenizer=tokenizer)

        for epoch in range(num_epochs):
            data_iter.new_epoch(seed=epoch)

            while True:
                batch = await data_iter.next_batch()
                if batch is None:
                    break  # epoch exhausted
                templates = batch["templates"]  # ready for vLLM

                # For math datasets:
                answers = batch["answers"]  # list of strings

                # For code datasets:
                problems = batch["problems"]  # list of dicts
    """

    def __init__(
        self,
        ds: ray.data.Dataset,
        batch_size: int,
        tokenizer,
        system_prompt: str = "",
        assistant_prefix: str = "",
    ):
        self.ds = ds
        self.batch_size = batch_size
        self.tokenizer = tokenizer
        self.assistant_prefix = assistant_prefix
        self.system_prompt = system_prompt
        self._iter: Iterator | None = None
        self._lock = threading.Lock()  # Protects _iter from concurrent access

    def _apply_template(self, prompt: str) -> str:
        """Apply chat template to a single prompt."""
        content = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,  # TODO: make configurable
        ) + self.assistant_prefix
        return content

    def new_epoch(self, seed: int | None = None) -> None:
        """Shuffle and reset iterator for a new epoch."""
        shuffled = self.ds.random_shuffle(seed=seed)
        self._iter = iter(shuffled.iter_batches(batch_size=self.batch_size))

    async def next_batch(self) -> dict | None:
        """
        Get next batch, or None if epoch exhausted.

        Returns dict with:
            - "templates": list of chat-formatted strings ready for vLLM
            - "answers": list of ground truth answers (if dataset has "answer" column)
            - "problems": list of problem dicts (if dataset has "problem" column)
        """
        if self._iter is None:
            raise RuntimeError("Call new_epoch() before next_batch()")

        def fetch():
            with self._lock:
                try:
                    return next(self._iter)  # type: ignore
                except StopIteration:
                    return None

        # Run blocking iterator in thread to avoid blocking the event loop
        loop = asyncio.get_event_loop()
        batch = await loop.run_in_executor(None, fetch)

        if batch is None:
            return None

        templates = [self._apply_template(p) for p in batch["prompt"]]
        result = {"templates": templates}

        # Support both math (answer) and code (problem) datasets
        if "answer" in batch:
            result["answers"] = list(batch["answer"])
        if "problem" in batch:
            result["problems"] = list(batch["problem"])

        return result
