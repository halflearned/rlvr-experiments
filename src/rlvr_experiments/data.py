from __future__ import annotations

import asyncio
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
        question =  row["question"].strip()
        answer = row["answer"].split("####")[-1].strip()
        return {
            "prompt": question,
            "answer": answer, 
        }

    return ds.map(preprocess)


class DataIterator:
    """
    Iterator over a Ray Dataset with epoch and batch support.
    Applies chat template to prompts for direct use with vLLM.

    Usage:
        data_iter = DataIterator(ds, batch_size=16, tokenizer=tokenizer)

        for epoch in range(num_epochs):
            data_iter.new_epoch(seed=epoch)

            while True:
                batch = data_iter.next_batch()
                if batch is None:
                    break  # epoch exhausted
                templates = batch["templates"]  # ready for vLLM
                answers = batch["answers"]
    """

    def __init__(
        self,
        ds: ray.data.Dataset,
        batch_size: int,
        tokenizer,
        system_prompt: str = "",
    ):
        self.ds = ds
        self.batch_size = batch_size
        self.tokenizer = tokenizer
        self.system_prompt = system_prompt
        self._iter: Iterator | None = None

    def _apply_template(self, prompt: str) -> str:
        """Apply chat template to a single prompt."""
        return self.tokenizer.apply_chat_template(
            [{"role": "user", "content": self.system_prompt + prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )

    def new_epoch(self, seed: int | None = None) -> None:
        """Shuffle and reset iterator for a new epoch."""
        shuffled = self.ds.random_shuffle(seed=seed)
        self._iter = iter(shuffled.iter_batches(batch_size=self.batch_size))

    async def next_batch(self) -> dict | None:
        """
        Get next batch, or None if epoch exhausted.

        Returns dict with:
            - "templates": list of chat-formatted strings ready for vLLM
            - "answers": list of ground truth answers
        """
        if self._iter is None:
            raise RuntimeError("Call new_epoch() before next_batch()")

        def fetch():
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
        return {
            "templates": templates,
            "answers": list(batch["answer"]),
        }
