"""Rollout types and epoch runner."""

import asyncio
from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn.functional as F

from .tracer import get_tracer, trace_span


# Sentinel to signal producer completion
class _Done:
    pass

DONE = _Done()


@dataclass
class Batch:
    """A training batch - generic container for rollout data."""
    input_ids: torch.Tensor       # [B, seq_len]
    completion_ids: torch.Tensor  # [B, completion_len]
    logprobs: torch.Tensor        # [B, completion_len]
    rewards: torch.Tensor         # [B]
    mask: torch.Tensor            # [B, completion_len]
    prompt_lens: torch.Tensor     # [B] - length of prompt for each sample


@dataclass
class RolloutSample:
    """One prompt with N completions."""
    input_ids: torch.Tensor
    completion_ids: torch.Tensor
    logprobs: torch.Tensor
    rewards: list[float]
    prompt_len: int  # Length of the prompt (same for all completions in this sample)

    @classmethod
    def from_vllm(cls, response, pad_token_id: int, rewards: list[float]):
        prompt = response.prompt_token_ids
        outputs = response.outputs
        n = len(outputs)
        prompt_len = len(prompt)

        seqs = [prompt + list(o.token_ids) for o in outputs]
        max_seq_len = max(len(s) for s in seqs)
        input_ids = torch.full((n, max_seq_len), pad_token_id, dtype=torch.long)
        for i, seq in enumerate(seqs):
            input_ids[i, :len(seq)] = torch.tensor(seq)

        max_completion_len = max(len(o.token_ids) for o in outputs)
        completion_ids = torch.full((n, max_completion_len), pad_token_id, dtype=torch.long)
        logprobs = torch.zeros((n, max_completion_len), dtype=torch.float32)

        for i, o in enumerate(outputs):
            L = len(o.token_ids)
            # Left-align to match input_ids (prompt + completion, then padding)
            completion_ids[i, :L] = torch.tensor(o.token_ids)
            logprobs[i, :L] = torch.tensor([o.logprobs[j][o.token_ids[j]].logprob for j in range(L)])

        return cls(input_ids, completion_ids, logprobs, rewards, prompt_len)


def make_batch(
    samples: list[RolloutSample],
    pad_token_id: int,
    max_seq_len: int | None = None,
    max_completion_len: int | None = None,
) -> Batch | None:
    """Combine samples into a training batch. Returns None if all zero-variance.

    Args:
        samples: List of RolloutSample objects
        pad_token_id: Token ID to use for padding
        max_seq_len: If provided, pad input_ids to this fixed length (avoids recompilation)
        max_completion_len: If provided, pad completion_ids to this fixed length
    """
    valid = [s for s in samples if torch.tensor(s.rewards, dtype=torch.float32).std() > 1e-6]
    if not valid:
        return None

    def pad_cat(tensors, pad_value=0, fixed_len=None):
        max_len = fixed_len if fixed_len is not None else max(t.shape[1] for t in tensors)
        return torch.cat([F.pad(t, (0, max_len - t.shape[1]), value=pad_value) for t in tensors])

    # Build prompt_lens: each sample has N completions with the same prompt_len
    prompt_lens_list = []
    for s in valid:
        n_completions = s.input_ids.size(0)
        prompt_lens_list.extend([s.prompt_len] * n_completions)

    return Batch(
        input_ids=pad_cat([s.input_ids for s in valid], pad_value=pad_token_id, fixed_len=max_seq_len),
        completion_ids=pad_cat([s.completion_ids for s in valid], pad_value=pad_token_id, fixed_len=max_completion_len),
        logprobs=pad_cat([s.logprobs for s in valid], fixed_len=max_completion_len),
        rewards=torch.cat([torch.tensor(s.rewards, dtype=torch.float32) for s in valid]),
        mask=pad_cat([(s.completion_ids != pad_token_id).float() for s in valid], fixed_len=max_completion_len),
        prompt_lens=torch.tensor(prompt_lens_list, dtype=torch.long),
    )


async def run_epoch(
    rollout,
    data_iter,
    buffer,
    *,
    reward: Callable,
    pad_token_id: int,
    batch_size: int,
    sampling_params: dict | None = None,
    epoch: int = 0,
    max_seq_len: int | None = None,
    max_completion_len: int | None = None,
):
    """Run generation/verification and yield training batches.

    Args:
        rollout: VLLMHandle for generation
        data_iter: DataIterator yielding {"templates": [...], "problems": [...]}
        buffer: DataBuffer for producer/consumer coordination
        reward: async (problem, completions) -> list[float]
        pad_token_id: For padding tensors
        batch_size: Samples per training batch
        sampling_params: vLLM sampling parameters
        epoch: Current epoch (for buffer versioning)
        max_seq_len: If provided, pad input_ids to fixed length (avoids recompilation)
        max_completion_len: If provided, pad completions to fixed length

    Yields:
        (step, batch) tuples. Zero-variance batches are filtered.

    Mid-epoch sync: sync_titan_to_vllm can be called from another task at any time.
    It will pause generation, wait for in-flight requests, sync, then resume.
    """
    tracer = get_tracer()
    sp = {**(sampling_params or {}), "logprobs": 0}
    num_producers = rollout.num_replicas
    done_count = 0

    async def produce():
        """Generate completions, compute rewards, push to buffer."""
        async def process_one(response, problem):
            completions = [out.text for out in response.outputs]
            rewards = await reward(problem, completions)
            sample = RolloutSample.from_vllm(response, pad_token_id, rewards)
            await buffer.put(sample, epoch)

        async with asyncio.TaskGroup() as tg:
            while True:
                data_batch = await data_iter.next_batch()
                if data_batch is None:
                    break
                responses = await rollout.generate(data_batch["templates"], **sp)
                for response, problem in zip(responses, data_batch["problems"]):
                    tg.create_task(process_one(response, problem))

        # Signal completion
        await buffer.put(DONE, epoch)

    # Start one producer per vLLM replica
    producers = [asyncio.create_task(produce()) for _ in range(num_producers)]

    step = 0
    samples = []
    try:
        while done_count < num_producers:
            item = await buffer.pop(epoch)

            if isinstance(item, _Done):
                done_count += 1
                continue

            samples.append(item)

            if len(samples) >= batch_size:
                with trace_span("make_batch"):
                    all_rewards = [r for s in samples for r in s.rewards]
                    tracer.counter("rewards", {
                        "mean": sum(all_rewards) / len(all_rewards) if all_rewards else 0,
                        "max": max(all_rewards) if all_rewards else 0,
                        "num_positive": sum(1 for r in all_rewards if r > 0),
                    })

                    batch = make_batch(samples, pad_token_id, max_seq_len, max_completion_len)
                    samples = []

                    if batch is None:
                        tracer.counter("skipped", {"zero_variance_batches": 1})
                        continue

                    step += 1
                    yield step, batch

        # Yield any remaining samples as final batch
        if samples:
            batch = make_batch(samples, pad_token_id, max_seq_len, max_completion_len)
            if batch is not None:
                step += 1
                yield step, batch

    finally:
        for t in producers:
            t.cancel()
        await asyncio.gather(*producers, return_exceptions=True)
