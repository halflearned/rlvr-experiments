import logging
from dataclasses import dataclass
from typing import Generic, TypeVar

import torch
from ray.util.queue import Queue

from .tracer import traced

logger = logging.getLogger(__name__)

T = TypeVar("T")


# TODO: Neither of the following two functions belong here. 
# Let's move them to vllm_utils or a new utils file.
def pad_and_cat(tensors: list[torch.Tensor], pad_value: float = 0.0) -> torch.Tensor:
    """Pad tensors to max length in dim=1, then concatenate along dim=0."""
    max_len = max(t.shape[1] for t in tensors)
    padded = []
    for t in tensors:
        if t.shape[1] < max_len:
            pad_size = max_len - t.shape[1]
            padding = torch.full((t.shape[0], pad_size), pad_value, dtype=t.dtype, device=t.device)
            t = torch.cat([t, padding], dim=1)
        padded.append(t)
    return torch.cat(padded, dim=0)


def compute_group_advantages(rewards: torch.Tensor, group_sizes: list[int]) -> torch.Tensor:
    """
    Compute normalized advantages per group.

    Args:
        rewards: [total_completions] flat tensor of rewards
        group_sizes: list of sizes for each group (prompt)

    Returns:
        advantages: [total_completions] normalized within each group
    """
    advantages = torch.zeros_like(rewards)
    offset = 0
    for size in group_sizes:
        group_rewards = rewards[offset : offset + size]
        mean = group_rewards.mean()
        std = group_rewards.std(unbiased=False).clamp(min=1e-6)
        advantages[offset : offset + size] = (group_rewards - mean) / std
        offset += size
    return advantages

"""
TODO: (?)
One subtlety with the current implementation: if we have 2 consumers and max_reads=2, consumer A might read an item, 
re-queue it, and consumer A might read it again (rather than consumer B). 
The queue doesn't guarantee round-robin distribution. 
If we need strict "each consumer sees every item" semantics, that would require fan-out/broadcast.
"""

@dataclass
class RolloutEntry(Generic[T]):
    """Wrapper that adds version and read-count metadata to queue items."""
    item: T
    version: int
    reads_remaining: int


class RolloutBuffer(Generic[T]):
    """
    A rollout buffer built on ray.util.queue.Queue.

    Supports:
    - Blocking put/pop across processes
    - Version filtering (stale entries discarded on pop)
    - Read-count replay (items re-queued until reads exhausted)
    - Multiple producers and consumers
    """

    def __init__(self, maxsize: int = 0, max_reads: int = 1):
        """
        Args:
            maxsize: Max queue size. 0 = unbounded.
            max_reads: Number of times each item can be read before eviction.
        """
        self._queue: Queue = Queue(maxsize=maxsize)
        self._max_reads = max_reads
        self._stats = {"put": 0, "popped": 0, "skipped_zero_var": 0, "evicted_stale": 0, "evicted_exhausted": 0}

    @traced("buffer.put")
    async def put(self, item: T, version: int) -> None:
        """Add an item to the buffer. Blocks if queue is full."""
        logger.debug(f"buffer.put: version={version}, queue_size={self.size()}")
        entry = RolloutEntry(item=item, version=version, reads_remaining=self._max_reads)
        await self._queue.put_async(entry)
        self._stats["put"] += 1
        logger.debug(f"buffer.put: complete, queue_size={self.size()}")

    async def _pop_one(self, min_version: int | None = None) -> T:
        """
        Pop a single item, blocking until available.
        Discards items with version < min_version.
        Re-queues items that have remaining reads.
        """
        while True:
            logger.debug(f"_pop_one: waiting, min_version={min_version}, queue_size={self.size()}")
            entry: RolloutEntry[T] = await self._queue.get_async()
            logger.debug(f"_pop_one: got entry version={entry.version}, reads_remaining={entry.reads_remaining}")

            # Discard if stale
            if min_version is not None and entry.version < min_version:
                self._stats["evicted_stale"] += 1
                logger.debug(f"_pop_one: evicted stale entry (version {entry.version} < {min_version})")
                continue

            self._stats["popped"] += 1
            entry.reads_remaining -= 1

            # Re-queue if reads remaining, otherwise mark exhausted
            if entry.reads_remaining > 0:
                await self._queue.put_async(entry)
                logger.debug(f"_pop_one: re-queued entry, reads_remaining={entry.reads_remaining}")
            else:
                self._stats["evicted_exhausted"] += 1
                logger.debug("_pop_one: entry exhausted, evicted")

            return entry.item

    @traced("buffer.pop")
    async def pop(
        self,
        batch_size: int = 1,
        min_version: int | None = None,
    ) -> dict | None:
        """
        Pop up to batch_size items into a single batched dict.

        Each item is expected to have:
            - full_input_ids: [n, seq_len]
            - completion_ids: [n, T]
            - completion_mask: [n, T]
            - completion_logprobs: [n, T]
            - rewards: [n]

        Returns a dict with concatenated tensors and pre-computed advantages.
        Groups with zero reward variance are skipped (no gradient signal).
        Returns None if no valid items could be retrieved.
        """
        logger.debug(f"buffer.pop: batch_size={batch_size}, min_version={min_version}, queue_size={self.size()}")
        entries = []
        group_sizes = []
        max_attempts = batch_size * 10  # TODO: improve this?

        for attempt in range(max_attempts):
            if len(entries) >= batch_size:
                logger.debug(f"buffer.pop: got {len(entries)}/{batch_size} entries, done")
                break

            logger.debug(f"buffer.pop: attempt {attempt+1}/{max_attempts}, entries={len(entries)}, queue_size={self.size()}")
            item = await self._pop_one(min_version)
            rewards = item["rewards"]

            # Skip groups with no gradient signal (all rewards identical)
            reward_var = rewards.var().item() if len(rewards) > 1 else 0
            logger.debug(f"buffer.pop: rewards shape={rewards.shape}, var={reward_var:.4f}")
            if torch.allclose(rewards, rewards[0]):
                self._stats["skipped_zero_var"] += 1
                logger.debug(f"buffer.pop: skipped zero-variance entry (all rewards={rewards[0].item():.2f})")
                continue

            entries.append(item)
            group_sizes.append(rewards.shape[0])
            logger.debug(f"buffer.pop: accepted entry, now have {len(entries)}/{batch_size}")

        if not entries:
            logger.warning(f"buffer.pop: no valid batches after {max_attempts} attempts. Stats: {self.get_stats()}")
            return None

        if len(entries) < batch_size:
            logger.debug(f"buffer.pop: partial batch: got {len(entries)}/{batch_size} entries")

        # Pad to max length and concatenate (different prompts may have different lengths)
        batched = {
            "full_input_ids": pad_and_cat([e["full_input_ids"] for e in entries], pad_value=0),
            "completion_ids": pad_and_cat([e["completion_ids"] for e in entries], pad_value=0),
            "completion_mask": pad_and_cat([e["completion_mask"] for e in entries], pad_value=0),
            "completion_logprobs": pad_and_cat([e["completion_logprobs"] for e in entries], pad_value=0),
        }

        # Compute per-group normalized advantages
        all_rewards = torch.cat([e["rewards"] for e in entries], dim=0)
        batched["advantages"] = compute_group_advantages(all_rewards, group_sizes)

        return batched

    def size(self) -> int:
        return self._queue.qsize()

    def get_stats(self) -> dict:
        return {**self._stats, "current_size": self.size()}
