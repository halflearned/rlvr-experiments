from dataclasses import dataclass
from typing import Generic, TypeVar

from ray.util.queue import Queue

from .tracer import traced

T = TypeVar("T")

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
        self._stats = {"put": 0, "popped": 0, "evicted_stale": 0, "evicted_exhausted": 0}

    @traced("buffer.put")
    async def put(self, item: T, version: int) -> None:
        """Add an item to the buffer. Blocks if queue is full."""
        entry = RolloutEntry(item=item, version=version, reads_remaining=self._max_reads)
        await self._queue.put_async(entry)
        self._stats["put"] += 1

    @traced("buffer.pop")
    async def pop(self, min_version: int | None = None) -> T:
        """
        Pop an item, blocking until available.
        Discards items with version < min_version.
        Re-queues items that have remaining reads.
        """
        while True:
            entry: RolloutEntry[T] = await self._queue.get_async()

            # Discard if stale
            if min_version is not None and entry.version < min_version:
                self._stats["evicted_stale"] += 1
                continue

            self._stats["popped"] += 1
            entry.reads_remaining -= 1

            # Re-queue if reads remaining, otherwise mark exhausted
            if entry.reads_remaining > 0:
                await self._queue.put_async(entry)
            else:
                self._stats["evicted_exhausted"] += 1

            return entry.item

    def size(self) -> int:
        return self._queue.qsize()

    def get_stats(self) -> dict:
        return {**self._stats, "current_size": self.size()}
