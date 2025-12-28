import logging
from dataclasses import dataclass
from typing import Generic, TypeVar

from ray.util.queue import Queue

from .tracer import traced

logger = logging.getLogger(__name__)

T = TypeVar("T")


@dataclass
class Entry(Generic[T]):
    item: T
    version: int
    reads_remaining: int


class RolloutBuffer(Generic[T]):
    """Generic versioned queue with replay support."""

    def __init__(self, maxsize: int = 0, max_reads: int = 1):
        self._queue: Queue = Queue(maxsize=maxsize)
        self._max_reads = max_reads
        self._stats = {"put": 0, "popped": 0, "evicted_stale": 0, "evicted_exhausted": 0}

    @traced("buffer.put")
    async def put(self, item: T, version: int) -> None:
        entry = Entry(item=item, version=version, reads_remaining=self._max_reads)
        await self._queue.put_async(entry)
        self._stats["put"] += 1

    async def pop(self, min_version: int | None = None) -> T:
        """Pop one item. Discards stale entries, re-queues if reads remain."""
        while True:
            entry: Entry[T] = await self._queue.get_async()

            if min_version is not None and entry.version < min_version:
                self._stats["evicted_stale"] += 1
                continue

            self._stats["popped"] += 1
            entry.reads_remaining -= 1

            if entry.reads_remaining > 0:
                await self._queue.put_async(entry)
            else:
                self._stats["evicted_exhausted"] += 1

            return entry.item

    async def pop_batch(self, batch_size: int, min_version: int | None = None) -> list[T]:
        """Pop up to batch_size items."""
        return [await self.pop(min_version) for _ in range(batch_size)]

    def size(self) -> int:
        return self._queue.qsize()

    def get_stats(self) -> dict:
        return {**self._stats, "current_size": self.size()}
