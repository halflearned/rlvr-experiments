"""Versioned buffer for producer/consumer coordination."""

import asyncio
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Generic, TypeVar

from ray.util.queue import Queue

T = TypeVar("T")


@dataclass
class BufferStats:
    """Lightweight stats tracking for buffer visualization."""

    put: int = 0
    popped: int = 0
    evicted_stale: int = 0
    evicted_exhausted: int = 0
    by_version: dict[int, int] = field(default_factory=lambda: defaultdict(int))
    evicted_by_version: dict[int, int] = field(default_factory=lambda: defaultdict(int))

    def record_put(self, version: int) -> None:
        self.put += 1
        if version >= 0:
            self.by_version[version] += 1

    def record_pop(self, version: int, exhausted: bool) -> None:
        self.popped += 1
        if exhausted:
            self.evicted_exhausted += 1
            if version >= 0:
                self.by_version[version] = max(0, self.by_version[version] - 1)

    def record_evict_stale(self, version: int) -> None:
        self.evicted_stale += 1
        if version >= 0:
            self.evicted_by_version[version] += 1
            self.by_version[version] = max(0, self.by_version[version] - 1)

    def get_by_version(self) -> dict[int, int]:
        """Return per-version counts (excludes zeros)."""
        return {k: v for k, v in self.by_version.items() if v > 0}

    def get_evicted_by_version(self) -> dict[int, int]:
        """Return cumulative evicted counts per version."""
        return dict(self.evicted_by_version)

    def to_dict(self, current_size: int) -> dict:
        return {
            "put": self.put,
            "popped": self.popped,
            "evicted_stale": self.evicted_stale,
            "evicted_exhausted": self.evicted_exhausted,
            "current_size": current_size,
        }


@dataclass
class Entry(Generic[T]):
    item: T
    version: int
    reads_remaining: int


class DataBuffer(Generic[T]):
    """Generic versioned queue with replay support and per-version tracking."""

    def __init__(self, maxsize: int = 0, max_reads: int = 1):
        self._queue: Queue = Queue(maxsize=maxsize)
        self._max_reads = max_reads
        self.stats = BufferStats()

    async def put(self, item: T, version: int) -> None:
        entry = Entry(item=item, version=version, reads_remaining=self._max_reads)
        await self._queue.put_async(entry)
        self.stats.record_put(version)

    async def pop(self, min_version: int | None = None) -> T:
        """Pop one item. Discards stale entries, re-queues if reads remain.

        Args:
            min_version: Discard entries with version < min_version.
                         Entries with version < 0 are never evicted (control signals).
        """
        evicted_this_call = 0
        while True:
            entry: Entry[T] = await self._queue.get_async()

            # Negative version = never evict (control signals)
            if entry.version >= 0 and min_version is not None and entry.version < min_version:
                self.stats.record_evict_stale(entry.version)
                evicted_this_call += 1
                if evicted_this_call % 10 == 0:
                    print(f"[buffer] evicted {evicted_this_call} stale samples (version {entry.version} < {min_version})")
                continue

            entry.reads_remaining -= 1
            exhausted = entry.reads_remaining == 0

            if not exhausted:
                await self._queue.put_async(entry)

            self.stats.record_pop(entry.version, exhausted)
            return entry.item

    async def pop_batch(self, batch_size: int, min_version: int | None = None, timeout: float | None = None) -> list[T]:
        """Pop up to batch_size items. If timeout specified, returns partial batch on timeout."""
        items = []
        for _ in range(batch_size):
            try:
                if timeout is not None:
                    item = await asyncio.wait_for(self.pop(min_version), timeout=timeout)
                else:
                    item = await self.pop(min_version)
                items.append(item)
            except asyncio.TimeoutError:
                break
        return items

    def size(self) -> int:
        return self._queue.qsize()

    def get_stats(self) -> dict:
        return self.stats.to_dict(self.size())

    def get_by_version(self) -> dict[int, int]:
        return self.stats.get_by_version()

    def get_evicted_by_version(self) -> dict[int, int]:
        return self.stats.get_evicted_by_version()
