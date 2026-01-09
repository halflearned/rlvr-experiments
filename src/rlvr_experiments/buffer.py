"""Versioned buffer for producer/consumer coordination."""

import asyncio
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Generic, TypeVar

from ray.util.queue import Queue

from .tracer import get_tracer

T = TypeVar("T")


class BufferStats:
    """Lightweight stats tracking for buffer visualization.

    Tracks three item fates:
    - used: Items fully consumed (all reads exhausted) - good
    - wasted: Items evicted with all reads remaining (never used) - bad
    - partial: Items evicted with some reads used - in between

    Emits buffer events to tracer on every state change.
    """

    def __init__(self, get_size: callable):
        self._get_size = get_size  # Callback to get current queue size
        self.put = 0
        self.popped = 0
        self.by_version: dict[int, int] = defaultdict(int)
        # Per-version fate tracking (cumulative)
        self.used_by_version: dict[int, int] = defaultdict(int)
        self.wasted_by_version: dict[int, int] = defaultdict(int)
        self.partial_by_version: dict[int, int] = defaultdict(int)
        self.filtered_by_version: dict[int, int] = defaultdict(int)

    def _emit(self) -> None:
        """Emit current state to tracer."""
        tracer = get_tracer()
        if tracer is not None:
            tracer.buffer(
                size=self._get_size(),
                by_version=self.get_by_version(),
                fates=self.get_fates_by_version(),
            )

    def record_put(self, version: int) -> None:
        self.put += 1
        if version >= 0:
            self.by_version[version] += 1
        self._emit()

    def record_pop(self, version: int, exhausted: bool) -> None:
        self.popped += 1
        if exhausted:
            # Item fully consumed - good!
            if version >= 0:
                self.used_by_version[version] += 1
                self.by_version[version] = max(0, self.by_version[version] - 1)
            self._emit()

    def record_evict_stale(self, version: int, reads_remaining: int, max_reads: int) -> None:
        """Record a stale eviction with fate categorization."""
        if version >= 0:
            if reads_remaining == max_reads:
                # Never used at all - wasted
                self.wasted_by_version[version] += 1
            else:
                # Partially used
                self.partial_by_version[version] += 1
            self.by_version[version] = max(0, self.by_version[version] - 1)
        self._emit()

    def record_filtered(self, version: int) -> None:
        """Record a sample filtered due to zero-variance rewards."""
        if version >= 0:
            self.filtered_by_version[version] += 1
        self._emit()

    def get_by_version(self) -> dict[int, int]:
        """Return per-version counts (excludes zeros)."""
        return {k: v for k, v in self.by_version.items() if v > 0}

    def get_fates_by_version(self) -> dict[str, dict[int, int]]:
        """Return cumulative fate counts per version."""
        return {
            "used": {k: v for k, v in self.used_by_version.items() if v > 0},
            "wasted": {k: v for k, v in self.wasted_by_version.items() if v > 0},
            "partial": {k: v for k, v in self.partial_by_version.items() if v > 0},
            "filtered": {k: v for k, v in self.filtered_by_version.items() if v > 0},
        }

    def to_dict(self, current_size: int) -> dict:
        fates = self.get_fates_by_version()
        used = sum(fates["used"].values())
        wasted = sum(fates["wasted"].values())
        partial = sum(fates["partial"].values())
        filtered = sum(fates["filtered"].values())
        return {
            "put": self.put,
            "popped": self.popped,
            "used": used,
            "wasted": wasted,
            "partial": partial,
            "filtered": filtered,
            "current_size": current_size,
            # Legacy keys for backwards compatibility with tests
            "evicted_exhausted": used,
            "evicted_stale": wasted + partial,
        }


@dataclass
class Entry(Generic[T]):
    item: T
    version: int
    reads_remaining: int


class _Done:
    """Sentinel to signal producer completion."""
    pass


class DataBuffer(Generic[T]):
    """Generic versioned queue with replay support and per-version tracking."""

    def __init__(self, maxsize: int = 0, max_reads: int = 1):
        self._queue: Queue = Queue(maxsize=maxsize)
        self._max_reads = max_reads
        self.stats = BufferStats(get_size=self.size)

    async def put(self, item: T, version: int) -> None:
        entry = Entry(item=item, version=version, reads_remaining=self._max_reads)
        await self._queue.put_async(entry)
        self.stats.record_put(version)

    async def mark_done(self) -> None:
        """Signal that no more items will be added."""
        entry = Entry(item=_Done(), version=-1, reads_remaining=1)
        await self._queue.put_async(entry)

    async def pop(self, min_version: int | None = None) -> T | None:
        """Pop one item. Returns None when done signal received.

        Args:
            min_version: Discard entries with version < min_version.
        """
        while True:
            entry: Entry[T] = await self._queue.get_async()

            # Check for done signal
            if isinstance(entry.item, _Done):
                return None

            # Evict stale entries (version >= 0 only)
            if entry.version >= 0 and min_version is not None and entry.version < min_version:
                self.stats.record_evict_stale(entry.version, entry.reads_remaining, self._max_reads)
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

    def reset(self) -> None:
        """Reset buffer for a new epoch. Clears any remaining items."""
        # Drain any remaining items (shouldn't be many if epoch ended cleanly)
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except Exception:
                break
        # Stats carry over between epochs (cumulative)

    def get_stats(self) -> dict:
        return self.stats.to_dict(self.size())
