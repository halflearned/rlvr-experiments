"""Versioned buffer for producer/consumer coordination."""

from collections import defaultdict
from dataclasses import dataclass
from typing import Generic, TypeVar

from ray.util.queue import Queue

from .tracer import get_tracer

T = TypeVar("T")


class BufferStats:
    """Lightweight stats tracking for buffer.

    Buffer is now a simple FIFO - consumer handles staleness.
    We just track put/pop counts and per-version distribution.
    """

    def __init__(self, get_size: callable):
        self._get_size = get_size  # Callback to get current queue size
        self.put = 0
        self.popped = 0
        self.by_version: dict[int, int] = defaultdict(int)

    def _emit(self) -> None:
        """Emit current state to tracer."""
        tracer = get_tracer()
        if tracer is not None:
            tracer.buffer(
                size=self._get_size(),
                by_version=self.get_by_version(),
            )

    def record_put(self, version: int) -> None:
        self.put += 1
        if version >= 0:
            self.by_version[version] += 1
        self._emit()

    def record_pop(self, version: int, exhausted: bool) -> None:
        self.popped += 1
        if exhausted and version >= 0:
            self.by_version[version] = max(0, self.by_version[version] - 1)
        self._emit()

    def get_by_version(self) -> dict[int, int]:
        """Return per-version counts (excludes zeros)."""
        return {k: v for k, v in self.by_version.items() if v > 0}

    def to_dict(self, current_size: int) -> dict:
        return {
            "put": self.put,
            "popped": self.popped,
            "current_size": current_size,
        }


@dataclass
class Entry(Generic[T]):
    item: T
    version: int
    reads_remaining: int
    item_id: str  # For tracking wasted items


class _Done:
    """Sentinel to signal producer completion."""
    pass


class DataBuffer(Generic[T]):
    """Generic versioned queue with replay support and per-version tracking."""

    def __init__(self, maxsize: int = 0, max_reads: int = 1):
        self._queue: Queue = Queue(maxsize=maxsize)
        self._max_reads = max_reads
        self.stats = BufferStats(get_size=self.size)

    async def put(self, item: T, version: int, item_id: str) -> None:
        entry = Entry(item=item, version=version, reads_remaining=self._max_reads, item_id=item_id)
        await self._queue.put_async(entry)
        self.stats.record_put(version)

    async def mark_done(self) -> None:
        """Signal that no more items will be added."""
        entry = Entry(item=_Done(), version=-1, reads_remaining=1, item_id="__done__")
        await self._queue.put_async(entry)

    async def pop(self) -> tuple[T, str, int] | None:
        """Pop one item. Returns (item, item_id, version) tuple, or None when done.

        Consumer is responsible for staleness checks.

        Returns:
            Tuple of (item, item_id, version) or None if done signal received.
        """
        entry: Entry[T] = await self._queue.get_async()

        # Check for done signal
        if isinstance(entry.item, _Done):
            return None

        entry.reads_remaining -= 1
        exhausted = entry.reads_remaining == 0

        if not exhausted:
            await self._queue.put_async(entry)

        self.stats.record_pop(entry.version, exhausted)
        return (entry.item, entry.item_id, entry.version)

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
