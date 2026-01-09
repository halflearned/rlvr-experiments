"""Unit tests for DataBuffer."""

import pytest
import asyncio

from rlvr_experiments.buffer import DataBuffer, Entry


class TestDataBufferBasic:
    def test_init(self, ray_context):
        buffer = DataBuffer(maxsize=10, max_reads=2)
        assert buffer._max_reads == 2
        assert buffer.size() == 0

    def test_entry_dataclass(self):
        entry = Entry(item="test", version=1, reads_remaining=2)
        assert entry.item == "test"
        assert entry.version == 1
        assert entry.reads_remaining == 2


class TestDataBufferPutPop:
    @pytest.mark.asyncio
    async def test_put_increments_size(self, ray_context):
        buffer = DataBuffer()

        await buffer.put("item1", version=0)
        assert buffer.size() == 1

        await buffer.put("item2", version=0)
        assert buffer.size() == 2

    @pytest.mark.asyncio
    async def test_pop_decrements_size(self, ray_context):
        buffer = DataBuffer()
        await buffer.put("item", version=0)

        await buffer.pop()
        assert buffer.size() == 0

    @pytest.mark.asyncio
    async def test_fifo_order(self, ray_context):
        buffer = DataBuffer()

        await buffer.put("first", version=0)
        await buffer.put("second", version=0)
        await buffer.put("third", version=0)

        assert await buffer.pop() == "first"
        assert await buffer.pop() == "second"
        assert await buffer.pop() == "third"

    @pytest.mark.asyncio
    async def test_stats_tracking(self, ray_context):
        buffer = DataBuffer()

        await buffer.put("item", version=0)
        await buffer.pop()

        stats = buffer.get_stats()
        assert stats["put"] == 1
        assert stats["popped"] == 1


class TestDataBufferVersioning:
    @pytest.mark.asyncio
    async def test_discards_stale_entries(self, ray_context):
        buffer = DataBuffer()

        await buffer.put("old", version=0)
        await buffer.put("new", version=1)

        # Pop with min_version=1 should skip the old entry
        result = await buffer.pop(min_version=1)
        assert result == "new"

        stats = buffer.get_stats()
        assert stats["evicted_stale"] == 1

    @pytest.mark.asyncio
    async def test_accepts_matching_version(self, ray_context):
        buffer = DataBuffer()

        await buffer.put("item", version=5)

        result = await buffer.pop(min_version=5)
        assert result == "item"

    @pytest.mark.asyncio
    async def test_accepts_higher_version(self, ray_context):
        buffer = DataBuffer()

        await buffer.put("item", version=10)

        result = await buffer.pop(min_version=5)
        assert result == "item"

    @pytest.mark.asyncio
    async def test_multiple_stale_entries(self, ray_context):
        buffer = DataBuffer()

        await buffer.put("old1", version=0)
        await buffer.put("old2", version=1)
        await buffer.put("old3", version=2)
        await buffer.put("new", version=5)

        result = await buffer.pop(min_version=5)
        assert result == "new"

        stats = buffer.get_stats()
        assert stats["evicted_stale"] == 3


class TestDataBufferMaxReads:
    @pytest.mark.asyncio
    async def test_single_read_removes_entry(self, ray_context):
        buffer = DataBuffer(max_reads=1)

        await buffer.put("item", version=0)
        await buffer.pop()

        assert buffer.size() == 0
        stats = buffer.get_stats()
        assert stats["evicted_exhausted"] == 1

    @pytest.mark.asyncio
    async def test_multi_read_keeps_entry(self, ray_context):
        buffer = DataBuffer(max_reads=3)

        await buffer.put("item", version=0)

        # First two reads should keep the entry
        result1 = await buffer.pop()
        assert result1 == "item"
        assert buffer.size() == 1

        result2 = await buffer.pop()
        assert result2 == "item"
        assert buffer.size() == 1

        # Third read should remove it
        result3 = await buffer.pop()
        assert result3 == "item"
        assert buffer.size() == 0

    @pytest.mark.asyncio
    async def test_multi_read_same_item_returned(self, ray_context):
        buffer = DataBuffer(max_reads=2)

        await buffer.put("item", version=0)

        result1 = await buffer.pop()
        result2 = await buffer.pop()

        assert result1 == result2 == "item"


class TestDataBufferPopBatch:
    @pytest.mark.asyncio
    async def test_pop_batch_full(self, ray_context):
        buffer = DataBuffer()

        for i in range(5):
            await buffer.put(f"item{i}", version=0)

        batch = await buffer.pop_batch(3)
        assert len(batch) == 3
        assert batch == ["item0", "item1", "item2"]

    @pytest.mark.asyncio
    async def test_pop_batch_partial_with_timeout(self, ray_context):
        buffer = DataBuffer()

        await buffer.put("item1", version=0)
        await buffer.put("item2", version=0)

        # Request 5 items but only 2 available
        batch = await buffer.pop_batch(5, timeout=0.1)
        assert len(batch) == 2
        assert batch == ["item1", "item2"]

    @pytest.mark.asyncio
    async def test_pop_batch_with_version(self, ray_context):
        buffer = DataBuffer()

        await buffer.put("old", version=0)
        await buffer.put("new1", version=1)
        await buffer.put("new2", version=1)

        batch = await buffer.pop_batch(3, min_version=1, timeout=0.1)
        assert batch == ["new1", "new2"]


class TestDataBufferConcurrency:
    @pytest.mark.asyncio
    async def test_concurrent_puts(self, ray_context):
        buffer = DataBuffer()

        async def put_items(start, count):
            for i in range(count):
                await buffer.put(f"item{start + i}", version=0)

        await asyncio.gather(
            put_items(0, 10),
            put_items(10, 10),
            put_items(20, 10),
        )

        assert buffer.size() == 30

    @pytest.mark.asyncio
    async def test_concurrent_put_pop(self, ray_context):
        buffer = DataBuffer()
        results = []

        async def producer():
            for i in range(10):
                await buffer.put(i, version=0)
                await asyncio.sleep(0.001)

        async def consumer():
            for _ in range(10):
                item = await buffer.pop()
                results.append(item)

        await asyncio.gather(producer(), consumer())

        assert len(results) == 10
        assert set(results) == set(range(10))


class TestDataBufferEpochHandling:
    @pytest.mark.asyncio
    async def test_mark_done_returns_none(self, ray_context):
        buffer = DataBuffer()

        await buffer.put("item", version=0)
        await buffer.mark_done()

        assert await buffer.pop() == "item"
        assert await buffer.pop() is None  # Done signal

    @pytest.mark.asyncio
    async def test_reset_clears_buffer(self, ray_context):
        buffer = DataBuffer()

        await buffer.put("item1", version=0)
        await buffer.put("item2", version=0)
        await buffer.mark_done()

        buffer.reset()
        assert buffer.size() == 0

    @pytest.mark.asyncio
    async def test_reset_allows_new_epoch(self, ray_context):
        buffer = DataBuffer()

        # First epoch
        await buffer.put("epoch1_item", version=0)
        await buffer.mark_done()
        assert await buffer.pop() == "epoch1_item"
        assert await buffer.pop() is None  # Done

        # Reset for next epoch
        buffer.reset()

        # Second epoch - should work normally
        await buffer.put("epoch2_item", version=0)
        await buffer.mark_done()
        assert await buffer.pop() == "epoch2_item"
        assert await buffer.pop() is None  # Done
