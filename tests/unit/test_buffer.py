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
        entry = Entry(item="test", version=1, reads_remaining=2, item_id="test_id")
        assert entry.item == "test"
        assert entry.version == 1
        assert entry.reads_remaining == 2
        assert entry.item_id == "test_id"


class TestDataBufferPutPop:
    @pytest.mark.asyncio
    async def test_put_increments_size(self, ray_context):
        buffer = DataBuffer()

        await buffer.put("item1", version=0, item_id="id1")
        assert buffer.size() == 1

        await buffer.put("item2", version=0, item_id="id2")
        assert buffer.size() == 2

    @pytest.mark.asyncio
    async def test_pop_decrements_size(self, ray_context):
        buffer = DataBuffer()
        await buffer.put("item", version=0, item_id="id1")

        await buffer.pop()
        assert buffer.size() == 0

    @pytest.mark.asyncio
    async def test_fifo_order(self, ray_context):
        buffer = DataBuffer()

        await buffer.put("first", version=0, item_id="id1")
        await buffer.put("second", version=0, item_id="id2")
        await buffer.put("third", version=0, item_id="id3")

        entry1 = await buffer.pop()
        assert entry1.item == "first"
        assert entry1.item_id == "id1"
        assert entry1.version == 0

        entry2 = await buffer.pop()
        assert entry2.item == "second"

        entry3 = await buffer.pop()
        assert entry3.item == "third"

    @pytest.mark.asyncio
    async def test_pop_returns_entry(self, ray_context):
        """pop() returns Entry with item, item_id, version."""
        buffer = DataBuffer()
        await buffer.put("item", version=5, item_id="prompt_123")

        entry = await buffer.pop()
        assert entry.item == "item"
        assert entry.item_id == "prompt_123"
        assert entry.version == 5

    @pytest.mark.asyncio
    async def test_stats_tracking(self, ray_context):
        buffer = DataBuffer()

        await buffer.put("item", version=0, item_id="id1")
        await buffer.pop()

        stats = buffer.get_stats()
        assert stats["put"] == 1
        assert stats["popped"] == 1


class TestDataBufferVersionTracking:
    """Buffer tracks versions but doesn't filter - consumer handles staleness."""

    @pytest.mark.asyncio
    async def test_returns_version_with_item(self, ray_context):
        buffer = DataBuffer()

        await buffer.put("old", version=0, item_id="id1")
        await buffer.put("new", version=5, item_id="id2")

        # Buffer returns all items with their version - consumer decides staleness
        entry1 = await buffer.pop()
        assert entry1.item == "old"
        assert entry1.version == 0

        entry2 = await buffer.pop()
        assert entry2.item == "new"
        assert entry2.version == 5

    @pytest.mark.asyncio
    async def test_by_version_stats(self, ray_context):
        buffer = DataBuffer()

        await buffer.put("item1", version=0, item_id="id1")
        await buffer.put("item2", version=0, item_id="id2")
        await buffer.put("item3", version=1, item_id="id3")

        by_version = buffer.stats.get_by_version()
        assert by_version[0] == 2
        assert by_version[1] == 1

        # Pop one version-0 item
        await buffer.pop()
        by_version = buffer.stats.get_by_version()
        assert by_version[0] == 1  # Decremented


class TestDataBufferMaxReads:
    @pytest.mark.asyncio
    async def test_single_read_removes_entry(self, ray_context):
        buffer = DataBuffer(max_reads=1)

        await buffer.put("item", version=0, item_id="id1")
        await buffer.pop()

        assert buffer.size() == 0

    @pytest.mark.asyncio
    async def test_multi_read_keeps_entry(self, ray_context):
        buffer = DataBuffer(max_reads=3)

        await buffer.put("item", version=0, item_id="id1")

        # First two reads should keep the entry
        entry1 = await buffer.pop()
        assert entry1.item == "item"
        assert buffer.size() == 1

        entry2 = await buffer.pop()
        assert entry2.item == "item"
        assert buffer.size() == 1

        # Third read should remove it
        entry3 = await buffer.pop()
        assert entry3.item == "item"
        assert buffer.size() == 0

    @pytest.mark.asyncio
    async def test_multi_read_same_item_returned(self, ray_context):
        buffer = DataBuffer(max_reads=2)

        await buffer.put("item", version=0, item_id="id1")

        entry1 = await buffer.pop()
        entry2 = await buffer.pop()

        assert entry1.item == entry2.item == "item"


class TestDataBufferConcurrency:
    @pytest.mark.asyncio
    async def test_concurrent_puts(self, ray_context):
        buffer = DataBuffer()

        async def put_items(start, count):
            for i in range(count):
                await buffer.put(f"item{start + i}", version=0, item_id=f"id{start + i}")

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
                await buffer.put(i, version=0, item_id=f"id{i}")
                await asyncio.sleep(0.001)

        async def consumer():
            for _ in range(10):
                entry = await buffer.pop()
                results.append(entry.item)

        await asyncio.gather(producer(), consumer())

        assert len(results) == 10
        assert set(results) == set(range(10))


class TestDataBufferEpochHandling:
    @pytest.mark.asyncio
    async def test_mark_done_returns_none(self, ray_context):
        buffer = DataBuffer()

        await buffer.put("item", version=0, item_id="id1")
        await buffer.mark_done()

        entry = await buffer.pop()
        assert entry.item == "item"
        assert await buffer.pop() is None  # Done signal

    @pytest.mark.asyncio
    async def test_reset_clears_buffer(self, ray_context):
        buffer = DataBuffer()

        await buffer.put("item1", version=0, item_id="id1")
        await buffer.put("item2", version=0, item_id="id2")
        await buffer.mark_done()

        buffer.reset()
        assert buffer.size() == 0

    @pytest.mark.asyncio
    async def test_reset_allows_new_epoch(self, ray_context):
        buffer = DataBuffer()

        # First epoch
        await buffer.put("epoch1_item", version=0, item_id="id1")
        await buffer.mark_done()
        entry = await buffer.pop()
        assert entry.item == "epoch1_item"
        assert await buffer.pop() is None  # Done

        # Reset for next epoch
        buffer.reset()

        # Second epoch - should work normally
        await buffer.put("epoch2_item", version=1, item_id="id2")
        await buffer.mark_done()
        entry = await buffer.pop()
        assert entry.item == "epoch2_item"
        assert await buffer.pop() is None  # Done
