"""Unit tests for VLLMHandle pause/resume logic and load-aware routing."""

import pytest
import asyncio
from unittest.mock import MagicMock, AsyncMock

from rlvr_experiments.vllm_engine_actor import VLLMHandle, LoadAwareRouter


class TestVLLMHandleBasic:
    def test_init(self):
        actors = [MagicMock(), MagicMock()]
        handle = VLLMHandle(actors, name="test_vllm")

        assert handle.num_replicas == 2
        assert handle.name == "test_vllm"
        assert handle._in_flight == 0
        assert handle._paused.is_set()  # Not paused initially

    def test_num_replicas(self):
        actors = [MagicMock() for _ in range(4)]
        handle = VLLMHandle(actors)
        assert handle.num_replicas == 4


class TestLoadAwareRouter:
    @pytest.mark.asyncio
    async def test_routes_to_least_loaded(self):
        router = LoadAwareRouter(num_replicas=3, max_concurrent_per_replica=10)

        # Acquire slots and verify they go to least loaded
        # acquire_slot returns (replica_idx, slot_idx)
        replica_idx0, slot_idx0 = await router.acquire_slot()
        assert replica_idx0 == 0  # All equal, picks first
        assert router.get_load() == [1, 0, 0]

        replica_idx1, slot_idx1 = await router.acquire_slot()
        assert replica_idx1 == 1  # replica 1 has 0 load
        assert router.get_load() == [1, 1, 0]

        replica_idx2, slot_idx2 = await router.acquire_slot()
        assert replica_idx2 == 2  # replica 2 has 0 load
        assert router.get_load() == [1, 1, 1]

        replica_idx3, slot_idx3 = await router.acquire_slot()
        assert replica_idx3 == 0  # All equal again, picks first
        assert router.get_load() == [2, 1, 1]

    @pytest.mark.asyncio
    async def test_release_slot(self):
        router = LoadAwareRouter(num_replicas=2, max_concurrent_per_replica=10)

        replica_idx, slot_idx = await router.acquire_slot()
        assert router.get_load()[replica_idx] == 1

        await router.release_slot(replica_idx, slot_idx)
        assert router.get_load()[replica_idx] == 0

    @pytest.mark.asyncio
    async def test_respects_capacity_limit(self):
        router = LoadAwareRouter(num_replicas=1, max_concurrent_per_replica=2)

        # Fill to capacity
        slot1 = await router.acquire_slot()
        slot2 = await router.acquire_slot()
        assert router.get_load() == [2]

        # Next acquire should block until we release
        acquire_task = asyncio.create_task(router.acquire_slot())
        await asyncio.sleep(0.01)
        assert not acquire_task.done()

        # Release one slot
        await router.release_slot(slot1[0], slot1[1])

        # Now acquire should complete
        replica_idx, slot_idx = await asyncio.wait_for(acquire_task, timeout=1.0)
        assert replica_idx == 0

    @pytest.mark.asyncio
    async def test_balances_across_replicas(self):
        router = LoadAwareRouter(num_replicas=3, max_concurrent_per_replica=100)

        # Acquire 9 slots
        for _ in range(9):
            await router.acquire_slot()

        # Should be evenly distributed
        assert router.get_load() == [3, 3, 3]


class TestVLLMHandleGenerateSingle:
    @pytest.fixture
    def mock_actors(self):
        """Create mock actors that return vLLM-like response objects."""
        def make_mock_response(text):
            """Create a mock response object with outputs that have token_ids."""
            output = MagicMock()
            output.token_ids = [1, 2, 3]  # Non-empty token_ids
            output.text = text
            response = MagicMock()
            response.outputs = [output]
            return response

        actors = []
        for i in range(2):
            actor = MagicMock()
            actor.generate = MagicMock()
            # Return a list containing a mock response object
            actor.generate.remote = AsyncMock(return_value=[make_mock_response(f"response_{i}")])
            actors.append(actor)
        return actors

    @pytest.mark.asyncio
    async def test_generate_single_returns_single_result(self, mock_actors):
        handle = VLLMHandle(mock_actors)
        result = await handle.generate_single("prompt", temperature=0.7)

        # Should return unwrapped result (the response object, not the list)
        assert result.outputs[0].text == "response_0"

    @pytest.mark.asyncio
    async def test_generate_single_load_aware_routing(self, mock_actors):
        """Create a mock response object with outputs that have token_ids."""
        def make_mock_response(text):
            output = MagicMock()
            output.token_ids = [1, 2, 3]
            output.text = text
            response = MagicMock()
            response.outputs = [output]
            return response

        # Create actors with different completion times
        slow_event = asyncio.Event()

        async def slow_generate(*args, **kwargs):
            await slow_event.wait()
            return [make_mock_response("slow_response")]

        mock_actors[0].generate.remote = slow_generate

        handle = VLLMHandle(mock_actors, max_concurrent_per_replica=10)

        # Start a slow request to actor 0
        slow_task = asyncio.create_task(handle.generate_single("slow"))
        await asyncio.sleep(0.01)

        # Next request should go to actor 1 (less loaded)
        result = await handle.generate_single("fast")
        assert result.outputs[0].text == "response_1"

        # Verify loads
        loads = handle.get_replica_loads()
        assert loads[0] == 1  # slow request still in flight
        assert loads[1] == 0  # fast request completed

        # Cleanup
        slow_event.set()
        await slow_task

    @pytest.mark.asyncio
    async def test_generate_single_blocks_when_paused(self, mock_actors):
        handle = VLLMHandle(mock_actors)
        await handle.stop()

        task = asyncio.create_task(handle.generate_single("prompt"))
        await asyncio.sleep(0.01)
        assert not task.done()

        handle.resume()
        result = await asyncio.wait_for(task, timeout=1.0)
        # Result should be a response object with outputs
        assert result.outputs[0].text in ["response_0", "response_1"]

    @pytest.mark.asyncio
    async def test_generate_single_exception_releases_slot(self, mock_actors):
        mock_actors[0].generate.remote = AsyncMock(side_effect=RuntimeError("boom"))
        mock_actors[1].generate.remote = AsyncMock(side_effect=RuntimeError("boom"))
        handle = VLLMHandle(mock_actors)

        with pytest.raises(RuntimeError):
            await handle.generate_single("prompt")

        # Slot should be released
        assert handle.get_replica_loads() == [0, 0]
        assert handle._in_flight == 0

    @pytest.mark.asyncio
    async def test_in_flight_tracking(self, mock_actors):
        def make_mock_response(text):
            output = MagicMock()
            output.token_ids = [1, 2, 3]
            output.text = text
            response = MagicMock()
            response.outputs = [output]
            return response

        handle = VLLMHandle(mock_actors)

        # Create a slow generate that we can control
        generate_started = asyncio.Event()
        generate_continue = asyncio.Event()

        async def slow_generate(*args, **kwargs):
            generate_started.set()
            await generate_continue.wait()
            return [make_mock_response("response")]

        mock_actors[0].generate.remote = slow_generate

        # Start generate but don't await it
        task = asyncio.create_task(handle.generate_single("prompt"))

        # Wait for generate to start
        await generate_started.wait()
        assert handle._in_flight == 1
        assert not handle._in_flight_zero.is_set()

        # Let it complete
        generate_continue.set()
        await task

        assert handle._in_flight == 0
        assert handle._in_flight_zero.is_set()


def _make_mock_response(text="response"):
    """Create a mock response object with outputs that have token_ids."""
    output = MagicMock()
    output.token_ids = [1, 2, 3]
    output.text = text
    response = MagicMock()
    response.outputs = [output]
    return response


class TestVLLMHandleStopResume:
    @pytest.fixture
    def mock_actors(self):
        actors = []
        for _ in range(2):
            actor = MagicMock()
            actor.generate = MagicMock()
            actor.generate.remote = AsyncMock(return_value=[_make_mock_response()])
            actors.append(actor)
        return actors

    @pytest.mark.asyncio
    async def test_stop_closes_gate(self, mock_actors):
        handle = VLLMHandle(mock_actors)

        await handle.stop()

        assert not handle._paused.is_set()

    @pytest.mark.asyncio
    async def test_resume_opens_gate(self, mock_actors):
        handle = VLLMHandle(mock_actors)

        await handle.stop()
        handle.resume()

        assert handle._paused.is_set()

    @pytest.mark.asyncio
    async def test_stop_waits_for_in_flight(self, mock_actors):
        handle = VLLMHandle(mock_actors)

        # Create a slow generate
        generate_continue = asyncio.Event()

        async def slow_generate(*args, **kwargs):
            await generate_continue.wait()
            return [_make_mock_response()]

        mock_actors[0].generate.remote = slow_generate

        # Start a generate
        gen_task = asyncio.create_task(handle.generate_single("prompt"))
        await asyncio.sleep(0.01)  # Let it start

        # Start stop - it should block waiting for in-flight
        stop_task = asyncio.create_task(handle.stop())
        await asyncio.sleep(0.01)

        assert not stop_task.done()  # Should be waiting

        # Let the generate complete
        generate_continue.set()
        await gen_task

        # Now stop should complete
        await asyncio.wait_for(stop_task, timeout=1.0)

    @pytest.mark.asyncio
    async def test_concurrent_generates_tracked(self, mock_actors):
        handle = VLLMHandle(mock_actors)

        events = [asyncio.Event() for _ in range(3)]
        continue_event = asyncio.Event()

        call_count = [0]

        async def tracked_generate(*args, **kwargs):
            idx = call_count[0]
            call_count[0] += 1
            events[idx].set()
            await continue_event.wait()
            return [_make_mock_response()]

        for actor in mock_actors:
            actor.generate.remote = tracked_generate

        # Start 3 concurrent generates
        tasks = [asyncio.create_task(handle.generate_single(f"p{i}")) for i in range(3)]

        # Wait for all to start
        for e in events:
            await e.wait()

        assert handle._in_flight == 3

        # Let them all complete
        continue_event.set()
        await asyncio.gather(*tasks)

        assert handle._in_flight == 0

    @pytest.mark.asyncio
    async def test_stop_resume_cycle(self, mock_actors):
        handle = VLLMHandle(mock_actors)

        # Multiple stop/resume cycles should work
        for _ in range(3):
            await handle.stop()
            assert not handle._paused.is_set()

            handle.resume()
            assert handle._paused.is_set()

            # Generate should work after resume
            result = await handle.generate_single("prompt")
            assert result.outputs[0].text == "response"


class TestVLLMHandleEdgeCases:
    @pytest.mark.asyncio
    async def test_stop_when_no_in_flight(self):
        actor = MagicMock()
        actor.generate.remote = AsyncMock(return_value=["r"])
        handle = VLLMHandle([actor])

        # Should complete immediately
        await asyncio.wait_for(handle.stop(), timeout=0.1)
