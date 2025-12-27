from __future__ import annotations

from typing import Any, Dict, Optional, Sequence
import asyncio
import time
import uuid
import ray

from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm import SamplingParams
from vllm.sampling_params import RequestOutputKind

from .tracer import traced

# Hardcode these for now as discussed.
WORKER_CLS = "rlvr_experiments.vllm_worker.WeightSyncVLLMWorker"
EXECUTOR_BACKEND = "ray"


@ray.remote(num_gpus=0)
class VLLMEngineRank:
    def __init__(
        self,
        engine_kwargs: Dict[str, Any],
    ) -> None:
        # Remove our custom fields before passing to vLLM
        engine_kwargs = dict(engine_kwargs)
        engine_kwargs.pop("data_parallel_size", None)

        engine_args = AsyncEngineArgs(
            worker_cls=WORKER_CLS,
            distributed_executor_backend=EXECUTOR_BACKEND,
            **engine_kwargs,
        )
        self.engine = AsyncLLMEngine.from_engine_args(engine_args)

    async def add_sync_channel(self, *, channel_name: str, host: str, port: int, world_size: int, rank: int) -> None:
        # Connect all workers to the external NCCL sync group.
        await self.engine.collective_rpc(
            "add_sync_channel",
            kwargs={"channel_name": channel_name, "host": host, "port": port, "world_size": world_size, "rank": rank},
        )

    async def generate(
        self,
        prompts: Sequence[str],
        **sampling_params: Optional[Dict[str, Any]],
    ):
        t_start = time.perf_counter()
        sp = SamplingParams(**sampling_params)
        if sp.output_kind is None:
            sp.output_kind = RequestOutputKind.FINAL_ONLY
        t_sp = time.perf_counter()

        tasks = [self._gen_single(p, sp.clone(), str(uuid.uuid4())) for p in prompts]
        t_tasks = time.perf_counter()

        results = await asyncio.gather(*tasks)
        t_gen = time.perf_counter()

        # Count total tokens generated
        total_tokens = sum(
            len(out.token_ids)
            for r in results
            for out in r.outputs
        )

        print(f"[VLLM TIMING] sp_setup: {(t_sp - t_start)*1000:.1f}ms, "
              f"task_create: {(t_tasks - t_sp)*1000:.1f}ms, "
              f"generation: {(t_gen - t_tasks)*1000:.1f}ms, "
              f"total_tokens: {total_tokens}, "
              f"n_prompts: {len(prompts)}, n_samples: {sp.n}")

        return results

    async def _gen_single(self, prompt, sp, req_id):
        final = None
        async for out in self.engine.generate(prompt, sp, req_id):
            if final is None:
                final = out
            else:
                final.add(out, aggregate=False)
        return final

    async def recv_chunk(self, chunk, dtype_str: str, src_rank: int):
        # Your chunked worker-side receiver; left as-is in your codebase.
        await self.engine.collective_rpc(
            "recv_chunk_from_hf",
            kwargs={"chunk": chunk, "dtype_str": dtype_str, "src_rank": src_rank},
        )

    def ready(self) -> bool:
        return True


class VLLMHandle:
    """
    Handle for one or more vLLM instances (data parallel replicas).

    Each instance may have multiple TP workers internally. This class provides
    a unified interface regardless of how many replicas are configured.
    """

    def __init__(self, actors: list, name: str = "vllm"):
        """
        Args:
            actors: List of VLLMEngineRank actor references (one per DP replica).
            name: Name for this vLLM role.
        """
        if not actors:
            raise ValueError("VLLMHandle requires at least one actor")
        self._actors = actors
        self.name = name
        self._stop_event = asyncio.Event()
        self._active_tasks: list[asyncio.Task] = []
        self._current_replica = 0

    @property
    def num_replicas(self) -> int:
        return len(self._actors)

    def is_stopped(self) -> bool:
        """Check if the stop signal has been set."""
        return self._stop_event.is_set()

    def start_producer(self, producer_coro_factory) -> list[asyncio.Task]:
        """
        Start rollout producers, one per replica.

        Args:
            producer_coro_factory: A callable (replica_id) -> coroutine.
                The coroutine should check handle.is_stopped() to know when to exit.

        Returns:
            List of created asyncio.Tasks (one per replica).
        """
        self._active_tasks = []
        for replica_id in range(self.num_replicas):
            coro = producer_coro_factory(replica_id)
            task = asyncio.create_task(coro)
            self._active_tasks.append(task)
        return self._active_tasks

    async def stop(self) -> None:
        """
        Signal rollout producers to stop and wait for them to finish.
        Call this before syncing weights.
        """
        self._stop_event.set()
        if self._active_tasks:
            print(f"[VLLMHandle] Waiting for {len(self._active_tasks)} producer(s) to stop...")
            await asyncio.gather(*self._active_tasks)
            self._active_tasks = []
            print("[VLLMHandle] All producers stopped.")

    def resume(self) -> None:
        """Reset the stop event so new rollout producers can run."""
        self._stop_event.clear()

    @traced("vllm.generate")
    async def generate(self, prompts, **sampling_params):
        """Generate using round-robin replica selection."""
        actor = self._actors[self._current_replica % len(self._actors)]
        self._current_replica += 1
        return await actor.generate.remote(prompts, **sampling_params)
