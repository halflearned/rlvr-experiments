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
        #model_name: str,
        engine_kwargs: Dict[str, Any],
    ) -> None:

        engine_args = AsyncEngineArgs(
            #model=model_name,
            worker_cls=WORKER_CLS,
            distributed_executor_backend=EXECUTOR_BACKEND,
            **engine_kwargs,
        )
        self.engine = AsyncLLMEngine.from_engine_args(engine_args)

    async def join_sync(self, *, host: str, port: int, world_size: int, rank: int) -> None:
        # Connect all workers to the external NCCL sync group.
        await self.engine.collective_rpc(
            "init_weight_sync",
            kwargs={"host": host, "port": port, "world_size": world_size, "rank": rank},
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
    def __init__(self, actor, name: str = "vllm"):
        self._actor = actor
        self.name = name
        self._stop_event = asyncio.Event()
        self._active_task: asyncio.Task | None = None

    def is_stopped(self) -> bool:
        """Check if the stop signal has been set."""
        return self._stop_event.is_set()

    def start_producer(self, producer_coro) -> asyncio.Task:
        """
        Start a rollout producer coroutine and track it internally.
        The producer will be automatically stopped when sync_titan_to_vllm is called.

        Args:
            producer_coro: A coroutine object that produces rollouts.
                           Should check is_stopped() to know when to exit.

        Returns:
            The created asyncio.Task (now running in background)
        """
        self._active_task = asyncio.create_task(producer_coro)
        return self._active_task

    async def stop(self) -> None:
        """
        Signal rollout producers to stop and wait for them to finish.
        Call this before syncing weights.
        """
        self._stop_event.set()
        if self._active_task is not None:
            print("[VLLM] Waiting for rollout producer to stop...")
            await self._active_task
            self._active_task = None
            print("[VLLM] Rollout producer stopped.")

    def resume(self) -> None:
        """Reset the stop event so new rollout producers can run."""
        self._stop_event.clear()

    @traced("vllm.generate")
    async def generate(self, prompts, **sampling_params):
        return await self._actor.generate.remote(prompts, **sampling_params)
