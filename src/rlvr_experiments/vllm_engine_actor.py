import asyncio
import uuid
import torch
import ray

from typing import Any, AsyncIterator, Callable, Sequence
from vllm import SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.sampling_params import RequestOutputKind
from vllm.v1.worker.gpu_worker import Worker

from .weight_sync import WeightSyncManager
from .tracer import traced, trace_span

WORKER_CLS = "rlvr_experiments.vllm_engine_actor.WeightSyncVLLMWorker"


@ray.remote(num_gpus=0)
class VLLMEngineRank:
    def __init__(self, engine_kwargs: dict[str, Any]) -> None:
        engine_kwargs = dict(engine_kwargs)
        engine_kwargs.pop("data_parallel_size", None)
        engine_args = AsyncEngineArgs(
            worker_cls=WORKER_CLS, distributed_executor_backend="ray", **engine_kwargs
        )
        self.engine = AsyncLLMEngine.from_engine_args(engine_args)

    def ready(self) -> bool:
        return True

    async def add_sync_channel(self, *, channel_name: str, host: str, port: int, world_size: int, rank: int) -> None:
        await self.engine.collective_rpc(
            "add_sync_channel",
            kwargs={"channel_name": channel_name, "host": host, "port": port, "world_size": world_size, "rank": rank},
        )

    async def generate(self, prompts: Sequence[str], **sampling_params):
        sp = SamplingParams(**sampling_params)
        if sp.output_kind is None:
            sp.output_kind = RequestOutputKind.FINAL_ONLY

        async def gen_single(prompt):
            final = None
            async for out in self.engine.generate(prompt, sp.clone(), str(uuid.uuid4())):
                final = out if final is None else final.add(out, aggregate=False) or final
            return final

        return await asyncio.gather(*[gen_single(p) for p in prompts])

    async def recv_chunk(self, channel: str, chunk: dict, dtype_str: str, src_rank: int):
        # channel is unused - vLLM only has one sync channel
        await self.engine.collective_rpc(
            "recv_chunk", kwargs={"chunk": chunk, "dtype_str": dtype_str, "src_rank": src_rank}
        )


class VLLMHandle:
    """Handle for one or more vLLM instances (data parallel replicas).

    Manages GPU generation and producer lifecycle for rollout collection.

    Usage:
        async for samples in rollout.run_epoch(producer_fn, buffer, batch_size=8, ...):
            train_on(samples)
    """

    def __init__(self, actors: list, name: str = "vllm"):
        self._actors = actors
        self.name = name
        self._stop_event = asyncio.Event()
        self._current_replica = 0
        self._producer_tasks: list[asyncio.Task] = []

    def is_stopped(self) -> bool:
        return self._stop_event.is_set()

    @traced("vllm.generate")
    async def generate(self, prompts, **sampling_params):
        actor = self._actors[self._current_replica % len(self._actors)]
        self._current_replica += 1
        return await actor.generate.remote(prompts, **sampling_params)

    # ---- Producer lifecycle ----

    def _producers_done(self) -> bool:
        return len(self._producer_tasks) > 0 and all(t.done() for t in self._producer_tasks)

    def _check_producer_errors(self) -> None:
        for i, t in enumerate(self._producer_tasks):
            if t.done() and not t.cancelled():
                exc = t.exception()
                if exc is not None:
                    raise RuntimeError(f"Producer {i} failed") from exc

    async def run_epoch(
        self,
        producer_fn: Callable,
        buffer,
        batch_size: int,
        min_version: int = 0,
        **producer_kwargs,
    ) -> AsyncIterator[list]:
        """Run producers and yield batches until done.

        Args:
            producer_fn: Coroutine that generates samples. Called as:
                         producer_fn(rollout, buffer, **producer_kwargs)
            buffer: RolloutBuffer to collect samples into
            batch_size: Number of samples per batch
            min_version: Minimum version for buffer.pop()
            **producer_kwargs: Additional args passed to producer_fn

        Yields:
            Lists of samples, batch_size each (may be smaller at end)
        """
        self._stop_event.clear()
        self._producer_tasks = [
            asyncio.create_task(producer_fn(self, buffer, **producer_kwargs))
            for _ in self._actors
        ]

        try:
            while True:
                with trace_span(None, "wait_for_batch"):
                    self._check_producer_errors()

                    if self._producers_done() and buffer.size() == 0:
                        return

                    samples = []
                    while len(samples) < batch_size:
                        self._check_producer_errors()
                        if self._producers_done() and buffer.size() == 0:
                            break
                        # Wait longer when buffer empty (verifications take several seconds)
                        timeout = 10.0 if buffer.size() == 0 else 5.0
                        batch = await buffer.pop_batch(batch_size - len(samples), min_version=min_version, timeout=timeout)
                        samples.extend(batch)

                if not samples:
                    return

                yield samples
        finally:
            self._stop_event.set()
            for t in self._producer_tasks:
                t.cancel()
            await asyncio.gather(*self._producer_tasks, return_exceptions=True)
            self._producer_tasks = []



class WeightSyncVLLMWorker(Worker):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.weight_sync: WeightSyncManager | None = None

    def add_sync_channel(self, channel_name: str, host: str, port: int, world_size: int, rank: int):
        sync_rank = rank + self.rank
        self.weight_sync = WeightSyncManager(
            host=host, port=port, world_size=world_size, rank=sync_rank, device=self.device
        )

    def recv_chunk(self, chunk: dict, dtype_str: str, src_rank: int):
        """Receive a chunk via NCCL broadcast and load weights into vLLM model."""
        dtype = getattr(torch, dtype_str)
        flat = torch.empty(chunk["total_numel"], dtype=dtype, device=self.device)
        self.weight_sync.communicator.broadcast(flat, src=src_rank)

        offset = 0
        for p in chunk["params"]:
            weight = flat[offset : offset + p["numel"]].view(p["shape"])
            offset += p["numel"]
            self.model_runner.model.load_weights(weights=[(p["name"], weight)])
