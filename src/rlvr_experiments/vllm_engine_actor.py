import asyncio
import uuid
import torch
import ray

from typing import Any, Sequence
from vllm import SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.sampling_params import RequestOutputKind
from vllm.v1.worker.gpu_worker import Worker

from .syncing import WeightSyncManager
from .tracer import traced

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
    """Handle for one or more vLLM instances (data parallel replicas)."""

    def __init__(self, actors: list, name: str = "vllm"):
        self._actors = actors
        self._stop_event = asyncio.Event()
        self._current_replica = 0
        self.name = name

    @property
    def num_replicas(self) -> int:
        return len(self._actors)

    def is_stopped(self) -> bool:
        return self._stop_event.is_set()

    def stop(self) -> None:
        self._stop_event.set()

    def reset(self) -> None:
        self._stop_event.clear()

    @traced("vllm.generate")
    async def generate(self, prompts, **sampling_params):
        actor = self._actors[self._current_replica % len(self._actors)]
        self._current_replica += 1
        return await actor.generate.remote(prompts, **sampling_params)


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
