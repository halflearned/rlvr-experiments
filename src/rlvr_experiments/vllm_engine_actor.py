from __future__ import annotations

from typing import Any, Dict, Optional, Sequence
import asyncio
import uuid
import ray

from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm import SamplingParams


# Hardcode these for now as discussed.
WORKER_CLS = "rlvr_experiments.vllm_worker.WeightSyncVLLMWorker"
EXECUTOR_BACKEND = "ray"


@ray.remote(num_gpus=1)
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
        sp = SamplingParams(**sampling_params)
        tasks = [self._gen_single(p, sp, str(uuid.uuid4())) for p in prompts]
        return await asyncio.gather(*tasks)

    async def _gen_single(self, prompt, sp, req_id):
        final = None
        async for out in self.engine.generate(prompt, sp, req_id):
            final = out
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

    async def generate(self, prompts, **sampling_params):
        return await self._actor.generate.remote(prompts, **sampling_params)
