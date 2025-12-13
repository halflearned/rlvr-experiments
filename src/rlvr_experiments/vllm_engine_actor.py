from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple
import asyncio
import uuid
import ray
import torch

from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm import SamplingParams

from .syncing import ChunkMeta


@ray.remote(num_gpus=1)
class VLLMEngineRank:
    def __init__(
        self,
        sync_host: str,
        sync_port: int,
        sync_world_size: int,
        sync_rank: int,
        model_name: str,
        dtype: str = "bfloat16",
        max_model_len: int = 4096,
        tensor_parallel_size: int = 1,
        trust_remote_code: bool = True,
        **vllm_kwargs: Any,
    ) -> None:
        
        print(f"[VLLMEngineRank] Initializing vLLM V1 (TP={tensor_parallel_size})...")
        
        engine_args = AsyncEngineArgs(
            model=model_name,
            dtype=dtype,
            max_model_len=max_model_len,
            tensor_parallel_size=tensor_parallel_size,
            trust_remote_code=trust_remote_code,
            worker_cls="rlvr_experiments.vllm_worker.WeightSyncVLLMWorker",
            **vllm_kwargs,
        )
        
        # Create Engine (This starts the internal loop/processes)
        self.engine = AsyncLLMEngine.from_engine_args(engine_args)
        
        # Save sync config for the async initialization step
        self.sync_args = {
            "host": sync_host,
            "port": sync_port,
            "world_size": sync_world_size,
            "rank": sync_rank
        }
        
        print(f"[VLLMEngineRank] Engine created. Waiting for init_workers() call...")

    async def init_workers(self) -> None:
        """
        Async initialization phase.
        Connects all vLLM workers to the NCCL group.
        This will BLOCK until the Trainer also joins the group!
        """
        print("[VLLMEngineRank] Sending init_weight_sync RPC to workers...")
        
        # Direct await (Valid because we are now in an async method)
        await self.engine.collective_rpc(
            "init_weight_sync",
            kwargs=self.sync_args
        )
        
        print(f"[VLLMEngineRank] Ready. Workers joined sync group.")

    async def generate(self, prompts: Sequence[str], sampling_params: Optional[Dict[str, Any]] = None):
        if sampling_params is None: sampling_params = {"max_tokens": 16}
        sp = SamplingParams(**sampling_params)
        tasks = [self._gen_single(p, sp, str(uuid.uuid4())) for p in prompts]
        return await asyncio.gather(*tasks)

    async def _gen_single(self, prompt, sp, req_id):
        final = None
        async for out in self.engine.generate(prompt, sp, req_id): final = out
        return final

    async def recv_chunk(self, chunk: ChunkMeta, dtype_str: str, src_rank: int = 0):
        await self.engine.collective_rpc(
            "recv_chunk_from_hf",
            kwargs={
                "chunk": chunk,
                "dtype_str": dtype_str,
                "src_rank": src_rank,
            },
        )

    def ready(self) -> bool:
        return True



class VLLMHandle:
    def __init__(self, actor, name="vllm"):
        self._actor = actor
        self.name = name

    async def generate(self, prompts, sampling_params=None):
        return await self._actor.generate.remote(prompts, sampling_params)
