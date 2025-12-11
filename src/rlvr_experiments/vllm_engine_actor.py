from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple
import asyncio
import uuid

import ray
import torch

from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm import SamplingParams, RequestOutput

from rlvr_experiments.weight_sync import WeightSyncManager

# -----------------------------------------------------------------------------
# Helper function for TP > 1 Weight Sync
# -----------------------------------------------------------------------------
def _update_weight_on_worker(worker: Any, name: str, weight: torch.Tensor) -> None:
    """
    This function runs INSIDE the vLLM worker processes (both driver and remotes).
    It locates the underlying model and applies the weight update.
    """
    # Navigate: Worker -> ModelRunner -> Model
    # This path is stable for vLLM v0.6.0+
    if hasattr(worker, "model_runner"):
        model = worker.model_runner.model
    elif hasattr(worker, "model"):
        model = worker.model
    else:
        raise RuntimeError(f"Could not locate model on vLLM worker: {type(worker)}")

    # Load the weight (vLLM's load_weights expects a list of tuples)
    model.load_weights(weights=[(name, weight)])


@ray.remote(num_gpus=1)
class VLLMEngineRank:
    """
    Async vLLM engine rank with NCCL-based weight sync.
    Supports Tensor Parallelism > 1 by broadcasting updates to worker actors.
    """

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
        
        # vLLM Driver (this actor) uses the primary GPU
        self.device = torch.device("cuda", 0)

        # 1. Initialize shared weight-sync communicator
        self.weight_sync = WeightSyncManager()
        self.weight_sync.init_communicator(
            host=sync_host,
            port=sync_port,
            world_size=sync_world_size,
            my_rank=sync_rank,
            device=self.device,
        )

        # 2. Initialize AsyncLLMEngine
        print(
            f"[VLLMEngineRank] Initializing AsyncLLMEngine model={model_name}, "
            f"dtype={dtype}, tp={tensor_parallel_size}"
        )

        engine_args = AsyncEngineArgs(
            model=model_name,
            dtype=dtype,
            max_model_len=max_model_len,
            tensor_parallel_size=tensor_parallel_size,
            trust_remote_code=trust_remote_code,
            **vllm_kwargs,
        )
        
        # Initialize the engine (this spawns TP workers if tp > 1)
        self.engine = AsyncLLMEngine.from_engine_args(engine_args)

        print(
            f"[VLLMEngineRank] Rank {sync_rank}/{sync_world_size} ready "
            f"with AsyncLLMEngine (TP={tensor_parallel_size})"
        )

    # ------------------------------------------------------------------ #
    # Async Generation API
    # ------------------------------------------------------------------ #
    async def generate(
        self,
        prompts: Sequence[str],
        sampling_params: Optional[Dict[str, Any]] = None,
    ) -> List[Any]:
        if sampling_params is None:
            sampling_params = {"max_tokens": 16}
        
        sp = SamplingParams(**sampling_params)
        
        tasks = []
        for prompt in prompts:
            # vLLM requires unique request_ids
            request_id = str(uuid.uuid4())
            tasks.append(self._generate_single(prompt, sp, request_id))
            
        return await asyncio.gather(*tasks)

    async def _generate_single(self, prompt: str, sampling_params: SamplingParams, request_id: str) -> RequestOutput:
        results_generator = self.engine.generate(prompt, sampling_params, request_id)
        
        final_output = None
        async for request_output in results_generator:
            final_output = request_output
            
        return final_output

    # ------------------------------------------------------------------ #
    # Weight sync API
    # ------------------------------------------------------------------ #
    def recv_named_param(
        self,
        name: str,
        dtype_str: str,
        shape: Tuple[int, ...],
        src_rank: int = 0,
    ) -> None:
        """
        Receives weight on Driver, then pushes to ALL TP workers.
        """
        if not self.weight_sync.is_initialized:
            raise RuntimeError("[VLLMEngineRank] Weight sync communicator not initialized.")

        dtype_map = {
            "float16": torch.float16, "fp16": torch.float16, "half": torch.float16,
            "bfloat16": torch.bfloat16, "bf16": torch.bfloat16,
            "float32": torch.float32, "fp32": torch.float32,
        }
        dtype = dtype_map[dtype_str]

        # 1. Receive parameter on the Driver (Rank 0 of the TP group)
        #    This is a blocking NCCL call on the driver's GPU.
        weight = torch.empty(shape, dtype=dtype, device=self.device)
        self.weight_sync.communicator.broadcast(weight, src=src_rank)

        # 2. Get the Model Executor (manages TP workers)
        #    Path: AsyncLLMEngine -> LLMEngine -> RayGPUExecutor (for TP)
        model_executor = self.engine.engine.model_executor

        # 3. Update the Local Driver Worker
        #    The driver worker is a local object, not a Ray actor.
        _update_weight_on_worker(model_executor.driver_worker, name, weight)

        # 4. Update Remote Workers (if TP > 1)
        if hasattr(model_executor, "workers") and model_executor.workers:
            # RayGPUExecutor workers allow executing a callable via 'execute_method'
            # We pass the helper function and arguments.
            # Ray will auto-serialize the 'weight' tensor (using shared memory).
            futures = [
                worker.execute_method.remote(_update_weight_on_worker, name=name, weight=weight)
                for worker in model_executor.workers
            ]
            # Wait for all workers to apply weights before proceeding
            ray.get(futures)

    def recv_many_named_params(
        self,
        params: Sequence[Tuple[str, str, Tuple[int, ...]]],
        src_rank: int = 0,
    ) -> None:
        for name, dtype_str, shape in params:
            self.recv_named_param(name=name, dtype_str=dtype_str, shape=shape, src_rank=src_rank)


class VLLMHandle:
    """
    Async wrapper for the VLLMEngineRank actor.
    """
    def __init__(self, actor: ray.actor.ActorHandle, name: str = "vllm") -> None:
        self._actor = actor
        self.name = name

    async def generate(
        self,
        prompts: Sequence[str],
        sampling_params: Dict[str, Any] | None = None,
    ):
        # Directly await the ObjectRef returned by the async actor method
        ref = self._actor.generate.remote(prompts, sampling_params)
        return await ref