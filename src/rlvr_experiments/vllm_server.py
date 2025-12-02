#!/usr/bin/env python
import argparse
import logging
import os
from dataclasses import dataclass, field
from typing import List, Sequence

import torch

# Make sure vLLM uses spawn for CUDA + multiprocessing
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

from fastapi import FastAPI  # type: ignore
from pydantic import BaseModel  # type: ignore
import uvicorn  # type: ignore

from vllm import LLM, SamplingParams
from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
from vllm.distributed.parallel_state import get_world_group
from vllm.distributed.utils import StatelessProcessGroup

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


# ---------------------------------------------------------------------------
# Worker extension: runs inside each vLLM worker process
# ---------------------------------------------------------------------------

class WeightSyncWorkerExtension:
    """
    Minimal worker extension enabling NCCL-based weight synchronization
    between an external trainer process (client) and all vLLM workers.
    """

    communicator = None   # type: ignore
    client_rank = None    # type: ignore

    def init_communicator(self, host: str, port: int, world_size: int, client_device_uuid: str) -> None:
        """
        Initialize the NCCL communicator for this worker.

        Args:
            host: address of the rendezvous host (the client side).
            port: TCP port used for the StatelessProcessGroup/TCPStore.
            world_size: total number of participants (engine ranks + 1 client).
            client_device_uuid: UUID of the client's CUDA device, to ensure
                                we don't share a device between client and worker.
        """
        if self.communicator is not None:
            raise RuntimeError("Weight update group already initialized. Call close_communicator first.")

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for this minimal WeightSyncWorkerExtension.")

        # Basic safety: ensure client uses a different GPU than this worker
        worker_uuid = str(torch.cuda.get_device_properties(self.device).uuid)
        if worker_uuid == client_device_uuid:
            raise RuntimeError(
                f"Client device UUID {client_device_uuid} matches worker device {worker_uuid}. "
                "Trainer and vLLM must use different devices."
            )

        # Rank of this worker in vLLM's global world group
        rank = get_world_group().rank

        # Create a stateless process group for the weight-update communicator
        pg = StatelessProcessGroup.create(
            host=host,
            port=port,
            rank=rank,
            world_size=world_size,
        )
        self.communicator = PyNcclCommunicator(pg, device=self.device)

        # Client is the last rank
        self.client_rank = world_size - 1

        logger.info(
            "Worker rank %d initialized weight communicator (world_size=%d, client_rank=%d)",
            rank, world_size, self.client_rank
        )

    def update_named_param(self, name: str, dtype: str, shape: Sequence[int]) -> None:
        """
        Receive updated weights from the client and load them into the model.

        Args:
            name: parameter name (must match vLLM model param name).
            dtype: string like 'torch.float16', 'torch.bfloat16', etc.
            shape: tensor shape.
        """
        if self.communicator is None:
            raise RuntimeError("Communicator not initialized. Call init_communicator first.")

        # Parse dtype string
        torch_dtype = getattr(torch, dtype.split(".")[-1])

        # Allocate buffer on this worker's device
        weight = torch.empty(shape, dtype=torch_dtype, device=self.device)

        # Broadcast tensor contents from client rank to all workers
        self.communicator.broadcast(weight, src=self.client_rank)
        self.communicator.group.barrier()

        # Inject updated weights into the vLLM model
        # vLLM expects a sequence of (name, tensor) pairs
        self.model_runner.model.load_weights(weights=[(name, weight)])

        logger.debug("Updated parameter %s with shape %s", name, tuple(shape))

    def close_communicator(self) -> None:
        """
        Tear down the communicator.
        """
        if self.communicator is not None:
            del self.communicator
            self.communicator = None
            self.client_rank = None
            logger.info("Closed weight update communicator in worker")


class InitCommunicatorRequest(BaseModel):
    host: str          # client rendezvous host (reachable from workers)
    port: int          # client rendezvous port
    world_size: int    # client will send something; server will recompute
    client_device_uuid: str


class UpdateWeightsRequest(BaseModel):
    name: str
    dtype: str
    shape: list[int]




def sanitize_logprob(logprob):
    import math

    value = logprob.logprob
    if math.isnan(value):
        logger.warning(f"Generated NaN logprob, token logprob '{logprob}' will be ignored")
        return None

    return value


def create_app(vllm_args) -> FastAPI:
    # Instantiate vLLM engine with our worker extension
    llm = LLM(
        **vllm_args,
        worker_extension_cls="rlvr_experiments.vllm_server.WeightSyncWorkerExtension",
    )
    # At this point, vLLM will have spawned its engine workers.

    # Number of engine ranks = TP * DP; here DP is effectively 1
    engine_world_size = vllm_args.get("tensor_parallel_size", 1) * 1  # TODO: Assuming DP=1 for now 

    app = FastAPI()

    @app.get("/health/")
    async def health():
        return {"status": "ok"}

    @app.get("/get_world_size/")
    async def get_world_size():
        """
        Return the vLLM engine world size (without client).

        Client should treat:
          vllm_world_size = response["world_size"]
          world_size = vllm_world_size + 1   # client rank is last
        """
        return {"world_size": engine_world_size}

    @app.post("/init_communicator/")
    async def init_communicator(req: InitCommunicatorRequest):
        """
        Initialize NCCL communicator across all engine ranks + client.
        """
        # Ignore req.world_size and recompute deterministically
        world_size = engine_world_size + 1

        llm.collective_rpc(
            method="init_communicator",
            args=(req.host, req.port, world_size, req.client_device_uuid),
        )
        return {"message": "Initializing communicator", "world_size": world_size}

    @app.post("/update_named_param/")
    async def update_named_param(req: UpdateWeightsRequest):
        """
        Notify workers to expect a broadcast for a named parameter.

        After this returns, the client must call NCCL broadcast with
        the actual tensor contents.
        """
        # TODO: will need to remap names back to HF namespace!
        llm.collective_rpc(
            method="update_named_param",
            args=(req.name, req.dtype, tuple(req.shape)),
        )
        return {"message": f"Updating parameter {req.name}"}

    @app.post("/close_communicator/")
    async def close_communicator():
        """
        Close the communicator in all workers.
        """
        llm.collective_rpc(method="close_communicator", args=())
        return {"message": "Closing communicator"}


    from vllm.sampling_params import GuidedDecodingParams  # only if you want regex; can omit


    class GenerateRequest(BaseModel):
        prompts: list[str]
        n: int = 1
        repetition_penalty: float = 1.0
        temperature: float = 1.0
        top_p: float = 1.0
        top_k: int = -1
        min_p: float = 0.0
        max_tokens: int = 16
        truncate_prompt_tokens: int | None = None
        guided_decoding_regex: str | None = None
        # keep this for future flexibility, but optional for now
        generation_kwargs: dict = field(default_factory=dict)


    class GenerateResponse(BaseModel):
        prompt_ids: list[list[int]]
        completion_ids: list[list[int]]
        logprobs: list[list[float]]
        

    @app.post("/generate/", response_model=GenerateResponse)
    async def generate(request: GenerateRequest):
        """
        Simple text-generation endpoint for vLLM.

        - Accepts plain prompts.
        - Returns token ids and logprobs similar to TRL's vllm-serve.
        """

        # (optional) guided decoding
        if request.guided_decoding_regex is not None:
            guided_decoding = GuidedDecodingParams(regex=request.guided_decoding_regex)
        else:
            guided_decoding = None

        generation_kwargs = {
            "n": request.n,
            "repetition_penalty": request.repetition_penalty,
            "temperature": request.temperature,
            "top_p": request.top_p,
            "top_k": request.top_k,
            "min_p": request.min_p,
            "max_tokens": request.max_tokens,
            "truncate_prompt_tokens": request.truncate_prompt_tokens,
            "guided_decoding": guided_decoding,
            "logprobs": 0,  # vLLM: 0 = return logprobs for sampled tokens
        }
        generation_kwargs.update(request.generation_kwargs)

        sampling_params = SamplingParams(**generation_kwargs)

        # Directly call vLLM; no custom DP pipes
        outputs = llm.generate(request.prompts, sampling_params)

        # Shape semantics intentionally match TRL:
        # - prompt_ids: one per input prompt
        # - completion_ids/logprobs: flattened across all completions
        prompt_ids: list[list[int]] = [out.prompt_token_ids for out in outputs]

        completion_ids: list[list[int]] = []
        logprobs: list[list[float]] = []

        for out in outputs:            # one per prompt
            for comp in out.outputs:   # one per completion
                completion_ids.append(list(comp.token_ids))
                # comp.logprobs is a list[dict[token_id -> TokenLogProb]]
                comp_logprobs: list[float] = []
                for lp_dict in comp.logprobs:
                    tok_lp = next(iter(lp_dict.values()))
                    val = sanitize_logprob(tok_lp)
                    if val is not None:
                        comp_logprobs.append(val)
                logprobs.append(comp_logprobs)

        return GenerateResponse(
            prompt_ids=prompt_ids,
            completion_ids=completion_ids,
            logprobs=logprobs,
        )


    return app

