#!/usr/bin/env python
import argparse
import logging
import os
from dataclasses import dataclass, field
from typing import Sequence

import torch

# Make sure vLLM uses spawn for CUDA + multiprocessing
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

from fastapi import FastAPI  # type: ignore
from pydantic import BaseModel  # type: ignore
import uvicorn  # type: ignore

from vllm import LLM
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


# ---------------------------------------------------------------------------
# CLI args
# ---------------------------------------------------------------------------

@dataclass
class ScriptArgs:
    model: str = field(
        metadata={"help": "HF model ID or local path (e.g. Qwen/Qwen3-0.6B)."}
    )
    host: str = field(
        default="0.0.0.0",
        metadata={"help": "Host to bind the HTTP server on."},
    )
    port: int = field(
        default=8000,
        metadata={"help": "Port to bind the HTTP server on."},
    )
    tensor_parallel_size: int = field(
        default=1,
        metadata={"help": "vLLM tensor parallel degree."},
    )
    gpu_memory_utilization: float = field(
        default=0.7,
        metadata={"help": "vLLM gpu_memory_utilization."},
    )
    dtype: str = field(
        default="bfloat16",
        metadata={"help": "Model dtype for vLLM (e.g. float16, bfloat16, auto)."},
    )
    max_model_len: int | None = field(
        default=1024,
        metadata={"help": "Max model length for vLLM (context length)."},
    )
    enforce_eager: bool = field(
        default=False,
        metadata={"help": "If True, disable CUDA graphs and run in eager mode."},
    )
    kv_cache_dtype: str = field(
        default="auto",
        metadata={"help": "KV cache dtype (auto, float16, bfloat16, etc.)."},
    )
    trust_remote_code: bool = field(
        default=False,
        metadata={"help": "Allow remote code when loading models."},
    )
    log_level: str = field(
        default="info",
        metadata={"help": "uvicorn log level."},
    )


def parse_args() -> ScriptArgs:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.7)
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--max-model-len", type=int, default=1024)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--kv-cache-dtype", type=str, default="auto")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--log-level", type=str, default="info")
    ns = parser.parse_args()
    return ScriptArgs(
        model=ns.model,
        host=ns.host,
        port=ns.port,
        tensor_parallel_size=ns.tensor_parallel_size,
        gpu_memory_utilization=ns.gpu_memory_utilization,
        dtype=ns.dtype,
        max_model_len=ns.max_model_len,
        enforce_eager=ns.enforce_eager,
        kv_cache_dtype=ns.kv_cache_dtype,
        trust_remote_code=ns.trust_remote_code,
        log_level=ns.log_level,
    )


# ---------------------------------------------------------------------------
# HTTP schema
# ---------------------------------------------------------------------------

class InitCommunicatorRequest(BaseModel):
    host: str          # client rendezvous host (reachable from workers)
    port: int          # client rendezvous port
    world_size: int    # client will send something; server will recompute
    client_device_uuid: str


class UpdateWeightsRequest(BaseModel):
    name: str
    dtype: str
    shape: list[int]


# ---------------------------------------------------------------------------
# Main setup
# ---------------------------------------------------------------------------

def create_app(script_args: ScriptArgs) -> FastAPI:
    # Instantiate vLLM engine with our worker extension
    logger.info("Initializing vLLM LLM(model=%s)...", script_args.model)
    llm = LLM(
        model=script_args.model,
        tensor_parallel_size=script_args.tensor_parallel_size,
        gpu_memory_utilization=script_args.gpu_memory_utilization,
        dtype=script_args.dtype,
        max_model_len=script_args.max_model_len,
        enforce_eager=script_args.enforce_eager,
        kv_cache_dtype=script_args.kv_cache_dtype,
        trust_remote_code=script_args.trust_remote_code,
        worker_extension_cls=WeightSyncWorkerExtension,
    )
    # At this point, vLLM will have spawned its engine workers.

    # Number of engine ranks = TP * DP; here DP is effectively 1
    engine_world_size = script_args.tensor_parallel_size

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

    return app


def main():
    script_args = parse_args()
    app = create_app(script_args)
    uvicorn.run(app, host=script_args.host, port=script_args.port, log_level=script_args.log_level)


if __name__ == "__main__":
    main()
