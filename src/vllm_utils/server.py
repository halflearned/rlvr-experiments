# stolen from trl

# Copyright 2025 The HuggingFace Team and Contributors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import base64
import importlib.util
import logging
import os
import sys
from collections.abc import Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from io import BytesIO
from itertools import chain
from multiprocessing import Pipe, Process
from multiprocessing.connection import Connection

import torch
import torch.distributed.distributed_c10d as c10d
from transformers import HfArgumentParser, is_torch_xpu_available, is_vision_available


# --- Local Import Utils (Replacements for trl.import_utils) ---
def is_package_available(pkg_name: str) -> bool:
    return importlib.util.find_spec(pkg_name) is not None

def is_fastapi_available():
    return is_package_available("fastapi")

def is_pydantic_available():
    return is_package_available("pydantic")

def is_uvicorn_available():
    return is_package_available("uvicorn")

def is_vllm_available():
    return is_package_available("vllm")

def is_vllm_ascend_available():
    return is_package_available("vllm_ascend")
# -------------------------------------------------------------


if is_fastapi_available():
    from fastapi import FastAPI

if is_pydantic_available():
    from pydantic import BaseModel

if is_uvicorn_available():
    import uvicorn

if is_vision_available():
    from PIL import Image

if is_vllm_available():
    from vllm import LLM, SamplingParams
    from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
    from vllm.distributed.parallel_state import get_world_group
    from vllm.distributed.utils import StatelessProcessGroup
    from vllm.sampling_params import GuidedDecodingParams
    from vllm.utils import get_open_port

    if is_vllm_ascend_available():
        from vllm_ascend.distributed.device_communicators.pyhccl import PyHcclCommunicator as PyNcclCommunicator


logger = logging.getLogger(__name__)

# We use CUDA with multiprocessing, so we must use the 'spawn' start method.
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"


class WeightSyncWorkerExtension:
    """
    A vLLM worker extension that enables weight synchronization between a client and multiple server workers.
    """

    communicator = None
    client_rank = None

    def init_communicator(self, host: str, port: int, world_size: int, client_device_uuid: str) -> None:
        if self.communicator is not None:
            raise RuntimeError("Weight update group already initialized. Call close_communicator first.")

        if torch.cuda.is_available() or (
            is_torch_xpu_available() and hasattr(torch.xpu.get_device_properties(self.device), "uuid")
        ):
            accelerator_module = torch.xpu if is_torch_xpu_available() else torch.cuda
            if client_device_uuid == str(accelerator_module.get_device_properties(self.device).uuid):
                raise RuntimeError(
                    f"Attempting to use the same device (UUID: {client_device_uuid}) for multiple distinct "
                    "roles/ranks. Ensure trainer uses different devices than vLLM server."
                )
        
        rank = get_world_group().rank

        if is_torch_xpu_available():
            store = torch.distributed.TCPStore(host_name=host, port=port, world_size=world_size, is_master=(rank == 0))
            prefixed_store = c10d.PrefixStore("client2server", store)
            pg = c10d.ProcessGroupXCCL(
                store=prefixed_store,
                rank=rank,
                size=world_size,
            )
            self.communicator = pg
        else:
            pg = StatelessProcessGroup.create(host=host, port=port, rank=rank, world_size=world_size)
            self.communicator = PyNcclCommunicator(pg, device=self.device)

        self.client_rank = world_size - 1

    def update_named_param(self, name: str, dtype: str, shape: Sequence[int]) -> None:
        if self.communicator is None:
            raise RuntimeError("Communicator not initialized. Call `init_communicator` first.")

        dtype = getattr(torch, dtype.split(".")[-1])
        weight = torch.empty(shape, dtype=dtype, device=self.device)

        if is_torch_xpu_available():
            self.communicator.broadcast(weight, root=self.client_rank)
            self.communicator.barrier()
        else:
            self.communicator.broadcast(weight, src=self.client_rank)
            self.communicator.group.barrier()

        self.model_runner.model.load_weights(weights=[(name, weight)])

    def close_communicator(self) -> None:
        if self.communicator is not None:
            del self.communicator
            self.communicator = None
            self.client_rank = None


@dataclass
class ScriptArguments:
    """
    Arguments for the script.
    """
    model: str = field(
        metadata={"help": "Model name or path to load the model from."},
    )
    revision: str | None = field(
        default=None,
        metadata={"help": "Revision to use for the model. If not specified, the default branch will be used."},
    )
    tensor_parallel_size: int = field(
        default=1,
        metadata={"help": "Number of tensor parallel workers to use."},
    )
    data_parallel_size: int = field(
        default=1,
        metadata={"help": "Number of data parallel workers to use."},
    )
    host: str = field(
        default="0.0.0.0",
        metadata={"help": "Host address to run the server on."},
    )
    port: int = field(
        default=8000,
        metadata={"help": "Port to run the server on."},
    )
    gpu_memory_utilization: float = field(
        default=0.9,
        metadata={
            "help": "Ratio (between 0 and 1) of GPU memory to reserve for the model weights, activations, and KV "
            "cache."
        },
    )
    dtype: str = field(
        default="auto",
        metadata={
            "help": "Data type to use for vLLM generation."
        },
    )
    max_model_len: int | None = field(
        default=None,
        metadata={
            "help": "If set, the `max_model_len` to use for vLLM."
        },
    )
    enable_prefix_caching: bool | None = field(
        default=None,
        metadata={
            "help": "Whether to enable prefix caching in vLLM."
        },
    )
    enforce_eager: bool | None = field(
        default=False,
        metadata={
            "help": "Whether to enforce eager execution."
        },
    )
    kv_cache_dtype: str = field(
        default="auto",
        metadata={
            "help": "Data type to use for KV cache."
        },
    )
    trust_remote_code: bool = field(
        default=False,
        metadata={
            "help": "Whether to trust remote code when loading models."
        },
    )
    log_level: str = field(
        default="info",
        metadata={
            "help": "Log level for uvicorn."
        },
    )
    vllm_model_impl: str = field(
        default="vllm",
        metadata={
            "help": "Model implementation to use for vLLM. Must be one of `transformers` or `vllm`."
        },
    )


def llm_worker(
    script_args: ScriptArguments, data_parallel_rank: int, master_port: int, connection: Connection
) -> None:
    # Set required environment variables for DP to work with vLLM
    os.environ["VLLM_DP_RANK"] = str(data_parallel_rank)
    os.environ["VLLM_DP_RANK_LOCAL"] = str(data_parallel_rank)
    os.environ["VLLM_DP_SIZE"] = str(script_args.data_parallel_size)
    os.environ["VLLM_DP_MASTER_PORT"] = str(master_port)

    # Determine the module path dynamically to allow import in spawned workers
    current_module_name = os.path.splitext(os.path.basename(__file__))[0]
    worker_extension_name = f"{current_module_name}.WeightSyncWorkerExtension"

    llm = LLM(
        model=script_args.model,
        revision=script_args.revision,
        tensor_parallel_size=script_args.tensor_parallel_size,
        gpu_memory_utilization=script_args.gpu_memory_utilization,
        enforce_eager=script_args.enforce_eager,
        dtype=script_args.dtype,
        enable_prefix_caching=script_args.enable_prefix_caching,
        kv_cache_dtype=script_args.kv_cache_dtype,
        max_model_len=script_args.max_model_len,
        worker_extension_cls=worker_extension_name,
        trust_remote_code=script_args.trust_remote_code,
        model_impl=script_args.vllm_model_impl,
    )

    connection.send({"status": "ready"})

    while True:
        try:
            command = connection.recv()
        except KeyboardInterrupt:
            llm.collective_rpc(method="close_communicator")
            break

        if command["type"] in ["call", "fire_and_forget"]:
            method_name = command["method"]
            args, kwargs = command.get("args", ()), command.get("kwargs", {})
            method = getattr(llm, method_name)
            result = method(*args, **kwargs)
            if command["type"] == "call":
                connection.send(result)
        elif command["type"] == "shutdown":
            break


def chunk_list(lst: list, n: int) -> list[list]:
    k, r = divmod(len(lst), n)
    return [lst[i * k + min(i, r) : (i + 1) * k + min(i + 1, r)] for i in range(n)]


def sanitize_logprob(logprob):
    import math
    value = logprob.logprob
    if math.isnan(value):
        logger.warning(f"Generated NaN logprob, token logprob '{logprob}' will be ignored")
        return None
    return value


def main(script_args: ScriptArguments):
    if not is_fastapi_available():
        raise ImportError("FastAPI is required. Please install it using `pip install fastapi`.")
    if not is_pydantic_available():
        raise ImportError("Pydantic is required. Please install it using `pip install pydantic`.")
    if not is_uvicorn_available():
        raise ImportError("Uvicorn is required. Please install it using `pip install uvicorn`.")
    if not is_vllm_available():
        raise ImportError("vLLM is required. Please install it using `pip install vllm`.")

    master_port = get_open_port()
    connections = []
    processes = []
    for data_parallel_rank in range(script_args.data_parallel_size):
        parent_connection, child_connection = Pipe()
        process = Process(target=llm_worker, args=(script_args, data_parallel_rank, master_port, child_connection))
        process.start()
        connections.append(parent_connection)
        processes.append(process)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        ready_connections = set()
        while len(ready_connections) < script_args.data_parallel_size:
            for connection in connections:
                msg = connection.recv()
                if isinstance(msg, dict) and msg.get("status") == "ready":
                    ready_connections.add(connection)
        yield
        for process in processes:
            process.join(timeout=10)
            if process.is_alive():
                logger.warning(f"Process {process} is still alive, attempting to terminate...")
                process.terminate()
                process.join()

    app = FastAPI(lifespan=lifespan)

    @app.get("/health/")
    async def health():
        return {"status": "ok"}

    @app.get("/get_world_size/")
    async def get_world_size():
        return {"world_size": script_args.tensor_parallel_size * script_args.data_parallel_size}

    class GenerateRequest(BaseModel):
        prompts: list[str]
        images: list[str] | None = None
        n: int = 1
        repetition_penalty: float = 1.0
        temperature: float = 1.0
        top_p: float = 1.0
        top_k: int = -1
        min_p: float = 0.0
        max_tokens: int = 16
        truncate_prompt_tokens: int | None = None
        guided_decoding_regex: str | None = None
        generation_kwargs: dict = field(default_factory=dict)

    class GenerateResponse(BaseModel):
        prompt_ids: list[list[int]]
        completion_ids: list[list[int]]
        logprobs: list[list[float]]

    @app.post("/generate/", response_model=GenerateResponse)
    async def generate(request: GenerateRequest):
        request.images = request.images or [None] * len(request.prompts)
        prompts = []
        for prompt, image in zip(request.prompts, request.images, strict=True):
            row = {"prompt": prompt}
            if image is not None:
                row["multi_modal_data"] = {"image": Image.open(BytesIO(base64.b64decode(image)))}
            prompts.append(row)

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
            "logprobs": 0,
        }
        generation_kwargs.update(request.generation_kwargs)
        sampling_params = SamplingParams(**generation_kwargs)

        chunked_prompts = chunk_list(prompts, script_args.data_parallel_size)

        for connection, prompts in zip(connections, chunked_prompts, strict=True):
            if not prompts:
                prompts = ["<placeholder>"]
            kwargs = {"prompts": prompts, "sampling_params": sampling_params}
            connection.send({"type": "call", "method": "generate", "kwargs": kwargs})

        all_outputs = [connection.recv() for connection in connections]
        all_outputs = [output for output, prompts in zip(all_outputs, chunked_prompts, strict=True) if prompts]
        all_outputs = list(chain.from_iterable(all_outputs))
        prompt_ids = [output.prompt_token_ids for output in all_outputs]
        completion_ids = [list(output.token_ids) for outputs in all_outputs for output in outputs.outputs]
        logprobs = [
            [sanitize_logprob(next(iter(logprob.values()))) for logprob in output.logprobs]
            for outputs in all_outputs
            for output in outputs.outputs
        ]
        return {"prompt_ids": prompt_ids, "completion_ids": completion_ids, "logprobs": logprobs}

    class ChatRequest(BaseModel):
        messages: list[list[dict]]
        n: int = 1
        repetition_penalty: float = 1.0
        temperature: float = 1.0
        top_p: float = 1.0
        top_k: int = -1
        min_p: float = 0.0
        max_tokens: int = 16
        truncate_prompt_tokens: int | None = None
        guided_decoding_regex: str | None = None
        generation_kwargs: dict = field(default_factory=dict)
        chat_template_kwargs: dict = field(default_factory=dict)

    class ChatResponse(BaseModel):
        prompt_ids: list[list[int]]
        completion_ids: list[list[int]]
        logprobs: list[list[float]]

    @app.post("/chat/", response_model=ChatResponse)
    async def chat(request: ChatRequest):
        for message_list in request.messages:
            for message in message_list:
                if isinstance(message["content"], list):
                    for part in message["content"]:
                        if part["type"] == "image_pil":
                            part["image_pil"] = Image.open(BytesIO(base64.b64decode(part["image_pil"])))

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
            "logprobs": 0,
        }
        generation_kwargs.update(request.generation_kwargs)
        sampling_params = SamplingParams(**generation_kwargs)

        chunked_messages = chunk_list(request.messages, script_args.data_parallel_size)

        for connection, messages in zip(connections, chunked_messages, strict=True):
            if not messages:
                messages = [[{"role": "user", "content": "<placeholder>"}]]
            kwargs = {
                "messages": messages,
                "sampling_params": sampling_params,
                "chat_template_kwargs": request.chat_template_kwargs,
            }
            connection.send({"type": "call", "method": "chat", "kwargs": kwargs})

        all_outputs = [connection.recv() for connection in connections]
        all_outputs = [output for output, prompts in zip(all_outputs, chunked_messages, strict=True) if prompts]
        all_outputs = list(chain.from_iterable(all_outputs))
        prompt_ids = [output.prompt_token_ids for output in all_outputs]
        completion_ids = [list(output.token_ids) for outputs in all_outputs for output in outputs.outputs]
        logprobs = [
            [sanitize_logprob(next(iter(logprob.values()))) for logprob in output.logprobs]
            for outputs in all_outputs
            for output in outputs.outputs
        ]
        return {"prompt_ids": prompt_ids, "completion_ids": completion_ids, "logprobs": logprobs}

    class InitCommunicatorRequest(BaseModel):
        host: str
        port: int
        world_size: int
        client_device_uuid: str

    @app.post("/init_communicator/")
    async def init_communicator(request: InitCommunicatorRequest):
        world_size = script_args.tensor_parallel_size * script_args.data_parallel_size + 1
        kwargs = {
            "method": "init_communicator",
            "args": (request.host, request.port, world_size, request.client_device_uuid),
        }
        for connection in connections:
            connection.send({"type": "fire_and_forget", "method": "collective_rpc", "kwargs": kwargs})
        return {"message": "Request received, initializing communicator"}

    class UpdateWeightsRequest(BaseModel):
        name: str
        dtype: str
        shape: list[int]

    @app.post("/update_named_param/")
    async def update_named_param(request: UpdateWeightsRequest):
        kwargs = {"method": "update_named_param", "args": (request.name, request.dtype, tuple(request.shape))}
        print("Received update_named_param request for:", request.name)
        for connection in connections:
            connection.send({"type": "fire_and_forget", "method": "collective_rpc", "kwargs": kwargs})
        print("Sent update_named_param to all connections for:", request.name)
        return {"message": "Request received, updating named parameter"}

    @app.post("/reset_prefix_cache/")
    async def reset_prefix_cache():
        for connection in connections:
            connection.send({"type": "call", "method": "reset_prefix_cache"})
        all_outputs = [connection.recv() for connection in connections]
        success = all(output for output in all_outputs)
        return {"message": "Request received, resetting prefix cache status: " + str(success)}

    @app.post("/close_communicator/")
    async def close_communicator():
        kwargs = {"method": "close_communicator"}
        for connection in connections:
            connection.send({"type": "fire_and_forget", "method": "collective_rpc", "kwargs": kwargs})
        return {"message": "Request received, closing communicator"}

    uvicorn.run(app, host=script_args.host, port=script_args.port, log_level=script_args.log_level)


def make_parser(subparsers: argparse._SubParsersAction | None = None):
    # Replaced TrlParser with HfArgumentParser
    if subparsers is not None:
        parser = subparsers.add_parser("vllm-serve", help="Run the vLLM serve script", dataclass_types=[ScriptArguments])
        # HfArgumentParser requires dataclass_types to be a list/tuple or passed to parse_args_into_dataclasses directly
        # But subparsers.add_parser returns a standard ArgumentParser. 
        # Since HfArgumentParser wraps argparse, if we use subparsers, we rely on standard argparse logic usually.
        # However, to be fully compatible with HfArgumentParser logic below:
        return parser
    else:
        parser = HfArgumentParser(ScriptArguments)
    return parser


if __name__ == "__main__":
    parser = make_parser()
    # HfArgumentParser.parse_args_and_config() handles config files and CLI args
    (script_args,) = parser.parse_args_into_dataclasses()
    main(script_args)