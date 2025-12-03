import atexit
import logging
import socket
import time
from urllib.parse import urlparse

import requests
import torch
from torch import nn

from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
from vllm.distributed.utils import StatelessProcessGroup

logger = logging.getLogger(__name__)


class VLLMClient:
    """Client for custom vLLM RLVR server (generate + weight updates)."""

    def __init__(
        self,
        model: str,
        base_url: str | None = None,
        host: str = "0.0.0.0",
        server_port: int = 8000,
        group_port: int = 51216,
        connection_timeout: float = 60.0,
    ):
        self.session = requests.Session()
        self.model_name = model
        self.group_port = group_port

        # communicator-related state
        self.communicator: PyNcclCommunicator | None = None
        self.rank: int | None = None

        if base_url is not None:
            parsed_url = urlparse(base_url)
            # Resolve hostname to IP for the NCCL rendezvous host later
            self.host = socket.gethostbyname(parsed_url.hostname)
            scheme = parsed_url.scheme or "http"
            self.base_url = f"{scheme}://{parsed_url.netloc}{parsed_url.path}"
        else:
            self.host = host
            self.base_url = f"http://{self.host}:{server_port}"

        self.check_server(total_timeout=connection_timeout)

    # ---------- health check ----------

    def check_server(self, total_timeout: float = 60.0, retry_interval: float = 5.0):
        """Block until server /health/ is reachable or timeout."""
        url = f"{self.base_url}/health/"
        start_time = time.time()

        while True:
            try:
                response = self.session.get(url, timeout=5)
                if response.status_code == 200:
                    logger.info("vLLM server is up!")
                    return
            except requests.exceptions.RequestException as exc:
                elapsed_time = time.time() - start_time
                if elapsed_time >= total_timeout:
                    raise ConnectionError(
                        f"Can't reach vLLM server at {self.base_url} after {total_timeout}s"
                    ) from exc

            logger.info(f"Server not ready. Retrying in {retry_interval}s...")
            time.sleep(retry_interval)

    # ---------- generation ----------

    def generate(
        self,
        prompts: str | list[str],
        n: int = 1,
        temperature: float = 1.0,
        top_p: float = 1.0,
        max_tokens: int = 16,
        repetition_penalty: float = 1.0,
        top_k: int = -1,
        min_p: float = 0.0,
        truncate_prompt_tokens: int | None = None,
        **generation_kwargs,
    ) -> dict:
        """Call custom /generate/ endpoint on the vLLM server."""
        url = f"{self.base_url}/generate/"

        if isinstance(prompts, str):
            prompts = [prompts]

        payload = {
            "prompts": prompts,
            "n": n,
            "repetition_penalty": repetition_penalty,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "min_p": min_p,
            "max_tokens": max_tokens,
            "truncate_prompt_tokens": truncate_prompt_tokens,
            "generation_kwargs": generation_kwargs,
        }

        response = self.session.post(url, json=payload, timeout=120)
        if response.status_code != 200:
            raise Exception(f"Request failed: {response.status_code}, {response.text}")
        return response.json()

    # ---------- weight update communicator ----------

    def init_communicator(self, device: torch.device | str | int = 0):
        """
        Initialize the NCCL communicator for weight synchronization.

        device: trainer main-process CUDA device (e.g. 0, 'cuda:0', torch.device('cuda:0')).
        """

    
        # 1) Get vLLM engine world size (TP * DP = TP here)
        url = f"{self.base_url}/get_world_size/"
        response = self.session.get(url, timeout=5)
        if response.status_code != 200:
            raise Exception(f"Request failed: {response.status_code}, {response.text}")
        vllm_world_size = response.json()["world_size"]

        # Client is the last rank
        world_size = vllm_world_size + 1
        self.rank = vllm_world_size  # client rank

        # 2) Normalize device
        if isinstance(device, int):
            device = torch.device(f"cuda:{device}")
        elif isinstance(device, str):
            device = torch.device(device)
        assert device.type == "cuda", "init_communicator currently assumes CUDA."

        # 3) Get client device UUID
        client_device_uuid = str(torch.cuda.get_device_properties(device).uuid)

        # 4) Ask server to initialize its side of the communicator
        url = f"{self.base_url}/init_communicator/"
        resp = self.session.post(
            url,
            json={
                "host": "0.0.0.0",  # server side uses this for its TCPStore / ProcessGroup
                "port": self.group_port,
                "world_size": world_size,
                "client_device_uuid": client_device_uuid,
            },
            timeout=10,
        )
        if resp.status_code != 200:
            raise Exception(f"Request failed: {resp.status_code}, {resp.text}")

        # small sleep to avoid ugly NCCL logs complaining about racey connect
        time.sleep(0.1)

        # 5) Build client-side StatelessProcessGroup + PyNcclCommunicator
        #    Note: TRL uses self.host here (resolved from base_url), not "0.0.0.0".
        pg = StatelessProcessGroup.create(
            host=self.host,
            port=self.group_port,
            rank=self.rank,
            world_size=world_size,
        )
        self.communicator = PyNcclCommunicator(pg, device=device)

        # auto-close on interpreter exit
        atexit.register(self.close_communicator)
        logger.info(f"Initialized weight-sync communicator with rank {self.rank}/{world_size}")

    def update_named_param(self, name: str, weights: torch.Tensor):
        """
        Update a specific named parameter in the vLLM model.

        The protocol:
          1. POST /update_named_param/ with name, dtype, shape
          2. vLLM workers allocate tensors and call communicator.broadcast(...)
          3. Client does communicator.broadcast(weights, src=self.rank)
        """
        if self.communicator is None:
            raise RuntimeError("Communicator not initialized. Call init_communicator() first.")

        dtype, shape = str(weights.dtype), tuple(weights.shape)
        url = f"{self.base_url}/update_named_param/"
        resp = self.session.post(
            url,
            json={"name": name, "dtype": dtype, "shape": list(shape)},
            timeout=10,
        )
        if resp.status_code != 200:
            raise Exception(f"Request failed: {resp.status_code}, {resp.text}")

        # NCCL broadcast: client is src, all engine ranks are receivers
        self.communicator.broadcast(weights, src=self.rank)
        self.communicator.group.barrier()

    def update_model_params(self, model: nn.Module):
        """
        Update all parameters of a local nn.Module in the vLLM server.

        Assumes param names match server-side `load_weights` names or are mapped
        appropriately in your WeightSyncWorkerExtension.
        """
        if self.communicator is None:
            raise RuntimeError("Communicator not initialized. Call init_communicator() first.")

        logger.info("Pushing model parameters to vLLM server...")
        for name, param in model.named_parameters():
            self.update_named_param(name, param.data)
        logger.info("Finished pushing model parameters.")

    def close_communicator(self):
        """Tell server to close communicator and drop our NCCL group."""
        if self.communicator is None:
            return

        try:
            url = f"{self.base_url}/close_communicator/"
            resp = self.session.post(url, timeout=5)
            if resp.status_code != 200:
                logger.warning(f"close_communicator server error: {resp.status_code}, {resp.text}")
        except requests.ConnectionError:
            # server might already be gone; ignore
            pass

        self.communicator = None
        logger.info("Weight-sync communicator closed.")
