import os
import ray
import torch

from typing import Any, Dict, List, Callable

from torch.distributed.tensor import DTensor, distribute_tensor

from .weight_sync import WeightSyncManager
from .ops import compute_logprobs

import asyncio
import logging
import tempfile

import tomli_w as toml

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@ray.remote(num_gpus=1, num_cpus=2)
class TitanModelRank:
    def __init__(
        self,
        rank: int,
        world_size: int,
        config_path: str,
        master_addr: str,
        master_port: str,
        group_name: str = "default",
        trainable: bool = True,
    ) -> None:
        self.rank = rank
        self.world_size = world_size
        self.group_name = group_name
        self.sync_managers: Dict[str, WeightSyncManager] = {}
        self._sync_cache: Dict[str, torch.Tensor] | None = None

        os.environ["RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["LOCAL_RANK"] = "0"
        os.environ["MASTER_ADDR"] = master_addr
        os.environ["MASTER_PORT"] = master_port

        from .model import TitanModel
        from torchtitan.config import ConfigManager

        job_config = ConfigManager().parse_args(["--job.config-file", config_path])
        self.model = TitanModel(job_config, trainable=trainable)
        logger.info(f"{group_name} Rank {rank}: Initialized Titan")

    def add_sync_channel(self, channel_name: str, host: str, port: int, world_size: int, rank: int) -> None:
        if channel_name in self.sync_managers:
            return
        self.sync_managers[channel_name] = WeightSyncManager(
            host=host, port=port, world_size=world_size, rank=rank, device=self.model.device
        )

    def call_method(self, attr: str, *args, **kwargs):
        """Dispatch a method call (used by DistributedModelHandle)."""
        fn = getattr(self, attr, None) or getattr(self.model, attr)
        return fn(*args, **kwargs)

    def prepare_sync_state(self) -> None:
        """Cache HF state dict (involves DTensor collectives, all ranks must call)."""
        self._sync_cache = self.model.hf_state_dict()

    def clear_sync_state(self) -> None:
        self._sync_cache = None
        torch.cuda.empty_cache()

    def build_chunk_plan(self, max_chunk_elems: int) -> list[dict]:
        """Build a chunk plan from cached HF state (metadata only, single rank)."""
        chunks = []
        current_params = []
        current_total = 0

        for name, tensor in self._sync_cache.items():
            numel = tensor.numel()
            shape = tuple(tensor.shape)

            if current_params and current_total + numel > max_chunk_elems:
                chunks.append({"total_numel": current_total, "params": current_params})
                current_params = []
                current_total = 0

            current_params.append({"name": name, "numel": numel, "shape": shape})
            current_total += numel

        if current_params:
            chunks.append({"total_numel": current_total, "params": current_params})
        return chunks

    def broadcast_chunk(self, channel: str, chunk: dict, dtype_str: str, src_rank: int) -> None:
        """Broadcast a chunk to receivers (all ranks must call)."""
        manager = self.sync_managers[channel]
        is_src = manager.sync_group_rank == src_rank
        dtype = getattr(torch, dtype_str)
        device = self.model.device

        # Collect full tensors (all Titan ranks participate in DTensor collectives)
        tensors = []
        for p in chunk["params"]:
            t = self._sync_cache[p["name"]]
            t_full = t.full_tensor() if isinstance(t, DTensor) else t
            tensors.append((p["name"], p["numel"], t_full))

        # Pack into flat buffer (only src fills with real data)
        flat = torch.empty(chunk["total_numel"], dtype=dtype, device=device)
        if is_src:
            offset = 0
            for name, numel, t_full in tensors:
                buf = t_full.to(device=device, dtype=dtype).contiguous().view(-1)
                flat[offset : offset + numel].copy_(buf)
                offset += numel

        manager.communicator.broadcast(flat, src=src_rank)

    def recv_chunk(self, channel: str, chunk: dict, dtype_str: str, src_rank: int) -> None:
        """Receive a chunk and load into model parameters."""
        manager = self.sync_managers[channel]
        dtype = getattr(torch, dtype_str)
        device = self.model.device

        flat = torch.empty(chunk["total_numel"], dtype=dtype, device=device)
        manager.communicator.broadcast(flat, src=src_rank)

        with torch.no_grad():
            model_params = dict(self.model.model_parts[0].named_parameters())
            offset = 0

            for p in chunk["params"]:
                full_tensor = flat[offset : offset + p["numel"]].view(p["shape"])
                offset += p["numel"]

                titan_state = self.model.sd_adapter.from_hf({p["name"]: full_tensor})
                for titan_name, incoming_val in titan_state.items():
                    if titan_name not in model_params:
                        continue
                    target = model_params[titan_name]
                    if isinstance(target, DTensor):
                        sharded = distribute_tensor(incoming_val, target.device_mesh, target.placements)
                        target.copy_(sharded)
                    else:
                        target.copy_(incoming_val.to(device=target.device, dtype=target.dtype))

    def compute_logprobs(self, input_ids: torch.Tensor, completion_ids: torch.Tensor) -> torch.Tensor | None:
        """Forward pass returning logprobs (only rank 0 returns, others return None)."""
        logits = self.model.forward(input_ids)
        if isinstance(logits, DTensor):
            logits = logits.full_tensor()

        completion_ids = completion_ids.to(self.model.device)
        logprobs = compute_logprobs(logits, completion_ids)
        return logprobs.detach().cpu() if self.rank == 0 else None

    def forward_backward(
        self,
        loss_fn: Callable[..., torch.Tensor],
        input_ids: torch.Tensor,
        loss_args: tuple = (),
        loss_kwargs: dict | None = None,
    ) -> float:
        """Forward pass, compute loss, and backward pass. Returns loss value."""
        logits = self.model.forward(input_ids)
        device = self.model.device

        def to_device(v: Any) -> Any:
            if torch.is_tensor(v) and not isinstance(v, DTensor):
                return v.to(device, non_blocking=True)
            return v

        args_local = tuple(to_device(a) for a in loss_args)
        kwargs_local = {k: to_device(v) for k, v in (loss_kwargs or {}).items()}

        with self.model.train_context_mgr(None):
            loss = loss_fn(logits, *args_local, **kwargs_local)

            if not torch.is_tensor(loss) or loss.ndim != 0:
                raise RuntimeError("loss_fn must return a scalar torch.Tensor")

            loss.backward()

        return float(loss.detach().item())


class DistributedModelHandle:
    """
    Client-side handle for a distributed Titan model.

    Dispatches method calls to all actor ranks in parallel (required for collectives),
    returns rank 0's result.
    """

    def __init__(self, actors: List, name: str = "model"):
        self.actors = actors
        self.name = name

    def __getattr__(self, attr):
        if attr.startswith("_"):
            raise AttributeError(attr)

        async def proxy(*args, **kwargs):
            refs = [a.call_method.remote(attr, *args, **kwargs) for a in self.actors]
            results = await asyncio.gather(*[self._resolve(r) for r in refs])
            return results[0]

        return proxy

    async def _resolve(self, ref):
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, ray.get, ref)


def create_titan_group(config: dict, name: str, world_size: int, port: int) -> DistributedModelHandle:
    master_addr = ray.util.get_node_ip_address()
    trainable = bool(config.get("trainable", True))

    cfg = dict(config)
    cfg.pop("trainable", None)

    # Ray actors may be on different nodes - use shared filesystem for config
    config_dir = os.environ.get("RLVR_TITAN_CONFIG_DIR", os.path.join(os.getcwd(), ".rlvr_titan_job_configs"))
    os.makedirs(config_dir, exist_ok=True)

    with tempfile.NamedTemporaryFile(mode="wb", suffix=f".{name}.toml", delete=False, dir=config_dir) as f:
        toml.dump(cfg, f)
        config_path = f.name

    logger.info(f"Creating Titan group '{name}' with world_size={world_size} on {master_addr}:{port}")
    actors = [
        TitanModelRank.options(num_gpus=1, num_cpus=2).remote(
            rank=r,
            world_size=world_size,
            config_path=config_path,
            master_addr=master_addr,
            master_port=str(port),
            group_name=name,
            trainable=trainable,
        )
        for r in range(world_size)
    ]
    ray.get([a.__ray_ready__.remote() for a in actors])

    if os.environ.get("RLVR_KEEP_TITAN_CONFIG") != "1":
        try:
            os.remove(config_path)
        except FileNotFoundError:
            pass

    return DistributedModelHandle(actors, name=name)
