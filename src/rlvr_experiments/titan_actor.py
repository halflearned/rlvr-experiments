import os
import ray
import torch

from typing import Any, Dict, List, Tuple

from torch.distributed.tensor import DTensor, distribute_tensor

from .weight_sync import WeightSyncManager
from .syncing import ChunkMeta, ParamMeta

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@ray.remote(num_gpus=1, num_cpus=2)
class TitanModelRank:
    def __init__(self, rank: int, world_size: int, config_path: str, group_name: str = "default", trainable: bool = True) -> None:
        self.rank = rank
        self.world_size = world_size
        self.group_name = group_name
        self.config_path = config_path
        self.trainable = trainable # TODO: improve this handling later

        self.model = None
        self.sync_managers: Dict[str, WeightSyncManager] = {}
        self._sync_cache: Dict[str, torch.Tensor] | None = None

        # Titan world env
        os.environ["RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["LOCAL_RANK"] = "0"  # okay since 1 actor per gpu

    # ------------------------------------------------------------------ #
    # Titan init
    # ------------------------------------------------------------------ #
    def initialize_process_group(self, master_addr: str, master_port: str) -> Dict[str, Any]:
        os.environ["MASTER_ADDR"] = master_addr
        os.environ["MASTER_PORT"] = master_port

        from .model import TitanModel
        from torchtitan.config import ConfigManager

        job_config = ConfigManager().parse_args(["--job.config-file", self.config_path])
        self.model = TitanModel(job_config, trainable=self.trainable)

        print(f"[{self.group_name} Rank {self.rank}] Initialized Titan on port {master_port}")
        return {"status": "ready"}

    # ------------------------------------------------------------------ #
    # Weight-sync PG
    # ------------------------------------------------------------------ #
    def add_sync_channel(self, channel_name: str, host: str, port: int, world_size: int, my_rank: int) -> None:
        if channel_name in self.sync_managers:
            return
        manager = WeightSyncManager()
        manager.init_communicator(
            host=host,
            port=port,
            world_size=world_size,
            my_rank=my_rank,
            device=self.model.device,
        )
        self.sync_managers[channel_name] = manager
        print(f"[{self.group_name} Rank {self.rank}] Joined channel '{channel_name}' as sync_rank={my_rank}")

    # ------------------------------------------------------------------ #
    # HF metadata helpers
    # ------------------------------------------------------------------ #
    def get_hf_param_names_and_shapes(self) -> List[Tuple[str, str, Tuple[int, ...]]]:
        """Collective in Titan world: used only for metadata."""
        if self.model is None:
            raise RuntimeError("Model not initialized")
        hf_state = self.model.hf_state_dict()
        return [
            (k, str(v.dtype).split(".")[-1], tuple(v.shape))
            for k, v in hf_state.items()
        ]

    # ------------------------------------------------------------------ #
    # Sync cache + chunk planning
    # ------------------------------------------------------------------ #
    def prepare_sync_state(self) -> None:
        """
        Prepare HF state once. This will:
          * run the Titan/DCP collectives,
          * give each Titan rank a local view of HF tensors.
        We keep the dict per-rank to keep DTensor semantics simple.
        """
        if self.model is None:
            raise RuntimeError("Model not initialized")
        self._sync_cache = self.model.hf_state_dict()
        print(f"[{self.group_name} Rank {self.rank}] Prepared sync cache with {len(self._sync_cache)} entries.")

    def clear_sync_state(self) -> None:
        self._sync_cache = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def build_chunk_plan(
        self,
        max_chunk_elems: int,
    ) -> List[ChunkMeta]:
        """
        Build a chunk plan from the HF cache.

        This is *metadata only* (no NCCL, no device copies). It can be called
        on a single rank (e.g., trainer rank 0) and the result shared via Ray.
        """
        if self._sync_cache is None:
            raise RuntimeError("Sync state not prepared.")

        chunks: List[ChunkMeta] = []
        current_params: List[ParamMeta] = []
        current_total = 0

        # Deterministic order
        for name, tensor in self._sync_cache.items():
            # NOTE: tensor may be DTensor or plain Tensor; .numel() and .shape
            # are global for DTensor as well.
            numel = tensor.numel()
            shape = tuple(tensor.shape)

            if current_params and current_total + numel > max_chunk_elems:
                chunks.append(ChunkMeta(total_numel=current_total, params=current_params))
                current_params = []
                current_total = 0

            current_params.append(ParamMeta(name=name, numel=numel, shape=shape))
            current_total += numel

        if current_params:
            chunks.append(ChunkMeta(total_numel=current_total, params=current_params))

        print(f"[{self.group_name} Rank {self.rank}] Built chunk plan with {len(chunks)} chunks.")
        return chunks

    # ------------------------------------------------------------------ #
    # Chunk broadcast (Titan side)
    # ------------------------------------------------------------------ #
    def broadcast_chunk(
        self,
        channel: str,
        chunk,         # ChunkMeta or dict
        dtype_str: str,
        src_rank: int,
    ) -> None:
        if self._sync_cache is None:
            raise RuntimeError("Sync cache not prepared.")
        if channel not in self.sync_managers:
            raise ValueError(f"Channel '{channel}' not found.")

        manager = self.sync_managers[channel]
        my_sync_rank = manager.sync_group_rank
        if my_sync_rank is None:
            raise RuntimeError("WeightSyncManager has no sync_group_rank.")

        is_src = (my_sync_rank == src_rank)
        dtype = getattr(torch, dtype_str)
        device = self.model.device

        # Normalize chunk
        if hasattr(chunk, "total_numel") and hasattr(chunk, "params"):
            total_numel = int(chunk.total_numel)
            params_iter = chunk.params
        else:
            total_numel = int(chunk["total_numel"])
            params_iter = chunk["params"]

        def iter_params():
            for p in params_iter:
                if hasattr(p, "name"):
                    yield p.name, int(p.numel), tuple(p.shape)
                else:
                    yield p["name"], int(p["numel"]), tuple(p["shape"])

        # 1. Collect full tensors (all Titan ranks participate)
        local_full_tensors = []
        for name, numel, shape in iter_params():
            t = self._sync_cache[name]
            if isinstance(t, DTensor):
                t_full = t.full_tensor()
            else:
                t_full = t
            local_full_tensors.append((name, numel, shape, t_full))

        # 2. Allocate flat buffer
        flat = torch.empty(total_numel, dtype=dtype, device=device)

        # Only src copies real data; others' contents will be overwritten by NCCL.
        if is_src:
            offset = 0
            for name, numel, shape, t_full in local_full_tensors:
                buf = t_full.to(device=device, dtype=dtype).contiguous().view(-1)
                assert buf.numel() == numel, f"Numel mismatch for {name}"
                flat[offset : offset + numel].copy_(buf)
                offset += numel

        manager.communicator.broadcast(flat, src=src_rank)

        del flat
        del local_full_tensors


    # ------------------------------------------------------------------ #
    # Chunk receive (Titan reference side)
    # ------------------------------------------------------------------ #
    def recv_chunk_from_hf(
        self,
        channel: str,
        chunk: ChunkMeta,
        dtype_str: str,
        src_rank: int,
    ) -> None:
        """
        Called on *all* reference Titan ranks in the channel.
        - Participates in NCCL broadcast of the flat buffer.
        - Slices/reconstructs HF tensors.
        - Uses sd_adapter.from_hf to translate HF → Titan internas, then
          loads into parameters (handling DTensors via distribute_tensor).
        """
        if self.model is None:
            raise RuntimeError("Model not initialized.")
        if channel not in self.sync_managers:
            raise ValueError(f"Channel '{channel}' not found.")

        manager = self.sync_managers[channel]
        dtype = getattr(torch, dtype_str)
        device = self.model.device

        # 1. Receive flat buffer via NCCL.
        flat = torch.empty(chunk.total_numel, dtype=dtype, device=device)
        manager.communicator.broadcast(flat, src=src_rank)
        torch.cuda.current_stream().synchronize()

        # 2. Slice and load.
        with torch.no_grad():
            offset = 0
            model_params = dict(self.model.model_parts[0].named_parameters())

            for pm in chunk.params:
                flat_slice = flat[offset : offset + pm.numel]
                full_tensor = flat_slice.view(pm.shape)
                offset += pm.numel

                # HF name is pm.name
                titan_state = self.model.sd_adapter.from_hf({pm.name: full_tensor})
                for titan_name, incoming_val in titan_state.items():
                    if titan_name not in model_params:
                        continue
                    target = model_params[titan_name]
                    if isinstance(target, DTensor):
                        sharded_src = distribute_tensor(
                            incoming_val,
                            target.device_mesh,
                            target.placements,
                        )
                        target.copy_(sharded_src)
                    else:
                        target.copy_(incoming_val.to(device=target.device, dtype=target.dtype))

        del flat


import asyncio

class DistributedModelHandle:
    _simple_attrs = {"tokenizer", "device", "step", "job_config", "model_args"}

    def __init__(self, actors: List, name: str = "model"):
        self.actors = actors
        self.name = name

    async def resolve(self, ref):
        if hasattr(ref, "__await__"):
            return await ref
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, ray.get, ref)

    async def get_hf_param_info(self):
        futures = [a.get_hf_param_names_and_shapes.remote() for a in self.actors]
        results = await asyncio.gather(*futures)
        return results[0]


import tempfile
import tomli_w as toml

def create_titan_group(config: dict, name: str, world_size: int, port: int) -> DistributedModelHandle:
    master_addr = ray.util.get_node_ip_address()

    cfg = dict(config)
    cfg.pop("trainable", None) 

    with tempfile.NamedTemporaryFile(mode="wb", suffix=".toml", delete=False) as f:
        toml.dump(cfg, f)
        config_path = f.name

    try:
        logger.info(f"Creating Titan group '{name}' with world_size={world_size} on {master_addr}:{port}")
        actors = [
            TitanModelRank.options(num_gpus=1, num_cpus=2).remote(
                rank=r,
                world_size=world_size,
                config_path=config_path,
                group_name=name,
            )
            for r in range(world_size)
        ]

        ray.get([a.initialize_process_group.remote(master_addr, str(port)) for a in actors])
        return DistributedModelHandle(actors, name=name)

    finally:
        try:
            os.remove(config_path)
        except FileNotFoundError:
            pass