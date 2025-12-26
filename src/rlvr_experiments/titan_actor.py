import os
import ray
import time
import torch

from typing import Any, Dict, List, Tuple, Callable

from torch.distributed.tensor import DTensor, distribute_tensor

from .weight_sync import WeightSyncManager
from .syncing import ChunkMeta, ParamMeta
from .ops import compute_logprobs
from .tracer import traced

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

        if self.rank == 0:
            p = job_config.parallelism
            tp = int(getattr(p, "tensor_parallel_degree", 1))
            dp_rep = int(getattr(p, "data_parallel_replicate_degree", 1))
            dp_shard = int(getattr(p, "data_parallel_shard_degree", 1))
            expected_ws = dp_rep * dp_shard * tp
            if expected_ws != int(os.environ["WORLD_SIZE"]):
                logger.warning(
                    f"{self.group_name}: config/world_size mismatch: "
                    f"expected_ws={expected_ws} (dp_rep={dp_rep}, dp_shard={dp_shard}, tp={tp}) "
                    f"but env WORLD_SIZE={os.environ['WORLD_SIZE']} config_path={self.config_path}"
                )
            else:
                logger.info(
                    f"{self.group_name}: Loaded config: "
                    f"dp_rep={dp_rep} dp_shard={dp_shard} tp={tp} "
                    f"(WORLD_SIZE={os.environ['WORLD_SIZE']}) config_path={self.config_path}"
                )

        self.model = TitanModel(job_config, trainable=self.trainable)

        if self.rank == 0:
            pd = getattr(self.model, "parallel_dims", None)
            logger.info(
                f"{self.group_name}: parallel_dims: "
                f"dp_rep={getattr(pd, 'dp_replicate', None)} "
                f"dp_shard={getattr(pd, 'dp_shard', None)} "
                f"tp={getattr(pd, 'tp', None)} "
                f"world_size={getattr(pd, 'world_size', None)}"
            )

        logger.info(f"{self.group_name} Rank {self.rank}: Initialized Titan on port {master_port}")
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
        logger.info(f"{self.group_name} Rank {self.rank}: Joined channel '{channel_name}' as sync_rank={my_rank}")

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
    # Generic attribute/method forwarding
    # ------------------------------------------------------------------ #
    def get_attr(self, attr: str):
        if self.model is None:
            raise RuntimeError("Model not initialized")
        return getattr(self.model, attr)

    def call_method(self, attr: str, *args, **kwargs):
        if self.model is None:
            raise RuntimeError("Model not initialized")
        if hasattr(self, attr):
            fn = getattr(self, attr)
        else:
            fn = getattr(self.model, attr)
        if not callable(fn):
            raise AttributeError(f"{attr} is not callable")
        return fn(*args, **kwargs)


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
        logger.debug(f"{self.group_name} Rank {self.rank}: prepare_sync_state starting...")
        if self.model is None:
            raise RuntimeError("Model not initialized")
        self._sync_cache = self.model.hf_state_dict()
        logger.debug(f"{self.group_name} Rank {self.rank}: prepare_sync_state done, {len(self._sync_cache)} entries")

    def clear_sync_state(self) -> None:
        logger.debug(f"{self.group_name} Rank {self.rank}: clear_sync_state...")
        self._sync_cache = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.debug(f"{self.group_name} Rank {self.rank}: clear_sync_state done")

    def build_chunk_plan(
        self,
        max_chunk_elems: int,
    ) -> List[ChunkMeta]:
        """
        Build a chunk plan from the HF cache.

        This is *metadata only* (no NCCL, no device copies). It can be called
        on a single rank (e.g., trainer rank 0) and the result shared via Ray.
        """
        logger.debug(f"{self.group_name} Rank {self.rank}: build_chunk_plan, max_chunk_elems={max_chunk_elems}")
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

        logger.debug(f"{self.group_name} Rank {self.rank}: build_chunk_plan done, {len(chunks)} chunks")
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
        logger.debug(f"{self.group_name} Rank {self.rank}: broadcast_chunk, channel={channel}")
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

        logger.debug(f"{self.group_name} Rank {self.rank}: broadcast_chunk NCCL broadcast...")
        manager.communicator.broadcast(flat, src=src_rank)
        logger.debug(f"{self.group_name} Rank {self.rank}: broadcast_chunk done")

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
        logger.debug(f"{self.group_name} Rank {self.rank}: recv_chunk_from_hf, channel={channel}")
        if self.model is None:
            raise RuntimeError("Model not initialized.")
        if channel not in self.sync_managers:
            raise ValueError(f"Channel '{channel}' not found.")

        manager = self.sync_managers[channel]
        dtype = getattr(torch, dtype_str)
        device = self.model.device

        # 1. Receive flat buffer via NCCL.
        logger.debug(f"{self.group_name} Rank {self.rank}: recv_chunk_from_hf NCCL broadcast...")
        flat = torch.empty(chunk.total_numel, dtype=dtype, device=device)
        manager.communicator.broadcast(flat, src=src_rank)
        torch.cuda.current_stream().synchronize()
        logger.debug(f"{self.group_name} Rank {self.rank}: recv_chunk_from_hf NCCL broadcast received")

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
        logger.debug(f"{self.group_name} Rank {self.rank}: recv_chunk_from_hf done")

    def forward_step(self, input_dict):
        """
        One name for all models.

        - Trainable models: return raw outputs (may include DTensors).
        - Non-trainable models: return a plain torch.Tensor materialized on rank 0.
          Other ranks return None. This method is collective if the output is DTensor.
        """
        logger.debug(f"{self.group_name} Rank {self.rank}: forward_step starting...")
        if self.model is None:
            raise RuntimeError("Model not initialized")

        torch.cuda.synchronize()
        t_start = time.perf_counter()
        logger.debug(f"{self.group_name} Rank {self.rank}: forward_step calling model.forward_step...")
        out = self.model.forward_step(input_dict)
        torch.cuda.synchronize()
        t_end = time.perf_counter()
        logger.debug(f"{self.group_name} Rank {self.rank}: forward_step GPU time {(t_end - t_start)*1000:.1f}ms")

        # What follows is necessary to handle the fact that different
        # Titan models may output DTensors that are on different meshes.
        # This can lead to issues when computing "across" meshes.
        # So we enforce that only trainable models return DTensors.
        # (And I suppose we assume there's only one trainable model)
        # Other models return plain tensors on rank 0 only.

        # Model is trainable: return DTensor output
        if self.trainable:
            return out

        # Non-trainable: export a plain tensor
        def materialize(value):
            if isinstance(value, DTensor):
                # Collective across this model's mesh, so all ranks must call forward().
                # Clone and move to CPU to ensure clean serialization through Ray
                return value.full_tensor().detach().clone().cpu()
            if torch.is_tensor(value):
                return value.detach().clone().cpu()
            return value

        plain_out = _tree_map(materialize, out)

        # Only leader returns to avoid shipping duplicates.
        if self.rank == 0:
            logger.debug(f"{self.group_name} Rank {self.rank}: forward_step done (returning plain_out)")
            return plain_out
        else:
            logger.debug(f"{self.group_name} Rank {self.rank}: forward_step done (returning None)")
            return None

    def compute_logprobs_step(self, input_dict: dict, completion_ids: torch.Tensor) -> torch.Tensor:
        """
        Forward pass + logprobs computation inside the actor.

        Returns only the small [B, T] logprobs tensor instead of full [B, seq_len, vocab] logits.
        This avoids serializing ~6GB of data through Ray for the reference model.
        """
        logger.debug(f"{self.group_name} Rank {self.rank}: compute_logprobs_step starting...")
        if self.model is None:
            raise RuntimeError("Model not initialized")

        torch.cuda.synchronize()
        t_start = time.perf_counter()

        # Forward pass
        logger.debug(f"{self.group_name} Rank {self.rank}: compute_logprobs_step forward...")
        logits = self.model.forward_step(input_dict)
        logger.debug(f"{self.group_name} Rank {self.rank}: compute_logprobs_step forward done")

        # Materialize DTensor if needed
        if isinstance(logits, DTensor):
            logger.debug(f"{self.group_name} Rank {self.rank}: compute_logprobs_step materializing DTensor...")
            logits = logits.full_tensor()

        # Compute logprobs inside the actor
        logger.debug(f"{self.group_name} Rank {self.rank}: compute_logprobs_step computing logprobs...")
        completion_ids = completion_ids.to(self.model.device)
        logprobs = compute_logprobs(logits, completion_ids)

        torch.cuda.synchronize()
        t_end = time.perf_counter()
        logger.debug(f"{self.group_name} Rank {self.rank}: compute_logprobs_step GPU time {(t_end - t_start)*1000:.1f}ms")

        # Return small tensor [B, T] instead of huge [B, seq_len, vocab]
        if self.rank == 0:
            logger.debug(f"{self.group_name} Rank {self.rank}: compute_logprobs_step done (returning logprobs)")
            return logprobs.detach().cpu()
        else:
            logger.debug(f"{self.group_name} Rank {self.rank}: compute_logprobs_step done (returning None)")
            return None

    def compute_loss_and_backward_step(
        self,
        loss_fn: Callable[..., torch.Tensor],
        trainer_output: Any,
        *args: Any,
        **kwargs: Any,
    ) -> float:
        logger.debug(f"{self.group_name} Rank {self.rank}: compute_loss_and_backward_step starting...")
        torch.cuda.synchronize()
        t_total_start = time.perf_counter()

        if isinstance(trainer_output, dict):
            logger.debug(f"{self.group_name} Rank {self.rank}: compute_loss_and_backward_step forward...")
            torch.cuda.synchronize()
            t_fwd_start = time.perf_counter()
            trainer_output = self.model.forward_step(trainer_output)
            torch.cuda.synchronize()
            t_fwd_end = time.perf_counter()
            logger.debug(f"{self.group_name} Rank {self.rank}: compute_loss_and_backward_step forward done, {(t_fwd_end - t_fwd_start)*1000:.1f}ms")

        device = self.model.device

        # Move tensors to device, keeping DTensors as-is (loss function will handle extraction)
        def move_to_device(value: Any, detach: bool = True) -> Any:
            if isinstance(value, DTensor):
                # Keep DTensor as-is, loss function will extract to local
                return value if not detach else value.detach()
            if torch.is_tensor(value):
                tensor = value.to(device, non_blocking=True)
                return tensor if not detach else tensor.detach()
            return value

        # Trainer output keeps gradients (DTensor), everything else is detached
        trainer_output_local = move_to_device(trainer_output, detach=False)
        args_local = tuple(move_to_device(arg, detach=True) for arg in args)
        kwargs_local = {k: move_to_device(v, detach=True) if torch.is_tensor(v) or isinstance(v, DTensor) else v
                       for k, v in kwargs.items()}

        # Wrap loss computation and backward in train_context to ensure proper
        # context is active throughout forward→loss→backward chain (matching torchtitan pattern)
        logger.debug(f"{self.group_name} Rank {self.rank}: compute_loss_and_backward_step computing loss...")
        torch.cuda.synchronize()
        t_loss_bwd_start = time.perf_counter()
        with self.model.train_context_mgr(None):
            loss = loss_fn(trainer_output_local, *args_local, **kwargs_local)

            if not torch.is_tensor(loss) or loss.ndim != 0:
                raise RuntimeError("loss_fn must return a scalar torch.Tensor")

            logger.debug(f"{self.group_name} Rank {self.rank}: compute_loss_and_backward_step backward...")
            loss.backward()
        torch.cuda.synchronize()
        t_loss_bwd_end = time.perf_counter()
        logger.debug(f"{self.group_name} Rank {self.rank}: compute_loss_and_backward_step backward done, loss+backward={(t_loss_bwd_end - t_loss_bwd_start)*1000:.1f}ms")

        torch.cuda.synchronize()
        t_total_end = time.perf_counter()
        logger.debug(f"{self.group_name} Rank {self.rank}: compute_loss_and_backward_step done, total={(t_total_end - t_total_start)*1000:.1f}ms")

        return float(loss.detach().item())

import asyncio

class DistributedModelHandle:
    _simple_attrs = {"tokenizer", "device", "step", "job_config", "model_args"}

    def __init__(self, actors: List, name: str = "model"):
        self.actors = actors
        self.name = name

    async def _call_all(self, attr, *args, **kwargs):
        logger.debug(f"DistributedModelHandle {self.name}: _call_all {attr} starting...")
        kwargs.pop("tracer", None)
        results = await asyncio.gather(
            *[self.resolve(a.call_method.remote(attr, *args, **kwargs)) for a in self.actors]
        )
        logger.debug(f"DistributedModelHandle {self.name}: _call_all {attr} done")
        return results[0]

    @traced("reference.compute_logprobs")
    async def compute_logprobs_step(self, input_dict: dict, completion_ids: torch.Tensor):
        logger.debug(f"DistributedModelHandle {self.name}: compute_logprobs_step starting...")
        result = await self._call_all("compute_logprobs_step", input_dict, completion_ids)
        logger.debug(f"DistributedModelHandle {self.name}: compute_logprobs_step done")
        return result

    @traced("trainer.forward_backward")
    async def compute_loss_and_backward_step(
        self,
        loss_fn: Callable[..., torch.Tensor],
        trainer_output: Any,
        *args: Any,
        **kwargs: Any,
    ) -> float:
        logger.debug(f"DistributedModelHandle {self.name}: compute_loss_and_backward_step starting...")
        result = await self._call_all(
            "compute_loss_and_backward_step",
            loss_fn,
            trainer_output,
            *args,
            **kwargs,
        )
        logger.debug(f"DistributedModelHandle {self.name}: compute_loss_and_backward_step done, loss={result}")
        return result

    @traced("trainer.optimizer_step")
    async def optimizer_step(self):
        logger.debug(f"DistributedModelHandle {self.name}: optimizer_step starting...")
        result = await self._call_all("optimizer_step")
        logger.debug(f"DistributedModelHandle {self.name}: optimizer_step done, grad_norm={result}")
        return result

    @traced("trainer.save_checkpoint")
    async def save_checkpoint(self, step: int | None = None, last_step: bool = False):
        return await self._call_all("save_checkpoint", step=step, last_step=last_step)

    @traced("trainer.export_to_hf")
    async def export_to_hf(self, output_path: str):
        logger.debug(f"DistributedModelHandle {self.name}: export_to_hf, output_path={output_path}")
        result = await self._call_all("export_to_hf", output_path=output_path)
        logger.debug(f"DistributedModelHandle {self.name}: export_to_hf done")
        return result

    def __getattr__(self, attr):
        if attr.startswith("_"):
            raise AttributeError(attr)
        if attr in self._simple_attrs:
            return ray.get(self.actors[0].get_attr.remote(attr))

        async def proxy(*args, **kwargs):
            kwargs.pop("tracer", None)
            results = await asyncio.gather(
                *[self.resolve(a.call_method.remote(attr, *args, **kwargs)) for a in self.actors]
            )
            return results[0]

        return proxy

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

    # `trainable` is an RLVR-only flag that should not be written into the
    # torchtitan JobConfig TOML, but it must still be propagated to the actor.
    trainable = bool(config.get("trainable", True))

    cfg = dict(config)
    cfg.pop("trainable", None)

    # NOTE: Ray actors may be scheduled on different nodes. Use a shared filesystem
    # path (repo/EFS) instead of the per-node default temp dir.
    config_dir = os.environ.get("RLVR_TITAN_CONFIG_DIR")
    if config_dir is None:
        config_dir = os.path.join(os.getcwd(), ".rlvr_titan_job_configs")
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
            group_name=name,
            trainable=trainable,
        )
        for r in range(world_size)
    ]

    try:
        ray.get([a.initialize_process_group.remote(master_addr, str(port)) for a in actors])
    finally:
        # Only delete after ALL actors have finished initialization
        keep_cfg = os.environ.get("RLVR_KEEP_TITAN_CONFIG") == "1"
        if not keep_cfg:
            try:
                os.remove(config_path)
            except FileNotFoundError:
                pass

    return DistributedModelHandle(actors, name=name)





from typing import Any, Callable
from dataclasses import is_dataclass, fields
import torch
from torch.distributed.tensor import DTensor

def _tree_map(fn, x):
    if x is None or isinstance(x, (str, bytes, int, float, bool)):
        return x
    if isinstance(x, dict):
        return {k: _tree_map(fn, v) for k, v in x.items()}
    if isinstance(x, list):
        return [_tree_map(fn, v) for v in x]
    if isinstance(x, tuple):
        return tuple(_tree_map(fn, v) for v in x)
    if is_dataclass(x):
        vals = {f.name: _tree_map(fn, getattr(x, f.name)) for f in fields(x)}
        return type(x)(**vals)
    return fn(x)
