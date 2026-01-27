import os
import ray
import torch

from typing import Any, Dict, List, Callable

from torch.distributed.tensor import DTensor, distribute_tensor

from .syncing import WeightSyncManager
from .ops import compute_logprobs

import asyncio
import logging
import tempfile

import tomli_w as toml

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set to True to enable detailed timing logs (requires cuda.synchronize, adds overhead)
_PROFILE_TITAN = os.environ.get("RLVR_PROFILE_TITAN", "0") == "1"


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
        self._recv_state_cache: Dict[str, torch.Tensor] | None = None

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
        """Cache HF state dict for sending (involves DTensor collectives, all ranks must call)."""
        self._sync_cache = self.model.hf_state_dict()

    def clear_sync_state(self) -> None:
        self._sync_cache = None
        # NOTE: Removed torch.cuda.empty_cache() - it was causing slowdown
        # on the next forward pass due to CUDA memory reorganization

    def prepare_recv_state(self) -> None:
        """Cache model state dict for receiving (avoids repeated get_model_state_dict calls)."""
        from torch.distributed.checkpoint.state_dict import get_model_state_dict
        model = self.model.model_parts[0]
        self._recv_state_cache = get_model_state_dict(model)

    def clear_recv_state(self) -> None:
        self._recv_state_cache = None

    def build_chunk_plan(self, max_chunk_elems: int) -> list[dict]:
        """Build a chunk plan from cached HF state (metadata only, single rank)."""
        chunks = []
        current_params = []
        current_total = 0

        # DEBUG: Comprehensive logging
        print(f"[SYNC DEBUG build_chunk_plan] START max_chunk_elems={max_chunk_elems:,}", flush=True)
        print(f"[SYNC DEBUG build_chunk_plan] _sync_cache has {len(self._sync_cache)} keys", flush=True)

        # Check actual dtype of first tensor
        first_name, first_tensor = next(iter(self._sync_cache.items()))
        actual_dtype = first_tensor.dtype if not isinstance(first_tensor, DTensor) else first_tensor._local_tensor.dtype
        print(f"[SYNC DEBUG build_chunk_plan] ACTUAL WEIGHT DTYPE: {actual_dtype}", flush=True)

        total_numel_all = 0
        dtensor_count = 0
        regular_count = 0

        for name, tensor in self._sync_cache.items():
            is_dtensor = isinstance(tensor, DTensor)
            numel = tensor.numel()
            total_numel_all += numel

            if is_dtensor:
                dtensor_count += 1
                local_numel = tensor._local_tensor.numel()
                local_dtype = tensor._local_tensor.dtype
                # Only log first few DTensors to reduce noise
                if dtensor_count <= 3:
                    print(f"[SYNC DEBUG DTensor] {name}: dtype={local_dtype} shape={tuple(tensor.shape)} global={numel:,} local={local_numel:,}", flush=True)
            else:
                regular_count += 1

        total_bytes = total_numel_all * 2  # bf16/fp16
        print(f"[SYNC DEBUG build_chunk_plan] TOTAL: {len(self._sync_cache)} params, {total_numel_all:,} elements, {total_bytes/1e9:.3f} GB", flush=True)
        print(f"[SYNC DEBUG build_chunk_plan] DTensors={dtensor_count}, Regular={regular_count}", flush=True)

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

        total_chunk_elems = sum(c["total_numel"] for c in chunks)
        print(f"[SYNC DEBUG build_chunk_plan] DONE: {len(chunks)} chunks, total_elems={total_chunk_elems:,}", flush=True)
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
            # DEBUG: Log first/last param in chunk
            if p == chunk["params"][0] or p == chunk["params"][-1]:
                print(f"[SYNC DEBUG broadcast] {p['name']}: expected={p['numel']:,} actual={t_full.numel():,} match={p['numel']==t_full.numel()}", flush=True)
            tensors.append((p["name"], p["numel"], t_full))

        # Pack into flat buffer (only src fills with real data)
        flat = torch.empty(chunk["total_numel"], dtype=dtype, device=device)
        if is_src:
            offset = 0
            for name, numel, t_full in tensors:
                buf = t_full.to(device=device, dtype=dtype).contiguous().view(-1)
                if buf.numel() != numel:
                    print(f"[SYNC DEBUG broadcast MISMATCH!] {name}: chunk_numel={numel:,} buf_numel={buf.numel():,}", flush=True)
                flat[offset : offset + numel].copy_(buf)
                offset += numel

        manager.communicator.broadcast(flat, src=src_rank)

    def recv_chunk(self, channel: str, chunk: dict, dtype_str: str, src_rank: int) -> None:
        """Receive a chunk and load into model parameters.

        Uses cached state dict from prepare_recv_state() if available,
        otherwise falls back to fetching it (slower).
        """
        manager = self.sync_managers[channel]
        dtype = getattr(torch, dtype_str)
        device = self.model.device

        flat = torch.empty(chunk["total_numel"], dtype=dtype, device=device)
        manager.communicator.broadcast(flat, src=src_rank)

        # Use cached state dict if available (much faster for many chunks)
        if self._recv_state_cache is not None:
            current_state = self._recv_state_cache
        else:
            # Fallback: fetch state dict (slow, logs warning)
            from torch.distributed.checkpoint.state_dict import get_model_state_dict
            logger.warning("recv_chunk called without prepare_recv_state - this is slow!")
            model = self.model.model_parts[0]
            current_state = get_model_state_dict(model)

        with torch.no_grad():
            offset = 0

            for p in chunk["params"]:
                full_tensor = flat[offset : offset + p["numel"]].view(p["shape"])
                offset += p["numel"]

                # Convert HF param name to Titan param name(s)
                titan_state = self.model.sd_adapter.from_hf({p["name"]: full_tensor})

                # Directly update matching params in current_state
                for titan_name, incoming_val in titan_state.items():
                    if titan_name in current_state:
                        target = current_state[titan_name]
                        if isinstance(target, DTensor):
                            # Re-shard the incoming tensor to match target's distribution
                            sharded = distribute_tensor(
                                incoming_val.to(device=device, dtype=target.dtype),
                                target.device_mesh,
                                target.placements,
                            )
                            target.copy_(sharded)
                        else:
                            target.copy_(incoming_val.to(device=target.device, dtype=target.dtype))

    def _get_dp_shard(self, tensor: torch.Tensor) -> torch.Tensor:
        """Shard a tensor by DP replicate rank (for DDP-style data parallelism)."""
        dp_degree = self.model.parallel_dims.dp_replicate
        if dp_degree <= 1:
            return tensor

        # Get local rank within dp_replicate dimension
        mesh = self.model.parallel_dims.world_mesh
        if "dp_replicate" not in mesh.mesh_dim_names:
            return tensor

        dp_rank = mesh.get_local_rank("dp_replicate")
        batch_size = tensor.shape[0]
        shard_size = batch_size // dp_degree

        if shard_size * dp_degree != batch_size:
            raise ValueError(
                f"Batch size {batch_size} not evenly divisible by dp_replicate={dp_degree}."
            )

        start = dp_rank * shard_size
        end = start + shard_size
        return tensor[start:end]

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor | None:
        """Forward pass returning raw logits (only rank 0 returns, others return None)."""
        logits = self.model.forward(input_ids)
        if isinstance(logits, DTensor):
            logits = logits.full_tensor()
        return logits.detach().cpu() if self.rank == 0 else None

    def get_rope_cache_info(self) -> dict:
        """Return info about rope_cache for debugging."""
        model = self.model.model_parts[0]
        if hasattr(model, 'rope_cache') and model.rope_cache is not None:
            return {
                "dtype": str(model.rope_cache.dtype),
                "shape": list(model.rope_cache.shape),
            }
        return {"error": "No rope_cache found"}

    def export_to_hf(self, output_path: str) -> None:
        """Export model to HuggingFace format. All ranks must call (collective for DTensors)."""
        import time
        import sys
        ts = time.strftime("%H:%M:%S")
        print(f"[{ts}] [TitanActor.export_to_hf] ENTERING: output_path={output_path}", flush=True)
        sys.stdout.flush()
        t0 = time.time()
        self.model.export_to_hf(output_path)
        ts = time.strftime("%H:%M:%S")
        print(f"[{ts}] [TitanActor.export_to_hf] RETURNING after {time.time()-t0:.2f}s", flush=True)
        sys.stdout.flush()

    def save_checkpoint(self, step: int | None = None, last_step: bool = False) -> None:
        """Save a distributed checkpoint. All ranks must call."""
        self.model.save_checkpoint(step=step, last_step=last_step)

    def get_step(self) -> int:
        """Get the current training step."""
        return self.model.step

    @torch.no_grad()
    def compute_logprobs(
        self,
        input_ids: torch.Tensor,
        completion_ids: torch.Tensor,
        prompt_lens: torch.Tensor | None = None,
        temperature: float = 1.0,
    ) -> torch.Tensor | None:
        """Forward pass returning logprobs (only rank 0 returns, others return None).

        Args:
            input_ids: [B, seq_len] - full sequence (prompt + completion + padding)
            completion_ids: [B, completion_len] - just the completion tokens
            prompt_lens: [B] - length of prompt for each sample, needed when sequences
                         have trailing padding to correctly slice logits
        """
        import time
        import torch.distributed as dist
        from torch.distributed.tensor.parallel import loss_parallel

        # Shard batch by DP replicate rank
        input_ids = self._get_dp_shard(input_ids)
        completion_ids = self._get_dp_shard(completion_ids)
        if prompt_lens is not None:
            prompt_lens = self._get_dp_shard(prompt_lens)

        if _PROFILE_TITAN:
            torch.cuda.synchronize()
            t0 = time.perf_counter()

        # DEBUG: Memory before forward
        if self.rank == 0:
            alloc_before = torch.cuda.memory_allocated() / 1e9
            reserved_before = torch.cuda.memory_reserved() / 1e9
            print(f"[MEM DEBUG {self.group_name}] BEFORE forward: allocated={alloc_before:.2f}GB, reserved={reserved_before:.2f}GB, input_shape={list(input_ids.shape)}", flush=True)

        logits = self.model.forward(input_ids)

        # DEBUG: Memory after forward
        if self.rank == 0:
            alloc_after = torch.cuda.memory_allocated() / 1e9
            reserved_after = torch.cuda.memory_reserved() / 1e9
            print(f"[MEM DEBUG {self.group_name}] AFTER forward: allocated={alloc_after:.2f}GB, reserved={reserved_after:.2f}GB, logits_shape={list(logits.shape) if hasattr(logits, 'shape') else 'DTensor'}", flush=True)

        if _PROFILE_TITAN:
            torch.cuda.synchronize()
            t1 = time.perf_counter()

        # Use loss_parallel to keep logits sharded on vocab dimension
        # This saves ~75% memory by avoiding the full_tensor() all-gather
        completion_ids = completion_ids.to(self.model.device)
        if isinstance(logits, DTensor) and prompt_lens is not None:
            # DTensor path: compute logprobs with vocab-sharded logits
            with loss_parallel():
                logprobs = compute_logprobs(
                    logits,
                    completion_ids,
                    temperature=temperature,
                    prompt_lens=prompt_lens,
                )
                # DEBUG: Memory after logprobs
                if self.rank == 0:
                    alloc_lp = torch.cuda.memory_allocated() / 1e9
                    reserved_lp = torch.cuda.memory_reserved() / 1e9
                    print(f"[MEM DEBUG {self.group_name}] AFTER logprobs: allocated={alloc_lp:.2f}GB, reserved={reserved_lp:.2f}GB", flush=True)
        else:
            # Regular path: gather logits first if DTensor
            if isinstance(logits, DTensor):
                logits = logits.full_tensor()
            logprobs = compute_logprobs(
                logits,
                completion_ids,
                temperature=temperature,
                prompt_lens=prompt_lens,
            )

        if _PROFILE_TITAN:
            torch.cuda.synchronize()
            t2 = time.perf_counter()

        # Gather logprobs from all DP replicate ranks to reconstruct full batch
        dp_degree = self.model.parallel_dims.dp_replicate
        if dp_degree > 1 and "dp_replicate" in self.model.parallel_dims.world_mesh.mesh_dim_names:
            mesh = self.model.parallel_dims.world_mesh
            dp_mesh = mesh["dp_replicate"]
            dp_group = dp_mesh.get_group()

            # All-gather logprobs across DP ranks
            gathered = [torch.zeros_like(logprobs) for _ in range(dp_degree)]
            dist.all_gather(gathered, logprobs, group=dp_group)
            logprobs = torch.cat(gathered, dim=0)

        if _PROFILE_TITAN:
            torch.cuda.synchronize()
            t3 = time.perf_counter()

            if self.rank == 0:
                print(
                    f"[PROFILE compute_logprobs] "
                    f"forward={1000*(t1-t0):.0f}ms, "
                    f"logprobs={1000*(t2-t1):.0f}ms, "
                    f"gather={1000*(t3-t2):.0f}ms, "
                    f"TOTAL={1000*(t3-t0):.0f}ms | "
                    f"shape={list(input_ids.shape)}",
                    flush=True
                )

        return logprobs.detach().cpu() if self.rank == 0 else None

    def forward_backward(
        self,
        loss_fn: Callable[..., torch.Tensor],
        input_ids: torch.Tensor,
        loss_args: tuple = (),
        loss_kwargs: dict | None = None,
        scale_loss: float = 1.0,
        micro_batch_size: int | None = None,
    ) -> tuple[float, dict | None]:
        """Forward pass, compute loss, and backward pass. Returns (loss, debug_metrics).

        Args:
            scale_loss: Scale factor for loss before backward. Use 1/accumulation_steps
                       for gradient accumulation so gradients average correctly.
            micro_batch_size: If provided, split the batch into micro-batches of this size
                             and accumulate gradients. Must evenly divide batch size.
                             This allows processing larger batches than GPU memory permits.
        """
        import time

        # Shard batch by DP replicate rank
        input_ids = self._get_dp_shard(input_ids)
        loss_args = tuple(
            self._get_dp_shard(a) if torch.is_tensor(a) else a
            for a in loss_args
        )
        if loss_kwargs:
            loss_kwargs = {
                k: self._get_dp_shard(v) if torch.is_tensor(v) else v
                for k, v in loss_kwargs.items()
            }

        batch_size = input_ids.shape[0]

        # Determine micro-batching
        if micro_batch_size is None or micro_batch_size >= batch_size:
            # No micro-batching, process entire batch at once
            num_micro_batches = 1
            micro_batch_size = batch_size
        else:
            if batch_size % micro_batch_size != 0:
                raise ValueError(
                    f"Batch size {batch_size} must be divisible by micro_batch_size {micro_batch_size}"
                )
            num_micro_batches = batch_size // micro_batch_size

        if _PROFILE_TITAN:
            torch.cuda.synchronize()
            t0 = time.perf_counter()

        device = self.model.device

        def to_device(v: Any) -> Any:
            if torch.is_tensor(v) and not isinstance(v, DTensor):
                return v.to(device, non_blocking=True)
            return v

        # Combined scale factor: outer accumulation * inner micro-batch accumulation
        combined_scale = scale_loss / num_micro_batches

        total_loss = 0.0
        debug_metrics = None

        for mb_idx in range(num_micro_batches):
            start = mb_idx * micro_batch_size
            end = start + micro_batch_size

            # Slice input_ids
            mb_input_ids = input_ids[start:end]

            # Slice loss_args (tensors only)
            mb_loss_args = tuple(
                a[start:end] if torch.is_tensor(a) else a
                for a in loss_args
            )

            # Slice loss_kwargs (tensors only)
            mb_loss_kwargs = {
                k: v[start:end] if torch.is_tensor(v) else v
                for k, v in (loss_kwargs or {}).items()
            }

            # Forward pass
            logits = self.model.forward(mb_input_ids)

            # Move args to device
            args_local = tuple(to_device(a) for a in mb_loss_args)
            kwargs_local = {k: to_device(v) for k, v in mb_loss_kwargs.items()}

            with self.model.train_context_mgr(None):
                # Note: loss_parallel is handled by train_context_mgr when
                # config has disable_loss_parallel=false. This enables
                # vocab-sharded cross_entropy without all-gathering logits.
                loss = loss_fn(logits, *args_local, **kwargs_local)

                if not torch.is_tensor(loss) or loss.ndim != 0:
                    raise RuntimeError("loss_fn must return a scalar torch.Tensor")

                # Scale loss for gradient accumulation (both outer and inner)
                scaled_loss = loss * combined_scale
                scaled_loss.backward()

            total_loss += float(loss.detach().item())

            # Collect debug metrics from last micro-batch
            if mb_idx == num_micro_batches - 1 and hasattr(loss_fn, 'get_debug_metrics'):
                debug_metrics = loss_fn.get_debug_metrics()

            # Free memory between micro-batches
            del logits, loss, scaled_loss, args_local, kwargs_local

        if _PROFILE_TITAN:
            torch.cuda.synchronize()
            t1 = time.perf_counter()

            if self.rank == 0:
                print(
                    f"[PROFILE forward_backward] "
                    f"TOTAL={1000*(t1-t0):.0f}ms | "
                    f"shape={list(input_ids.shape)}, "
                    f"micro_batches={num_micro_batches}x{micro_batch_size}",
                    flush=True
                )

        # Return average loss across micro-batches
        avg_loss = total_loss / num_micro_batches
        return avg_loss, debug_metrics

    def log_metrics(self, loss: float, grad_norm: float, ntokens: int) -> dict | None:
        """Log metrics using torchtitan's MetricsProcessor (only rank 0 logs)."""
        if self.rank == 0:
            return self.model.log_metrics(loss, grad_norm, ntokens)
        return None


class DistributedModelHandle:
    """
    Client-side handle for a distributed Titan model.

    Dispatches method calls to all actor ranks in parallel (required for collectives),
    returns rank 0's result.

    Tracks in-flight calls for coordination with weight sync operations.
    """

    def __init__(self, actors: List, name: str = "model"):
        self.actors = actors
        self.name = name
        self.version = 0  # Incremented after each optimizer step
        self._in_flight = 0
        self._in_flight_zero = asyncio.Event()
        self._in_flight_zero.set()  # Initially no in-flight requests

    def __getattr__(self, attr):
        if attr.startswith("_"):
            raise AttributeError(attr)

        async def proxy(*args, **kwargs):
            # Track in-flight calls
            self._in_flight += 1
            if self._in_flight == 1:
                self._in_flight_zero.clear()

            try:
                refs = [a.call_method.remote(attr, *args, **kwargs) for a in self.actors]
                results = await asyncio.gather(*[self._resolve(r) for r in refs])
                # Auto-increment version after optimizer step
                if attr == "optim_step":
                    self.version += 1
                return results[0]
            finally:
                self._in_flight -= 1
                if self._in_flight == 0:
                    self._in_flight_zero.set()

        return proxy

    async def wait_idle(self) -> None:
        """Wait until all in-flight calls complete."""
        if self._in_flight > 0:
            print(f"[{self.name}] wait_idle: {self._in_flight} calls in flight, waiting...")
        await self._in_flight_zero.wait()
        print(f"[{self.name}] wait_idle: done")

    async def _resolve(self, ref):
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, ray.get, ref)


def create_titan_group(config: dict, name: str, world_size: int, port: int) -> DistributedModelHandle:
    master_addr = ray.util.get_node_ip_address()
    trainable = bool(config.get("trainable", True))

    cfg = dict(config)
    cfg.pop("trainable", None)

    # RLVR always needs to load weights - ensure checkpoint.enable is true
    # (torchtitan silently skips loading HF weights when enable=false)
    if "checkpoint" not in cfg:
        cfg["checkpoint"] = {}
    if not cfg["checkpoint"].get("enable", False):
        cfg["checkpoint"]["enable"] = True
        logger.info("Forcing checkpoint.enable=true (required for HF weight loading)")

    # On SageMaker, use SM_MODEL_DIR for checkpoints so they get uploaded to S3
    model_dir = os.environ.get("SM_MODEL_DIR")
    if model_dir:
        # Set job.dump_folder to MODEL_DIR - checkpoints are saved relative to this
        if "job" not in cfg:
            cfg["job"] = {}
        cfg["job"]["dump_folder"] = model_dir
        logger.info(f"SageMaker detected: setting dump_folder to {model_dir}")

    # Ray actors may be on different nodes - use shared filesystem for config
    config_dir = os.environ.get("RLVR_TITAN_CONFIG_DIR", os.path.join(os.getcwd(), ".rlvr_titan_job_configs"))
    os.makedirs(config_dir, exist_ok=True)

    with tempfile.NamedTemporaryFile(mode="wb", suffix=f".{name}.toml", delete=False, dir=config_dir) as f:
        toml.dump(cfg, f)
        config_path = f.name

    logger.info(f"Creating Titan group '{name}' with world_size={world_size} on {master_addr}:{port}")

    actors = [
        TitanModelRank.options(
            num_gpus=1,
            num_cpus=2,
        ).remote(
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
