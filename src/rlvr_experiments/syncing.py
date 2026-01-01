"""Weight synchronization between models.

Provides NCCL-based weight transfer between:
- Trainer → vLLM (for rollout generation)
- Trainer → Reference (for KL computation)
"""

import asyncio
import logging

import ray
import torch
from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
from vllm.distributed.utils import StatelessProcessGroup

from .tracer import traced, trace_span

logger = logging.getLogger(__name__)


# --- Low-level NCCL manager ---

class WeightSyncManager:
    """NCCL communicator for weight sync between models in separate distributed worlds."""

    def __init__(self, host: str, port: int, world_size: int, rank: int, device: torch.device) -> None:
        pg = StatelessProcessGroup.create(host=host, port=port, rank=rank, world_size=world_size)
        self.communicator = PyNcclCommunicator(pg, device=device)
        self.sync_group_rank = rank
        self.device = device
        self.world_size = world_size


# --- High-level sync functions ---

_BYTES_PER_ELEM = {"bfloat16": 2, "float16": 2, "float32": 4, "int8": 1}


def _chunk_elems_from_mb(chunk_mb: int, dtype_str: str) -> int:
    return (chunk_mb * 1024 * 1024) // _BYTES_PER_ELEM[dtype_str]


async def _sync_chunks(src, dst_actors, channel, chunk_mb, dtype_str, src_rank, label):
    """Sync weights from src to dst_actors via chunked NCCL broadcast."""
    loop = asyncio.get_event_loop()

    async def resolve(ref):
        return await loop.run_in_executor(None, ray.get, ref)

    await asyncio.gather(*[resolve(a.prepare_sync_state.remote()) for a in src.actors])

    try:
        max_chunk_elems = _chunk_elems_from_mb(chunk_mb, dtype_str)
        chunks = await resolve(src.actors[0].build_chunk_plan.remote(max_chunk_elems=max_chunk_elems))

        for chunk in chunks:
            src_futs = [resolve(a.broadcast_chunk.remote(channel, chunk, dtype_str, src_rank)) for a in src.actors]
            dst_futs = [resolve(a.recv_chunk.remote(channel, chunk, dtype_str, src_rank)) for a in dst_actors]
            await asyncio.gather(*src_futs, *dst_futs)
    finally:
        await asyncio.gather(*[resolve(a.clear_sync_state.remote()) for a in src.actors])

    logger.info(label)


@traced("sync.trainer_to_vllm")
async def sync_titan_to_vllm(trainer, vllm, chunk_mb=100, src_rank=0, wire_dtype="bfloat16"):
    """Sync weights from trainer to vLLM. Caller must stop producers first."""
    if not vllm.is_stopped():
        raise RuntimeError("Cannot sync weights while vLLM is generating. Call stop_producers() first.")
    channel = f"{trainer.name}_to_{vllm.name}"
    with trace_span("sync.titan_to_vllm"):
        await _sync_chunks(trainer, vllm._actors, channel, chunk_mb, wire_dtype, src_rank,
                           f"synced {trainer.name} -> {vllm.name}")


@traced("sync.trainer_to_reference")
async def sync_titan_to_titan(src, dst, chunk_mb=100, src_rank=0, wire_dtype="bfloat16"):
    """Sync weights from src Titan model to dst Titan model."""
    channel = f"{src.name}_to_{dst.name}"
    with trace_span("sync.titan_to_titan"):
        await _sync_chunks(src, dst.actors, channel, chunk_mb, wire_dtype, src_rank,
                           f"synced {src.name} -> {dst.name}")
