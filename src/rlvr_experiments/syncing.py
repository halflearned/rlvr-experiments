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


async def _sync_chunks(src, dst_actors, channel, chunk_mb, dtype_str, src_rank, label, *, dst_is_titan=False):
    """Sync weights from src to dst_actors via chunked NCCL broadcast.

    Args:
        dst_is_titan: If True, dst_actors are Titan actors that support prepare_recv_state/clear_recv_state
                      for faster chunk reception. vLLM actors don't need this optimization.
    """
    loop = asyncio.get_event_loop()

    async def resolve(ref):
        return await loop.run_in_executor(None, ray.get, ref)

    # Prepare source actors (cache HF state dict)
    with trace_span("sync.prepare_src"):
        await asyncio.gather(*[resolve(a.prepare_sync_state.remote()) for a in src.actors])

    # Prepare destination actors if they support it (Titan only)
    if dst_is_titan:
        with trace_span("sync.prepare_dst"):
            await asyncio.gather(*[resolve(a.prepare_recv_state.remote()) for a in dst_actors])

    try:
        max_chunk_elems = _chunk_elems_from_mb(chunk_mb, dtype_str)
        chunks = await resolve(src.actors[0].build_chunk_plan.remote(max_chunk_elems=max_chunk_elems))

        with trace_span("sync.nccl_broadcast", args={"num_chunks": len(chunks)}):
            for chunk in chunks:
                src_futs = [resolve(a.broadcast_chunk.remote(channel, chunk, dtype_str, src_rank)) for a in src.actors]
                dst_futs = [resolve(a.recv_chunk.remote(channel, chunk, dtype_str, src_rank)) for a in dst_actors]
                await asyncio.gather(*src_futs, *dst_futs)
    finally:
        with trace_span("sync.cleanup"):
            await asyncio.gather(*[resolve(a.clear_sync_state.remote()) for a in src.actors])
            if dst_is_titan:
                await asyncio.gather(*[resolve(a.clear_recv_state.remote()) for a in dst_actors])

    logger.info(label)


@traced("sync.trainer_to_vllm")
async def sync_titan_to_vllm(trainer, vllm, chunk_mb=100, src_rank=0, wire_dtype="bfloat16", abort_in_flight=True, trainer_version=0):
    """Sync weights from trainer to vLLM.

    Pauses generation, aborts or waits for in-flight requests, syncs, then resumes.
    Updates vllm.trainer_version AFTER sync so new samples are tagged with this version.

    Args:
        abort_in_flight: If True (default), abort in-flight requests instead of waiting.
                         This avoids wasted generation/verification work for stale samples.
        trainer_version: Current trainer version - samples generated after this sync will be tagged with this version.
    """
    with trace_span("sync.waiting_for_vllm_pause"):
        await vllm.stop(abort=abort_in_flight)

    try:
        channel = f"{trainer.name}_to_{vllm.name}"
        with trace_span("sync.titan_to_vllm"):
            await _sync_chunks(trainer, vllm._actors, channel, chunk_mb, wire_dtype, src_rank,
                               f"synced {trainer.name} -> {vllm.name}")
    finally:
        # Update trainer version AFTER sync - new samples will be tagged with this version
        vllm.set_trainer_version(trainer_version)
        vllm.resume()


@traced("sync.trainer_to_reference")
async def sync_titan_to_titan(src, dst, chunk_mb=100, src_rank=0, wire_dtype="bfloat16"):
    """Sync weights from src Titan model to dst Titan model.

    Waits for any in-flight calls on dst to complete before syncing.
    """
    # Wait for in-flight calls on destination model to complete
    print(f"[sync_titan_to_titan] Waiting for {dst.name} to be idle (in_flight={dst._in_flight})")
    with trace_span("sync.waiting_for_dst_idle"):
        await dst.wait_idle()
    print(f"[sync_titan_to_titan] {dst.name} is idle, starting sync")

    channel = f"{src.name}_to_{dst.name}"
    with trace_span("sync.titan_to_titan"):
        await _sync_chunks(src, dst.actors, channel, chunk_mb, wire_dtype, src_rank,
                           f"synced {src.name} -> {dst.name}", dst_is_titan=True)
