import asyncio

from dataclasses import dataclass
from typing import Any, Sequence

from .tracer import traced

@dataclass
class ParamMeta:
    name: str
    numel: int
    shape: tuple[int, ...]


@dataclass
class ChunkMeta:
    """Metadata for one flat chunk."""
    total_numel: int
    params: list[ParamMeta]


_BYTES_PER_ELEM = {
    "bfloat16": 2,
    "float16": 2,
    "float32": 4,
    "int8": 1,
}

def _chunk_elems_from_mb(chunk_mb: int, dtype_str: str) -> int:
    # Keep it simple; assume caller passes sane values.
    return (int(chunk_mb) * 1024 * 1024) // _BYTES_PER_ELEM[dtype_str]


async def _sync_chunks(
    *,
    src: "DistributedModelHandle",
    channel: str,
    chunk_mb: int,
    dtype_str: str,
    src_rank: int,
    receiver_per_chunk,  # async fn(chunk) -> awaitable
    label: str,
) -> None:
    print(f"--- [{label}] ---")

    # 1) Prepare HF cache on all src ranks (collectives happen here)
    await asyncio.gather(*[a.prepare_sync_state.remote() for a in src.actors])

    try:
        # 2) Build chunk plan on src rank 0 (requires cache)
        max_chunk_elems = _chunk_elems_from_mb(chunk_mb, dtype_str)
        chunk_plan: Sequence[Any] = await src.resolve(
            src.actors[0].build_chunk_plan.remote(max_chunk_elems=max_chunk_elems)
        )

        # 3) For each chunk: src broadcast (all ranks) + receiver
        for chunk in chunk_plan:
            src_futs = [
                a.broadcast_chunk.remote(channel, chunk, dtype_str, src_rank)
                for a in src.actors
            ]
            await asyncio.gather(*src_futs, receiver_per_chunk(chunk))

    finally:
        # Always clear even if a chunk fails
        await asyncio.gather(*[a.clear_sync_state.remote() for a in src.actors])

    print(f"✓ {label} complete.")


def _infer_channel_name(src_name: str, dst_name: str) -> str:
    # TODO: this is brittle; improve later
    return f"{src_name}_to_{dst_name}"


@traced("sync.trainer_to_vllm")
async def sync_titan_to_vllm(
    trainer: "DistributedModelHandle",
    vllm,
    channel: str | None = None,
    chunk_mb: int = 100,
    src_rank: int = 0,
    wire_dtype: str = "bfloat16",
) -> None:
    if channel is None:
        channel = _infer_channel_name(trainer.name, vllm.name)

    async def recv(chunk: Any):
        # Engine actor should forward to collective_rpc on workers
        return await vllm._actor.recv_chunk.remote(chunk, wire_dtype, src_rank)

    await _sync_chunks(
        src=trainer,
        channel=channel,
        chunk_mb=chunk_mb,
        dtype_str=wire_dtype,
        src_rank=src_rank,
        receiver_per_chunk=recv,
        label=(
            f"CHUNKED: {trainer.name} -> {vllm.name} "
            f"(channel={channel}, chunk_mb={chunk_mb}, dtype={wire_dtype})"
        ),
    )


@traced("sync.trainer_to_reference")
async def sync_titan_to_titan(
    src: "DistributedModelHandle",
    dst: "DistributedModelHandle",
    channel: str | None = None,
    chunk_mb: int = 100,
    src_rank: int = 0,
    wire_dtype: str = "bfloat16",
) -> None:
    
    if channel is None:
        channel = _infer_channel_name(src.name, dst.name)

    async def recv(chunk: Any):
        dst_futs = [
            a.recv_chunk_from_hf.remote(channel, chunk, wire_dtype, src_rank)
            for a in dst.actors
        ]
        return await asyncio.gather(*dst_futs)

    await _sync_chunks(
        src=src,
        channel=channel,
        chunk_mb=chunk_mb,
        dtype_str=wire_dtype,
        src_rank=src_rank,
        receiver_per_chunk=recv,
        label=(
            f"CHUNKED: {src.name} -> {dst.name} "
            f"(channel={channel}, chunk_mb={chunk_mb}, dtype={wire_dtype})"
        ),
    )
