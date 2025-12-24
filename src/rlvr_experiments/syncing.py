import asyncio

from dataclasses import dataclass
from typing import Any, Sequence

from .tracer import traced, trace_span

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
    print(f"[TEMPLOG _sync_chunks] Starting sync, src={src.name}, channel={channel}, chunk_mb={chunk_mb}")

    # 1) Prepare HF cache on all src ranks (collectives happen here)
    print(f"[TEMPLOG _sync_chunks] Step 1: Preparing sync state on {len(src.actors)} src actors...")
    await asyncio.gather(*[a.prepare_sync_state.remote() for a in src.actors])
    print(f"[TEMPLOG _sync_chunks] Step 1 complete: Sync state prepared")

    try:
        # 2) Build chunk plan on src rank 0 (requires cache)
        print(f"[TEMPLOG _sync_chunks] Step 2: Building chunk plan...")
        max_chunk_elems = _chunk_elems_from_mb(chunk_mb, dtype_str)
        chunk_plan: Sequence[Any] = await src.resolve(
            src.actors[0].build_chunk_plan.remote(max_chunk_elems=max_chunk_elems)
        )
        print(f"[TEMPLOG _sync_chunks] Step 2 complete: Got {len(chunk_plan)} chunks to sync")

        # 3) For each chunk: src broadcast (all ranks) + receiver
        print(f"[TEMPLOG _sync_chunks] Step 3: Broadcasting {len(chunk_plan)} chunks...")
        for i, chunk in enumerate(chunk_plan):
            print(f"[TEMPLOG _sync_chunks] Chunk {i+1}/{len(chunk_plan)}: Starting broadcast...")
            src_futs = [
                a.broadcast_chunk.remote(channel, chunk, dtype_str, src_rank)
                for a in src.actors
            ]
            print(f"[TEMPLOG _sync_chunks] Chunk {i+1}/{len(chunk_plan)}: Waiting for src broadcast + receiver...")
            await asyncio.gather(*src_futs, receiver_per_chunk(chunk))
            print(f"[TEMPLOG _sync_chunks] Chunk {i+1}/{len(chunk_plan)}: Complete")

    finally:
        # Always clear even if a chunk fails
        print(f"[TEMPLOG _sync_chunks] Clearing sync state...")
        await asyncio.gather(*[a.clear_sync_state.remote() for a in src.actors])
        print(f"[TEMPLOG _sync_chunks] Sync state cleared")

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
    """
    Sync weights from trainer to vLLM.

    Automatically stops any rollout producers (via vllm.stop()), syncs weights,
    then resumes (via vllm.resume()) so new producers can run.
    """
    print(f"[TEMPLOG sync_titan_to_vllm] Starting sync: {trainer.name} -> {vllm.name}")

    print(f"[TEMPLOG sync_titan_to_vllm] Stopping vLLM producer...")
    with trace_span(None, "sync.stop_producer"):
        await vllm.stop()
    print(f"[TEMPLOG sync_titan_to_vllm] vLLM producer stopped")

    if channel is None:
        channel = _infer_channel_name(trainer.name, vllm.name)
    print(f"[TEMPLOG sync_titan_to_vllm] Using channel={channel}")

    async def recv(chunk: Any):
        # Engine actor should forward to collective_rpc on workers
        print(f"[TEMPLOG sync_titan_to_vllm] recv_chunk called for chunk")
        result = await vllm._actor.recv_chunk.remote(chunk, wire_dtype, src_rank)
        print(f"[TEMPLOG sync_titan_to_vllm] recv_chunk complete")
        return result

    print(f"[TEMPLOG sync_titan_to_vllm] Starting NCCL sync...")
    with trace_span(None, "sync.nccl_sync"):
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
    print(f"[TEMPLOG sync_titan_to_vllm] NCCL sync complete")

    print(f"[TEMPLOG sync_titan_to_vllm] Resuming vLLM...")
    vllm.resume()
    print(f"[TEMPLOG sync_titan_to_vllm] Done")


@traced("sync.trainer_to_reference")
async def sync_titan_to_titan(
    src: "DistributedModelHandle",
    dst: "DistributedModelHandle",
    channel: str | None = None,
    chunk_mb: int = 100,
    src_rank: int = 0,
    wire_dtype: str = "bfloat16",
) -> None:
    print(f"[TEMPLOG sync_titan_to_titan] Starting sync: {src.name} -> {dst.name}")

    if channel is None:
        channel = _infer_channel_name(src.name, dst.name)
    print(f"[TEMPLOG sync_titan_to_titan] Using channel={channel}, dst has {len(dst.actors)} actors")

    async def recv(chunk: Any):
        print(f"[TEMPLOG sync_titan_to_titan] recv_chunk_from_hf called, dispatching to {len(dst.actors)} dst actors...")
        dst_futs = [
            a.recv_chunk_from_hf.remote(channel, chunk, wire_dtype, src_rank)
            for a in dst.actors
        ]
        result = await asyncio.gather(*dst_futs)
        print(f"[TEMPLOG sync_titan_to_titan] recv_chunk_from_hf complete for all dst actors")
        return result

    print(f"[TEMPLOG sync_titan_to_titan] Starting _sync_chunks...")
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
    print(f"[TEMPLOG sync_titan_to_titan] Done")
