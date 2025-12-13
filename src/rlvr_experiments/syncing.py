import asyncio
import torch
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class ParamMeta:
    name: str
    numel: int
    shape: Tuple[int, ...]


@dataclass
class ChunkMeta:
    """Metadata for one flat chunk."""
    total_numel: int
    params: List[ParamMeta]



async def sync_titan_to_vllm(
    trainer: 'DistributedModelHandle',
    vllm,
    channel: str = "fast_vllm",
    max_chunk_elems: int = 50_000_000,  # ~100MB at bf16
    src_rank: int = 0,
):
    """
    HF-dense, chunked sync: Trainer → vLLM.
    - Wire type: bf16.
    - Chunking to avoid huge flat buffers.
    """
    print(f"--- [FAST CHUNKED: {trainer.name} -> vLLM] ---")

    # 1. Prepare HF state on all trainer ranks.
    await asyncio.gather(*[a.prepare_sync_state.remote() for a in trainer.actors])

    # 2. Build chunk plan on trainer rank 0.
    chunk_plan = await trainer.resolve(
        trainer.actors[0].build_chunk_plan.remote(max_chunk_elems=max_chunk_elems)
    )

    dtype_str = "bfloat16"

    # 3. For each chunk, run Titan broadcast + vLLM recv concurrently.
    for chunk in chunk_plan:
        # Trainer-side broadcasts (all trainer ranks)
        t_futs = [
            a.broadcast_chunk.remote(channel, chunk, dtype_str, src_rank)
            for a in trainer.actors
        ]

        # vLLM-side receive (all workers via collective_rpc)
        v_fut = vllm._actor.recv_chunk.remote(chunk, dtype_str, src_rank)

        await asyncio.gather(*t_futs, v_fut)

    # 4. Cleanup trainer caches.
    await asyncio.gather(*[a.clear_sync_state.remote() for a in trainer.actors])

    print(f"✓ {trainer.name} -> vLLM (chunked) complete.")


async def sync_titan_to_titan(
    src: 'DistributedModelHandle',
    dst: 'DistributedModelHandle',
    channel: str = "slow_ref",
    max_chunk_elems: int = 50_000_000,
    src_rank: int = 0,
    dtype_str: str = "bfloat16",  # or your training dtype, e.g. "float16"
):
    """
    HF-dense, chunked sync: Trainer (src) → Reference (dst).
    """
    print(f"--- [FAST CHUNKED: {src.name} -> {dst.name}] ---")

    # 1. Prepare HF state on all src ranks.
    await asyncio.gather(*[a.prepare_sync_state.remote() for a in src.actors])

    # 2. Build chunk plan on src rank 0.
    chunk_plan = await src.resolve(
        src.actors[0].build_chunk_plan.remote(max_chunk_elems=max_chunk_elems)
    )

    # 3. For each chunk, run src broadcast + dst recv concurrently.
    for chunk in chunk_plan:
        s_futs = [
            a.broadcast_chunk.remote(channel, chunk, dtype_str, src_rank)
            for a in src.actors
        ]
        d_futs = [
            a.recv_chunk_from_hf.remote(channel, chunk, dtype_str, src_rank)
            for a in dst.actors
        ]

        await asyncio.gather(*s_futs, *d_futs)

    # 4. Cleanup src caches.
    await asyncio.gather(*[a.clear_sync_state.remote() for a in src.actors])

    print(f"✓ {src.name} -> {dst.name} (chunked) complete.")
