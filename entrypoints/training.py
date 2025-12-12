import argparse
import asyncio
import time
import ray
import torch
from torchtitan.tools.logging import init_logger

from rlvr_experiments.distributed_titan_actor import (
    create_titan_group, 
    sync_titan_to_titan, 
    sync_titan_to_vllm
)
from rlvr_experiments.vllm_engine_actor import VLLMEngineRank, VLLMHandle



async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str)
    args, _ = parser.parse_known_args()
    init_logger()
    ray.init(address="auto")
    
    # --- Configuration ---
    TRAINER_RANKS = 2
    VLLM_TP_SIZE = 1
    REF_RANKS = 2
    
    # Calculate Correct World Sizes
    # Fast Channel: Trainer + All vLLM Workers
    FAST_WORLD_SIZE = TRAINER_RANKS + VLLM_TP_SIZE 
    
    # Slow Channel: Trainer + Reference
    SLOW_WORLD_SIZE = TRAINER_RANKS + REF_RANKS

    master_addr = ray.util.get_node_ip_address()
    
    # 1. Init Actors
    print(f"--- Phase 1: Allocation (Fast World Size={FAST_WORLD_SIZE}) ---")
    trainer = create_titan_group(args.config, "trainer", TRAINER_RANKS, 29500)
    reference = create_titan_group(args.config, "reference", REF_RANKS, 29600)

    # vLLM: Must know the TOTAL world size (Trainer + Workers)
    print(f"Launching vLLM (TP={VLLM_TP_SIZE})...")
    vllm_actor = VLLMEngineRank.options(num_gpus=1).remote(
        sync_host=master_addr,
        sync_port=51216,
        sync_world_size=FAST_WORLD_SIZE,
        sync_rank=TRAINER_RANKS,
        model_name="Qwen/Qwen3-0.6B",
        dtype="bfloat16",
        tensor_parallel_size=VLLM_TP_SIZE,
        trust_remote_code=True,
        distributed_executor_backend="ray",
    )
    vllm = VLLMHandle(vllm_actor)

    # 2. Trigger Async Connection (Fire and Forget)
    print("Triggering vLLM worker connection...")
    init_future = vllm_actor.init_workers.remote()

    # 3. Wiring (Connection Phase)
    print("--- Phase 2: Wiring ---")
    
    # Update topology helper call to use the correct sizes
    # We need to manually call add_sync_channel here or update the helper
    # because the helper might still be using the old 'len + 1' math.
    
    # Let's do it explicitly here to be safe and transparent:
    
    # A. Wire Fast Channel (Trainer joins vLLM)
    print(f"Wiring Trainer to vLLM (Port 51216, Size={FAST_WORLD_SIZE})...")
    futures = []
    for i, actor in enumerate(trainer.actors):
        futures.append(actor.add_sync_channel.remote(
            channel_name="fast_vllm",
            host=master_addr,
            port=51216,
            world_size=FAST_WORLD_SIZE, # <--- FIXED
            my_rank=i
        ))
    await asyncio.gather(*futures)
    
    # B. Wire Slow Channel (Trainer joins Reference)
    print(f"Wiring Trainer to Reference (Port 51217, Size={SLOW_WORLD_SIZE})...")
    futures = []
    # Trainer
    for i, actor in enumerate(trainer.actors):
        futures.append(actor.add_sync_channel.remote(
            channel_name="slow_ref",
            host=master_addr,
            port=51217,
            world_size=SLOW_WORLD_SIZE,
            my_rank=i
        ))
    # Reference (Starts after Trainer ranks)
    for i, actor in enumerate(reference.actors):
        futures.append(actor.add_sync_channel.remote(
            channel_name="slow_ref",
            host=master_addr,
            port=51217,
            world_size=SLOW_WORLD_SIZE,
            my_rank=TRAINER_RANKS + i
        ))
    await asyncio.gather(*futures)
    
    # 4. Finish Init
    print("Waiting for vLLM to finish connecting...")
    await init_future
    print("✓ vLLM Connected.")


    # 5. Testing
    print("--- Phase 3: Testing ---")
    
    # SYNC 1: Trainer to Reference
    print("Syncing Trainer -> Reference...")
    from time import time
    start = time()
    await sync_titan_to_titan(trainer, reference, channel="slow_ref")
    end = time()
    print(f"✓ Titan <> Titan sync complete. Time taken: {end - start:.10f}s")

    # --- CRITICAL BARRIER ---
    # SYNC 2: Trainer to vLLM
    print("Syncing Trainer -> vLLM...")
    start = time()
    await sync_titan_to_vllm(trainer, vllm, channel="fast_vllm")
    end = time()
    print(f"✓ Titan <> vLLM sync complete. Time taken: {end - start:.10f}s")

    output = await vllm.generate(["The capital of France is"])
    
    print("Generation:", output[0].outputs[0].text)
    print("✓ Success.")

if __name__ == "__main__":
    asyncio.run(main())