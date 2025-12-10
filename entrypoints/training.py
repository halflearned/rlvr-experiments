"""
RLVR Orchestrator - Single trainer version for testing
"""

import argparse
import torch
import asyncio
from torchtitan.tools.logging import init_logger
import ray

from rlvr_experiments.distributed_titan_actor import create_distributed_model


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str)
    args, _ = parser.parse_known_args()
    
    init_logger()
    
    print("Connecting to Ray cluster...")
    ray.init(address="auto")
    
    print(f"Ray cluster resources: {ray.cluster_resources()}")
    
    # Create distributed trainer (8 GPUs)
    print("\nCreating trainer model with 8 GPUs...")
    trainer = create_distributed_model(
        config_path=args.config,
        world_size=8,
        group_name="trainer",
        master_port=29500
    )
    
    print("✓ Trainer initialized")
    
    # Test: Get tokenizer
    print("\nTesting trainer.get_tokenizer()...")
    tokenizer = trainer.tokenizer
    print(f"✓ Got tokenizer: {type(tokenizer)}")
    
    # Test: Create dummy input
    print("\nTesting forward pass...")
    dummy_input = {
        "input": torch.randint(0, 1000, (2, 128)).cuda()
    }
    
    logits = await trainer.forward_step(dummy_input)
    print(f"✓ Forward pass successful, logits shape: {logits.shape}")
    
    # Test: Backward pass
    print("\nTesting backward pass...")
    dummy_loss = logits.mean()
    await trainer.backward_step(dummy_loss)
    print("✓ Backward pass successful")
    
    # Test: Optimizer step
    print("\nTesting optimizer step...")
    loss_val = await trainer.optimizer_step()
    print(f"✓ Optimizer step successful, loss: {loss_val}")
    
    print("\n" + "="*60)
    print("ALL TESTS PASSED!")
    print("="*60)
    
    # Now you can add your actual training loop here
    # For now, just exit
    print("\nTraining loop would start here...")


if __name__ == "__main__":
    asyncio.run(main())