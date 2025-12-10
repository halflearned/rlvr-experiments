import argparse
import torch
import asyncio
from torchtitan.tools.logging import init_logger
import ray

from rlvr_experiments.distributed_titan_actor import create_two_distributed_models_with_sync


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str)
    args, _ = parser.parse_known_args()
    
    init_logger()
    
    print("Connecting to Ray cluster...")
    ray.init(address="auto")
    
    print(f"Ray cluster resources: {ray.cluster_resources()}")
    
    # print("\nCreating trainer model with 4 GPUs...")
    # trainer = create_distributed_model(
    #     config_path=args.config,
    #     world_size=4,
    #     group_name="trainer",
    #     master_port=29500
    # )

    # print("\nCreating trainer model with 4 GPUs...")
    # reference = create_distributed_model(
    #     config_path=args.config,
    #     world_size=4,
    #     group_name="reference",
    #     master_port=29501
    # )
    # Create both models with weight sync enabled
    trainer, reference = create_two_distributed_models_with_sync(
        config_path=args.config,
        ranks_per_model=4,
        trainer_master_port=29500,
        reference_master_port=29600,
        weight_sync_port=51216,
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

    logits_ref = await reference.forward_step(dummy_input)
    print(f"✓ Reference forward pass successful, logits shape: {logits_ref.shape}")
    
    # Test: Backward pass
    print("\nTesting backward pass...")
    dummy_loss = logits.mean()
    await trainer.backward_step(dummy_loss)
    print("✓ Backward pass successful")
    
    # Test: Optimizer step
    print("\nTesting optimizer step...")
    loss_val = await trainer.optimizer_step()
    print(f"✓ Optimizer step successful, loss: {loss_val}")

    # Test: Sync weights to reference model
    print("\nTesting weight sync to reference model...")
    await trainer.sync_weights_to(reference)

    print("\n" + "="*60)
    print("ALL TESTS PASSED!")
    print("="*60)
    
    # Now you can add your actual training loop here
    # For now, just exit


if __name__ == "__main__":
    asyncio.run(main())