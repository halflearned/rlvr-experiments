"""
Minimal TorchTitan training script for RLVR experiments.
Based on actual torchtitan patterns and the real Qwen3Model implementation.
"""

import os
import time
from typing import Optional

import torch
import torch.nn as nn
import torch.distributed as dist

# TorchTitan imports
from torchtitan.config_manager import JobConfig
from torchtitan.parallelisms import ParallelDims

# Import actual Qwen3 model and parallelization
from torchtitan.models.qwen3.model import Qwen3Model
from torchtitan.models.qwen3.args import Qwen3ModelArgs
from torchtitan.models.qwen3.infra.parallelize import parallelize_qwen3


# ==============================================================================
# Model Building
# ==============================================================================

def build_model_args(job_config: JobConfig) -> Qwen3ModelArgs:
    """
    Build Qwen3ModelArgs from JobConfig.
    This extracts model configuration from the TOML config.
    """
    model_config = job_config.model
    
    return Qwen3ModelArgs(
        dim=model_config.dim,
        n_layers=model_config.n_layers,
        n_heads=model_config.n_heads,
        n_kv_heads=getattr(model_config, 'n_kv_heads', None),
        vocab_size=model_config.vocab_size,
        hidden_dim=model_config.hidden_dim,
        head_dim=getattr(model_config, 'head_dim', model_config.dim // model_config.n_heads),
        max_seq_len=model_config.max_seq_len,
        rope_theta=getattr(model_config, 'rope_theta', 1000000.0),
        norm_eps=getattr(model_config, 'norm_eps', 1e-6),
        qk_norm=getattr(model_config, 'qk_norm', True),
        depth_init=getattr(model_config, 'depth_init', True),
        moe_enabled=getattr(model_config, 'moe_enabled', False),
        attn_type=getattr(model_config, 'attn_type', 'sdpa'),
        attn_mask_type=getattr(model_config, 'attn_mask_type', 'causal'),
    )


def build_model(job_config: JobConfig, parallel_dims: ParallelDims) -> Qwen3Model:
    """
    Build Qwen3Model on meta device and apply parallelization.
    
    This follows the actual TorchTitan pattern:
    1. Build model on meta device (no memory allocation)
    2. Apply parallelization (TP, FSDP, etc.)
    3. Materialize to GPU
    4. Initialize weights
    """
    # Build model arguments from config
    model_args = build_model_args(job_config)
    
    # Build on meta device (no memory allocation yet)
    with torch.device("meta"):
        model = Qwen3Model(model_args)
    
    rank = dist.get_rank()
    if rank == 0:
        print(f"Model created on meta device: {model_args.n_layers} layers, {model_args.dim} dim")
    
    # Apply parallelization using the actual TorchTitan function
    model = parallelize_qwen3(
        model=model,
        parallel_dims=parallel_dims,
        job_config=job_config,
    )
    
    if rank == 0:
        print("Applied parallelization to model")
    
    # Materialize model to GPU (allocate memory)
    model = model.to_empty(device="cuda")
    
    if rank == 0:
        print("Model materialized on cuda")
    
    # Initialize weights (CRITICAL - without this, weights are uninitialized!)
    model.init_weights(buffer_device=torch.device("cuda"))
    
    if rank == 0:
        print("Weights initialized")
    
    return model



# ==============================================================================
# Training Loop
# ==============================================================================

def create_dummy_batch(
    batch_size: int,
    seq_len: int,
    vocab_size: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Create dummy training batch.
    Returns just input_ids - labels are computed from shifted inputs.
    """
    input_ids = torch.randint(
        0, vocab_size, (batch_size, seq_len), device=device, dtype=torch.long
    )
    return input_ids


def compute_loss(logits: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
    """
    Compute cross-entropy loss for causal language modeling.
    
    For RLVR: Replace this with your actual RL loss.
    This is just a standard LM loss for testing the training infrastructure.
    """
    # Shift so that tokens < n predict n
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = input_ids[..., 1:].contiguous()
    
    # Flatten the tokens
    loss_fct = nn.CrossEntropyLoss()
    loss = loss_fct(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1)
    )
    
    return loss


def train_step(
    model: Qwen3Model,
    optimizer: torch.optim.Optimizer,
    batch_size: int,
    seq_len: int,
    vocab_size: int,
) -> float:
    """
    Execute one training step with dummy data.
    
    For actual RLVR training:
    - Replace dummy data with real rollouts from vLLM
    - Replace loss computation with GRPO/PPO loss
    - Add value function if needed
    """
    model.train()
    
    # Create dummy batch
    input_ids = create_dummy_batch(
        batch_size, seq_len, vocab_size, torch.device("cuda")
    )
    
    # Forward pass - Qwen3Model just returns logits
    # attention_masks=None works for 'sdpa' attention type
    logits = model(input_ids, attention_masks=None)
    
    # Compute loss (replace with RLVR loss)
    loss = compute_loss(logits, input_ids)
    
    # Backward pass
    loss.backward()
    
    # Gradient clipping
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    
    # Optimizer step
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    
    return loss.item()


# ==============================================================================
# Checkpointing (Simplified)
# ==============================================================================

def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    checkpoint_dir: str,
) -> None:
    """
    Save checkpoint using standard torch.save.
    For distributed checkpointing, use:
    - torch.distributed.checkpoint for FSDP2 models
    - torchtitan.checkpointing utilities
    """
    if dist.get_rank() != 0:
        return
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f"step_{step}.pt")
    
    # For FSDP models, you'd use distributed checkpoint APIs
    torch.save({
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, checkpoint_path)
    
    logger.info(f"Saved checkpoint to {checkpoint_path}")



# ==============================================================================
# Main Training Function
# ==============================================================================

def main(config_path: str) -> None:
    """Main training loop following torchtitan patterns"""
    
    # 1. Parse configuration
    job_config = JobConfig()
    job_config.parse_args([f"--job.config_file={config_path}"])
    
    # 2. Initialize distributed training
    # torchtitan handles this through environment variables
    # Typically called via: torchrun --nproc_per_node=8 training.py --config=...
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    # 3. Set up parallel dimensions
    parallel_dims = ParallelDims(
        dp=getattr(job_config.training, 'data_parallel_degree', -1),
        tp=getattr(job_config.training, 'tensor_parallel_degree', 1),
        pp=getattr(job_config.training, 'pipeline_parallel_degree', 1),
        world_size=world_size,
        enable_loss_parallel=True,
    )
    
    if rank == 0:
        print(f"=" * 80)
        print(f"Starting RLVR Training")
        print(f"Config: {config_path}")
        print(f"World size: {world_size}")
        print(f"Parallel dims: dp={parallel_dims.dp}, tp={parallel_dims.tp}, pp={parallel_dims.pp}")
        print(f"=" * 80)
    
    # 4. Build model with parallelization (using actual TorchTitan function)
    if rank == 0:
        print("Building and parallelizing Qwen3 model...")
    
    model = build_model(job_config, parallel_dims)
    
    # 5. Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=job_config.optimizer.lr,
        betas=tuple(job_config.optimizer.betas),
        eps=job_config.optimizer.eps,
        weight_decay=job_config.optimizer.weight_decay,
        fused=getattr(job_config.optimizer, 'fused', True),
    )
    
    # 6. Training loop
    num_steps = job_config.training.steps
    batch_size = job_config.training.batch_size
    seq_len = job_config.training.seq_len
    vocab_size = job_config.model.vocab_size
    log_freq = job_config.metrics.log_freq
    checkpoint_interval = job_config.checkpoint.interval
    
    if rank == 0:
        print(f"\nStarting training for {num_steps} steps")
        print(f"Batch size: {batch_size}, Seq len: {seq_len}")
        print("=" * 80)
    
    for step in range(num_steps):
        t0 = time.perf_counter()
        
        # Training step with dummy data
        # TODO: Replace with actual RLVR data from vLLM rollouts
        loss = train_step(model, optimizer, batch_size, seq_len, vocab_size)
        
        dt = time.perf_counter() - t0
        
        # Logging
        if rank == 0 and step % log_freq == 0:
            tokens_per_sec = batch_size * seq_len * world_size / dt
            print(
                f"[Step {step:5d}/{num_steps}] "
                f"loss={loss:.4f} | "
                f"time={dt:.3f}s | "
                f"tok/s={tokens_per_sec:.0f}"
            )
        
        # Checkpointing
        if (step + 1) % checkpoint_interval == 0:
            save_checkpoint(
                model, optimizer, step + 1,
                job_config.checkpoint.folder
            )
    
    if rank == 0:
        print("=" * 80)
        print("Training completed successfully!")
        print("=" * 80)
    
    # Cleanup
    dist.destroy_process_group()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Minimal TorchTitan training for RLVR")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to TOML config file",
    )
    args = parser.parse_args()
    
    main(args.config)