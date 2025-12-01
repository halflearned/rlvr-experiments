import os
import argparse
import time
import torch
import torch.distributed as dist

from torch.distributed.device_mesh import init_device_mesh
from torch.distributed._composable.fsdp import fully_shard, MixedPrecisionPolicy
from torch.distributed._composable.replicate import replicate

# TorchTitan imports
from torchtitan.config import JobConfig
from torchtitan.distributed import ParallelDims
from torchtitan.utils import device_module, device_type
from torchtitan.parallelisms.parallel_dims import ParallelDims

# --- USER MODEL IMPORT ---
# Replace this with the actual path to your model class. 
# Since Qwen2.5/3 is architecturally Llama, you can often use Titan's Llama.
# from torchtitan.models.llama import Transformer as Qwen3Model 
# from torchtitan.models.llama import ModelArgs
from torchtitan.models.llama import Transformer, ModelArgs # Placeholder

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("job_config", type=str, required=True, help="Path to job config")
    return parser.parse_args()

def init_distributed_mesh(job_config: JobConfig):
    """
    Establish the DeviceMesh. 
    Critically, this controls the hierarchy of TP vs FSDP/DP.
    """
    # 1. Standard Torch Init
    dist.init_process_group(backend="nccl")
    
    # 2. Extract specific degrees from config
    #    (Assuming standard Titan JobConfig structure)
    p_config = job_config.parallelism
    world_size = dist.get_world_size()
    
    dims = ParallelDims(
        dp_shard=p_config.data_parallel_shard_degree,
        dp_replicate=p_config.data_parallel_replicate_degree,
        cp=p_config.context_parallel_degree,
        tp=p_config.tensor_parallel_degree,
        pp=p_config.pipeline_parallel_degree,
        world_size=world_size,
        enable_loss_parallel=not p_config.disable_loss_parallel
    )
    
    # 3. Build Mesh
    #    Standard hierarchy: (PP, CP, TP, DP) or (DP, TP) depending on needs.
    #    Titan's helper `build_mesh` handles the generic case well.
    mesh = dims.build_mesh(device_type)
    return mesh, dims

def apply_parallelisms(model: torch.nn.Module, mesh: dist.device_mesh.DeviceMesh, dims: ParallelDims):
    """
    The 'Hackable' High-MFU part. 
    Instead of a black-box function, we explicitly shard the layers.
    """
    
    # 1. Apply Tensor Parallelism (TP)
    #    If you are using Titan's Llama/Qwen definition, it likely handles TP 
    #    internal to the model class if passed a mesh, OR you apply it here 
    #    using `parallelize_module`.
    
    tp_mesh = mesh["tp"] if dims.tp > 1 else None
    
    if tp_mesh:
        # Pseudo-code for generic manual TP application if the model doesn't support auto-TP
        from torchtitan.parallelisms import parallelize_module
        from torchtitan.models.llama.parallelize import apply_tp
        
        # This function typically iterates over modules and applies 
        # ColwiseParallel/RowwiseParallel to Linear layers
        apply_tp(model, tp_mesh, verbose=True)

    # 2. Apply FSDP2 (Fully Sharded Data Parallel)
    #    This is preferred over FSDP1 for high MFU on H100s/A100s.
    
    dp_mesh = mesh["dp"] if "dp" in mesh.mesh_dim_names else None
    
    if dp_mesh:
        mp_policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
        
        # Shard every TransformerBlock
        for layer_id, transformer_block in model.layers.items():
            fully_shard(
                transformer_block, 
                mesh=dp_mesh, 
                mp_policy=mp_policy
            )
            
        # Shard the final output norm and head
        fully_shard(model.norm, mesh=dp_mesh, mp_policy=mp_policy)
        fully_shard(model.output, mesh=dp_mesh, mp_policy=mp_policy)
        
    return model

def main():
    args = get_args()
    
    # 1. Load Config
    job_config = JobConfig.from_file(args.job_config)
    
    # 2. Init Dist & Mesh
    mesh, dims = init_distributed_mesh(job_config)
    rank = dist.get_rank()
    
    if rank == 0:
        print(f"Distributed initialized. Mesh: {mesh}")

    # 3. Build Model (Meta Device first for memory efficiency)
    with torch.device("meta"):
        # Replace this with your Qwen3 config loading logic
        model_config = ModelArgs() 
        model = Transformer(model_config)

    # 4. Parallelize
    #    This moves parameters to GPU and applies sharding hooks
    apply_parallelisms(model, mesh, dims)
    
    # 5. Initialize Parameters
    #    Since we built on Meta, we need to materialize.
    #    If loading from checkpoint, use `load_distributed_checkpoint` here.
    #    For dummy training, we just materialize random weights.
    model.to_empty(device=device_type)
    model.init_weights() # Ensure your model has this, or manual init
    
    # 6. Optimizer
    #    Use FusedAdamW for speed
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=1e-4, 
        fused=True
    )
    
    # 7. Dummy Training Loop
    model.train()
    
    if rank == 0:
        print("Starting training loop...")

    # Dummy inputs
    bs, seq_len = 2, 1024
    dummy_input = torch.randint(0, 32000, (bs, seq_len), device=device_type)
    
    for step in range(10):
        t0 = time.perf_counter()
        
        optimizer.zero_grad()
        
        # Forward
        pred = model(dummy_input)
        
        # Dummy Loss (Must ensure valid gradient flow)
        loss = pred.mean() 
        
        # Backward
        loss.backward()
        
        # Clip Grad (Standard RLVR practice)
        model.clip_grad_norm_(1.0)
        
        # Step
        optimizer.step()
        
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        
        if rank == 0:
            print(f"Step {step} | Loss: {loss.item():.4f} | Time: {(t1-t0)*1000:.2f}ms")

    if rank == 0:
        print("Done.")

if __name__ == "__main__":
    main()