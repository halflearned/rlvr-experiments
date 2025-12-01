import os
import time
from typing import Optional

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh

# ---- TorchTitan imports ----
from torchtitan.config import JobConfig, TORCH_DTYPE_MAP
from torchtitan.distributed import dist_utils
from torchtitan.distributed.parallel_dims import ParallelDims
from torchtitan.tools.logging import logger

# Qwen3 model + infra; adjust paths to whatever your checkout uses
from torchtitan.models.qwen3.model.model import Qwen3Model

# Optional: Titan’s distributed checkpointing utils
from torchtitan.components.checkpoint import (
    save_distributed_checkpoint,
    load_distributed_checkpoint,
)


# ------------------------------------------------------------------
# 1. Distributed init: world_size + ParallelDims + DeviceMesh
# ------------------------------------------------------------------

def init_distributed(job_config: JobConfig) -> ParallelDims:
    """
    Titan-style distributed init that returns a ParallelDims instance.

    This mirrors what you saw in torchtitan/train.py:
      world_size = dist_utils.init_distributed(...)
      parallel_dims = ParallelDims(...)
    """

    world_size = dist_utils.init_distributed(
        job_config.comm,
        enable_cpu_backend=job_config.training.enable_cpu_offload,
        base_folder=job_config.job.dump_folder,
    )

    p = job_config.parallelism

    parallel_dims = ParallelDims(
        dp_shard=p.data_parallel_shard_degree,          # -1 means "infer"
        dp_replicate=p.data_parallel_replicate_degree,
        cp=p.context_parallel_degree,                   # we'll keep this = 1 for now
        tp=p.tensor_parallel_degree,                    # e.g. 1, 2, 4
        pp=p.pipeline_parallel_degree,                  # keep 1 for now
        world_size=world_size,
        enable_loss_parallel=not p.disable_loss_parallel,
    )

    # Build the world mesh and stash it for later (this is what parallelize_qwen3 expects)
    world_mesh: DeviceMesh = parallel_dims.build_mesh("cuda")
    parallel_dims.world_mesh = world_mesh  # type: ignore[attr-defined]

    if dist.get_rank() == 0:
        logger.info(f"Initialized distributed. WORLD_SIZE={world_size}")
        logger.info(f"ParallelDims: {parallel_dims}")

    return parallel_dims


# ------------------------------------------------------------------
# 2. Build Qwen3 on meta device
# ------------------------------------------------------------------

def build_qwen3_model_meta(job_config: JobConfig) -> torch.nn.Module:
    """
    Build the Qwen3 model on the meta device (no memory allocation yet).

    The exact constructor depends on how Qwen3 is wired in Titan.
    Here I assume a Llama-like pattern where JobConfig.model holds the
    architecture hyperparameters.
    """

    model_args = job_config.model  # usually a structured config / dataclass

    with torch.device("meta"):
        model = Qwen3ForCausalLM(model_args)

    return model


# ------------------------------------------------------------------
# 3. Apply Titan parallelisms (TP + FSDP2) for non-MoE Qwen3
# ------------------------------------------------------------------

def parallelize_qwen3_non_moe(
    model: torch.nn.Module,
    job_config: JobConfig,
    parallel_dims: ParallelDims,
) -> torch.nn.Module:
    """
    Apply Tensor Parallel + (H)FSDP2, no MoE, no Context Parallel.

    This is basically a thin wrapper over the `parallelize_qwen3` you pasted.
    We rely on TorchTitan’s own infra to:
      - set up tensor parallel on the Q/K/V/FFN linears
      - apply activation checkpointing
      - apply FSDP2 with dp_shard_cp mesh
    """

    model = parallelize_qwen3(  # WHAT SHOULD THIS FUNCTION BE?
        model=model,
        parallel_dims=parallel_dims,
        job_config=job_config,
    )

    # At this point:
    # - TP is applied if job_config.parallelism.tensor_parallel_degree > 1
    # - FSDP2 or HSDP is applied if dp_shard or dp_replicate > 1
    # - model is sharded and wrapped in FSDP2 modules where needed

    # Materialize parameters onto CUDA after parallelization
    model = model.to_empty(device="cuda")

    return model


# ------------------------------------------------------------------
# 4. Dummy training step (just exercise the distributed stack)
# ------------------------------------------------------------------

def training_step(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    seq_len: int,
    vocab_size: int,
) -> float:
    """
    Minimal step: synthetic data -> forward -> dummy loss -> backward -> step.

    We keep this extremely simple. The point is to stress the Titan
    parallelism stack, not the actual RLVR objective.
    """

    model.train()

    # Fake batch (batch size 1 local, you can bump it to probe MFU)
    bs = 1
    device = torch.device("cuda")

    input_ids = torch.randint(
        0,
        vocab_size,
        (bs, seq_len),
        device=device,
        dtype=torch.long,
    )

    # Standard LM loss: next-token prediction
    outputs = model(
        input_ids=input_ids,
        labels=input_ids,
    )

    # `outputs.loss` should already be a scalar on each rank
    loss = outputs.loss

    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    return float(loss.detach().cpu())


# ------------------------------------------------------------------
# 5. Checkpointing helpers (Titan-style)
# ------------------------------------------------------------------

def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    job_config: JobConfig,
    step: int,
) -> None:
    """
    Titan-style distributed checkpoint save.

    Titan uses DCP (distributed checkpoint) format so that you can
    restart with the same parallelism, or convert to HF weights, etc.
    """

    rank = dist.get_rank()
    ckpt_dir = os.path.join(job_config.job.dump_folder, f"step_{step:08d}")
    os.makedirs(ckpt_dir, exist_ok=True)

    state = {
        "step": step,
        "optimizer": optimizer.state_dict(),
    }

    # Model state is handled by save_distributed_checkpoint, not stuffed into `state`
    save_distributed_checkpoint(
        state=state,
        model=model,
        checkpoint_dir=ckpt_dir,
        tag=f"step_{step:08d}",
    )

    if rank == 0:
        logger.info(f"Saved distributed checkpoint to {ckpt_dir}")


def maybe_load_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    job_config: JobConfig,
) -> int:
    """
    Optionally resume from the latest checkpoint in job_config.job.dump_folder.

    Implementation left simple; you can add "find latest" logic.
    Returns the starting step.
    """

    # For now: no resume
    return 0


# ------------------------------------------------------------------
# 6. Main: tie everything together
# ------------------------------------------------------------------

def main(config_path: str) -> None:
    # 1) Parse JobConfig from TOML / whatever Titan uses
    job_config = JobConfig.from_file(config_path)  # adjust to your API

    # 2) Init distributed + mesh
    parallel_dims = init_distributed(job_config)
    rank = dist.get_rank()

    # 3) Build Qwen3 on meta device
    if rank == 0:
        logger.info("Building Qwen3 model on meta device...")
    model = build_qwen3_model_meta(job_config)

    # 4) Apply TP + FSDP2
    if rank == 0:
        logger.info("Applying TorchTitan parallelisms (TP + FSDP2)...")
    model = parallelize_qwen3_non_moe(model, job_config, parallel_dims)

    # 5) Optimizer
    mp_param_dtype = TORCH_DTYPE_MAP[job_config.training.mixed_precision_param]
    lr = job_config.training.learning_rate
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        fused=True,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=job_config.training.weight_decay,
    )

    # 6) Optional: resume
    start_step = maybe_load_checkpoint(model, optimizer, job_config)

    # 7) Dummy training loop
    num_steps = job_config.training.num_steps
    vocab_size = job_config.model.vocab_size
    seq_len = job_config.training.seq_len

    if rank == 0:
        logger.info(f"Starting dummy training: {num_steps} steps from step {start_step}")

    for step in range(start_step, num_steps):
        t0 = time.time()
        loss = training_step(model, optimizer, seq_len, vocab_size)
        dt = time.time() - t0

        if rank == 0 and (step % job_config.training.log_interval == 0):
            logger.info(
                f"[step {step}] loss={loss:.4f}, step_time={dt:.3f}s"
            )

        if rank == 0 and (step + 1) % job_config.training.ckpt_interval == 0:
            save_checkpoint(model, optimizer, job_config, step + 1)

    if rank == 0:
        logger.info("Finished dummy run.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to TorchTitan JobConfig TOML",
    )
    args = parser.parse_args()

    main(args.config)
