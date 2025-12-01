# entrypoints/training.py

import argparse
import os
import time
from typing import Optional

import torch
import torch.distributed as dist

from torch.distributed.device_mesh import DeviceMesh

# ---- TorchTitan imports ----
from torchtitan.config_manager import JobConfig, TORCH_DTYPE_MAP
from torchtitan.distributed import dist_utils
from torchtitan.distributed.parallel_dims import ParallelDims
from torchtitan.tools.logging import logger
from torchtitan.components.checkpoint import (
    save_distributed_checkpoint,
    load_distributed_checkpoint,
)

# Qwen3 model + infra
from torchtitan.models.qwen3.model.model import Qwen3Model, Qwen3ModelArgs
from torchtitan.models.qwen3.infra.parallelize import parallelize_qwen3  # adjust if needed


# ------------------------------------------------------------------
# 1. Distributed init: world_size + ParallelDims + DeviceMesh
# ------------------------------------------------------------------

def init_distributed(job_config: JobConfig) -> ParallelDims:
    """Mirror TorchTitan's init path and construct ParallelDims."""

    world_size = dist_utils.init_distributed(
        job_config.comm,
        enable_cpu_backend=job_config.training.enable_cpu_offload,
        base_folder=job_config.job.dump_folder,
    )

    p = job_config.parallelism

    parallel_dims = ParallelDims(
        dp_shard=p.data_parallel_shard_degree,       # -1 means "infer"
        dp_replicate=p.data_parallel_replicate_degree,
        cp=p.context_parallel_degree,
        tp=p.tensor_parallel_degree,
        pp=p.pipeline_parallel_degree,
        ep=p.expert_parallel_degree,
        etp=p.expert_tensor_parallel_degree,
        world_size=world_size,
        enable_loss_parallel=not p.disable_loss_parallel,
    )

    world_mesh: DeviceMesh = parallel_dims.build_mesh("cuda")
    parallel_dims.world_mesh = world_mesh  # for parallelize_qwen3

    if dist.get_rank() == 0:
        logger.info(f"Initialized distributed. WORLD_SIZE={world_size}")
        logger.info(f"ParallelDims: {parallel_dims}")

    return parallel_dims


# ------------------------------------------------------------------
# 2. Build Qwen3 on the meta device
# ------------------------------------------------------------------

def build_qwen3_model_meta(job_config: JobConfig) -> Qwen3Model:
    """
    Construct Qwen3ModelArgs + Qwen3Model on meta device.

    You need to decide how to map `job_config.model.flavor` etc
    into a concrete Qwen3ModelArgs instance. This function is the
    only place you have to touch for that.
    """

    # TODO: replace this with your real config wiring.
    # For example, if torchtitan provides a registry like:
    #   from torchtitan.models.qwen3.model.args import QWEN3_MODEL_CONFIGS
    #   model_args = QWEN3_MODEL_CONFIGS[job_config.model.flavor]
    #
    # For now, we assume you pass a fully-built Qwen3ModelArgs
    # via a custom import / extension.
    if not hasattr(job_config, "qwen3"):
        raise RuntimeError(
            "Expected job_config.qwen3 to hold a Qwen3ModelArgs-compatible dict "
            "(or add your own mapping in build_qwen3_model_meta)."
        )

    # Example: job_config.qwen3 is a simple namespace or dict
    qwen3_cfg = job_config.qwen3
    if isinstance(qwen3_cfg, dict):
        model_args = Qwen3ModelArgs(**qwen3_cfg)
    else:
        # argparse Namespace-ish: turn into kwargs
        model_args = Qwen3ModelArgs(**vars(qwen3_cfg))

    with torch.device("meta"):
        model = Qwen3Model(model_args)

    return model


# ------------------------------------------------------------------
# 3. Apply TorchTitan parallelisms (TP + FSDP2/HSDP/etc.)
# ------------------------------------------------------------------

def parallelize_qwen3_non_moe(
    model: Qwen3Model,
    job_config: JobConfig,
    parallel_dims: ParallelDims,
) -> Qwen3Model:
    """
    Wrap Qwen3 with Titan's TP/FSDP/etc using the official helper.

    Assumes a non-MoE config, but parallelize_qwen3 itself can handle MoE
    if/when you pass a MoE-enabled Qwen3ModelArgs.
    """

    model = parallelize_qwen3(
        model=model,
        parallel_dims=parallel_dims,
        job_config=job_config,
    )

    # Materialize parameters on CUDA *after* parallelization
    model = model.to_empty(device="cuda")
    model.init_weights(buffer_device=torch.device("cuda"))

    return model


# ------------------------------------------------------------------
# 4. Dummy LM training step (no RL yet, just stress the stack)
# ------------------------------------------------------------------

def dummy_lm_step(
    model: Qwen3Model,
    optimizer: torch.optim.Optimizer,
    seq_len: int,
    vocab_size: int,
) -> float:
    """
    Synthetic next-token prediction:

      - batch of random tokens
      - forward through Qwen3Model
      - manual cross entropy loss
      - backward + step

    This is intentionally small and stupid; the point is to ensure that
    your Titan parallelism + optimizer + checkpoint stack works.
    """

    model.train()
    device = torch.device("cuda")

    # Local batch size per rank; bump this to probe MFU later.
    bs = 1

    tokens = torch.randint(
        low=0,
        high=vocab_size,
        size=(bs, seq_len),
        device=device,
        dtype=torch.long,
    )

    # For now assume attn_type == "sdpa", so we don't need masks.
    # If you switch to "flex"/"varlen", you'll want:
    #   attention_masks = model.get_attention_masks(tokens, tokenizer)
    # and pass it below.
    logits = model(tokens)  # [bs, seq_len, vocab_size]

    # Shift for next-token prediction
    # Flatten for CE: (bs*(seq_len-1), vocab)
    logits_flat = logits[:, :-1, :].contiguous().view(-1, vocab_size)
    targets_flat = tokens[:, 1:].contiguous().view(-1)

    loss = torch.nn.functional.cross_entropy(
        logits_flat,
        targets_flat,
        reduction="mean",
    )

    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    return float(loss.detach().cpu())


# ------------------------------------------------------------------
# 5. Checkpointing helpers (Titan DCP)
# ------------------------------------------------------------------

def save_checkpoint(
    model: Qwen3Model,
    optimizer: torch.optim.Optimizer,
    job_config: JobConfig,
    step: int,
) -> None:
    rank = dist.get_rank()
    ckpt_dir = os.path.join(job_config.job.dump_folder, f"step_{step:08d}")
    os.makedirs(ckpt_dir, exist_ok=True)

    state = {
        "step": step,
        "optimizer": optimizer.state_dict(),
    }

    save_distributed_checkpoint(
        state=state,
        model=model,
        checkpoint_dir=ckpt_dir,
        tag=f"step_{step:08d}",
    )

    if rank == 0:
        logger.info(f"Saved distributed checkpoint to {ckpt_dir}")


def maybe_load_checkpoint(
    model: Qwen3Model,
    optimizer: torch.optim.Optimizer,
    job_config: JobConfig,
) -> int:
    # For now: no resume; stub where you’ll plug in DCP restore logic.
    # e.g. scan job_config.job.dump_folder for latest step_* dir and:
    #   load_distributed_checkpoint(...); restore optimizer & return step.
    return 0


# ------------------------------------------------------------------
# 6. Main: glue everything together
# ------------------------------------------------------------------

def main() -> None:
    # Parse TorchTitan JobConfig (standard pattern)
    job_config = JobConfig()
    job_config.parse()  # fills job_config.* from CLI / toml

    # Init distributed and parallel dims
    parallel_dims = init_distributed(job_config)
    rank = dist.get_rank()

    if rank == 0:
        logger.info("Building Qwen3 on meta device...")
    model = build_qwen3_model_meta(job_config)

    if rank == 0:
        logger.info("Applying TorchTitan parallelisms (TP/FSDP/...)")
    model = parallelize_qwen3_non_moe(model, job_config, parallel_dims)

    # Optimizer setup
    mp_param_dtype = TORCH_DTYPE_MAP[job_config.training.mixed_precision_param]
    # Note: Titan usually casts params for FSDP; we just read LR & weight decay.
    lr = job_config.training.learning_rate
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        fused=True,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=job_config.training.weight_decay,
    )

    start_step = maybe_load_checkpoint(model, optimizer, job_config)

    num_steps = job_config.training.steps
    seq_len = job_config.training.seq_len
    # For now we trust the model args for vocab size
    vocab_size = model.vocab_size

    if rank == 0:
        logger.info(f"Starting dummy training: {num_steps} steps from {start_step}")

    for step in range(start_step, num_steps):
        t0 = time.time()
        loss = dummy_lm_step(model, optimizer, seq_len, vocab_size)
        dt = time.time() - t0

        if rank == 0 and (step % job_config.training.log_freq == 0):
            logger.info(
                f"[step {step}] loss={loss:.4f} step_time={dt:.3f}s"
            )

        if rank == 0 and (step + 1) % job_config.training.ckpt_interval == 0:
            save_checkpoint(model, optimizer, job_config, step + 1)

    if rank == 0:
        logger.info("Finished dummy Qwen3 Titan run.")


if __name__ == "__main__":
    # TorchTitan normally consumes args via JobConfig; you can still add your own
    # app-level flags here if you want, but keeping it minimal:
    main()
