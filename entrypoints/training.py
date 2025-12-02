import os
import glob
import torch
import torch.distributed as dist

from safetensors.torch import load_file
from torch.distributed.tensor import DTensor

from torchtitan.config import ConfigManager, TORCH_DTYPE_MAP
from torchtitan.tools.logging import init_logger, logger
from torchtitan.tools import utils
from torchtitan.distributed import ParallelDims, utils as dist_utils
import torchtitan.protocols.train_spec as train_spec_module

from torchtitan.models.qwen3.model.state_dict_adapter import Qwen3StateDictAdapter
from torchtitan.models.qwen3.infra.parallelize import parallelize_qwen3


def build_job_config(args) -> "JobConfig":
    cfg_mgr = ConfigManager()
    return cfg_mgr.parse_args(["--job.config-file", args.config])


def init_distributed(job_config) -> ParallelDims:
    world_size = dist_utils.init_distributed(
        job_config.comm,
        enable_cpu_backend=job_config.training.enable_cpu_offload,
        base_folder=job_config.job.dump_folder,
    )
    if world_size is None:
        world_size = dist.get_world_size()

    p = job_config.parallelism
    pdims = ParallelDims(
        dp_shard=p.data_parallel_shard_degree,
        dp_replicate=p.data_parallel_replicate_degree,
        cp=p.context_parallel_degree,
        tp=p.tensor_parallel_degree,
        pp=p.pipeline_parallel_degree,
        ep=p.expert_parallel_degree,
        etp=p.expert_tensor_parallel_degree,
        world_size=world_size,
    )
    logger.info(
        f"ParallelDims: world_size={pdims.world_size}, "
        f"dp_shard={pdims.dp_shard}, dp_replicate={pdims.dp_replicate}, "
        f"tp={pdims.tp}, cp={pdims.cp}, pp={pdims.pp}, ep={pdims.ep}, etp={pdims.etp}"
    )
    return pdims


def build_qwen3(job_config):
    train_spec = train_spec_module.get_train_spec(job_config.model.name)
    model_args = train_spec.model_args[job_config.model.flavor]
    model_args.update_from_config(job_config)

    logger.info(f"Building {job_config.model.name} {job_config.model.flavor} with args: {model_args}")
    with utils.set_default_dtype(TORCH_DTYPE_MAP[job_config.training.dtype]):
        model = train_spec.model_cls(model_args)

    return model, model_args


def load_hf_weights(model, model_args, job_config):
    hf_dir = job_config.model.hf_assets_path
    shard_paths = sorted(glob.glob(os.path.join(hf_dir, "*.safetensors")))

    logger.info(f"Loading HF weights from shards: {shard_paths}")
    hf_state = {}
    for shard in shard_paths:
        logger.info(f"  loading shard: {shard}")
        hf_state.update(load_file(shard))

    adapter = Qwen3StateDictAdapter(model_args, hf_assets_path=hf_dir)
    titan_state = adapter.from_hf(hf_state)

    missing, unexpected = model.load_state_dict(titan_state, strict=False)
    logger.info(f"Loaded Qwen3 HF weights. Missing keys: {missing}")
    logger.info(f"Unexpected keys: {unexpected}")
    return model


def run_smoke_test(model, job_config):

    device_type = utils.device_type
    device = torch.device(f"{device_type}:{int(os.environ.get('LOCAL_RANK', 0))}")

    model.to(device)
    model.train()

    bsz = job_config.training.local_batch_size
    seq_len = job_config.training.seq_len
    vocab_size = model.vocab_size

    tokens = torch.randint(0, vocab_size, (bsz, seq_len), device=device, dtype=torch.long)
    logits = model(tokens)
    loss = logits.mean()

    logger.info(f"Dummy loss (pre-backward): {loss.item():.6f}")
    loss.backward()
    logger.info("Backward pass completed.")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Start vLLM inference server")
    parser.add_argument("config", type=str, help="Path to config file")
    args, kwargs = parser.parse_known_args()

    init_logger()

    job_config = build_job_config(args)
    logger.info(
        f"JobConfig loaded: model={job_config.model.name}, "
        f"flavor={job_config.model.flavor}, hf_assets_path={job_config.model.hf_assets_path}"
    )

    device_module, device_type = utils.device_module, utils.device_type
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device(f"{device_type}:{local_rank}")
    device_module.set_device(device)
    logger.info(f"Using device {device} on LOCAL_RANK={local_rank}")

    pdims = init_distributed(job_config)

    model, model_args = build_qwen3(job_config)
    model = load_hf_weights(model, model_args, job_config)

    model = parallelize_qwen3(model, pdims, job_config)
    logger.info("Applied parallelize_qwen3 to model")

    run_smoke_test(model, job_config)
    logger.info("Qwen3 parallel fwd/bwd smoke test complete.")

    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
