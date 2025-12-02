import os
import glob
import torch
import torch.distributed as dist
import requests
from time import sleep
from safetensors.torch import load_file
from openai import OpenAI

from torchtitan.config import ConfigManager, TORCH_DTYPE_MAP
from torchtitan.tools.logging import init_logger, logger
from torchtitan.tools import utils
from torchtitan.distributed import ParallelDims, utils as dist_utils
import torchtitan.protocols.train_spec as train_spec_module
from torchtitan.models.qwen3.model.state_dict_adapter import Qwen3StateDictAdapter
from torchtitan.models.qwen3.infra.parallelize import parallelize_qwen3


def init_distributed(job_config):
    world_size = dist_utils.init_distributed(
        job_config.comm,
        enable_cpu_backend=job_config.training.enable_cpu_offload,
        base_folder=job_config.job.dump_folder,
    ) 
    
    if world_size is None:
        world_size = dist.get_world_size()

    p = job_config.parallelism
    return ParallelDims(
        dp_shard=p.data_parallel_shard_degree,
        dp_replicate=p.data_parallel_replicate_degree,
        cp=p.context_parallel_degree,
        tp=p.tensor_parallel_degree,
        pp=p.pipeline_parallel_degree,
        ep=p.expert_parallel_degree,
        etp=p.expert_tensor_parallel_degree,
        world_size=world_size,
    )


def build_qwen3(job_config):
    train_spec = train_spec_module.get_train_spec(job_config.model.name)
    model_args = train_spec.model_args[job_config.model.flavor]
    model_args.update_from_config(job_config)

    with utils.set_default_dtype(TORCH_DTYPE_MAP[job_config.training.dtype]):
        return train_spec.model_cls(model_args), model_args


def load_hf_weights(model, model_args, job_config):
    shard_paths = sorted(glob.glob(os.path.join(job_config.model.hf_assets_path, "*.safetensors")))
    
    hf_state = {}
    for shard in shard_paths:
        hf_state.update(load_file(shard))

    adapter = Qwen3StateDictAdapter(model_args, hf_assets_path=job_config.model.hf_assets_path)
    model.load_state_dict(adapter.from_hf(hf_state), strict=False)
    return model


def wait_for_vllm(max_retries=60, delay=2):
    base_url = os.environ.get("VLLM_BASE_URL", "http://vllm:8000/v1")
    health_url = base_url.replace("/v1", "/health")

    for i in range(max_retries):
        try:
            if requests.get(health_url, timeout=2).status_code == 200:
                return
        except Exception:
            pass
        sleep(delay)

    raise RuntimeError(f"vLLM not ready after {max_retries * delay} seconds")


def call_vllm(prompt):
    base_url = os.environ.get("VLLM_BASE_URL", "http://vllm:8000/v1")
    client = OpenAI(base_url=base_url, api_key="dummy")
    resp = client.chat.completions.create(
        model="Qwen/Qwen3-0.6B",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=64,
        temperature=0,
    )
    return resp.choices[0].message.content


def run_smoke_test(model, job_config):
    wait_for_vllm()
    logger.info(f"vLLM rollout: {call_vllm('Say hello from the RLVR trainer.')[:200]!r}")

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device(f"{utils.device_type}:{local_rank}")
    
    model.to(device).train()

    tokens = torch.randint(
        0, model.vocab_size,
        (job_config.training.local_batch_size, job_config.training.seq_len),
        device=device
    )
    loss = model(tokens).mean()
    
    logger.info(f"dummy loss: {loss.item():.6f}")
    loss.backward()


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str)
    args, _ = parser.parse_known_args()

    init_logger()
    job_config = ConfigManager().parse_args(["--job.config-file", args.config])

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device(f"{utils.device_type}:{local_rank}")
    utils.device_module.set_device(device)

    pdims = init_distributed(job_config)
    model, model_args = build_qwen3(job_config)
    model = load_hf_weights(model, model_args, job_config)
    model = parallelize_qwen3(model, pdims, job_config)

    run_smoke_test(model, job_config)
    logger.info("Smoke test complete.")

    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()