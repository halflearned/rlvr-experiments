import os
import torch

from torchtitan.config import ConfigManager, TORCH_DTYPE_MAP
from torchtitan.tools.logging import init_logger, logger
from torchtitan.tools import utils
import torchtitan.protocols.train_spec as train_spec_module

from torchtitan.models.qwen3.model.state_dict_adapter import Qwen3StateDictAdapter


def build_job_config() -> "JobConfig":
    """
    Use the same config path as torchtitan/train.py:
    - ConfigManager parses CLI args (including --config ...)
    - returns a fully-populated JobConfig with nested dataclasses
    """
    manual_args = [
        "--job.config-file",
        "configs/training.toml",
    ]
    config_manager = ConfigManager()
    job_config = config_manager.parse_args(manual_args)
    return job_config


def build_qwen3_model(job_config):
    """
    Mirror the Trainer's model construction, but stop before parallelism, loss, optimizer, etc.
    """
    # Get train spec for this model family (llama3, qwen3, etc.)
    train_spec = train_spec_module.get_train_spec(job_config.model.name)

    # Grab the correct flavor (e.g. "0.6B") args and update from config
    model_args = train_spec.model_args[job_config.model.flavor]
    model_args.update_from_config(job_config)

    logger.info(
        f"Building {job_config.model.name} {job_config.model.flavor} "
        f"with args: {model_args}"
    )

    # In Trainer they first build on meta to count params, then re-init; for our
    # simple script we can just build directly on CPU with the right dtype.
    with utils.set_default_dtype(TORCH_DTYPE_MAP[job_config.training.dtype]):
        model = train_spec.model_cls(model_args)

    return model, model_args


def load_hf_weights_into_qwen3(model, model_args, job_config):

    ckpt_path = os.path.join(job_config.model.hf_assets_path, "model.safetensors")
    hf_state_dict = torch.load(ckpt_path, map_location="cpu")

    adapter = Qwen3StateDictAdapter(model_args, hf_assets_path=job_config.model.hf_assets_path)
    titan_state_dict = adapter.from_hf(hf_state_dict)

    missing, unexpected = model.load_state_dict(titan_state_dict, strict=False)
    logger.info(f"Loaded Qwen3 HF weights. Missing keys: {missing}")
    logger.info(f"Unexpected keys: {unexpected}")

    return model


def main():
    init_logger()

    # 1) Config from TOML + CLI, the "native" torchtitan way
    job_config = build_job_config()
    print("Loaded model!")

    # 2) Build Qwen3 model & args
    model, model_args = build_qwen3_model(job_config)

    # 3) Load HF weights
    model = load_hf_weights_into_qwen3(model, model_args, job_config)

    model.eval()
    print("Qwen3 model ready:", job_config.model.name, job_config.model.flavor)


if __name__ == "__main__":
    main()
