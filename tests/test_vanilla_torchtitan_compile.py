"""Test whether vanilla TorchTitan compiled models produce identical outputs.

This test uses TorchTitan directly without the rlvr_experiments.model.TitanModel wrapper
to isolate whether the compile non-determinism bug is in TorchTitan itself or in
the wrapper.

Run as separate processes for compile=true and compile=false, then compare:
    python tests/test_vanilla_torchtitan_compile.py --compile false
    python tests/test_vanilla_torchtitan_compile.py --compile true
    python tests/test_vanilla_torchtitan_compile.py --compare
"""

import os
import sys
import tempfile
import torch
import tomli_w as toml

# Need to set up distributed environment for TorchTitan
os.environ["RANK"] = "0"
os.environ["WORLD_SIZE"] = "1"
os.environ["LOCAL_RANK"] = "0"
os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "29598"

from transformers import AutoTokenizer


def create_titan_config(compile_enabled: bool) -> str:
    """Create a TOML config file for vanilla TorchTitan and return its path."""
    config = {
        "job": {"dump_folder": "/tmp/titan_vanilla_test"},
        "profiling": {"enable_profiling": False},
        "metrics": {"log_freq": 0, "enable_tensorboard": False},
        "model": {
            "name": "qwen3",
            "flavor": "0.6B",
            "hf_assets_path": "/efs/rlvr-experiments/assets/hf/Qwen3-0.6B",
        },
        "optimizer": {"name": "AdamW", "lr": 1e-5, "eps": 1e-8},
        "lr_scheduler": {"warmup_steps": 2},
        "training": {
            "seq_len": 2048,
            "dtype": "bfloat16",
            "mixed_precision_param": "bfloat16",
            "mixed_precision_reduce": "float32",
            "steps": 100,
            "max_norm": 1.0,
        },
        "parallelism": {
            "data_parallel_replicate_degree": 1,
            "data_parallel_shard_degree": 1,
            "tensor_parallel_degree": 1,
            "context_parallel_degree": 1,
            "disable_loss_parallel": True,
        },
        "checkpoint": {
            "enable": True,
            "folder": "checkpoint",
            "initial_load_in_hf": True,
            "interval": 500,
        },
        "activation_checkpoint": {"mode": "none"},
        "compile": {"enable": compile_enabled, "components": ["model"]},
        "fault_tolerance": {},
    }

    with tempfile.NamedTemporaryFile(mode="wb", suffix=".toml", delete=False) as f:
        toml.dump(config, f)
        return f.name


def run_vanilla_torchtitan(compile_enabled: bool):
    """Run vanilla TorchTitan model and save output to file."""
    import torch.distributed as dist
    from torch.distributed.tensor import DTensor

    # TorchTitan imports
    import torchtitan.protocols.train_spec as train_spec_module
    from torchtitan.components.checkpoint import CheckpointManager
    from torchtitan.config import ConfigManager, TORCH_DTYPE_MAP
    from torchtitan.distributed import ParallelDims, utils as dist_utils
    from torchtitan.protocols.model_converter import build_model_converters
    from torchtitan.tools import utils as titan_utils

    config_path = create_titan_config(compile_enabled=compile_enabled)

    try:
        job_config = ConfigManager().parse_args(["--job.config-file", config_path])
        print(f"Loading vanilla TorchTitan model (compile={compile_enabled})...")

        # Device setup
        device_module, device_type = titan_utils.device_module, titan_utils.device_type
        device = torch.device(f"{device_type}:0")
        device_module.set_device(device)

        # Distributed init
        dist_utils.init_distributed(job_config.comm)

        # Parallel dimensions
        world_size = int(os.environ["WORLD_SIZE"])
        parallel_dims = ParallelDims(
            dp_shard=1,
            dp_replicate=1,
            cp=1,
            tp=1,
            pp=1,
            ep=1,
            etp=1,
            world_size=world_size,
        )

        # Get train spec and model args
        train_spec = train_spec_module.get_train_spec(job_config.model.name)
        model_args = train_spec.model_args[job_config.model.flavor]
        model_args.update_from_config(job_config)

        # State dict adapter
        sd_adapter = train_spec.state_dict_adapter(model_args, job_config.model.hf_assets_path)

        # Build model on meta device
        print("Building model...")
        with torch.device("meta"), titan_utils.set_default_dtype(
            TORCH_DTYPE_MAP[job_config.training.dtype]
        ):
            model = train_spec.model_cls(model_args)

        # Apply parallelism
        model_converters = build_model_converters(job_config, parallel_dims)
        model_converters.convert(model)
        model = train_spec.parallelize_fn(model, parallel_dims, job_config)

        # Initialize weights
        model.to_empty(device=device_type)
        print("Initializing model weights...")
        with torch.no_grad():
            model.init_weights(buffer_device=None)

        model_parts = [model]

        # Set up checkpointer and load weights
        checkpointer = CheckpointManager(
            dataloader=None,
            model_parts=model_parts,
            optimizers=None,
            lr_schedulers=None,
            states={},
            checkpoint_config=job_config.checkpoint,
            sd_adapter=sd_adapter,
            base_folder=job_config.job.dump_folder,
            ft_manager=None,
        )
        checkpointer.load(step=None)

        # Set to eval mode for inference
        model.eval()

        # Create test input
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
        text = "The quick brown fox jumps over the lazy dog."
        input_ids = tokenizer(text, return_tensors="pt")["input_ids"].to(device)

        print(f"Input shape: {input_ids.shape}")

        # Forward pass
        with torch.no_grad():
            out = model(input_ids)

        if isinstance(out, DTensor):
            out = out.full_tensor()

        # Save to file
        out_cpu = out.float().cpu()
        filename = f"/tmp/vanilla_titan_output_compile_{compile_enabled}.pt"
        torch.save(out_cpu, filename)
        print(f"Saved output to {filename}")
        print(f"Output shape: {out_cpu.shape}")
        print(f"Output range: [{out_cpu.min().item():.4f}, {out_cpu.max().item():.4f}]")
        print(f"First 5 logits: {out_cpu[0, 0, :5].tolist()}")

        # Cleanup
        checkpointer.close()
        dist.destroy_process_group()

    finally:
        os.unlink(config_path)


def compare_outputs():
    """Compare outputs from compile=true and compile=false runs."""
    try:
        out_true = torch.load("/tmp/vanilla_titan_output_compile_True.pt")
        out_false = torch.load("/tmp/vanilla_titan_output_compile_False.pt")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please run with --compile true and --compile false first")
        return

    diff = (out_true - out_false).abs().max().item()
    mean_diff = (out_true - out_false).abs().mean().item()

    print(f"\n=== Vanilla TorchTitan Compile Determinism Test ===")
    print(f"Max diff between compile=true and compile=false: {diff}")
    print(f"Mean diff: {mean_diff}")

    if diff == 0.0:
        print("PASS: Compiled and non-compiled outputs are identical")
    else:
        print(f"FAIL: Outputs differ")
        print(f"\nThis indicates that torch.compile produces different outputs")
        print(f"for vanilla TorchTitan, not just the wrapper.")
        print(f"\nFirst 5 logits comparison:")
        print(f"  compile=false: {out_false[0, 0, :5].tolist()}")
        print(f"  compile=true:  {out_true[0, 0, :5].tolist()}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--compile", type=str, choices=["true", "false"])
    parser.add_argument("--compare", action="store_true", help="Compare saved outputs")
    args = parser.parse_args()

    if args.compare:
        compare_outputs()
    elif args.compile is not None:
        compile_enabled = args.compile == "true"
        run_vanilla_torchtitan(compile_enabled)
    else:
        print("Usage:")
        print("  python tests/test_vanilla_torchtitan_compile.py --compile false")
        print("  python tests/test_vanilla_torchtitan_compile.py --compile true")
        print("  python tests/test_vanilla_torchtitan_compile.py --compare")
