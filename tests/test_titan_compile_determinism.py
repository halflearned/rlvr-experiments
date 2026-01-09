"""Test whether TorchTitan compiled models produce identical outputs for train vs eval mode.

Since we can't run two TitanModels in the same process (distributed init conflict),
we test a single model and check:
1. Whether forward passes are deterministic (same input -> same output)
2. Whether train/eval mode switch affects output
"""

import os
import sys
import tempfile
import torch
import tomli_w as toml

# Need to set up distributed environment for TitanModel
os.environ["RANK"] = "0"
os.environ["WORLD_SIZE"] = "1"
os.environ["LOCAL_RANK"] = "0"
os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "29599"

from transformers import AutoTokenizer


def create_titan_config(compile_enabled: bool) -> str:
    """Create a TOML config file for TitanModel and return its path."""
    config = {
        "job": {"dump_folder": "/tmp/titan_test"},
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


def test_titan_train_eval_mode(compile_enabled: bool):
    """Test if train/eval mode affects output for a single TitanModel."""
    from torchtitan.config import ConfigManager
    from rlvr_experiments.model import TitanModel
    from torch.distributed.tensor import DTensor

    config_path = create_titan_config(compile_enabled=compile_enabled)

    try:
        config = ConfigManager().parse_args(["--job.config-file", config_path])
        print(f"\nLoading model (compile={compile_enabled})...")
        model = TitanModel(config, trainable=True)
    finally:
        os.unlink(config_path)

    # Create test input
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    text = "The quick brown fox jumps over the lazy dog."
    input_ids = tokenizer(text, return_tensors="pt")["input_ids"].to(model.device)

    print(f"Input shape: {input_ids.shape}")

    # Test 1: Determinism in train mode
    print("\n=== Test 1: Determinism in train mode ===")
    model.model_parts[0].train()
    with torch.no_grad():
        out1 = model.forward(input_ids)
        out2 = model.forward(input_ids)

    if isinstance(out1, DTensor):
        out1 = out1.full_tensor()
    if isinstance(out2, DTensor):
        out2 = out2.full_tensor()

    diff_train = (out1 - out2).abs().max().item()
    print(f"Max diff (train mode, two calls): {diff_train}")
    test1_pass = diff_train == 0.0

    # Test 2: Determinism in eval mode
    print("\n=== Test 2: Determinism in eval mode ===")
    model.model_parts[0].eval()
    with torch.no_grad():
        out3 = model.forward(input_ids)
        out4 = model.forward(input_ids)

    if isinstance(out3, DTensor):
        out3 = out3.full_tensor()
    if isinstance(out4, DTensor):
        out4 = out4.full_tensor()

    diff_eval = (out3 - out4).abs().max().item()
    print(f"Max diff (eval mode, two calls): {diff_eval}")
    test2_pass = diff_eval == 0.0

    # Test 3: Train vs eval mode
    print("\n=== Test 3: Train vs eval mode (same weights) ===")
    diff_mode = (out1 - out3).abs().max().item()
    print(f"Max diff (train vs eval): {diff_mode}")
    test3_pass = diff_mode == 0.0

    if not test3_pass:
        print("\nDiagnostics:")
        print(f"  Train mode logits: [{out1.min().item():.4f}, {out1.max().item():.4f}]")
        print(f"  Eval mode logits:  [{out3.min().item():.4f}, {out3.max().item():.4f}]")
        diff_tensor = (out1 - out3).abs()
        print(f"  Mean diff: {diff_tensor.mean().item():.6f}")

    return test1_pass, test2_pass, test3_pass


def save_output(compile_enabled: bool):
    """Run model and save output to file for cross-process comparison."""
    from torchtitan.config import ConfigManager
    from rlvr_experiments.model import TitanModel
    from torch.distributed.tensor import DTensor
    import json

    config_path = create_titan_config(compile_enabled=compile_enabled)

    try:
        config = ConfigManager().parse_args(["--job.config-file", config_path])
        print(f"Loading model (compile={compile_enabled})...")
        model = TitanModel(config, trainable=True)
    finally:
        os.unlink(config_path)

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    text = "The quick brown fox jumps over the lazy dog."
    input_ids = tokenizer(text, return_tensors="pt")["input_ids"].to(model.device)

    # Run in eval mode (like reference model)
    model.model_parts[0].eval()
    with torch.no_grad():
        out = model.forward(input_ids)

    if isinstance(out, DTensor):
        out = out.full_tensor()

    # Save to file
    out_cpu = out.float().cpu()
    filename = f"/tmp/titan_output_compile_{compile_enabled}.pt"
    torch.save(out_cpu, filename)
    print(f"Saved output to {filename}")
    print(f"Output shape: {out_cpu.shape}")
    print(f"Output range: [{out_cpu.min().item():.4f}, {out_cpu.max().item():.4f}]")
    print(f"First 5 logits: {out_cpu[0, 0, :5].tolist()}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--compile", type=str, choices=["true", "false"])
    parser.add_argument("--compare", action="store_true", help="Compare saved outputs")
    args = parser.parse_args()

    if args.compare:
        # Compare mode - load both files and compare
        out_true = torch.load("/tmp/titan_output_compile_True.pt")
        out_false = torch.load("/tmp/titan_output_compile_False.pt")
        diff = (out_true - out_false).abs().max().item()
        print(f"Max diff between compile=true and compile=false: {diff}")
        if diff == 0.0:
            print("PASS: Compiled and non-compiled outputs are identical")
        else:
            print(f"FAIL: Outputs differ by {diff}")
            mean_diff = (out_true - out_false).abs().mean().item()
            print(f"Mean diff: {mean_diff}")
    else:
        # Run mode - save output
        compile_enabled = args.compile == "true"
        save_output(compile_enabled)
