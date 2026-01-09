"""Test if different TP degrees load the same weights from HF checkpoint.

This test checks if the weights themselves are identical (after full_tensor),
or if the numerical difference comes from the forward pass computation.
"""

import os
import asyncio
import pytest
import torch
import yaml

from rlvr_experiments.runtime import Runtime


def _create_config(tmp_path, assets_path, tp_degree, role_name="model"):
    """Create a config with specified parallelism."""
    roles = [
        {
            "name": role_name,
            "kind": "titan",
            "config": {
                "trainable": False,
                "profiling": {"enable_profiling": False},
                "metrics": {"log_freq": 1, "enable_tensorboard": False},
                "model": {
                    "name": "qwen3",
                    "flavor": "0.6B",
                    "hf_assets_path": assets_path,
                },
                "optimizer": {"name": "AdamW", "lr": 1e-5, "eps": 1e-4},
                "lr_scheduler": {"warmup_steps": 0},
                "training": {
                    "seq_len": 512,
                    "dtype": "bfloat16",
                    "mixed_precision_param": "bfloat16",
                    "mixed_precision_reduce": "float32",
                },
                "parallelism": {
                    "data_parallel_replicate_degree": 1,
                    "data_parallel_shard_degree": 1,
                    "fsdp_reshard_after_forward": "default",
                    "tensor_parallel_degree": tp_degree,
                    "context_parallel_degree": 1,
                    "disable_loss_parallel": True,
                },
                "checkpoint": {
                    "enable": True,
                    "initial_load_in_hf": True,
                    "initial_load_model_only": True,
                },
                "activation_checkpoint": {"mode": "none", "selective_ac_option": "op"},
                "compile": {"enable": False},
            },
        },
    ]

    config = {
        "run": {"name": f"tp_weight_test_{role_name}"},
        "model": {"path": assets_path},
        "tokenizer": {
            "pretrained_model_name_or_path": "Qwen/Qwen3-0.6B",
            "use_fast": False,
        },
        "training": {
            "num_epochs": 1,
            "iterations_per_epoch": 1,
            "sync_reference_every": 1,
            "train_batch_size": 1,
        },
        "verifier": {"num_workers": 1},
        "loss": {"beta": 0.1, "eps": 0.2},
        "data": {"dataset": "dummy"},
        "data_iter": {
            "batch_size": 1,
            "system_prompt": "",
            "assistant_prefix": "",
        },
        "sampling": {
            "temperature": 0.0,
            "max_tokens": 32,
            "n": 1,
            "logprobs": 0,
        },
        "buffer": {"max_reads": 1},
        "roles": roles,
        "sync": {
            "chunk_mb": 100,
            "wiring": [],
        },
    }

    config_path = tmp_path / f"tp_weight_test_{role_name}.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f)

    return str(config_path)


@pytest.fixture(scope="module")
def assets_path():
    """Path to 0.6B model assets."""
    return os.environ.get("NIGHTLY_ASSETS_PATH", "/efs/rlvr-experiments/assets/hf/Qwen3-0.6B")


@pytest.fixture(scope="module")
def event_loop():
    """Create event loop for module-scoped async fixtures."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.mark.integration
@pytest.mark.gpu
class TestTPWeights:
    """Test that weights are identical across different TP degrees."""

    @pytest.mark.asyncio
    async def test_tp1_vs_tp2_weights_match(self, tmp_path_factory, assets_path):
        """Check if TP=1 and TP=2 models have identical weights after HF loading."""
        tmp_path = tmp_path_factory.mktemp("tp_weights")

        config_tp1 = _create_config(tmp_path, assets_path, tp_degree=1, role_name="model_tp1")
        config_tp2 = _create_config(tmp_path, assets_path, tp_degree=2, role_name="model_tp2")

        runtime_tp1 = await Runtime.from_plan(config_tp1)
        await runtime_tp1.start()

        runtime_tp2 = await Runtime.from_plan(config_tp2)
        await runtime_tp2.start()

        try:
            model_tp1 = runtime_tp1.roles["model_tp1"]
            model_tp2 = runtime_tp2.roles["model_tp2"]

            # Get HF state dicts from both models
            # This calls full_tensor() on DTensors internally
            hf_sd_tp1 = await model_tp1.get_hf_state_dict()
            hf_sd_tp2 = await model_tp2.get_hf_state_dict()

            print(f"\n=== Comparing HF state dicts ===")
            print(f"TP1 has {len(hf_sd_tp1)} keys")
            print(f"TP2 has {len(hf_sd_tp2)} keys")

            # Check each parameter
            max_diffs = []
            for key in hf_sd_tp1.keys():
                if key not in hf_sd_tp2:
                    print(f"WARNING: Key {key} missing from TP2 state dict")
                    continue

                t1 = hf_sd_tp1[key]
                t2 = hf_sd_tp2[key]

                if t1.shape != t2.shape:
                    print(f"WARNING: Shape mismatch for {key}: {t1.shape} vs {t2.shape}")
                    continue

                diff = (t1.float() - t2.float()).abs()
                max_diff = diff.max().item()
                max_diffs.append((key, max_diff))

                if max_diff > 0:
                    print(f"  {key}: max_diff={max_diff:.10f}")

            # Sort by max_diff
            max_diffs.sort(key=lambda x: -x[1])

            print(f"\nTop 10 parameters with largest differences:")
            for key, diff in max_diffs[:10]:
                print(f"  {key}: {diff:.10f}")

            overall_max = max(d for _, d in max_diffs) if max_diffs else 0
            print(f"\nOverall max weight difference: {overall_max:.10f}")

            # If weights are identical but logprobs differ, the issue is in forward pass
            # If weights differ, the issue is in checkpoint loading with different TP
            if overall_max < 1e-6:
                print("CONCLUSION: Weights are identical - difference comes from forward pass computation")
            else:
                print("CONCLUSION: Weights differ - difference comes from checkpoint loading")

        finally:
            pass
