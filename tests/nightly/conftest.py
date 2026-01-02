"""Nightly test fixtures and utilities."""

import os
import pytest
import tempfile
import yaml


@pytest.fixture(scope="module")
def nightly_assets_path():
    """Path to model assets for nightly tests."""
    # Default path - override with env var if needed
    return os.environ.get("NIGHTLY_ASSETS_PATH", "/efs/rlvr-experiments/assets/hf/Qwen3-0.6B")


@pytest.fixture(scope="module")
def mini_config_template():
    """Base config template for nightly tests - minimal settings for fast runs."""
    return {
        "run": {"name": "nightly_test"},
        "model": {"path": None},  # Set by test
        "tokenizer": {
            "pretrained_model_name_or_path": "Qwen/Qwen3-0.6B",
            "use_fast": False,
        },
        "training": {
            "num_epochs": 2,
            "iterations_per_epoch": 10,  # Quick runs
            "sync_reference_every": 5,
            "train_batch_size": 1,
        },
        "verifier": {"num_workers": 2},
        "loss": {"beta": 0.1, "eps": 0.2},
        "data": {"dataset": "dummy"},
        "data_iter": {
            "batch_size": 4,
            "system_prompt": "",
            "assistant_prefix": "Let's think step by step.",
        },
        "sampling": {
            "temperature": 0.6,
            "top_p": 0.95,
            "top_k": 20,
            "max_tokens": 128,  # Short for speed
            "n": 4,  # Fewer completions
            "logprobs": 0,
        },
        "buffer": {"max_reads": 1},
        "roles": [
            {
                "name": "trainer",
                "kind": "titan",
                "config": {
                    "trainable": True,
                    "profiling": {"enable_profiling": False},
                    "metrics": {"log_freq": 1, "enable_tensorboard": False},
                    "model": {
                        "name": "qwen3",
                        "flavor": "0.6B",
                        "hf_assets_path": None,  # Set by test
                    },
                    "optimizer": {"name": "AdamW", "lr": 0.00001, "eps": 0.00000001},
                    "lr_scheduler": {"warmup_steps": 2},
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
                        "tensor_parallel_degree": 2,
                        "context_parallel_degree": 1,
                        "disable_loss_parallel": True,
                    },
                    "checkpoint": {
                        "enable": True,
                        "initial_load_in_hf": True,
                        "initial_load_model_only": True,
                    },
                    "activation_checkpoint": {"mode": "full", "selective_ac_option": "op"},
                    "compile": {"enable": False},
                },
            },
            {
                "name": "reference",
                "kind": "titan",
                "config": {
                    "trainable": False,
                    "model": {
                        "name": "qwen3",
                        "flavor": "0.6B",
                        "hf_assets_path": None,  # Set by test
                    },
                    "parallelism": {
                        "data_parallel_replicate_degree": 1,
                        "data_parallel_shard_degree": 1,
                        "fsdp_reshard_after_forward": "default",
                        "tensor_parallel_degree": 1,
                        "context_parallel_degree": 1,
                    },
                    "training": {
                        "seq_len": 512,
                        "dtype": "bfloat16",
                        "mixed_precision_param": "bfloat16",
                        "mixed_precision_reduce": "float32",
                    },
                    "checkpoint": {
                        "enable": True,
                        "initial_load_in_hf": True,
                        "initial_load_model_only": True,
                    },
                    "activation_checkpoint": {"mode": "full", "selective_ac_option": "op"},
                    "compile": {"enable": False},
                },
            },
            {
                "name": "rollout",
                "kind": "vllm",
                "config": {
                    "model": None,  # Set by test
                    "tensor_parallel_size": 1,
                    "data_parallel_size": 2,
                    "max_model_len": 512,
                    "gpu_memory_utilization": 0.85,
                    "dtype": "bfloat16",
                    "logprobs_mode": "raw_logprobs",
                    "max_num_seqs": 50,
                    "max_num_batched_tokens": 8192,
                    "enable_prefix_caching": True,
                    "enable_chunked_prefill": True,
                },
            },
        ],
        "sync": {
            "chunk_mb": 100,
            "wiring": [
                {"src": "trainer", "dst": "reference"},
                {"src": "trainer", "dst": "rollout"},
            ],
        },
    }


@pytest.fixture
def create_config(mini_config_template, nightly_assets_path, tmp_path):
    """Factory to create config files for tests."""

    def _create(overrides: dict = None):
        config = mini_config_template.copy()

        # Set model paths
        config["model"]["path"] = nightly_assets_path
        config["roles"][0]["config"]["model"]["hf_assets_path"] = nightly_assets_path
        config["roles"][1]["config"]["model"]["hf_assets_path"] = nightly_assets_path
        config["roles"][2]["config"]["model"] = nightly_assets_path

        # Apply overrides
        if overrides:
            _deep_update(config, overrides)

        # Write to temp file
        config_path = tmp_path / "test_config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config, f)

        return str(config_path)

    return _create


def _deep_update(d: dict, u: dict) -> dict:
    """Recursively update dict d with values from u."""
    for k, v in u.items():
        if isinstance(v, dict) and k in d and isinstance(d[k], dict):
            _deep_update(d[k], v)
        else:
            d[k] = v
    return d
