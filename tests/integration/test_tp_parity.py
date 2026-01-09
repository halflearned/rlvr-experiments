"""Test that different tensor parallelism degrees produce same logits.

This test investigates whether different TP configurations cause numerical
differences in model outputs even when loaded from the same checkpoint.
"""

import os
import asyncio
import pytest
import torch
import yaml

from rlvr_experiments.runtime import Runtime


def _create_config(tmp_path, assets_path, tp_degree, dp_shard=1, role_name="model"):
    """Create a config with specified parallelism."""
    roles = [
        {
            "name": role_name,
            "kind": "titan",
            "config": {
                "trainable": False,  # Both non-trainable for cleaner comparison
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
                    "data_parallel_shard_degree": dp_shard,
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
        "run": {"name": f"tp_parity_test_{role_name}"},
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

    config_path = tmp_path / f"tp_test_{role_name}.yaml"
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
class TestTPParity:
    """Test that different TP degrees produce same outputs when loaded from same checkpoint."""

    @pytest.mark.asyncio
    async def test_tp1_vs_tp1_same_logprobs(self, tmp_path_factory, assets_path):
        """Sanity check: two TP=1 models should produce identical logprobs."""
        tmp_path = tmp_path_factory.mktemp("tp1_vs_tp1")

        # Create two separate configs with TP=1
        config1_path = _create_config(tmp_path, assets_path, tp_degree=1, role_name="model1")
        config2_path = _create_config(tmp_path, assets_path, tp_degree=1, role_name="model2")

        # Load both runtimes
        runtime1 = await Runtime.from_plan(config1_path)
        await runtime1.start()

        runtime2 = await Runtime.from_plan(config2_path)
        await runtime2.start()

        try:
            model1 = runtime1.roles["model1"]
            model2 = runtime2.roles["model2"]

            # Test input (fixed seed for reproducibility)
            torch.manual_seed(42)
            input_ids = torch.randint(100, 1000, (2, 32))
            completion_ids = torch.randint(100, 1000, (2, 16))
            prompt_lens = torch.tensor([16, 16])

            # Get logprobs from both
            logprobs1 = await model1.compute_logprobs(input_ids, completion_ids, prompt_lens)
            logprobs2 = await model2.compute_logprobs(input_ids, completion_ids, prompt_lens)

            diff = (logprobs1 - logprobs2).abs()
            max_diff = diff.max().item()
            mean_diff = diff.mean().item()

            print(f"\n=== TP=1 vs TP=1 (sanity check) ===")
            print(f"Model1 logprobs[:5]: {logprobs1[0, :5]}")
            print(f"Model2 logprobs[:5]: {logprobs2[0, :5]}")
            print(f"Max diff: {max_diff:.10f}, Mean diff: {mean_diff:.10f}")

            # Should be exactly equal (or very close due to float32 rounding)
            assert max_diff < 1e-6, f"Two TP=1 models differ by {max_diff} - unexpected!"

        finally:
            pass  # Ray cleanup handled by test framework

    @pytest.mark.asyncio
    async def test_tp1_vs_tp2_logprobs(self, tmp_path_factory, assets_path):
        """Check if TP=1 and TP=2 models produce the same logprobs from same checkpoint."""
        tmp_path = tmp_path_factory.mktemp("tp1_vs_tp2")

        # Create configs with different TP
        config_tp1 = _create_config(tmp_path, assets_path, tp_degree=1, role_name="model_tp1")
        config_tp2 = _create_config(tmp_path, assets_path, tp_degree=2, role_name="model_tp2")

        # Load both runtimes
        runtime_tp1 = await Runtime.from_plan(config_tp1)
        await runtime_tp1.start()

        runtime_tp2 = await Runtime.from_plan(config_tp2)
        await runtime_tp2.start()

        try:
            model_tp1 = runtime_tp1.roles["model_tp1"]
            model_tp2 = runtime_tp2.roles["model_tp2"]

            # Test input (fixed seed for reproducibility)
            torch.manual_seed(42)
            input_ids = torch.randint(100, 1000, (2, 32))
            completion_ids = torch.randint(100, 1000, (2, 16))
            prompt_lens = torch.tensor([16, 16])

            # Get logprobs from both
            logprobs_tp1 = await model_tp1.compute_logprobs(input_ids, completion_ids, prompt_lens)
            logprobs_tp2 = await model_tp2.compute_logprobs(input_ids, completion_ids, prompt_lens)

            diff = (logprobs_tp1 - logprobs_tp2).abs()
            max_diff = diff.max().item()
            mean_diff = diff.mean().item()

            print(f"\n=== TP=1 vs TP=2 ===")
            print(f"TP1 logprobs[:5]: {logprobs_tp1[0, :5]}")
            print(f"TP2 logprobs[:5]: {logprobs_tp2[0, :5]}")
            print(f"Max diff: {max_diff:.10f}, Mean diff: {mean_diff:.10f}")

            # KL divergence
            log_ratio = logprobs_tp2 - logprobs_tp1
            kl = torch.exp(log_ratio) - log_ratio - 1.0
            kl_max = kl.max().item()
            print(f"KL max: {kl_max:.10f}")

            # Document whether there's a difference
            if max_diff > 1e-6:
                print(f"WARNING: TP=1 and TP=2 produce different logprobs (max_diff={max_diff})")
                print("This could explain the non-zero KL at step 1 with asymmetric parallelism!")

            # Expect them to be close (same checkpoint), but document if not
            assert max_diff < 0.1, f"TP=1 vs TP=2 differ too much: {max_diff}"

        finally:
            pass  # Ray cleanup handled by test framework
