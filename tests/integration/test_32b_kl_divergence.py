"""Test KL divergence behavior for 32B model with different parallelism configs.

This test investigates why kl_max is not zero at step 1 or after syncs for 32B.
The hypothesis is that different parallelism between trainer and reference causes
numerical differences.
"""

import os
import asyncio
import pytest
import torch
import yaml

from rlvr_experiments.runtime import Runtime
from rlvr_experiments.syncing import sync_titan_to_titan


def _create_32b_config(tmp_path, assets_path, trainer_tp, trainer_dp_shard, ref_tp, ref_dp_shard):
    """Create a config for 32B model with specified parallelism."""
    roles = [
        {
            "name": "trainer",
            "kind": "titan",
            "config": {
                "trainable": True,
                "profiling": {"enable_profiling": False},
                "metrics": {"log_freq": 1, "enable_tensorboard": False},
                "model": {
                    "name": "qwen3",
                    "flavor": "32B",
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
                    "data_parallel_shard_degree": trainer_dp_shard,
                    "fsdp_reshard_after_forward": "default",
                    "tensor_parallel_degree": trainer_tp,
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
                    "flavor": "32B",
                    "hf_assets_path": assets_path,
                },
                "parallelism": {
                    "data_parallel_replicate_degree": 1,
                    "data_parallel_shard_degree": ref_dp_shard,
                    "fsdp_reshard_after_forward": "default",
                    "tensor_parallel_degree": ref_tp,
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
                "activation_checkpoint": {"mode": "none", "selective_ac_option": "op"},
                "compile": {"enable": False},
            },
        },
    ]

    config = {
        "run": {"name": "kl_test_32b"},
        "model": {"path": assets_path},
        "tokenizer": {
            "pretrained_model_name_or_path": "Qwen/Qwen3-32B",
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
            "wiring": [{"src": "trainer", "dst": "reference"}],
        },
    }

    config_path = tmp_path / "kl_test_32b.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f)

    return str(config_path)


@pytest.fixture(scope="module")
def assets_path():
    """Path to 32B model assets."""
    path = os.environ.get("NIGHTLY_ASSETS_PATH", "/efs/rlvr-experiments/assets/hf/Qwen3-32B")
    if not os.path.exists(path):
        pytest.skip(f"32B model assets not found at {path}")
    return path


@pytest.fixture(scope="module")
def event_loop():
    """Create event loop for module-scoped async fixtures."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.mark.integration
@pytest.mark.gpu
class TestA_SymmetricParallelism:
    """Test with matching parallelism (TP=2, DP_shard=1 for both)."""

    @pytest.fixture(scope="class")
    def symmetric_config_path(self, tmp_path_factory, assets_path):
        """Create config with symmetric parallelism."""
        tmp_path = tmp_path_factory.mktemp("symmetric")
        # Both trainer and reference use TP=2, DP_shard=1
        return _create_32b_config(tmp_path, assets_path,
                                  trainer_tp=2, trainer_dp_shard=1,
                                  ref_tp=2, ref_dp_shard=1)

    @pytest.fixture(scope="class")
    async def symmetric_runtime(self, symmetric_config_path):
        """Runtime with symmetric parallelism."""
        runtime = await Runtime.from_plan(symmetric_config_path)
        await runtime.start()
        yield runtime

    @pytest.mark.asyncio
    async def test_logprobs_match_initially_symmetric(self, symmetric_runtime):
        """With symmetric parallelism, trainer and reference should produce same logprobs."""
        trainer = symmetric_runtime.roles["trainer"]
        reference = symmetric_runtime.roles["reference"]

        # Test input
        input_ids = torch.randint(100, 1000, (1, 32))
        completion_ids = torch.randint(100, 1000, (1, 16))
        prompt_lens = torch.tensor([16])

        # Get logprobs from both
        trainer_logprobs = await trainer.compute_logprobs(input_ids, completion_ids, prompt_lens)
        reference_logprobs = await reference.compute_logprobs(input_ids, completion_ids, prompt_lens)

        diff = (trainer_logprobs - reference_logprobs).abs()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()

        print(f"\n=== Symmetric parallelism (TP=2 both) ===")
        print(f"Trainer logprobs[:5]:   {trainer_logprobs[0, :5]}")
        print(f"Reference logprobs[:5]: {reference_logprobs[0, :5]}")
        print(f"Max diff: {max_diff:.6f}, Mean diff: {mean_diff:.6f}")

        # KL div unbiased estimator: kl = exp(log_ratio) - log_ratio - 1
        # where log_ratio = ref_logprobs - trainer_logprobs
        log_ratio = reference_logprobs - trainer_logprobs
        kl = torch.exp(log_ratio) - log_ratio - 1.0
        kl_max = kl.max().item()

        print(f"KL max: {kl_max:.6f}")

        # Should be very close (same checkpoint, same parallelism)
        assert max_diff < 1e-3, f"Logprobs differ by {max_diff} with symmetric parallelism"
        assert kl_max < 1e-4, f"KL divergence is {kl_max} with symmetric parallelism (should be ~0)"


@pytest.mark.integration
@pytest.mark.gpu
class TestB_AsymmetricParallelism:
    """Test with different parallelism (trainer TP=4, ref TP=2)."""

    @pytest.fixture(scope="class")
    def asymmetric_config_path(self, tmp_path_factory, assets_path):
        """Create config with asymmetric parallelism (like qwen3-32B-gsm8k.yaml)."""
        tmp_path = tmp_path_factory.mktemp("asymmetric")
        # Trainer: TP=4, DP_shard=2; Reference: TP=2, DP_shard=1
        return _create_32b_config(tmp_path, assets_path,
                                  trainer_tp=4, trainer_dp_shard=2,
                                  ref_tp=2, ref_dp_shard=1)

    @pytest.fixture(scope="class")
    async def asymmetric_runtime(self, asymmetric_config_path):
        """Runtime with asymmetric parallelism."""
        runtime = await Runtime.from_plan(asymmetric_config_path)
        await runtime.start()
        yield runtime

    @pytest.mark.asyncio
    async def test_logprobs_match_initially_asymmetric(self, asymmetric_runtime):
        """With asymmetric parallelism, check if logprobs still match."""
        trainer = asymmetric_runtime.roles["trainer"]
        reference = asymmetric_runtime.roles["reference"]

        # Test input
        input_ids = torch.randint(100, 1000, (1, 32))
        completion_ids = torch.randint(100, 1000, (1, 16))
        prompt_lens = torch.tensor([16])

        # Get logprobs from both
        trainer_logprobs = await trainer.compute_logprobs(input_ids, completion_ids, prompt_lens)
        reference_logprobs = await reference.compute_logprobs(input_ids, completion_ids, prompt_lens)

        diff = (trainer_logprobs - reference_logprobs).abs()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()

        print(f"\n=== Asymmetric parallelism (trainer TP=4, ref TP=2) ===")
        print(f"Trainer logprobs[:5]:   {trainer_logprobs[0, :5]}")
        print(f"Reference logprobs[:5]: {reference_logprobs[0, :5]}")
        print(f"Max diff: {max_diff:.6f}, Mean diff: {mean_diff:.6f}")

        # KL divergence
        log_ratio = reference_logprobs - trainer_logprobs
        kl = torch.exp(log_ratio) - log_ratio - 1.0
        kl_max = kl.max().item()

        print(f"KL max: {kl_max:.6f}")

        # This test documents whether asymmetric parallelism causes differences
        # If this fails, it confirms our hypothesis
        if max_diff > 1e-3:
            print(f"WARNING: Asymmetric parallelism causes logprob differences of {max_diff}")
        if kl_max > 0.01:
            print(f"WARNING: Asymmetric parallelism causes KL divergence of {kl_max}")

        # We expect this to pass if asymmetric parallelism is handled correctly
        # If it fails, it explains the bug
        assert max_diff < 1e-3, \
            f"Logprobs differ by {max_diff} with asymmetric parallelism - this may be the bug!"


@pytest.mark.integration
@pytest.mark.gpu
class TestC_SyncWithAsymmetric:
    """Test sync behavior with asymmetric parallelism."""

    @pytest.fixture(scope="class")
    def asymmetric_config_path(self, tmp_path_factory, assets_path):
        """Create config with asymmetric parallelism."""
        tmp_path = tmp_path_factory.mktemp("asymmetric_sync")
        return _create_32b_config(tmp_path, assets_path,
                                  trainer_tp=4, trainer_dp_shard=2,
                                  ref_tp=2, ref_dp_shard=1)

    @pytest.fixture(scope="class")
    async def asymmetric_runtime(self, asymmetric_config_path):
        """Runtime with asymmetric parallelism."""
        runtime = await Runtime.from_plan(asymmetric_config_path)
        await runtime.start()
        yield runtime

    @pytest.mark.asyncio
    async def test_sync_updates_reference_correctly(self, asymmetric_runtime):
        """After sync, reference should match trainer (even with different parallelism)."""
        trainer = asymmetric_runtime.roles["trainer"]
        reference = asymmetric_runtime.roles["reference"]

        # Sync trainer -> reference
        await sync_titan_to_titan(trainer, reference)

        # Test input
        input_ids = torch.randint(100, 1000, (1, 32))
        completion_ids = torch.randint(100, 1000, (1, 16))
        prompt_lens = torch.tensor([16])

        # Get logprobs from both after sync
        trainer_logprobs = await trainer.compute_logprobs(input_ids, completion_ids, prompt_lens)
        reference_logprobs = await reference.compute_logprobs(input_ids, completion_ids, prompt_lens)

        diff = (trainer_logprobs - reference_logprobs).abs()
        max_diff = diff.max().item()

        print(f"\n=== After sync (asymmetric parallelism) ===")
        print(f"Trainer logprobs[:5]:   {trainer_logprobs[0, :5]}")
        print(f"Reference logprobs[:5]: {reference_logprobs[0, :5]}")
        print(f"Max diff after sync: {max_diff:.6f}")

        # KL divergence
        log_ratio = reference_logprobs - trainer_logprobs
        kl = torch.exp(log_ratio) - log_ratio - 1.0
        kl_max = kl.max().item()

        print(f"KL max after sync: {kl_max:.6f}")

        # After sync, they should match
        assert kl_max < 1e-4, \
            f"KL divergence after sync is {kl_max} - sync not working correctly with asymmetric parallelism"
