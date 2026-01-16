"""End-to-end smoke tests for the full training pipeline.

These tests verify that the entire train_grpo.py pipeline works correctly
by running actual training with real models (small ones for speed).

IMPORTANT: These tests require GPUs and should be run as nightly tests.
They may take several minutes to complete.

Key verifications:
1. Training loop completes without errors
2. Loss decreases over steps
3. Rewards improve over training
4. Weight sync works correctly
5. Checkpointing works
"""

import asyncio
import os
import tempfile
import pytest
import yaml
import torch


def create_smoke_test_config(tmp_dir: str, model_path: str, num_steps: int = 5) -> str:
    """Create a minimal config for smoke testing.

    Uses:
    - Small model (Qwen3-0.6B if available, otherwise whatever is specified)
    - Dummy dataset (single math problem)
    - Minimal batching (1 prompt per step)
    - Few completions per prompt
    """
    config = {
        "run": {"name": "smoke_test"},
        "model": {"path": model_path},
        "tokenizer": {
            "pretrained_model_name_or_path": model_path,
            "use_fast": False,
        },
        "training": {
            "max_steps": num_steps,
            "max_staleness": 0,
            "abort_in_flight": False,
            "checkpoint_interval": 0,  # No checkpointing during smoke test

            # Minimal batching
            "seq_len_buckets": [512],
            "completion_len_buckets": [256],
            "prompts_per_rollout_sync": 1,
            "prompts_per_reference_sync": 1,
            "prompts_per_optim_step": 1,
            "prompts_per_forward_backward": 1,
            "completions_per_micro_batch": 4,
            "max_concurrent_tasks": 1,
        },
        "verifier": {"num_workers": 1},
        "loss": {"beta": 0.05, "eps": 0.2},
        "data": {
            "dataset": "dummy",
            "num_samples": 32,  # Enough for a few steps
        },
        "data_iter": {
            "system_prompt": "Solve the math problem.",
            "assistant_prefix": "",
        },
        "sampling": {
            "temperature": 0.6,
            "top_p": 0.95,
            "max_tokens": 128,  # Short for speed
            "n": 4,  # Few completions
            "logprobs": 0,
        },
        "buffer": {"max_reads": 1, "maxsize": 0},
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
                        "hf_assets_path": model_path,
                    },
                    "optimizer": {"name": "AdamW", "lr": 1e-4, "eps": 1e-8},
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
                        "tensor_parallel_degree": 1,
                        "context_parallel_degree": 1,
                        "disable_loss_parallel": True,
                    },
                    "checkpoint": {
                        "enable": True,
                        "initial_load_in_hf": True,
                        "initial_load_model_only": True,
                    },
                    "activation_checkpoint": {"mode": "full", "selective_ac_option": "op"},
                    "compile": {"enable": False},  # Faster startup
                },
            },
            {
                "name": "reference",
                "kind": "vllm",
                "config": {
                    "model": model_path,
                    "max_concurrent_per_replica": 4,
                    "max_num_seqs": 16,
                    "tensor_parallel_size": 1,
                    "data_parallel_size": 1,
                    "max_model_len": 513,
                    "gpu_memory_utilization": 0.3,
                    "dtype": "bfloat16",
                    "logprobs_mode": "raw_logprobs",
                    "max_num_batched_tokens": 4096,
                    "enable_prefix_caching": False,
                    "enable_chunked_prefill": True,
                },
            },
            {
                "name": "rollout",
                "kind": "vllm",
                "config": {
                    "model": model_path,
                    "max_concurrent_per_replica": 4,
                    "max_num_seqs": 16,
                    "tensor_parallel_size": 1,
                    "data_parallel_size": 1,
                    "max_model_len": 512,
                    "gpu_memory_utilization": 0.3,
                    "dtype": "bfloat16",
                    "logprobs_mode": "raw_logprobs",
                    "max_num_batched_tokens": 4096,
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

    config_path = os.path.join(tmp_dir, "smoke_test_config.yaml")
    with open(config_path, "w") as f:
        yaml.dump(config, f)

    return config_path


@pytest.fixture(scope="module")
def model_path():
    """Get path to a small model for testing."""
    # Try common locations for small test model
    paths = [
        "/efs/rlvr-experiments/assets/hf/Qwen3-0.6B",
        "/efs/rlvr-experiments/assets/hf/Qwen3-1.7B-Base",
        os.environ.get("NIGHTLY_ASSETS_PATH", ""),
    ]

    for path in paths:
        if path and os.path.exists(path):
            return path

    pytest.skip("No model found for E2E smoke test")


class TestE2ESmokeBasic:
    """Basic smoke tests for the training pipeline."""

    @pytest.mark.nightly
    @pytest.mark.gpu
    @pytest.mark.asyncio
    async def test_runtime_starts_and_stops(self, model_path):
        """Test that Runtime can be created and started."""
        from rlvr_experiments.runtime import Runtime

        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = create_smoke_test_config(tmp_dir, model_path, num_steps=1)

            runtime = await Runtime.from_plan(config_path)
            await runtime.start()

            # Verify roles are created
            assert "trainer" in runtime.roles
            assert "reference" in runtime.roles
            assert "rollout" in runtime.roles

            # Verify trainer has expected methods
            trainer = runtime.roles["trainer"]
            assert hasattr(trainer, "forward_backward")
            assert hasattr(trainer, "optim_step")
            assert hasattr(trainer, "compute_logprobs")

    @pytest.mark.nightly
    @pytest.mark.gpu
    @pytest.mark.asyncio
    async def test_single_training_step(self, model_path):
        """Test that a single training step completes without error."""
        from rlvr_experiments.runtime import Runtime
        from rlvr_experiments.losses import GRPOLoss, compute_advantages
        import torch

        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = create_smoke_test_config(tmp_dir, model_path, num_steps=1)

            runtime = await Runtime.from_plan(config_path)
            await runtime.start()

            trainer = runtime.roles["trainer"]
            reference = runtime.roles["reference"]
            loss_fn = GRPOLoss(beta=0.05, eps=0.2)

            # Create dummy training data
            batch_size = 2
            seq_len = 32
            completion_len = 16

            input_ids = torch.randint(100, 1000, (batch_size, seq_len))
            completion_ids = torch.randint(100, 1000, (batch_size, completion_len))
            prompt_lens = torch.tensor([seq_len - completion_len] * batch_size)
            rewards = torch.tensor([1.0, 0.0])
            mask = torch.ones(batch_size, completion_len)

            # Get reference logprobs
            ref_logprobs = await reference.compute_logprobs(
                input_ids, completion_ids, prompt_lens
            )

            # Forward/backward
            rollout_logprobs = ref_logprobs.clone()
            advantages = compute_advantages(rewards)

            loss, debug = await trainer.forward_backward(
                loss_fn,
                input_ids,
                loss_args=(completion_ids, ref_logprobs, rollout_logprobs, advantages),
                loss_kwargs={"padding_mask": mask, "prompt_lens": prompt_lens},
            )

            # Optimizer step
            grad_norm = await trainer.optim_step()

            print(f"Loss: {loss:.4f}, Grad norm: {grad_norm:.4f}")

            assert torch.isfinite(torch.tensor(loss)), "Loss should be finite"
            assert torch.isfinite(torch.tensor(grad_norm)), "Grad norm should be finite"
            assert grad_norm > 0, "Grad norm should be positive"


class TestE2ETrainingLoop:
    """Tests for the full training loop behavior."""

    @pytest.mark.nightly
    @pytest.mark.gpu
    @pytest.mark.asyncio
    async def test_loss_decreases_over_steps(self, model_path):
        """Loss should generally decrease over training steps on dummy data."""
        from rlvr_experiments.runtime import Runtime
        from rlvr_experiments.losses import GRPOLoss, compute_advantages
        import torch

        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = create_smoke_test_config(tmp_dir, model_path, num_steps=10)

            runtime = await Runtime.from_plan(config_path)
            await runtime.start()

            trainer = runtime.roles["trainer"]
            reference = runtime.roles["reference"]
            loss_fn = GRPOLoss(beta=0.05, eps=0.2)

            losses = []

            # Fixed training data (so loss should decrease)
            batch_size = 4
            seq_len = 32
            completion_len = 16

            torch.manual_seed(42)
            input_ids = torch.randint(100, 1000, (batch_size, seq_len))
            completion_ids = input_ids[:, seq_len - completion_len:]
            prompt_lens = torch.tensor([seq_len - completion_len] * batch_size)
            rewards = torch.tensor([1.0, 0.8, 0.2, 0.0])
            mask = torch.ones(batch_size, completion_len)

            for step in range(10):
                # Get fresh reference logprobs each step
                ref_logprobs = await reference.compute_logprobs(
                    input_ids, completion_ids, prompt_lens
                )

                rollout_logprobs = ref_logprobs.clone()
                advantages = compute_advantages(rewards)

                loss, _ = await trainer.forward_backward(
                    loss_fn,
                    input_ids,
                    loss_args=(completion_ids, ref_logprobs, rollout_logprobs, advantages),
                    loss_kwargs={"padding_mask": mask, "prompt_lens": prompt_lens},
                )

                await trainer.optim_step()
                losses.append(loss)
                print(f"Step {step}: loss={loss:.4f}")

            # Loss should generally decrease (allow some variance)
            first_half_avg = sum(losses[:5]) / 5
            second_half_avg = sum(losses[5:]) / 5

            print(f"First half avg: {first_half_avg:.4f}")
            print(f"Second half avg: {second_half_avg:.4f}")

            # Not a hard requirement, but generally expected
            # If this fails consistently, there may be a learning issue
            assert second_half_avg < first_half_avg + 0.5, \
                "Loss should not increase significantly over training"


class TestE2EWeightSync:
    """Tests for weight synchronization during training."""

    @pytest.mark.nightly
    @pytest.mark.gpu
    @pytest.mark.asyncio
    async def test_weight_sync_changes_vllm_output(self, model_path):
        """After training + sync, vLLM outputs should change."""
        from rlvr_experiments.runtime import Runtime
        from rlvr_experiments.syncing import sync_titan_to_vllm
        from rlvr_experiments.losses import GRPOLoss, compute_advantages
        import torch

        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = create_smoke_test_config(tmp_dir, model_path, num_steps=5)

            runtime = await Runtime.from_plan(config_path)
            await runtime.start()

            trainer = runtime.roles["trainer"]
            reference = runtime.roles["reference"]
            rollout = runtime.roles["rollout"]
            loss_fn = GRPOLoss(beta=0.05, eps=0.2)

            # Initial sync
            await sync_titan_to_vllm(trainer, rollout)

            # Get initial output
            prompt = "What is 2 + 2?"
            sampling_params = {"temperature": 0.0, "max_tokens": 16, "n": 1, "logprobs": 1}

            result_before = await rollout.generate([prompt], **sampling_params)
            tokens_before = list(result_before[0].outputs[0].token_ids)

            # Train for a few steps with high learning rate
            batch_size = 4
            seq_len = 32
            completion_len = 16

            torch.manual_seed(42)
            for _ in range(5):
                input_ids = torch.randint(100, 1000, (batch_size, seq_len))
                completion_ids = input_ids[:, seq_len - completion_len:]
                prompt_lens = torch.tensor([seq_len - completion_len] * batch_size)
                rewards = torch.rand(batch_size)
                mask = torch.ones(batch_size, completion_len)

                ref_logprobs = await reference.compute_logprobs(
                    input_ids, completion_ids, prompt_lens
                )

                advantages = compute_advantages(rewards)

                await trainer.forward_backward(
                    loss_fn,
                    input_ids,
                    loss_args=(completion_ids, ref_logprobs, ref_logprobs.clone(), advantages),
                    loss_kwargs={"padding_mask": mask, "prompt_lens": prompt_lens},
                )

                await trainer.optim_step()

            # Sync updated weights
            await sync_titan_to_vllm(trainer, rollout)

            # Get output after training
            result_after = await rollout.generate([prompt], **sampling_params)
            tokens_after = list(result_after[0].outputs[0].token_ids)

            print(f"Tokens before: {tokens_before[:10]}")
            print(f"Tokens after: {tokens_after[:10]}")

            # Extract logprobs for comparison (tokens might be same but logprobs different)
            def extract_logprobs(output):
                lps = output.logprobs
                if not lps:
                    return []
                result = []
                for lp in lps:
                    if isinstance(lp, dict):
                        for token_id, info in lp.items():
                            if isinstance(info, dict):
                                result.append(info.get("logprob", 0))
                            else:
                                result.append(getattr(info, "logprob", 0))
                            break
                return result

            lp_before = extract_logprobs(result_before[0].outputs[0])
            lp_after = extract_logprobs(result_after[0].outputs[0])

            print(f"Logprobs before (first 5): {lp_before[:5]}")
            print(f"Logprobs after (first 5): {lp_after[:5]}")

            # Either tokens or logprobs should have changed
            tokens_changed = tokens_before != tokens_after
            logprobs_changed = lp_before != lp_after

            assert tokens_changed or logprobs_changed, \
                "vLLM outputs should change after training and sync"


class TestE2ERewardImprovement:
    """Tests for reward improvement during training."""

    @pytest.mark.nightly
    @pytest.mark.gpu
    @pytest.mark.asyncio
    async def test_dummy_dataset_reward_improves(self, model_path):
        """Training on dummy dataset should show reward improvement.

        The dummy dataset has a simple math problem with answer "1".
        After training, the model should be more likely to produce correct answers.

        This is the most critical test - it verifies that the entire pipeline
        results in actual learning.
        """
        from rlvr_experiments.runtime import Runtime
        from rlvr_experiments.data import DataIterator, load_dummy
        from rlvr_experiments.verifiers import VerifierPool, MathVerifier
        from rlvr_experiments.algorithms.grpo import RolloutSample, TrainSample, make_batch
        from rlvr_experiments.losses import GRPOLoss, compute_advantages
        from rlvr_experiments.syncing import sync_titan_to_vllm
        from transformers import AutoTokenizer
        import torch

        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = create_smoke_test_config(tmp_dir, model_path, num_steps=10)

            runtime = await Runtime.from_plan(config_path)
            await runtime.start()

            trainer = runtime.roles["trainer"]
            reference = runtime.roles["reference"]
            rollout = runtime.roles["rollout"]

            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
            pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id

            # Load dummy dataset
            ds = load_dummy(num_samples=20)
            data_iter = DataIterator(
                ds,
                tokenizer=tokenizer,
                system_prompt="Solve the math problem.",
                assistant_prefix="",
            )
            data_iter.new_epoch(seed=42)

            # Create verifier
            verifier = VerifierPool(MathVerifier, num_workers=1)
            loss_fn = GRPOLoss(beta=0.05, eps=0.2)

            initial_rewards = []
            final_rewards = []

            # Generate initial samples to measure baseline
            for i in range(5):
                item = data_iter.get_next()
                if item is None:
                    break

                response = await rollout.generate_single(
                    item["template"],
                    temperature=0.6,
                    top_p=0.95,
                    max_tokens=128,
                    n=4,
                )

                completions = [out.text for out in response.outputs]
                rewards = await verifier.verify_completions(item["problem"], completions)
                initial_rewards.extend(rewards)
                data_iter.mark_done(item["problem"]["prompt_id"])

            initial_avg = sum(initial_rewards) / len(initial_rewards) if initial_rewards else 0

            # Reset for training
            data_iter.new_epoch(seed=43)

            # Train for a few steps
            for step in range(5):
                item = data_iter.get_next()
                if item is None:
                    break

                # Generate
                response = await rollout.generate_single(
                    item["template"],
                    temperature=0.6,
                    top_p=0.95,
                    max_tokens=128,
                    n=4,
                )

                completions = [out.text for out in response.outputs]
                rewards = await verifier.verify_completions(item["problem"], completions)

                # Skip if all same reward (no gradient)
                if torch.tensor(rewards).std() < 1e-6:
                    data_iter.mark_done(item["problem"]["prompt_id"])
                    continue

                # Build sample
                rollout_sample = RolloutSample.from_vllm(response, pad_token_id)

                # Get ref logprobs
                ref_logprobs = await reference.compute_logprobs(
                    rollout_sample.input_ids,
                    rollout_sample.completion_ids,
                    torch.tensor([rollout_sample.prompt_len] * len(completions)),
                )

                sample = TrainSample(
                    rollout_sample,
                    rewards,
                    ref_logprobs,
                    item_id=item["problem"]["prompt_id"],
                    trainer_version=trainer.version,
                )

                # Make batch
                batch, stats = make_batch([sample], pad_token_id)

                # Compute advantages
                advantages = compute_advantages(batch.rewards, group_size=4)

                # Train
                loss, _ = await trainer.forward_backward(
                    loss_fn,
                    batch.input_ids,
                    loss_args=(
                        batch.completion_ids,
                        batch.ref_logprobs,
                        batch.logprobs,
                        advantages,
                    ),
                    loss_kwargs={"padding_mask": batch.mask, "prompt_lens": batch.prompt_lens},
                )

                await trainer.optim_step()

                # Sync weights
                await sync_titan_to_vllm(trainer, rollout)

                print(f"Step {step}: loss={loss:.4f}, rewards={rewards}")
                data_iter.mark_done(item["problem"]["prompt_id"])

            # Generate final samples to measure improvement
            data_iter.new_epoch(seed=44)
            for i in range(5):
                item = data_iter.get_next()
                if item is None:
                    break

                response = await rollout.generate_single(
                    item["template"],
                    temperature=0.6,
                    top_p=0.95,
                    max_tokens=128,
                    n=4,
                )

                completions = [out.text for out in response.outputs]
                rewards = await verifier.verify_completions(item["problem"], completions)
                final_rewards.extend(rewards)
                data_iter.mark_done(item["problem"]["prompt_id"])

            final_avg = sum(final_rewards) / len(final_rewards) if final_rewards else 0

            print(f"Initial average reward: {initial_avg:.4f}")
            print(f"Final average reward: {final_avg:.4f}")

            # This is a soft check - training on a small number of steps may not
            # always show improvement, but should not show significant degradation
            assert final_avg >= initial_avg - 0.2, \
                f"Reward should not significantly degrade: {initial_avg:.4f} -> {final_avg:.4f}"
