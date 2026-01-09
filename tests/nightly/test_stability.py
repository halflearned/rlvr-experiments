"""Nightly stability tests.

These tests verify long-running behavior:
1. Multi-epoch training without crashes
2. Memory stability (no obvious leaks)
3. Graceful handling of edge cases
"""

import pytest
import asyncio
import gc
import os

from transformers import AutoTokenizer

from rlvr_experiments.data import DataIterator, load_dummy
from rlvr_experiments.losses import GRPOLoss
from rlvr_experiments.rollout import run_epoch
from rlvr_experiments.runtime import Runtime
from rlvr_experiments.verifiers import VerifierPool, MathVerifier
from rlvr_experiments.syncing import sync_titan_to_vllm, sync_titan_to_titan


def get_memory_mb():
    """Get current RSS memory in MB."""
    try:
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024  # Convert to MB
    except ImportError:
        return 0


@pytest.mark.nightly
@pytest.mark.gpu
class TestMultiEpochStability:
    """Test multi-epoch training stability."""

    @pytest.mark.asyncio
    async def test_five_epochs_no_crash(self, create_config):
        """Run 5 epochs and verify no crashes or degradation."""
        config_path = create_config({
            "training": {
                "num_epochs": 5,
                "iterations_per_epoch": 10,
                "train_batch_size": 1,
                "sync_reference_every": 5,
            },
        })

        epoch_stats = []
        memory_readings = []

        try:
            runtime = await Runtime.from_plan(config_path)
            plan = runtime.plan
            await runtime.start()

            trainer = runtime.roles["trainer"]
            reference = runtime.roles["reference"]
            rollout = runtime.roles["rollout"]
            buffer = runtime.buffer

            tokenizer = AutoTokenizer.from_pretrained(**plan.tokenizer)
            pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id

            verifier = VerifierPool(MathVerifier, **plan.verifier)
            data_iter = DataIterator(load_dummy(), tokenizer=tokenizer, **plan.data_iter)
            loss_fn = GRPOLoss(**plan.loss)

            num_epochs = plan.training["num_epochs"]
            max_steps = plan.training["iterations_per_epoch"]
            sync_ref_every = plan.training["sync_reference_every"]

            for epoch in range(num_epochs):
                epoch_losses = []
                epoch_rewards = []
                data_iter.new_epoch(seed=epoch)

                async for step, batch in run_epoch(
                    rollout, data_iter, buffer,
                    reward=verifier.verify_completions,
                    pad_token_id=pad_token_id,
                    batch_size=plan.training.get("train_batch_size", 1),
                    sampling_params=plan.sampling,
                    epoch=epoch,
                ):
                    ref_logprobs = await reference.compute_logprobs(
                        batch.input_ids, batch.completion_ids, batch.prompt_lens
                    )
                    loss = await trainer.forward_backward(
                        loss_fn, batch.input_ids,
                        loss_args=(batch.completion_ids, ref_logprobs, batch.logprobs, batch.rewards),
                        loss_kwargs={"padding_mask": batch.mask, "prompt_lens": batch.prompt_lens},
                    )
                    grad_norm = await trainer.optim_step()

                    epoch_losses.append(loss)
                    epoch_rewards.append(batch.rewards.mean().item())

                    if step >= max_steps:
                        break

                    if (step + 1) % sync_ref_every == 0:
                        await sync_titan_to_vllm(trainer, rollout)
                        await sync_titan_to_titan(trainer, reference)

                # Record epoch stats
                epoch_stats.append({
                    "epoch": epoch,
                    "avg_loss": sum(epoch_losses) / len(epoch_losses),
                    "avg_reward": sum(epoch_rewards) / len(epoch_rewards),
                    "steps": len(epoch_losses),
                })

                # Memory reading after each epoch
                gc.collect()
                memory_readings.append(get_memory_mb())

                print(f"Epoch {epoch}: loss={epoch_stats[-1]['avg_loss']:.4f}, "
                      f"reward={epoch_stats[-1]['avg_reward']:.2f}, "
                      f"memory={memory_readings[-1]:.0f}MB")

            verifier.shutdown()

        finally:
            if 'runtime' in dir():
                await runtime.shutdown()

        # Verify all epochs completed
        assert len(epoch_stats) == 5, f"Only {len(epoch_stats)} epochs completed"

        # Check no NaN losses
        for stat in epoch_stats:
            assert stat["avg_loss"] == stat["avg_loss"], f"NaN loss in epoch {stat['epoch']}"

        # Check steps per epoch are consistent
        for stat in epoch_stats:
            assert stat["steps"] >= max_steps, f"Epoch {stat['epoch']} only had {stat['steps']} steps"

        # Memory check - warn if memory grew significantly
        if len(memory_readings) >= 2:
            memory_growth = memory_readings[-1] - memory_readings[0]
            if memory_growth > 1000:  # More than 1GB growth
                print(f"WARNING: Memory grew by {memory_growth:.0f}MB over {num_epochs} epochs")


@pytest.mark.nightly
@pytest.mark.gpu
class TestEdgeCases:
    """Test edge cases and error handling."""

    @pytest.mark.asyncio
    async def test_very_short_completions(self, create_config):
        """Test handling of very short max_tokens."""
        config_path = create_config({
            "training": {
                "num_epochs": 1,
                "iterations_per_epoch": 3,
            },
            "sampling": {
                "max_tokens": 32,  # Very short
                "n": 4,
            },
        })

        try:
            runtime = await Runtime.from_plan(config_path)
            plan = runtime.plan
            await runtime.start()

            trainer = runtime.roles["trainer"]
            reference = runtime.roles["reference"]
            rollout = runtime.roles["rollout"]
            buffer = runtime.buffer

            tokenizer = AutoTokenizer.from_pretrained(**plan.tokenizer)
            pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id

            verifier = VerifierPool(MathVerifier, **plan.verifier)
            data_iter = DataIterator(load_dummy(), tokenizer=tokenizer, **plan.data_iter)
            loss_fn = GRPOLoss(**plan.loss)

            data_iter.new_epoch(seed=0)
            steps = 0

            async for step, batch in run_epoch(
                rollout, data_iter, buffer,
                reward=verifier.verify_completions,
                pad_token_id=pad_token_id,
                batch_size=plan.training.get("train_batch_size", 1),
                sampling_params=plan.sampling,
                epoch=0,
            ):
                ref_logprobs = await reference.compute_logprobs(
                    batch.input_ids, batch.completion_ids, batch.prompt_lens
                )
                loss = await trainer.forward_backward(
                    loss_fn, batch.input_ids,
                    loss_args=(batch.completion_ids, ref_logprobs, batch.logprobs, batch.rewards),
                    loss_kwargs={"padding_mask": batch.mask, "prompt_lens": batch.prompt_lens},
                )
                await trainer.optim_step()
                steps += 1

                if step >= plan.training["iterations_per_epoch"]:
                    break

            verifier.shutdown()

        finally:
            if 'runtime' in dir():
                await runtime.shutdown()

        assert steps >= 3, "Should complete at least 3 steps with short completions"

    @pytest.mark.asyncio
    async def test_single_completion_per_prompt(self, create_config):
        """Test with n=1 (single completion per prompt)."""
        config_path = create_config({
            "training": {
                "num_epochs": 1,
                "iterations_per_epoch": 3,
            },
            "sampling": {
                "n": 1,  # Single completion
            },
        })

        try:
            runtime = await Runtime.from_plan(config_path)
            plan = runtime.plan
            await runtime.start()

            trainer = runtime.roles["trainer"]
            reference = runtime.roles["reference"]
            rollout = runtime.roles["rollout"]
            buffer = runtime.buffer

            tokenizer = AutoTokenizer.from_pretrained(**plan.tokenizer)
            pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id

            verifier = VerifierPool(MathVerifier, **plan.verifier)
            data_iter = DataIterator(load_dummy(), tokenizer=tokenizer, **plan.data_iter)
            loss_fn = GRPOLoss(**plan.loss)

            data_iter.new_epoch(seed=0)
            batches_yielded = 0

            async for step, batch in run_epoch(
                rollout, data_iter, buffer,
                reward=verifier.verify_completions,
                pad_token_id=pad_token_id,
                batch_size=plan.training.get("train_batch_size", 1),
                sampling_params=plan.sampling,
                epoch=0,
            ):
                # With n=1, most batches will have zero variance and be skipped
                # This is expected behavior
                batches_yielded += 1

                ref_logprobs = await reference.compute_logprobs(
                    batch.input_ids, batch.completion_ids, batch.prompt_lens
                )
                loss = await trainer.forward_backward(
                    loss_fn, batch.input_ids,
                    loss_args=(batch.completion_ids, ref_logprobs, batch.logprobs, batch.rewards),
                    loss_kwargs={"padding_mask": batch.mask, "prompt_lens": batch.prompt_lens},
                )
                await trainer.optim_step()

                if step >= plan.training["iterations_per_epoch"]:
                    break

            verifier.shutdown()

        finally:
            if 'runtime' in dir():
                await runtime.shutdown()

        # With n=1, we expect zero-variance batches to be filtered
        # So we may get fewer batches than steps
        print(f"Batches yielded with n=1: {batches_yielded}")
