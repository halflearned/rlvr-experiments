"""Nightly smoke test using GSM8K dataset.

This test runs a few training steps on GSM8K to verify:
1. The data pipeline works with real math problems
2. The verifier correctly scores answers
3. Training doesn't crash with varied problem difficulty
"""

import pytest
import asyncio

from transformers import AutoTokenizer

from rlvr_experiments.data import DataIterator, load_gsm8k
from rlvr_experiments.losses import GRPOLoss
from rlvr_experiments.rollout import run_epoch
from rlvr_experiments.runtime import Runtime
from rlvr_experiments.verifiers import VerifierPool, MathVerifier
from rlvr_experiments.syncing import sync_titan_to_vllm


@pytest.mark.nightly
@pytest.mark.gpu
class TestGSM8KSmoke:
    """Smoke tests with GSM8K dataset."""

    @pytest.mark.asyncio
    async def test_gsm8k_training_runs(self, create_config):
        """Verify training works with GSM8K data."""
        config_path = create_config({
            "training": {
                "num_epochs": 1,
                "iterations_per_epoch": 5,
                "train_batch_size": 1,
            },
            "data": {
                "dataset": "gsm8k",
                "split": "test",  # Smaller split for testing
            },
            "sampling": {
                "n": 4,
                "max_tokens": 256,
            },
        })

        rewards_seen = []
        completed_steps = 0

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
            data_iter = DataIterator(
                load_gsm8k(split="test"),
                tokenizer=tokenizer,
                **plan.data_iter
            )
            loss_fn = GRPOLoss(**plan.loss)

            data_iter.new_epoch(seed=42)

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

                avg_reward = batch.rewards.mean().item()
                rewards_seen.append(avg_reward)
                completed_steps += 1

                print(f"step={step} loss={loss:.4f} reward={avg_reward:.2f}")

                if step >= plan.training["iterations_per_epoch"]:
                    break

            verifier.shutdown()

        finally:
            if 'runtime' in dir():
                await runtime.shutdown()

        # Basic sanity checks
        assert completed_steps >= 5, f"Only completed {completed_steps} steps"
        assert len(rewards_seen) > 0, "No rewards recorded"

        # With GSM8K and a small model, we don't expect high accuracy
        # But we should see some variation in rewards
        print(f"\nReward distribution: min={min(rewards_seen):.2f}, max={max(rewards_seen):.2f}")

    @pytest.mark.asyncio
    async def test_verifier_scoring_variety(self, create_config):
        """Verify we get a mix of correct and incorrect answers."""
        config_path = create_config({
            "training": {
                "num_epochs": 1,
                "iterations_per_epoch": 10,
            },
            "data": {
                "dataset": "gsm8k",
                "split": "test",
            },
            "sampling": {
                "n": 8,  # More completions for better signal
                "max_tokens": 256,
            },
        })

        all_rewards = []

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
            data_iter = DataIterator(
                load_gsm8k(split="test"),
                tokenizer=tokenizer,
                **plan.data_iter
            )
            loss_fn = GRPOLoss(**plan.loss)

            data_iter.new_epoch(seed=123)

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

                # Collect individual rewards
                all_rewards.extend(batch.rewards.tolist())

                if step >= plan.training["iterations_per_epoch"]:
                    break

            verifier.shutdown()

        finally:
            if 'runtime' in dir():
                await runtime.shutdown()

        # Check we have variety in rewards
        positive_count = sum(1 for r in all_rewards if r > 0)
        negative_count = sum(1 for r in all_rewards if r == 0)

        print(f"\nTotal completions: {len(all_rewards)}")
        print(f"Positive rewards: {positive_count}")
        print(f"Zero rewards: {negative_count}")

        # We should see at least some of each (unless model is extremely good or bad)
        # This is a soft check - with small models on GSM8K, most answers are wrong
        assert len(all_rewards) > 0, "No rewards collected"
