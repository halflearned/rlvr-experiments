"""Unit tests for chunked loss computation.

These tests verify that:
1. compute_advantages produces correct normalized values
2. Grouped advantage computation works for multi-prompt batches
3. Chunked forward-backward produces the same gradients as full batch
4. Edge cases (uneven chunks, single sample) are handled correctly
"""

import pytest
import torch
import torch.nn as nn


class TestComputeAdvantages:
    """Test that compute_advantages works correctly."""

    def test_advantages_zero_mean_unit_var(self):
        """Advantages should have zero mean and unit variance."""
        from rlvr_experiments.losses import compute_advantages

        rewards = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        adv = compute_advantages(rewards)

        assert adv.mean().abs() < 1e-5
        assert (adv.std() - 1.0).abs() < 1e-5

    def test_advantages_with_identical_rewards(self):
        """Edge case: all identical rewards should not crash (clamped std)."""
        from rlvr_experiments.losses import compute_advantages

        rewards = torch.tensor([1.0, 1.0, 1.0, 1.0])
        adv = compute_advantages(rewards)

        # With clamped std, this should produce zeros
        assert torch.allclose(adv, torch.zeros_like(adv))

    def test_grouped_advantages(self):
        """Grouped advantages should normalize within each group."""
        from rlvr_experiments.losses import compute_advantages

        # Two prompts, 4 completions each
        rewards = torch.tensor([1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0])

        # Compute with group_size=4
        adv = compute_advantages(rewards, group_size=4)

        # Check each group has zero mean and unit variance
        group1 = adv[:4]
        group2 = adv[4:]

        assert group1.mean().abs() < 1e-5
        assert (group1.std() - 1.0).abs() < 1e-5
        assert group2.mean().abs() < 1e-5
        assert (group2.std() - 1.0).abs() < 1e-5

    def test_grouped_vs_ungrouped_differ(self):
        """Grouped advantages should differ from ungrouped when groups have different scales."""
        from rlvr_experiments.losses import compute_advantages

        # Two prompts with very different reward scales
        rewards = torch.tensor([1.0, 2.0, 3.0, 4.0, 100.0, 200.0, 300.0, 400.0])

        ungrouped = compute_advantages(rewards)
        grouped = compute_advantages(rewards, group_size=4)

        # Should NOT be the same
        assert not torch.allclose(ungrouped, grouped, atol=1e-3)

    def test_grouped_batch_not_divisible_raises(self):
        """Should raise error if batch not divisible by group_size."""
        from rlvr_experiments.losses import compute_advantages

        rewards = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])  # 5 elements

        with pytest.raises(ValueError):
            compute_advantages(rewards, group_size=4)  # 5 not divisible by 4


class TestGRPOLossWithPrecomputedAdvantages:
    """Test that GRPOLoss works correctly with pre-computed advantages."""

    @pytest.fixture
    def simple_inputs(self):
        """Create simple test inputs for loss function."""
        batch_size = 4
        seq_len = 10
        completion_len = 5
        vocab_size = 100

        # Create fake logits that produce predictable logprobs
        logits = torch.randn(batch_size, seq_len, vocab_size)
        response = torch.randint(0, vocab_size, (batch_size, completion_len))
        ref_logprobs = torch.randn(batch_size, completion_len)
        rollout_logprobs = torch.randn(batch_size, completion_len)
        rewards = torch.tensor([1.0, 0.5, 0.0, -0.5])
        padding_mask = torch.ones(batch_size, completion_len)
        prompt_lens = torch.tensor([5, 5, 5, 5])  # All start at same position

        return {
            "logits": logits,
            "response": response,
            "ref_logprobs": ref_logprobs,
            "rollout_logprobs": rollout_logprobs,
            "rewards": rewards,
            "padding_mask": padding_mask,
            "prompt_lens": prompt_lens,
        }

    def test_loss_runs_with_precomputed_advantages(self, simple_inputs):
        """Loss should run with pre-computed advantages."""
        from rlvr_experiments.losses import GRPOLoss, compute_advantages

        loss_fn = GRPOLoss(beta=0.01, eps=0.2)

        # Compute advantages first
        advantages = compute_advantages(simple_inputs["rewards"])

        # Compute loss
        loss = loss_fn(
            simple_inputs["logits"].clone(),
            simple_inputs["response"],
            simple_inputs["ref_logprobs"],
            simple_inputs["rollout_logprobs"],
            advantages,
            simple_inputs["padding_mask"],
            prompt_lens=simple_inputs["prompt_lens"],
        )

        assert torch.is_tensor(loss)
        assert loss.ndim == 0  # Scalar

    def test_loss_gradients_flow(self, simple_inputs):
        """Gradients should flow through the loss."""
        from rlvr_experiments.losses import GRPOLoss, compute_advantages

        loss_fn = GRPOLoss(beta=0.01, eps=0.2)

        logits = simple_inputs["logits"].clone().requires_grad_(True)
        advantages = compute_advantages(simple_inputs["rewards"])

        loss = loss_fn(
            logits,
            simple_inputs["response"],
            simple_inputs["ref_logprobs"],
            simple_inputs["rollout_logprobs"],
            advantages,
            simple_inputs["padding_mask"],
            prompt_lens=simple_inputs["prompt_lens"],
        )
        loss.backward()

        assert logits.grad is not None
        assert not torch.isnan(logits.grad).any()


class TestChunkedLossComputation:
    """Test that chunked loss gives same results as full batch."""

    @pytest.fixture
    def model_and_inputs(self):
        """Create a simple model and inputs for gradient testing."""
        batch_size = 8
        seq_len = 10
        completion_len = 5
        vocab_size = 100
        hidden_size = 32

        # Simple linear model (no transformer, just for gradient testing)
        model = nn.Linear(hidden_size, vocab_size)

        # Inputs
        hidden_states = torch.randn(batch_size, seq_len, hidden_size)
        response = torch.randint(0, vocab_size, (batch_size, completion_len))
        ref_logprobs = torch.randn(batch_size, completion_len)
        rollout_logprobs = torch.randn(batch_size, completion_len)
        rewards = torch.randn(batch_size)  # Random rewards
        padding_mask = torch.ones(batch_size, completion_len)
        prompt_lens = torch.full((batch_size,), 5, dtype=torch.long)

        return {
            "model": model,
            "hidden_states": hidden_states,
            "response": response,
            "ref_logprobs": ref_logprobs,
            "rollout_logprobs": rollout_logprobs,
            "rewards": rewards,
            "padding_mask": padding_mask,
            "prompt_lens": prompt_lens,
        }

    def test_chunked_same_gradients_as_full_batch(self, model_and_inputs):
        """Chunked computation should give same gradients as full batch."""
        from rlvr_experiments.losses import GRPOLoss, compute_advantages

        loss_fn = GRPOLoss(beta=0.01, eps=0.2)
        inputs = model_and_inputs
        batch_size = inputs["hidden_states"].size(0)
        chunk_size = 4  # Process in 2 chunks

        # Pre-compute advantages on full batch (critical for correctness)
        advantages = compute_advantages(inputs["rewards"])

        # Full batch computation
        model_full = nn.Linear(32, 100)
        model_full.load_state_dict(inputs["model"].state_dict())

        logits_full = model_full(inputs["hidden_states"])
        loss_full = loss_fn(
            logits_full,
            inputs["response"],
            inputs["ref_logprobs"],
            inputs["rollout_logprobs"],
            advantages,
            inputs["padding_mask"],
            prompt_lens=inputs["prompt_lens"],
        )
        loss_full.backward()
        grad_full = model_full.weight.grad.clone()

        # Chunked computation
        model_chunked = nn.Linear(32, 100)
        model_chunked.load_state_dict(inputs["model"].state_dict())

        num_chunks = (batch_size + chunk_size - 1) // chunk_size
        for chunk_idx in range(num_chunks):
            start = chunk_idx * chunk_size
            end = min(start + chunk_size, batch_size)

            chunk_hidden = inputs["hidden_states"][start:end]
            chunk_response = inputs["response"][start:end]
            chunk_ref = inputs["ref_logprobs"][start:end]
            chunk_rollout = inputs["rollout_logprobs"][start:end]
            chunk_adv = advantages[start:end]
            chunk_mask = inputs["padding_mask"][start:end]
            chunk_prompt_lens = inputs["prompt_lens"][start:end]

            logits_chunk = model_chunked(chunk_hidden)
            loss_chunk = loss_fn(
                logits_chunk,
                chunk_response,
                chunk_ref,
                chunk_rollout,
                chunk_adv,
                chunk_mask,
                prompt_lens=chunk_prompt_lens,
            )
            # Scale loss by 1/num_chunks for gradient averaging
            (loss_chunk / num_chunks).backward()

        grad_chunked = model_chunked.weight.grad.clone()

        # Gradients should be very close
        assert torch.allclose(grad_full, grad_chunked, atol=1e-4), \
            f"Max diff: {(grad_full - grad_chunked).abs().max()}"

    def test_chunked_with_uneven_chunks(self, model_and_inputs):
        """Chunked computation should work with uneven chunk sizes.

        With uneven chunks, we must weight each chunk's loss by its actual size
        relative to the total batch size, not just 1/num_chunks.
        """
        from rlvr_experiments.losses import GRPOLoss, compute_advantages

        loss_fn = GRPOLoss(beta=0.01, eps=0.2)
        inputs = model_and_inputs
        batch_size = inputs["hidden_states"].size(0)
        chunk_size = 3  # 8 / 3 = 2 chunks of 3 + 1 chunk of 2

        # Pre-compute advantages on full batch
        advantages = compute_advantages(inputs["rewards"])

        # Full batch
        model_full = nn.Linear(32, 100)
        model_full.load_state_dict(inputs["model"].state_dict())

        logits_full = model_full(inputs["hidden_states"])
        loss_full = loss_fn(
            logits_full,
            inputs["response"],
            inputs["ref_logprobs"],
            inputs["rollout_logprobs"],
            advantages,
            inputs["padding_mask"],
            prompt_lens=inputs["prompt_lens"],
        )
        loss_full.backward()
        grad_full = model_full.weight.grad.clone()

        # Chunked with uneven sizes - weight by actual chunk size
        model_chunked = nn.Linear(32, 100)
        model_chunked.load_state_dict(inputs["model"].state_dict())

        num_chunks = (batch_size + chunk_size - 1) // chunk_size

        for chunk_idx in range(num_chunks):
            start = chunk_idx * chunk_size
            end = min(start + chunk_size, batch_size)
            actual_chunk_size = end - start

            chunk_hidden = inputs["hidden_states"][start:end]
            chunk_response = inputs["response"][start:end]
            chunk_ref = inputs["ref_logprobs"][start:end]
            chunk_rollout = inputs["rollout_logprobs"][start:end]
            chunk_adv = advantages[start:end]
            chunk_mask = inputs["padding_mask"][start:end]
            chunk_prompt_lens = inputs["prompt_lens"][start:end]

            logits_chunk = model_chunked(chunk_hidden)
            loss_chunk = loss_fn(
                logits_chunk,
                chunk_response,
                chunk_ref,
                chunk_rollout,
                chunk_adv,
                chunk_mask,
                prompt_lens=chunk_prompt_lens,
            )
            # Weight by actual chunk size / total batch size
            chunk_weight = actual_chunk_size / batch_size
            (loss_chunk * chunk_weight).backward()

        grad_chunked = model_chunked.weight.grad.clone()

        assert torch.allclose(grad_full, grad_chunked, atol=1e-4)


class TestChunkedAdvantagesCorrectness:
    """Test that chunking advantages incorrectly gives WRONG results.

    This is a negative test to verify that our approach is necessary.
    If we compute advantages per-chunk instead of on the full batch,
    we get different results.
    """

    def test_per_chunk_advantages_differ_from_full_batch(self):
        """Per-chunk advantages should give different results than full-batch."""
        from rlvr_experiments.losses import compute_advantages

        # Create rewards that span a wide range
        rewards = torch.tensor([1.0, 0.9, 0.8, 0.7, 0.1, 0.0, -0.5, -1.0])

        # Full batch advantages
        full_advantages = compute_advantages(rewards)

        # Per-chunk advantages (wrong approach)
        chunk_size = 4
        chunk1_adv = compute_advantages(rewards[:chunk_size])
        chunk2_adv = compute_advantages(rewards[chunk_size:])
        per_chunk_advantages = torch.cat([chunk1_adv, chunk2_adv])

        # These should NOT be equal
        assert not torch.allclose(full_advantages, per_chunk_advantages, atol=1e-3), \
            "Per-chunk and full-batch advantages should differ!"
