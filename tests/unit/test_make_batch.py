"""Unit tests for make_batch padding, bucketing, and clamping logic.

These tests verify critical edge cases that could cause silent failures:
1. Sequence length bucketing for torch.compile cache efficiency
2. Completion length clamping when seq_len hits max bucket
3. Padding mask correctness at boundaries
4. Prompt length tracking across samples
"""

import pytest
import torch

from rlvr_experiments.algorithms.grpo import RolloutSample, TrainSample, make_batch


def make_rollout(prompt_len: int, completion_lens: list[int], pad_token_id: int = 0) -> RolloutSample:
    """Helper to create RolloutSample with specific lengths."""
    n = len(completion_lens)
    max_comp_len = max(completion_lens)
    max_seq_len = prompt_len + max_comp_len

    # Build input_ids: prompt + completion + padding
    input_ids = torch.full((n, max_seq_len), pad_token_id, dtype=torch.long)
    completion_ids = torch.full((n, max_comp_len), pad_token_id, dtype=torch.long)
    logprobs = torch.zeros((n, max_comp_len), dtype=torch.float32)

    for i, comp_len in enumerate(completion_lens):
        # Fill prompt (same for all)
        input_ids[i, :prompt_len] = torch.arange(1, prompt_len + 1)
        # Fill completion with unique tokens
        comp_tokens = torch.arange(100 + i * 100, 100 + i * 100 + comp_len)
        input_ids[i, prompt_len:prompt_len + comp_len] = comp_tokens
        completion_ids[i, :comp_len] = comp_tokens
        logprobs[i, :comp_len] = -torch.arange(1, comp_len + 1).float()

    return RolloutSample(
        input_ids=input_ids,
        completion_ids=completion_ids,
        logprobs=logprobs,
        prompt_len=prompt_len,
        finish_reasons=["stop"] * n,
        completion_lens=completion_lens,
    )


def make_train_sample(
    prompt_len: int,
    completion_lens: list[int],
    rewards: list[float],
    pad_token_id: int = 0,
    trainer_version: int = 0,
    item_id: str = "test",
) -> TrainSample:
    """Helper to create TrainSample with specific lengths."""
    rollout = make_rollout(prompt_len, completion_lens, pad_token_id)
    n = len(completion_lens)
    max_comp_len = max(completion_lens)
    ref_logprobs = torch.randn((n, max_comp_len))
    return TrainSample(
        rollout=rollout,
        rewards=rewards,
        ref_logprobs=ref_logprobs,
        item_id=item_id,
        trainer_version=trainer_version,
    )


class TestMakeBatchBucketing:
    """Tests for sequence and completion length bucketing."""

    def test_seq_len_rounds_up_to_bucket(self):
        """Sequence length should round up to nearest bucket."""
        # Natural seq_len = 100 (50 prompt + 50 completion)
        sample = make_train_sample(prompt_len=50, completion_lens=[50], rewards=[1.0])

        # Use buckets that will force rounding up
        batch, stats = make_batch(
            [sample],
            pad_token_id=0,
            seq_len_buckets=[64, 128, 256],
            completion_len_buckets=[64, 128],
        )

        # 100 should round up to 128
        assert batch.input_ids.shape[1] == 128
        assert stats.padded_seq_len == 128

    def test_completion_len_rounds_up_to_bucket(self):
        """Completion length should round up to nearest bucket."""
        sample = make_train_sample(prompt_len=10, completion_lens=[30], rewards=[1.0])

        batch, stats = make_batch(
            [sample],
            pad_token_id=0,
            seq_len_buckets=[256],
            completion_len_buckets=[32, 64, 128],
        )

        # 30 should round up to 32
        assert batch.completion_ids.shape[1] == 32
        assert stats.padded_completion_len == 32

    def test_seq_len_clamps_to_max_bucket(self):
        """Sequence length should clamp to max bucket when too large."""
        # Very long completion that exceeds max bucket
        sample = make_train_sample(prompt_len=100, completion_lens=[500], rewards=[1.0])

        batch, stats = make_batch(
            [sample],
            pad_token_id=0,
            seq_len_buckets=[128, 256],  # Max is 256
            completion_len_buckets=[128, 256, 512],
        )

        # Should clamp to 256
        assert batch.input_ids.shape[1] == 256
        assert stats.padded_seq_len == 256


class TestMakeBatchCompletionClamping:
    """Tests for completion length clamping when seq_len hits max bucket.

    This is a critical edge case: if seq_len is clamped to max bucket,
    completion_len must also be clamped so that prompt_len + completion_len <= seq_len.
    Otherwise, compute_logprobs will try to access positions beyond seq_len.
    """

    def test_completion_clamped_when_seq_clamped(self):
        """Completion length should be clamped when sequence is clamped."""
        # prompt=200, completion=300 -> natural seq=500
        # But max seq bucket is 256
        # So completion must be clamped to 256 - 200 = 56
        sample = make_train_sample(prompt_len=200, completion_lens=[300], rewards=[1.0])

        batch, stats = make_batch(
            [sample],
            pad_token_id=0,
            seq_len_buckets=[256],
            completion_len_buckets=[128, 256, 512],
        )

        # Seq clamped to 256
        assert batch.input_ids.shape[1] == 256
        # Completion must fit: 256 - 200 = 56
        assert batch.completion_ids.shape[1] == 56
        assert stats.padded_completion_len == 56

    def test_prompt_plus_completion_fits_in_seq(self):
        """prompt_len + completion_len should always fit in seq_len."""
        # Various prompt/completion combinations
        test_cases = [
            (100, 200),  # 300 total, needs clamping if max=256
            (200, 100),  # 300 total
            (50, 400),   # 450 total
        ]

        for prompt_len, comp_len in test_cases:
            sample = make_train_sample(
                prompt_len=prompt_len,
                completion_lens=[comp_len],
                rewards=[1.0]
            )

            batch, stats = make_batch(
                [sample],
                pad_token_id=0,
                seq_len_buckets=[256],
                completion_len_buckets=[512],
            )

            # This should never fail
            max_prompt = batch.prompt_lens.max().item()
            assert max_prompt + stats.padded_completion_len <= stats.padded_seq_len, \
                f"prompt={max_prompt} + completion={stats.padded_completion_len} > seq={stats.padded_seq_len}"


class TestMakeBatchMixedPromptLens:
    """Tests for batching samples with different prompt lengths."""

    def test_multiple_prompt_lens_tracked(self):
        """Multiple samples with different prompt_lens should be tracked correctly."""
        sample1 = make_train_sample(prompt_len=10, completion_lens=[20, 25], rewards=[1.0, 0.0])
        sample2 = make_train_sample(prompt_len=30, completion_lens=[15, 20], rewards=[0.5, 0.5])

        batch, stats = make_batch(
            [sample1, sample2],
            pad_token_id=0,
            seq_len_buckets=[256],
            completion_len_buckets=[128],
        )

        # 4 completions total
        assert batch.prompt_lens.shape == (4,)
        # First 2 have prompt_len=10, next 2 have prompt_len=30
        assert batch.prompt_lens[0] == 10
        assert batch.prompt_lens[1] == 10
        assert batch.prompt_lens[2] == 30
        assert batch.prompt_lens[3] == 30

    def test_seq_len_accommodates_longest_prompt(self):
        """Sequence length should accommodate the longest prompt + completion."""
        # Short prompt, long completion
        sample1 = make_train_sample(prompt_len=10, completion_lens=[100], rewards=[1.0])
        # Long prompt, short completion
        sample2 = make_train_sample(prompt_len=150, completion_lens=[20], rewards=[1.0])

        batch, stats = make_batch(
            [sample1, sample2],
            pad_token_id=0,
            seq_len_buckets=[256],
            completion_len_buckets=[128],
        )

        # Need to fit: max(10+100, 150+20) = max(110, 170) = 170
        # But also need max_prompt + padded_completion = 150 + 100 (bucketed to 128) = 278
        # So seq_len should be >= 278, clamped to 256
        # And completion should be clamped to 256 - 150 = 106
        assert stats.padded_seq_len == 256
        assert stats.padded_completion_len == 106


class TestMakeBatchPaddingMask:
    """Tests for padding mask correctness."""

    def test_mask_zeros_padding(self):
        """Mask should be 0 for padding positions."""
        sample = make_train_sample(
            prompt_len=10,
            completion_lens=[5, 10],  # Different lengths
            rewards=[1.0, 0.0],
            pad_token_id=0,
        )

        batch, _ = make_batch([sample], pad_token_id=0)

        # First completion has 5 tokens, rest is padding
        # Mask should be [1,1,1,1,1,0,0,0,0,0] for first row (if padded to 10)
        assert batch.mask[0, :5].sum() == 5  # First 5 are non-padding
        assert batch.mask[0, 5:].sum() == 0  # Rest is padding

        # Second completion has 10 tokens
        assert batch.mask[1, :10].sum() == 10

    def test_mask_matches_completion_not_pad_token(self):
        """Mask should match where completion_ids != pad_token_id."""
        sample = make_train_sample(
            prompt_len=10,
            completion_lens=[3, 7],
            rewards=[1.0, 0.0],
            pad_token_id=99,  # Use different pad token
        )

        batch, _ = make_batch([sample], pad_token_id=99)

        # Mask should exactly match non-padding positions
        expected_mask = (batch.completion_ids != 99).float()
        assert torch.allclose(batch.mask, expected_mask)


class TestMakeBatchEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_single_token_completion(self):
        """Handle single-token completions."""
        sample = make_train_sample(prompt_len=10, completion_lens=[1], rewards=[1.0])

        batch, stats = make_batch(
            [sample],
            pad_token_id=0,
            seq_len_buckets=[256],
            completion_len_buckets=[1, 8, 16],
        )

        assert batch.completion_ids.shape[1] == 1
        assert stats.padded_completion_len == 1

    def test_empty_completion_buckets_uses_natural_len(self):
        """Empty bucket list should use natural lengths."""
        sample = make_train_sample(prompt_len=10, completion_lens=[23], rewards=[1.0])

        # This should work but bucket to defaults
        batch, stats = make_batch([sample], pad_token_id=0)

        # Should round up to some default bucket
        assert stats.padded_completion_len >= 23

    def test_identical_completions(self):
        """Handle all completions having identical length."""
        sample = make_train_sample(
            prompt_len=10,
            completion_lens=[20, 20, 20],
            rewards=[1.0, 0.5, 0.0]
        )

        batch, stats = make_batch([sample], pad_token_id=0)

        # All mask rows should be identical
        assert (batch.mask[0] == batch.mask[1]).all()
        assert (batch.mask[1] == batch.mask[2]).all()


class TestMakeBatchRefLogprobs:
    """Tests for reference logprobs handling."""

    def test_ref_logprobs_padded_to_completion_len(self):
        """Reference logprobs should be padded to match completion length."""
        rollout = make_rollout(prompt_len=10, completion_lens=[5, 10])
        # Ref logprobs with original lengths
        ref_logprobs = torch.tensor([
            [1.0, 2.0, 3.0, 4.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # 5 real + 5 padding
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],  # 10 real
        ])
        sample = TrainSample(rollout, [1.0, 0.0], ref_logprobs, item_id="test", trainer_version=0)

        batch, stats = make_batch(
            [sample],
            pad_token_id=0,
            completion_len_buckets=[16],
        )

        # Ref logprobs should be padded to bucketed completion length
        assert batch.ref_logprobs.shape[1] == stats.padded_completion_len

    def test_ref_logprobs_values_preserved(self):
        """Original ref_logprobs values should be preserved after padding."""
        rollout = make_rollout(prompt_len=10, completion_lens=[3])
        ref_logprobs = torch.tensor([[1.5, 2.5, 3.5]])
        sample = TrainSample(rollout, [1.0], ref_logprobs, item_id="test", trainer_version=0)

        batch, _ = make_batch([sample], pad_token_id=0, completion_len_buckets=[8])

        # Original values should be preserved at the start
        assert batch.ref_logprobs[0, 0] == pytest.approx(1.5)
        assert batch.ref_logprobs[0, 1] == pytest.approx(2.5)
        assert batch.ref_logprobs[0, 2] == pytest.approx(3.5)


class TestBatchStatsAccuracy:
    """Tests for BatchStats accuracy."""

    def test_stats_track_actual_lengths(self):
        """BatchStats should track actual (non-padded) lengths."""
        sample1 = make_train_sample(prompt_len=10, completion_lens=[5, 15], rewards=[1.0, 0.0])
        sample2 = make_train_sample(prompt_len=20, completion_lens=[10], rewards=[0.5])

        _, stats = make_batch([sample1, sample2], pad_token_id=0)

        # seq_lens = [10+5, 10+15, 20+10] = [15, 25, 30]
        assert stats.seq_lens == [15, 25, 30]
        # completion_lens = [5, 15, 10]
        assert stats.completion_lens == [5, 15, 10]

    def test_stats_track_finish_reasons(self):
        """BatchStats should count finish reasons."""
        rollout1 = make_rollout(prompt_len=10, completion_lens=[5, 10])
        rollout1.finish_reasons = ["stop", "length"]
        sample1 = TrainSample(
            rollout1,
            [1.0, 0.0],
            torch.zeros((2, 10)),
            item_id="test1",
            trainer_version=0,
        )

        rollout2 = make_rollout(prompt_len=10, completion_lens=[8])
        rollout2.finish_reasons = ["stop"]
        sample2 = TrainSample(
            rollout2,
            [0.5],
            torch.zeros((1, 8)),
            item_id="test2",
            trainer_version=0,
        )

        _, stats = make_batch([sample1, sample2], pad_token_id=0)

        assert stats.finish_reasons["stop"] == 2
        assert stats.finish_reasons["length"] == 1
