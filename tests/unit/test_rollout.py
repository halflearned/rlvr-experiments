"""Unit tests for rollout types and batch creation."""

import pytest
import torch
from unittest.mock import MagicMock

from rlvr_experiments.algorithms.grpo import RolloutSample, TrainSample, make_batch, Batch


def make_train_sample(rollout: RolloutSample, rewards: list[float]) -> TrainSample:
    """Helper to create TrainSample with dummy ref_logprobs."""
    n = rollout.input_ids.size(0)
    comp_len = rollout.completion_ids.size(1)
    ref_logprobs = torch.zeros((n, comp_len), dtype=torch.float32)
    return TrainSample(rollout, rewards, ref_logprobs, item_id="test", trainer_version=0)


class TestRolloutSampleFromVllm:
    def test_basic_conversion(self, mock_vllm_output, mock_vllm_response):
        outputs = [
            mock_vllm_output([10, 11, 12], "hello", [-0.1, -0.2, -0.3]),
            mock_vllm_output([20, 21], "hi", [-0.5, -0.6]),
        ]
        response = mock_vllm_response([1, 2, 3], outputs)

        sample = RolloutSample.from_vllm(response, pad_token_id=0)

        # Check shapes
        assert sample.input_ids.shape == (2, 6)  # 2 outputs, max_len = 3 + 3 = 6
        assert sample.completion_ids.shape == (2, 3)  # max completion len = 3
        assert sample.logprobs.shape == (2, 3)

    def test_padding(self, mock_vllm_output, mock_vllm_response):
        outputs = [
            mock_vllm_output([10, 11, 12, 13], "long"),  # 4 tokens
            mock_vllm_output([20], "s"),  # 1 token
        ]
        response = mock_vllm_response([1, 2], outputs)

        sample = RolloutSample.from_vllm(response, pad_token_id=99)

        # Longer sequence: [1, 2, 10, 11, 12, 13] = 6 tokens (prompt + completion)
        # Shorter sequence: [1, 2, 20] + padding = [1, 2, 20, 99, 99, 99]
        # Padding is at the END of shorter sequences
        assert sample.input_ids[0, 0] == 1  # First token is prompt
        assert sample.input_ids[1, 0] == 1  # Same prompt for shorter seq
        assert sample.input_ids[1, -1] == 99  # Last positions are padding

    def test_completion_ids_left_aligned(self, mock_vllm_output, mock_vllm_response):
        outputs = [
            mock_vllm_output([10, 11, 12], "abc"),
            mock_vllm_output([20], "x"),
        ]
        response = mock_vllm_response([1], outputs)

        sample = RolloutSample.from_vllm(response, pad_token_id=0)

        # completion_ids should be left-aligned (padding on right)
        # For the short completion [20], it should be [20, 0, 0] in a length-3 tensor
        assert sample.completion_ids[1, 0] == 20
        assert sample.completion_ids[1, -1] == 0  # padding

    def test_logprobs_extracted(self, mock_vllm_output, mock_vllm_response):
        outputs = [
            mock_vllm_output([10, 11], "ab", [-1.0, -2.0]),
        ]
        response = mock_vllm_response([1], outputs)

        sample = RolloutSample.from_vllm(response, pad_token_id=0)

        # Logprobs should match what we provided (left-aligned)
        assert sample.logprobs[0, 0] == pytest.approx(-1.0)
        assert sample.logprobs[0, 1] == pytest.approx(-2.0)


class TestMakeBatch:
    def test_combines_samples(self, mock_vllm_output, mock_vllm_response):
        # Create two samples with varying rewards (both must have variance)
        outputs1 = [mock_vllm_output([10], "a"), mock_vllm_output([11], "b")]
        response1 = mock_vllm_response([1], outputs1)
        rollout1 = RolloutSample.from_vllm(response1, pad_token_id=0)
        sample1 = make_train_sample(rollout1, rewards=[1.0, 0.0])

        outputs2 = [mock_vllm_output([20], "c"), mock_vllm_output([21], "d")]
        response2 = mock_vllm_response([2], outputs2)
        rollout2 = RolloutSample.from_vllm(response2, pad_token_id=0)
        sample2 = make_train_sample(rollout2, rewards=[0.8, 0.2])

        batch, stats = make_batch([sample1, sample2], pad_token_id=0)

        assert batch is not None
        # 2 samples x 2 completions each = 4 total
        assert batch.input_ids.shape[0] == 4
        assert batch.rewards.shape[0] == 4

    def test_includes_all_samples(self, mock_vllm_output, mock_vllm_response):
        # make_batch no longer filters - filtering is done upstream in grpo_samples
        # This test verifies all samples are included regardless of variance
        outputs1 = [mock_vllm_output([10], "a"), mock_vllm_output([11], "b")]
        response1 = mock_vllm_response([1], outputs1)
        rollout1 = RolloutSample.from_vllm(response1, pad_token_id=0)
        sample1 = make_train_sample(rollout1, rewards=[1.0, 1.0])  # zero variance

        outputs2 = [mock_vllm_output([20], "c"), mock_vllm_output([21], "d")]
        response2 = mock_vllm_response([2], outputs2)
        rollout2 = RolloutSample.from_vllm(response2, pad_token_id=0)
        sample2 = make_train_sample(rollout2, rewards=[1.0, 0.0])  # has variance

        batch, stats = make_batch([sample1, sample2], pad_token_id=0)

        # Both samples included (4 completions total)
        assert batch.input_ids.shape[0] == 4

    def test_pads_to_max_length(self, mock_vllm_output, mock_vllm_response):
        # Short completion
        outputs1 = [mock_vllm_output([10], "a"), mock_vllm_output([11], "b")]
        response1 = mock_vllm_response([1], outputs1)
        rollout1 = RolloutSample.from_vllm(response1, pad_token_id=0)
        sample1 = make_train_sample(rollout1, rewards=[1.0, 0.0])

        # Long completion
        outputs2 = [
            mock_vllm_output([20, 21, 22, 23], "long"),
            mock_vllm_output([30], "s"),
        ]
        response2 = mock_vllm_response([2], outputs2)
        rollout2 = RolloutSample.from_vllm(response2, pad_token_id=0)
        sample2 = make_train_sample(rollout2, rewards=[0.0, 1.0])

        batch, stats = make_batch([sample1, sample2], pad_token_id=0)

        # All completions padded to at least max actual length (4)
        # Note: bucketing may round up to the nearest bucket, so check >= 4
        assert batch.completion_ids.shape[1] >= 4
        assert stats.padded_completion_len >= 4

    def test_mask_matches_non_padding(self, mock_vllm_output, mock_vllm_response):
        outputs = [mock_vllm_output([10, 11], "ab"), mock_vllm_output([20], "c")]
        response = mock_vllm_response([1], outputs)
        rollout = RolloutSample.from_vllm(response, pad_token_id=0)
        sample = make_train_sample(rollout, rewards=[1.0, 0.0])

        batch, stats = make_batch([sample], pad_token_id=0)

        # Mask should be 1 where completion_ids != pad_token_id
        expected_mask = (batch.completion_ids != 0).float()
        assert torch.allclose(batch.mask, expected_mask)

    def test_batch_dataclass_fields(self, mock_vllm_output, mock_vllm_response):
        outputs = [mock_vllm_output([10], "a"), mock_vllm_output([11], "b")]
        response = mock_vllm_response([1], outputs)
        rollout = RolloutSample.from_vllm(response, pad_token_id=0)
        sample = make_train_sample(rollout, rewards=[1.0, 0.0])

        batch, stats = make_batch([sample], pad_token_id=0)

        assert isinstance(batch, Batch)
        assert hasattr(batch, "input_ids")
        assert hasattr(batch, "completion_ids")
        assert hasattr(batch, "logprobs")
        assert hasattr(batch, "rewards")
        assert hasattr(batch, "mask")
        assert hasattr(batch, "prompt_lens")
        assert hasattr(batch, "ref_logprobs")

    def test_prompt_lens_tracked(self, mock_vllm_output, mock_vllm_response):
        # Prompt length 2
        outputs1 = [mock_vllm_output([10], "a"), mock_vllm_output([11], "b")]
        response1 = mock_vllm_response([1, 2], outputs1)  # prompt = [1, 2]
        rollout1 = RolloutSample.from_vllm(response1, pad_token_id=0)
        sample1 = make_train_sample(rollout1, rewards=[1.0, 0.0])

        # Prompt length 3
        outputs2 = [mock_vllm_output([20], "c"), mock_vllm_output([21], "d")]
        response2 = mock_vllm_response([1, 2, 3], outputs2)  # prompt = [1, 2, 3]
        rollout2 = RolloutSample.from_vllm(response2, pad_token_id=0)
        sample2 = make_train_sample(rollout2, rewards=[0.8, 0.2])

        batch, stats = make_batch([sample1, sample2], pad_token_id=0)

        # 4 completions total: 2 with prompt_len=2, 2 with prompt_len=3
        assert batch.prompt_lens.shape == (4,)
        assert batch.prompt_lens[0] == 2
        assert batch.prompt_lens[1] == 2
        assert batch.prompt_lens[2] == 3
        assert batch.prompt_lens[3] == 3

    def test_ref_logprobs_concatenated(self, mock_vllm_output, mock_vllm_response):
        outputs1 = [mock_vllm_output([10], "a"), mock_vllm_output([11], "b")]
        response1 = mock_vllm_response([1], outputs1)
        rollout1 = RolloutSample.from_vllm(response1, pad_token_id=0)
        ref1 = torch.tensor([[0.1], [0.2]])
        sample1 = TrainSample(rollout1, [1.0, 0.0], ref1, item_id="test1", trainer_version=0)

        outputs2 = [mock_vllm_output([20], "c"), mock_vllm_output([21], "d")]
        response2 = mock_vllm_response([2], outputs2)
        rollout2 = RolloutSample.from_vllm(response2, pad_token_id=0)
        ref2 = torch.tensor([[0.3], [0.4]])
        sample2 = TrainSample(rollout2, [0.8, 0.2], ref2, item_id="test2", trainer_version=0)

        batch, stats = make_batch([sample1, sample2], pad_token_id=0)

        # ref_logprobs should be concatenated and in batch
        assert batch.ref_logprobs.shape[0] == 4
        assert batch.ref_logprobs[0, 0] == pytest.approx(0.1)
        assert batch.ref_logprobs[1, 0] == pytest.approx(0.2)
        assert batch.ref_logprobs[2, 0] == pytest.approx(0.3)
        assert batch.ref_logprobs[3, 0] == pytest.approx(0.4)
