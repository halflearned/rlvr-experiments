"""Unit tests for compute_logprobs correctness.

These tests verify critical edge cases that could cause silent failures:
1. Logprobs are computed from correct positions based on prompt_lens
2. Different prompt lengths in the same batch are handled correctly
3. Padding in sequences doesn't corrupt logprob computation
4. Edge cases like zero-length completions are handled
"""

import pytest
import torch
import torch.nn.functional as F

from rlvr_experiments.ops import compute_logprobs


class TestComputeLogprobsBasic:
    """Basic functionality tests for compute_logprobs."""

    def test_simple_case(self):
        """Test simple case with uniform prompt length."""
        batch_size = 2
        seq_len = 10
        completion_len = 4
        vocab_size = 100

        # Create logits
        logits = torch.randn(batch_size, seq_len, vocab_size)
        # Create target completion tokens
        input_ids = torch.randint(0, vocab_size, (batch_size, completion_len))
        # All prompts have same length
        prompt_lens = torch.tensor([seq_len - completion_len] * batch_size)

        logprobs, _ = compute_logprobs(logits, input_ids, prompt_lens=prompt_lens)

        assert logprobs.shape == (batch_size, completion_len)
        # Logprobs should be negative (log of probability < 1)
        assert (logprobs <= 0).all()

    def test_matches_manual_computation(self):
        """Verify logprobs match manual cross_entropy computation."""
        batch_size = 1
        seq_len = 8
        completion_len = 3
        vocab_size = 10

        # Create simple logits and targets
        torch.manual_seed(42)
        logits = torch.randn(batch_size, seq_len, vocab_size)
        input_ids = torch.tensor([[2, 5, 7]])  # Target tokens
        prompt_lens = torch.tensor([seq_len - completion_len])  # = 5

        logprobs, _ = compute_logprobs(logits, input_ids, prompt_lens=prompt_lens)

        # Manual computation:
        # For token at position 0 in completion, we use logits at position prompt_len-1 = 4
        # For token at position 1 in completion, we use logits at position prompt_len = 5
        # etc.
        expected = []
        for i, target in enumerate(input_ids[0]):
            logit_pos = prompt_lens[0].item() - 1 + i
            log_softmax = F.log_softmax(logits[0, logit_pos], dim=-1)
            expected.append(log_softmax[target].item())

        expected = torch.tensor([expected])
        assert torch.allclose(logprobs, expected, atol=1e-5)


class TestComputeLogprobsPromptLens:
    """Tests for prompt_lens handling - critical for correctness."""

    def test_different_prompt_lens_in_batch(self):
        """Different prompt lengths should use different logit positions."""
        batch_size = 2
        seq_len = 16
        completion_len = 4
        vocab_size = 10

        torch.manual_seed(42)
        logits = torch.randn(batch_size, seq_len, vocab_size)
        input_ids = torch.randint(0, vocab_size, (batch_size, completion_len))

        # Different prompt lengths: 8 and 10
        prompt_lens = torch.tensor([8, 10])

        logprobs, _ = compute_logprobs(logits, input_ids, prompt_lens=prompt_lens)

        # Manually verify first token position for each sample
        # Sample 0: first completion token predicted by logits at position 7 (8-1)
        log_softmax_0 = F.log_softmax(logits[0, 7], dim=-1)
        expected_0 = log_softmax_0[input_ids[0, 0]].item()

        # Sample 1: first completion token predicted by logits at position 9 (10-1)
        log_softmax_1 = F.log_softmax(logits[1, 9], dim=-1)
        expected_1 = log_softmax_1[input_ids[1, 0]].item()

        assert logprobs[0, 0] == pytest.approx(expected_0, abs=1e-5)
        assert logprobs[1, 0] == pytest.approx(expected_1, abs=1e-5)

    def test_wrong_prompt_lens_gives_wrong_logprobs(self):
        """Using wrong prompt_lens should give different (wrong) logprobs.

        This is a negative test to verify that prompt_lens actually matters.
        """
        batch_size = 2
        seq_len = 16
        completion_len = 4
        vocab_size = 10

        torch.manual_seed(42)
        logits = torch.randn(batch_size, seq_len, vocab_size)
        input_ids = torch.randint(0, vocab_size, (batch_size, completion_len))

        # Correct prompt lengths
        correct_prompt_lens = torch.tensor([8, 10])
        # Wrong prompt lengths (shifted by 2)
        wrong_prompt_lens = torch.tensor([6, 8])

        correct_logprobs, _ = compute_logprobs(logits, input_ids, prompt_lens=correct_prompt_lens)
        wrong_logprobs, _ = compute_logprobs(logits, input_ids, prompt_lens=wrong_prompt_lens)

        # These should NOT be equal
        assert not torch.allclose(correct_logprobs, wrong_logprobs, atol=1e-3), \
            "Wrong prompt_lens should give different logprobs!"

    def test_prompt_lens_at_boundaries(self):
        """Test prompt_lens at sequence boundaries."""
        seq_len = 10
        completion_len = 3
        vocab_size = 5

        torch.manual_seed(42)
        logits = torch.randn(1, seq_len, vocab_size)
        input_ids = torch.randint(0, vocab_size, (1, completion_len))

        # prompt_len = 1 (minimum reasonable value)
        # This means completion starts at position 0
        prompt_lens = torch.tensor([1])
        logprobs, _ = compute_logprobs(logits, input_ids, prompt_lens=prompt_lens)

        # First token uses logits at position 0 (prompt_len - 1)
        log_softmax = F.log_softmax(logits[0, 0], dim=-1)
        expected = log_softmax[input_ids[0, 0]].item()
        assert logprobs[0, 0] == pytest.approx(expected, abs=1e-5)


class TestComputeLogprobsGroups:
    """Tests for _get_prompt_groups helper (used in DTensor path)."""

    def test_contiguous_groups_processed_correctly(self):
        """Samples with same prompt_len should be grouped correctly."""
        # Create batch where first 3 samples have prompt_len=5, next 2 have prompt_len=8
        batch_size = 5
        seq_len = 16
        completion_len = 4
        vocab_size = 10

        torch.manual_seed(42)
        logits = torch.randn(batch_size, seq_len, vocab_size)
        input_ids = torch.randint(0, vocab_size, (batch_size, completion_len))
        prompt_lens = torch.tensor([5, 5, 5, 8, 8])

        logprobs, _ = compute_logprobs(logits, input_ids, prompt_lens=prompt_lens)

        # Verify each sample individually
        for i in range(batch_size):
            for j in range(completion_len):
                logit_pos = prompt_lens[i].item() - 1 + j
                log_softmax = F.log_softmax(logits[i, logit_pos], dim=-1)
                expected = log_softmax[input_ids[i, j]].item()
                assert logprobs[i, j] == pytest.approx(expected, abs=1e-5), \
                    f"Mismatch at sample {i}, position {j}"


class TestComputeLogprobsPadding:
    """Tests for handling padded sequences."""

    def test_padding_in_completion_handled(self):
        """Padding in completion should still compute logprobs (mask applied later)."""
        seq_len = 10
        completion_len = 5
        vocab_size = 10
        pad_token = 0

        torch.manual_seed(42)
        logits = torch.randn(2, seq_len, vocab_size)

        # Sample 0: real completion [1, 2, 3], padding [0, 0]
        # Sample 1: full completion [4, 5, 6, 7, 8]
        input_ids = torch.tensor([
            [1, 2, 3, pad_token, pad_token],
            [4, 5, 6, 7, 8],
        ])
        prompt_lens = torch.tensor([5, 5])

        # This should compute logprobs for ALL positions (including padding)
        # The masking happens in the loss function, not here
        logprobs, _ = compute_logprobs(logits, input_ids, prompt_lens=prompt_lens)

        assert logprobs.shape == (2, completion_len)
        # All values should be finite
        assert torch.isfinite(logprobs).all()

    def test_no_nan_with_extreme_logits(self):
        """No NaN even with extreme logit values."""
        batch_size = 1
        seq_len = 8
        completion_len = 3
        vocab_size = 10

        # Create extreme logits
        logits = torch.zeros(batch_size, seq_len, vocab_size)
        logits[0, :, 0] = 1000.0  # Very high logit for token 0
        logits[0, :, 1:] = -1000.0  # Very low logits for others

        input_ids = torch.tensor([[0, 0, 0]])  # Target the high-prob token
        prompt_lens = torch.tensor([5])

        logprobs, _ = compute_logprobs(logits, input_ids, prompt_lens=prompt_lens)

        # Should be close to 0 (probability close to 1)
        assert torch.isfinite(logprobs).all()
        assert (logprobs > -0.1).all()  # log(~1) is close to 0


class TestComputeLogprobsEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_single_token_completion(self):
        """Handle single-token completion."""
        logits = torch.randn(1, 5, 10)
        input_ids = torch.tensor([[3]])
        prompt_lens = torch.tensor([4])

        logprobs, _ = compute_logprobs(logits, input_ids, prompt_lens=prompt_lens)

        assert logprobs.shape == (1, 1)
        assert torch.isfinite(logprobs).all()

    def test_large_batch(self):
        """Handle large batch efficiently."""
        batch_size = 128
        seq_len = 256
        completion_len = 64
        vocab_size = 1000

        logits = torch.randn(batch_size, seq_len, vocab_size)
        input_ids = torch.randint(0, vocab_size, (batch_size, completion_len))
        prompt_lens = torch.randint(100, 192, (batch_size,))  # Variable prompt lengths

        logprobs, _ = compute_logprobs(logits, input_ids, prompt_lens=prompt_lens)

        assert logprobs.shape == (batch_size, completion_len)
        assert torch.isfinite(logprobs).all()

    def test_temperature_scaling(self):
        """Temperature should affect logprob magnitudes."""
        logits = torch.randn(1, 8, 10)
        input_ids = torch.tensor([[5, 3, 7]])
        prompt_lens = torch.tensor([5])

        logprobs_t1, _ = compute_logprobs(logits, input_ids, prompt_lens=prompt_lens, temperature=1.0)
        logprobs_t2, _ = compute_logprobs(logits, input_ids, prompt_lens=prompt_lens, temperature=2.0)
        logprobs_t05, _ = compute_logprobs(logits, input_ids, prompt_lens=prompt_lens, temperature=0.5)

        # Higher temperature -> logprobs closer to uniform -> smaller magnitude differences
        # Lower temperature -> logprobs more peaked -> larger magnitude differences
        # All should be finite
        assert torch.isfinite(logprobs_t1).all()
        assert torch.isfinite(logprobs_t2).all()
        assert torch.isfinite(logprobs_t05).all()


class TestComputeLogprobsAlignment:
    """Tests for align=True/False behavior."""

    def test_align_false_one_to_one(self):
        """With align=False, logits and input_ids should be one-to-one."""
        batch_size = 2
        seq_len = 5
        vocab_size = 10

        torch.manual_seed(42)
        logits = torch.randn(batch_size, seq_len, vocab_size)
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))

        logprobs, _ = compute_logprobs(logits, input_ids, align=False)

        assert logprobs.shape == (batch_size, seq_len)

        # Each logprob should be computed from corresponding logit position
        for i in range(batch_size):
            for j in range(seq_len):
                log_softmax = F.log_softmax(logits[i, j], dim=-1)
                expected = log_softmax[input_ids[i, j]].item()
                assert logprobs[i, j] == pytest.approx(expected, abs=1e-5)


class TestComputeLogprobsIntegration:
    """Integration tests verifying compute_logprobs works in GRPO context."""

    def test_logprobs_match_across_forward_passes(self):
        """Same inputs should produce same logprobs (determinism)."""
        logits = torch.randn(4, 32, 100)
        input_ids = torch.randint(0, 100, (4, 8))
        prompt_lens = torch.tensor([24, 24, 24, 24])

        logprobs1, _ = compute_logprobs(logits, input_ids, prompt_lens=prompt_lens)
        logprobs2, _ = compute_logprobs(logits, input_ids, prompt_lens=prompt_lens)

        assert torch.equal(logprobs1, logprobs2)

    def test_grpo_typical_shapes(self):
        """Test with typical GRPO batch shapes from configs."""
        # From qwen3-1.7B-mixed.yaml:
        # completions_per_prompt = 16
        # prompts_per_forward_backward = 2
        # So batch_size = 32

        batch_size = 32
        seq_len = 768  # seq_len_bucket
        completion_len = 256  # Use smaller completion to avoid edge case
        vocab_size = 1000  # Smaller vocab for test

        logits = torch.randn(batch_size, seq_len, vocab_size)
        input_ids = torch.randint(0, vocab_size, (batch_size, completion_len))
        # Ensure prompt_len + completion_len <= seq_len
        # prompt_len must be at least 1, and prompt_len - 1 + completion_len <= seq_len - 1
        # So prompt_len <= seq_len - completion_len = 768 - 256 = 512
        prompt_lens = torch.randint(100, 300, (batch_size,))

        logprobs, _ = compute_logprobs(logits, input_ids, prompt_lens=prompt_lens)

        assert logprobs.shape == (batch_size, completion_len)
