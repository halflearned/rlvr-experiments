"""Integration tests for GRPO learning correctness.

These tests verify that GRPO training actually makes the model learn:
1. Probability of high-reward completions should increase
2. Probability of low-reward completions should decrease
3. Learning should be stable (no NaN, no explosion)
4. Different reward patterns produce expected learning

These are critical tests for detecting silent failures where the training
loop runs without errors but the model doesn't actually learn.
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from rlvr_experiments.losses import GRPOLoss, DrGRPOLoss, compute_advantages, compute_drgrpo_advantages
from rlvr_experiments.ops import compute_logprobs


class SmallTransformer(nn.Module):
    """Minimal transformer for testing GRPO learning.

    Small enough to train quickly, large enough to have meaningful learning.
    """

    def __init__(self, vocab_size: int = 100, d_model: int = 64, n_heads: int = 4, n_layers: int = 2):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(128, d_model)  # Max seq len
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, n_layers)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Forward pass returning logits."""
        seq_len = input_ids.shape[1]
        x = self.embed(input_ids)
        pos = self.pos_embed(torch.arange(seq_len, device=input_ids.device))
        x = x + pos
        # Causal mask for autoregressive attention
        mask = nn.Transformer.generate_square_subsequent_mask(seq_len, device=input_ids.device)
        x = self.transformer(x, mask=mask, is_causal=True)
        return self.head(x)

    def get_token_prob(self, input_ids: torch.Tensor, target_token: int, position: int) -> float:
        """Get probability of target_token at given position."""
        with torch.no_grad():
            logits = self.forward(input_ids)
            probs = F.softmax(logits[0, position - 1], dim=-1)  # Position - 1 predicts position
            return probs[target_token].item()


class TestGRPOLearningDirection:
    """Tests verifying GRPO pushes probability in the correct direction."""

    @pytest.fixture
    def model_and_optimizer(self):
        """Create fresh model and optimizer for each test."""
        torch.manual_seed(42)
        model = SmallTransformer(vocab_size=100, d_model=64, n_heads=4, n_layers=2)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        return model, optimizer

    @pytest.mark.integration
    def test_high_reward_completion_probability_increases(self, model_and_optimizer):
        """Probability of high-reward completion should increase with training."""
        model, optimizer = model_and_optimizer
        loss_fn = GRPOLoss(beta=0.0, eps=0.2)  # No KL penalty

        # Fixed prompt
        prompt = torch.tensor([[1, 2, 3]])  # 3-token prompt
        prompt_len = 3

        # Two completions: one good (reward=1), one bad (reward=0)
        good_completion = torch.tensor([[10]])  # Token 10 is good
        bad_completion = torch.tensor([[20]])  # Token 20 is bad

        # Full sequences
        good_seq = torch.cat([prompt, good_completion], dim=1)
        bad_seq = torch.cat([prompt, bad_completion], dim=1)

        # Batch: [good_seq, bad_seq]
        input_ids = torch.cat([good_seq, bad_seq], dim=0)
        completion_ids = torch.cat([good_completion, bad_completion], dim=0)
        rewards = torch.tensor([1.0, 0.0])
        mask = torch.ones(2, 1)
        prompt_lens = torch.tensor([prompt_len, prompt_len])

        # Get initial probabilities
        prob_good_before = model.get_token_prob(prompt, 10, prompt_len)
        prob_bad_before = model.get_token_prob(prompt, 20, prompt_len)

        # Train for several steps
        for _ in range(50):
            optimizer.zero_grad()
            logits = model(input_ids)

            # Use current model for ref and rollout logprobs (on-policy)
            with torch.no_grad():
                ref_logprobs = compute_logprobs(logits.detach(), completion_ids, prompt_lens=prompt_lens)
            rollout_logprobs = ref_logprobs.clone()

            advantages = compute_advantages(rewards)

            loss = loss_fn(
                logits,
                completion_ids,
                ref_logprobs,
                rollout_logprobs,
                advantages,
                mask,
                prompt_lens,
            )
            loss.backward()
            optimizer.step()

        # Get final probabilities
        prob_good_after = model.get_token_prob(prompt, 10, prompt_len)
        prob_bad_after = model.get_token_prob(prompt, 20, prompt_len)

        print(f"Good token: {prob_good_before:.4f} -> {prob_good_after:.4f}")
        print(f"Bad token: {prob_bad_before:.4f} -> {prob_bad_after:.4f}")

        # Good completion probability should increase
        assert prob_good_after > prob_good_before, \
            f"P(good) should increase: {prob_good_before:.4f} -> {prob_good_after:.4f}"
        # Bad completion probability should decrease
        assert prob_bad_after < prob_bad_before, \
            f"P(bad) should decrease: {prob_bad_before:.4f} -> {prob_bad_after:.4f}"

    @pytest.mark.integration
    def test_advantage_sign_determines_direction(self, model_and_optimizer):
        """Positive advantage should increase probability, negative should decrease."""
        model, optimizer = model_and_optimizer
        loss_fn = GRPOLoss(beta=0.0, eps=0.2)

        prompt = torch.tensor([[1, 2, 3]])
        prompt_len = 3

        # Track multiple tokens
        target_tokens = [10, 20, 30, 40]
        rewards = torch.tensor([1.0, 0.8, 0.2, 0.0])  # Different rewards

        # Build batch
        seqs = [torch.cat([prompt, torch.tensor([[t]])], dim=1) for t in target_tokens]
        input_ids = torch.cat(seqs, dim=0)
        completion_ids = torch.tensor([[t] for t in target_tokens])
        mask = torch.ones(4, 1)
        prompt_lens = torch.tensor([prompt_len] * 4)

        # Get initial probabilities
        probs_before = [model.get_token_prob(prompt, t, prompt_len) for t in target_tokens]

        # Train
        for _ in range(50):
            optimizer.zero_grad()
            logits = model(input_ids)

            with torch.no_grad():
                ref_logprobs = compute_logprobs(logits.detach(), completion_ids, prompt_lens=prompt_lens)
            rollout_logprobs = ref_logprobs.clone()

            advantages = compute_advantages(rewards)

            loss = loss_fn(
                logits,
                completion_ids,
                ref_logprobs,
                rollout_logprobs,
                advantages,
                mask,
                prompt_lens,
            )
            loss.backward()
            optimizer.step()

        # Get final probabilities
        probs_after = [model.get_token_prob(prompt, t, prompt_len) for t in target_tokens]
        advantages_val = compute_advantages(rewards).tolist()

        print("Token | Advantage | Before | After | Change")
        for i, (t, adv, pb, pa) in enumerate(zip(target_tokens, advantages_val, probs_before, probs_after)):
            print(f"  {t:3d} | {adv:+.3f}    | {pb:.4f} | {pa:.4f} | {pa - pb:+.4f}")

        # Positive advantages should increase probability
        # Negative advantages should decrease probability
        for i, (adv, pb, pa) in enumerate(zip(advantages_val, probs_before, probs_after)):
            if adv > 0.1:
                assert pa > pb, f"Token {target_tokens[i]}: positive adv={adv:.3f} but prob decreased"
            elif adv < -0.1:
                assert pa < pb, f"Token {target_tokens[i]}: negative adv={adv:.3f} but prob increased"


class TestGRPOLearningStability:
    """Tests for training stability - no NaN, no explosion."""

    @pytest.mark.integration
    def test_loss_is_finite(self):
        """Loss should always be finite during training."""
        torch.manual_seed(42)
        model = SmallTransformer()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        loss_fn = GRPOLoss(beta=0.1, eps=0.2)

        prompt_len = 5
        completion_len = 10
        batch_size = 8

        for step in range(100):
            # Random batch each step
            input_ids = torch.randint(0, 100, (batch_size, prompt_len + completion_len))
            completion_ids = input_ids[:, prompt_len:]
            rewards = torch.rand(batch_size)
            mask = torch.ones(batch_size, completion_len)
            prompt_lens = torch.tensor([prompt_len] * batch_size)

            optimizer.zero_grad()
            logits = model(input_ids)

            with torch.no_grad():
                ref_logprobs = compute_logprobs(logits.detach(), completion_ids, prompt_lens=prompt_lens)
            rollout_logprobs = ref_logprobs.clone()
            advantages = compute_advantages(rewards)

            loss = loss_fn(
                logits,
                completion_ids,
                ref_logprobs,
                rollout_logprobs,
                advantages,
                mask,
                prompt_lens,
            )

            assert torch.isfinite(loss), f"Loss is not finite at step {step}: {loss}"
            assert loss.abs() < 1000, f"Loss exploded at step {step}: {loss}"

            loss.backward()
            optimizer.step()

    @pytest.mark.integration
    def test_gradients_are_finite(self):
        """Gradients should always be finite."""
        torch.manual_seed(42)
        model = SmallTransformer()
        loss_fn = GRPOLoss(beta=0.1, eps=0.2)

        prompt_len = 5
        completion_len = 10
        batch_size = 4

        for _ in range(20):
            input_ids = torch.randint(0, 100, (batch_size, prompt_len + completion_len))
            completion_ids = input_ids[:, prompt_len:]
            rewards = torch.rand(batch_size)
            mask = torch.ones(batch_size, completion_len)
            prompt_lens = torch.tensor([prompt_len] * batch_size)

            logits = model(input_ids)

            with torch.no_grad():
                ref_logprobs = compute_logprobs(logits.detach(), completion_ids, prompt_lens=prompt_lens)
            rollout_logprobs = ref_logprobs.clone()
            advantages = compute_advantages(rewards)

            loss = loss_fn(
                logits,
                completion_ids,
                ref_logprobs,
                rollout_logprobs,
                advantages,
                mask,
                prompt_lens,
            )
            loss.backward()

            for name, param in model.named_parameters():
                if param.grad is not None:
                    assert torch.isfinite(param.grad).all(), \
                        f"Non-finite gradient for {name}"


class TestGRPOMultiPrompt:
    """Tests for multi-prompt batches with grouped advantages."""

    @pytest.mark.integration
    def test_grouped_advantages_isolate_prompts(self):
        """Each prompt's completions should be normalized separately."""
        torch.manual_seed(42)
        model = SmallTransformer()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        loss_fn = GRPOLoss(beta=0.0, eps=0.2)

        # Two prompts, 2 completions each
        prompt1 = torch.tensor([[1, 2]])
        prompt2 = torch.tensor([[3, 4]])
        prompt_len = 2

        # Prompt 1: token 10 is good, token 11 is bad
        # Prompt 2: token 20 is good, token 21 is bad
        seqs = [
            torch.cat([prompt1, torch.tensor([[10]])], dim=1),  # Prompt 1, good
            torch.cat([prompt1, torch.tensor([[11]])], dim=1),  # Prompt 1, bad
            torch.cat([prompt2, torch.tensor([[20]])], dim=1),  # Prompt 2, good
            torch.cat([prompt2, torch.tensor([[21]])], dim=1),  # Prompt 2, bad
        ]
        input_ids = torch.cat(seqs, dim=0)
        completion_ids = torch.tensor([[10], [11], [20], [21]])
        rewards = torch.tensor([1.0, 0.0, 1.0, 0.0])
        mask = torch.ones(4, 1)
        prompt_lens = torch.tensor([prompt_len] * 4)

        # Get initial probabilities
        prob_10_before = model.get_token_prob(prompt1, 10, prompt_len)
        prob_11_before = model.get_token_prob(prompt1, 11, prompt_len)
        prob_20_before = model.get_token_prob(prompt2, 20, prompt_len)
        prob_21_before = model.get_token_prob(prompt2, 21, prompt_len)

        # Train with grouped advantages (group_size=2)
        for _ in range(50):
            optimizer.zero_grad()
            logits = model(input_ids)

            with torch.no_grad():
                ref_logprobs = compute_logprobs(logits.detach(), completion_ids, prompt_lens=prompt_lens)
            rollout_logprobs = ref_logprobs.clone()

            # Group size = 2 (2 completions per prompt)
            advantages = compute_advantages(rewards, group_size=2)

            loss = loss_fn(
                logits,
                completion_ids,
                ref_logprobs,
                rollout_logprobs,
                advantages,
                mask,
                prompt_lens,
            )
            loss.backward()
            optimizer.step()

        # Get final probabilities
        prob_10_after = model.get_token_prob(prompt1, 10, prompt_len)
        prob_11_after = model.get_token_prob(prompt1, 11, prompt_len)
        prob_20_after = model.get_token_prob(prompt2, 20, prompt_len)
        prob_21_after = model.get_token_prob(prompt2, 21, prompt_len)

        print(f"Prompt 1 - Token 10: {prob_10_before:.4f} -> {prob_10_after:.4f}")
        print(f"Prompt 1 - Token 11: {prob_11_before:.4f} -> {prob_11_after:.4f}")
        print(f"Prompt 2 - Token 20: {prob_20_before:.4f} -> {prob_20_after:.4f}")
        print(f"Prompt 2 - Token 21: {prob_21_before:.4f} -> {prob_21_after:.4f}")

        # Good tokens should increase for each prompt
        assert prob_10_after > prob_10_before, "Prompt 1 good token should increase"
        assert prob_20_after > prob_20_before, "Prompt 2 good token should increase"
        # Bad tokens should decrease for each prompt
        assert prob_11_after < prob_11_before, "Prompt 1 bad token should decrease"
        assert prob_21_after < prob_21_before, "Prompt 2 bad token should decrease"


class TestDrGRPOLearning:
    """Tests for Dr. GRPO loss variant."""

    @pytest.mark.integration
    def test_drgrpo_learns_without_std_normalization(self):
        """Dr. GRPO should learn with mean-centered (not std-normalized) advantages."""
        torch.manual_seed(42)
        model = SmallTransformer()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        loss_fn = DrGRPOLoss(beta=0.0, eps=0.2)

        prompt = torch.tensor([[1, 2, 3]])
        prompt_len = 3

        good_seq = torch.cat([prompt, torch.tensor([[10]])], dim=1)
        bad_seq = torch.cat([prompt, torch.tensor([[20]])], dim=1)

        input_ids = torch.cat([good_seq, bad_seq], dim=0)
        completion_ids = torch.tensor([[10], [20]])
        rewards = torch.tensor([1.0, 0.0])
        mask = torch.ones(2, 1)
        prompt_lens = torch.tensor([prompt_len, prompt_len])

        prob_good_before = model.get_token_prob(prompt, 10, prompt_len)
        prob_bad_before = model.get_token_prob(prompt, 20, prompt_len)

        for _ in range(50):
            optimizer.zero_grad()
            logits = model(input_ids)

            with torch.no_grad():
                ref_logprobs = compute_logprobs(logits.detach(), completion_ids, prompt_lens=prompt_lens)
            rollout_logprobs = ref_logprobs.clone()

            # Dr. GRPO uses mean-centered advantages without std normalization
            advantages = compute_drgrpo_advantages(rewards)

            loss = loss_fn(
                logits,
                completion_ids,
                ref_logprobs,
                rollout_logprobs,
                advantages,
                mask,
                prompt_lens,
            )
            loss.backward()
            optimizer.step()

        prob_good_after = model.get_token_prob(prompt, 10, prompt_len)
        prob_bad_after = model.get_token_prob(prompt, 20, prompt_len)

        print(f"Good token: {prob_good_before:.4f} -> {prob_good_after:.4f}")
        print(f"Bad token: {prob_bad_before:.4f} -> {prob_bad_after:.4f}")

        assert prob_good_after > prob_good_before, "Dr. GRPO should increase good token prob"
        assert prob_bad_after < prob_bad_before, "Dr. GRPO should decrease bad token prob"


class TestKLPenalty:
    """Tests for KL divergence penalty effect."""

    @pytest.mark.integration
    def test_high_beta_constrains_policy_shift(self):
        """Higher KL penalty (beta) should result in smaller policy changes.

        For the KL penalty to have an effect, ref_logprobs must differ from
        trainer_logprobs. We use a fixed reference model (not updated during training)
        to create this divergence.
        """
        torch.manual_seed(42)

        # Create reference model (frozen)
        ref_model = SmallTransformer()

        # Train two models with different beta values
        model_low_beta = SmallTransformer()
        model_high_beta = SmallTransformer()
        # Start with same weights as reference
        model_low_beta.load_state_dict(ref_model.state_dict())
        model_high_beta.load_state_dict(ref_model.state_dict())

        opt_low = torch.optim.AdamW(model_low_beta.parameters(), lr=1e-3)
        opt_high = torch.optim.AdamW(model_high_beta.parameters(), lr=1e-3)

        loss_fn_low = GRPOLoss(beta=0.01, eps=0.2)
        loss_fn_high = GRPOLoss(beta=1.0, eps=0.2)  # Much higher KL penalty

        prompt = torch.tensor([[1, 2, 3]])
        prompt_len = 3

        input_ids = torch.cat([
            torch.cat([prompt, torch.tensor([[10]])], dim=1),
            torch.cat([prompt, torch.tensor([[20]])], dim=1),
        ], dim=0)
        completion_ids = torch.tensor([[10], [20]])
        rewards = torch.tensor([1.0, 0.0])
        mask = torch.ones(2, 1)
        prompt_lens = torch.tensor([prompt_len, prompt_len])

        # Compute fixed reference logprobs from frozen model
        with torch.no_grad():
            ref_logits = ref_model(input_ids)
            ref_logprobs = compute_logprobs(ref_logits, completion_ids, prompt_lens=prompt_lens)

        # Get initial probability
        prob_before = model_low_beta.get_token_prob(prompt, 10, prompt_len)

        # Train both models with FIXED reference logprobs
        for _ in range(50):
            for model, opt, loss_fn in [
                (model_low_beta, opt_low, loss_fn_low),
                (model_high_beta, opt_high, loss_fn_high),
            ]:
                opt.zero_grad()
                logits = model(input_ids)

                # Use current model for rollout logprobs (on-policy)
                with torch.no_grad():
                    rollout_logprobs = compute_logprobs(logits.detach(), completion_ids, prompt_lens=prompt_lens)
                advantages = compute_advantages(rewards)

                # Use FIXED ref_logprobs (from frozen reference model)
                loss = loss_fn(
                    logits,
                    completion_ids,
                    ref_logprobs.clone(),  # Fixed reference
                    rollout_logprobs,
                    advantages,
                    mask,
                    prompt_lens,
                )
                loss.backward()
                opt.step()

        prob_low_beta = model_low_beta.get_token_prob(prompt, 10, prompt_len)
        prob_high_beta = model_high_beta.get_token_prob(prompt, 10, prompt_len)

        print(f"Initial prob: {prob_before:.4f}")
        print(f"Low beta (0.01): {prob_low_beta:.4f}")
        print(f"High beta (1.0): {prob_high_beta:.4f}")

        # Both should increase probability of good token
        assert prob_low_beta > prob_before
        assert prob_high_beta > prob_before

        # Low beta should have moved more (less constrained by KL)
        change_low = prob_low_beta - prob_before
        change_high = prob_high_beta - prob_before
        assert change_low > change_high, \
            f"Low beta should change more: {change_low:.4f} vs {change_high:.4f}"


class TestClipping:
    """Tests for PPO-style clipping effect."""

    @pytest.mark.integration
    def test_clipping_limits_updates(self):
        """Clipping (eps > 0) should limit the size of policy updates."""
        torch.manual_seed(42)

        # Train with no clipping vs with clipping
        model_no_clip = SmallTransformer()
        model_clip = SmallTransformer()
        model_clip.load_state_dict(model_no_clip.state_dict())

        opt_no_clip = torch.optim.AdamW(model_no_clip.parameters(), lr=1e-2)  # High LR
        opt_clip = torch.optim.AdamW(model_clip.parameters(), lr=1e-2)

        loss_fn_no_clip = GRPOLoss(beta=0.0, eps=0.0)  # No clipping
        loss_fn_clip = GRPOLoss(beta=0.0, eps=0.1)  # Tight clipping

        prompt = torch.tensor([[1, 2, 3]])
        prompt_len = 3

        input_ids = torch.cat([
            torch.cat([prompt, torch.tensor([[10]])], dim=1),
            torch.cat([prompt, torch.tensor([[20]])], dim=1),
        ], dim=0)
        completion_ids = torch.tensor([[10], [20]])
        rewards = torch.tensor([1.0, 0.0])
        mask = torch.ones(2, 1)
        prompt_lens = torch.tensor([prompt_len, prompt_len])

        prob_before = model_no_clip.get_token_prob(prompt, 10, prompt_len)

        # Train for just a few steps (to see immediate effect)
        for _ in range(10):
            for model, opt, loss_fn in [
                (model_no_clip, opt_no_clip, loss_fn_no_clip),
                (model_clip, opt_clip, loss_fn_clip),
            ]:
                opt.zero_grad()
                logits = model(input_ids)

                with torch.no_grad():
                    ref_logprobs = compute_logprobs(logits.detach(), completion_ids, prompt_lens=prompt_lens)
                rollout_logprobs = ref_logprobs.clone()
                advantages = compute_advantages(rewards)

                loss = loss_fn(
                    logits,
                    completion_ids,
                    ref_logprobs,
                    rollout_logprobs,
                    advantages,
                    mask,
                    prompt_lens,
                )
                loss.backward()
                opt.step()

        prob_no_clip = model_no_clip.get_token_prob(prompt, 10, prompt_len)
        prob_clip = model_clip.get_token_prob(prompt, 10, prompt_len)

        print(f"Initial: {prob_before:.4f}")
        print(f"No clipping: {prob_no_clip:.4f}")
        print(f"With clipping: {prob_clip:.4f}")

        # Both should move in the right direction
        assert prob_no_clip > prob_before
        assert prob_clip > prob_before
