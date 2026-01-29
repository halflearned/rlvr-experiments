"""Nightly dummy-accuracy regression."""

import pytest
import torch
import torch.nn as nn

from rlvr_experiments.algorithms.grpo import RewardStats
from rlvr_experiments.losses import GRPOLoss, compute_advantages
from rlvr_experiments.ops import compute_logprobs


class TinyPolicy(nn.Module):
    """Minimal policy that predicts a single-token completion."""

    def __init__(self, vocab_size: int = 2):
        super().__init__()
        # Trainable logits for one timestep
        self.logits = nn.Parameter(torch.zeros(1, vocab_size))

    def forward(self, batch_size: int) -> torch.Tensor:
        # Repeat the single-step logits for the batch
        return self.logits.expand(batch_size, 1, -1)


@pytest.mark.nightly
def test_dummy_reward_improves_over_steps():
    """
    Smoke test that GRPO pushes probability mass toward high-reward completions
    on a trivial single-token task.
    """
    torch.manual_seed(0)
    model = TinyPolicy(vocab_size=2)
    opt = torch.optim.Adam(model.parameters(), lr=0.1)
    loss_fn = GRPOLoss(beta=0.0, eps=0.2)

    # Batch with two correct (token=1) and two incorrect (token=0) completions
    completion_ids = torch.tensor([[1], [1], [0], [0]], dtype=torch.long)
    rewards = torch.tensor([1.0, 1.0, 0.0, 0.0], dtype=torch.float32)
    padding_mask = torch.ones_like(completion_ids, dtype=torch.float32)
    prompt_lens = torch.ones(completion_ids.size(0), dtype=torch.long)  # completion starts at pos 0

    def policy_prob_of_correct():
        with torch.no_grad():
            logits = model.forward(batch_size=1)[0, 0]
            probs = torch.softmax(logits, dim=-1)
            return probs[1].item()

    prob_before = policy_prob_of_correct()

    # Train for a few steps; loss uses rollout/ref logprobs from current policy
    for _ in range(80):
        opt.zero_grad()
        logits = model.forward(batch_size=completion_ids.size(0))
        ref_logprobs, _ = compute_logprobs(logits.detach(), completion_ids, prompt_lens=prompt_lens)
        rollout_logprobs = ref_logprobs.clone()
        advantages = compute_advantages(rewards)
        loss = loss_fn(
            logits,
            completion_ids,
            ref_logprobs,
            rollout_logprobs,
            advantages,
            padding_mask,
            prompt_lens,
        )
        loss.backward()
        opt.step()

    prob_after = policy_prob_of_correct()

    assert prob_after > prob_before + 0.2  # moved meaningfully toward the good token
    assert prob_after > 0.8  # hits a high-accuracy regime on the dummy task


@pytest.mark.nightly
def test_compute_advantages_per_prompt_group():
    # Two prompts, two completions each -> group_size=2
    rewards = torch.tensor([0.0, 1.0, 0.2, 0.4], dtype=torch.float32)

    adv = compute_advantages(rewards, group_size=2)

    # Each prompt group should be zero-mean after normalization
    assert torch.allclose(adv[:2].mean(), torch.tensor(0.0), atol=1e-6)
    assert torch.allclose(adv[2:].mean(), torch.tensor(0.0), atol=1e-6)
    # Non-zero variance ensures gradients flow
    assert torch.any(adv != 0)


@pytest.mark.nightly
def test_reward_stats_tracks_filtered_and_used():
    stats = RewardStats()

    # First prompt: all completions correct -> filtered
    stats.record([1.0, 1.0, 1.0], used=False)
    # Second prompt: mixed -> used
    stats.record([1.0, 0.0, 0.0], used=True)

    metrics = stats.get_metrics()

    # used_reward should only reflect used prompt
    assert metrics["reward_used"] == pytest.approx(1.0 / 3.0, rel=1e-3)
    # overall averages include both prompts (4/6 = 0.666...)
    assert metrics["reward_overall"] == pytest.approx(2.0 / 3.0, rel=1e-3)
    assert metrics["frac_all_correct"] == pytest.approx(0.5)
    assert metrics["frac_all_wrong"] == pytest.approx(0.0)
