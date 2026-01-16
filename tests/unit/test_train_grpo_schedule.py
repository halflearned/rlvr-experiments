"""Tests for the training schedule helper in train_grpo."""

import pytest

from entrypoints.train_grpo import _compute_schedule


def test_raises_if_staleness_too_small_for_sync_gap():
    training_cfg = {
        "prompts_per_rollout_sync": 32,
        "prompts_per_reference_sync": 16,
        "prompts_per_optim_step": 8,
        "prompts_per_forward_backward": 2,
        "max_staleness": 0,
    }

    with pytest.raises(ValueError):
        _compute_schedule(training_cfg)


def test_accepts_configured_staleness_when_sufficient():
    training_cfg = {
        "prompts_per_rollout_sync": 8,
        "prompts_per_reference_sync": 8,
        "prompts_per_optim_step": 8,
        "prompts_per_forward_backward": 2,
        "max_staleness": 10,
    }

    schedule = _compute_schedule(training_cfg)

    assert schedule["sync_model_every"] == 1
    assert schedule["max_staleness"] == 10


def test_sync_intervals_never_zero():
    training_cfg = {
        "prompts_per_rollout_sync": 3,   # smaller than optim step to force floor
        "prompts_per_reference_sync": 0, # unset / zero should still become 1
        "prompts_per_optim_step": 8,
        "prompts_per_forward_backward": 4,
        "max_staleness": 1,
    }

    schedule = _compute_schedule(training_cfg)

    assert schedule["accumulation_steps"] == 2
    assert schedule["sync_model_every"] == 1
    assert schedule["sync_ref_every"] == 1
    # default staleness should be at least zero even if rollouts sync every step
    assert schedule["max_staleness"] >= 0


def test_invalid_batching_raises():
    training_cfg = {
        "prompts_per_rollout_sync": 8,
        "prompts_per_reference_sync": 8,
        "prompts_per_optim_step": 6,
        "prompts_per_forward_backward": 4,  # not divisible
    }

    with pytest.raises(ValueError):
        _compute_schedule(training_cfg)
