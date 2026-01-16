"""Integration-level checks for train_grpo schedule handling."""

import pytest

from entrypoints.train_grpo import _compute_schedule


@pytest.mark.integration
def test_errors_when_sync_gap_exceeds_allowed_staleness():
    training_cfg = {
        "prompts_per_rollout_sync": 20,
        "prompts_per_reference_sync": 10,
        "prompts_per_optim_step": 5,
        "prompts_per_forward_backward": 1,
        "max_staleness": 0,  # too small for sync_model_every=4
    }

    with pytest.raises(ValueError):
        _compute_schedule(training_cfg)


@pytest.mark.integration
def test_schedule_valid_when_staleness_covers_sync_gap():
    training_cfg = {
        "prompts_per_rollout_sync": 20,
        "prompts_per_reference_sync": 10,
        "prompts_per_optim_step": 5,
        "prompts_per_forward_backward": 1,
        "max_staleness": 3,
    }

    schedule = _compute_schedule(training_cfg)

    assert schedule["sync_model_every"] == 4
    assert schedule["max_staleness"] == 3
