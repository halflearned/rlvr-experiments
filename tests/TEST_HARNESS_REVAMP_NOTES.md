# Test Harness Revamp Notes

**Date:** January 2026
**Task:** Revamp test harness to approximate and test train_grpo.py entrypoint

## Summary

I've performed a comprehensive review of the codebase and revamped the test harness to better approximate the train_grpo.py entrypoint behavior. The focus was on identifying and testing for "silent failures" - situations where code runs without raising exceptions but the model doesn't learn as expected.

## New Test Files Created

### 1. `tests/unit/test_make_batch.py` (16 tests)
Tests for the critical batch-making logic in `algorithms/grpo.py`:
- **Bucketing tests**: Verify seq_len and completion_len are properly bucketed
- **Clamping tests**: Verify completion_len is clamped when seq_len hits max bucket
- **Mixed prompt lengths**: Verify batches with different prompt lengths work correctly
- **Padding mask**: Verify mask zeros out padding positions correctly
- **Edge cases**: Single token completions, empty buckets, identical completions

### 2. `tests/unit/test_compute_logprobs.py` (14 tests)
Tests for the critical logprob computation in `ops.py`:
- **Basic functionality**: Verify logprobs match manual computation
- **Prompt lengths**: Verify different prompt_lens in batch use correct logit positions
- **Negative test**: Verify wrong prompt_lens gives wrong logprobs (catches silent failures)
- **Padding handling**: Verify extreme logits don't cause NaN
- **Integration**: Verify typical GRPO batch shapes work

### 3. `tests/integration/test_grpo_learning.py` (8 tests)
Tests that verify the model actually learns:
- **Learning direction**: High-reward completions probability should increase
- **Advantage signs**: Positive advantages increase prob, negative decrease
- **Stability**: Loss and gradients should always be finite
- **Multi-prompt**: Grouped advantages isolate prompts correctly
- **Dr. GRPO**: Mean-centered (not std-normalized) advantages work
- **KL penalty**: Higher beta constrains policy shift
- **Clipping**: PPO-style clipping limits updates

### 4. `tests/nightly/test_e2e_smoke.py` (5 tests, require GPUs)
End-to-end tests that run with actual models:
- **Runtime startup**: Verify Runtime can be created and started
- **Single training step**: Verify a training step completes without error
- **Loss decreases**: Verify loss generally decreases over steps
- **Weight sync**: Verify training + sync changes vLLM output
- **Reward improvement**: Verify dummy dataset reward improves

## Fixed Stale Tests

### `tests/unit/test_rollout.py`
- Fixed `make_train_sample` helper to include new required args `item_id` and `trainer_version`
- Updated `test_pads_to_max_length` to account for bucketing (check >= instead of ==)
- Fixed direct `TrainSample` construction calls

### `tests/unit/test_vllm_handle.py`
- Updated `TestLoadAwareRouter` tests to use new `acquire_slot` return format `(replica_idx, slot_idx)`
- Updated `release_slot` calls to pass both arguments

### `tests/unit/test_verifiers.py`
- Updated duration assertions to check `>= 0` instead of `== 0.0` (actual timing is now returned)
- Updated timing span assertions to check validity rather than exact values

## Test Results Summary

**Non-Ray dependent unit tests:** 103 passed
**Integration tests (GRPO learning):** 8 passed
**Total new tests added:** 43

## Identified Silent Failure Modes

1. **Advantage computation with wrong group_size**: If `completions_per_prompt` doesn't match actual group size, advantages are computed incorrectly. Model trains but learns wrong gradients. **Covered by:** `test_grpo_learning.py::TestGRPOMultiPrompt`

2. **Wrong prompt_lens in compute_logprobs**: If prompt_lens don't match actual prompt lengths, logprobs are computed from wrong positions. **Covered by:** `test_compute_logprobs.py::TestComputeLogprobsPromptLens`

3. **Padding mask issues**: If mask computation fails for edge cases, loss is computed on padding tokens. **Covered by:** `test_make_batch.py::TestMakeBatchPaddingMask`

4. **Completion length clamping**: If seq_len is clamped to max bucket but completion_len isn't properly clamped, gather operations fail or compute wrong logprobs. **Covered by:** `test_make_batch.py::TestMakeBatchCompletionClamping`

5. **KL divergence with on-policy ref_logprobs**: If ref_logprobs always equal trainer_logprobs (on-policy), KL penalty has no effect. **Covered by:** `test_grpo_learning.py::TestKLPenalty` (uses fixed reference)

## Potential Issues Found in train_grpo.py

**No bugs found that would cause silent failures.** The entrypoint has good defensive checks:
- `_compute_schedule` validates staleness vs sync cadence
- Filter for zero-variance rewards before training
- Filter for sequences too long for trainer
- Staleness eviction in batches()
- Item tracking through data_iter

**Minor observation:** Line 401 uses `completions_per_prompt` from config for group_size in `compute_advantages`. If a sample has fewer completions due to filtering, this could cause issues. However, filtering happens before buffering, so all buffered samples should have the expected number of completions.

## Running the Tests

```bash
# Run all non-Ray unit tests
.venv/bin/python -m pytest tests/unit/test_make_batch.py tests/unit/test_compute_logprobs.py tests/unit/test_rollout.py tests/unit/test_vllm_handle.py tests/unit/test_verifiers.py tests/unit/test_chunked_loss.py tests/unit/test_multi_verifier.py -v

# Run GRPO learning integration tests
.venv/bin/python -m pytest tests/integration/test_grpo_learning.py -v

# Run Ray-dependent tests (requires Ray cluster)
.venv/bin/python -m pytest tests/unit/test_data.py tests/unit/test_buffer.py -v

# Run nightly E2E tests (requires GPUs and model)
.venv/bin/python -m pytest tests/nightly/test_e2e_smoke.py -v -m nightly
```

## Recommendations

1. **Continuous Integration**: Run `test_make_batch.py`, `test_compute_logprobs.py`, and `test_grpo_learning.py` on every PR - they're fast and catch critical issues.

2. **Nightly Tests**: Run `test_e2e_smoke.py` nightly with actual models to catch integration issues.

3. **Ray Tests**: The Ray-dependent tests in `test_data.py` and `test_buffer.py` error due to stale Ray session. These should be run in a fresh Ray environment.
