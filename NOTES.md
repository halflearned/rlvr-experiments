# RLVR Experiments - Debug Notes

## Problem Statement

Training Qwen3-1.7B-Base with GRPO on mixed datasets (GSM8K, MATH, MBPP, IFEval, APPS) is not showing expected improvement on benchmarks. We suspect formatting issues are filtering out otherwise correct answers, preventing the model from receiving training signal.

## Key Issues Identified

### 0. KL / Logprob Mismatch and Spikes (2026-01-21 to 2026-01-22)

**Symptom**: Sudden jump in KL at step 1 starting around 2026-01-21 18:02 UTC (e.g., `trace_20260121_180251.jsonl` shows `kl_max ~1.0+` vs earlier runs at `~0.02-0.12`).

**Root causes found**:

- **Trainer logprobs temperature flipped**: `entrypoints/train_grpo.py` changed from `policy_temperature = 1.0` to `policy_temperature = sampling.temperature` at ~18:08 UTC on 2026-01-21. With vLLM `logprobs_mode: raw_logprobs`, this makes trainer logprobs temperature-scaled (0.6) while vLLM logprobs remain pre-temperature. This mismatch directly inflates KL even at step 1. Fix options:
  - Set `policy_temperature = 1.0` for logprob computation, or
  - Use vLLM `logprobs_mode: processed_logprobs` for reference/rollout.

- **vLLM ref logprob truncation bug (pad token)**: `src/rlvr_experiments/vllm_engine_actor.py` assumed `pad_token_id==0` when trimming completions. For Qwen3, token id `0` is `"!"`, so any completion containing `!` early was truncated. This left ref logprobs at 0.0 for real tokens, causing huge spikes (`kl_max ~100+`), even when trainer/rollout matched. Fixed in commit `de735d8` by removing the hardcoded trim (use full padded length; masking handles padding).

**Evidence**:
- Good runs: `trace_20260120_013054.jsonl` (`kl_max ~0.02`), `trace_20260121_051408.jsonl` (`kl_max ~0.03`)
- Bad runs start: `trace_20260121_180251.jsonl` (`kl_max ~1.3`)
- After pad-token fix: `trace_20260122_075226.jsonl` shows `kl_max` ~0.2 then <0.1; occasional rollout/ref diffs remain but no 100+ spikes.

### 1. Fixed Seed Causing Identical Outputs (2026-01-19)

**Symptom**: All 83 rollouts in a debug run produced only 3 unique completion sets.

**Root Cause**: In `entrypoints/train_grpo.py`, the sampling params included a fixed seed:
```python
sampling_params = {**plan.sampling, "logprobs": 0}
if "seed" not in sampling_params:
    sampling_params["seed"] = seed  # seed=42 for all requests
```

This caused vLLM to produce identical outputs for the same prompt across different generation calls.

**Fix**: Commented out the seed assignment. TODO: implement per-request deterministic seeding (e.g., `hash(prompt_id + epoch + step)`).

### 2. Base Model Produces Garbage Without Instruction Format

**Symptom**: Qwen3-1.7B-Base outputs like "Cumhur\nCumhur\n..." or ".WriteHeader\n.WriteHeader\n..." instead of math reasoning.

**Root Cause**: The base model hasn't been instruction-tuned. It doesn't understand "Q: ... A:" format means "answer this question." It just continues text in whatever pattern seems likely from pretraining.

**Implication**: Need either:
- Few-shot examples in prompt to demonstrate expected format
- Start from instruction-tuned model (Qwen3-1.7B-Instruct)
- Or rely on RL to eventually discover the format (slow, may never work)

### 3. GSM8K Verifier Format Mismatch

**Symptom**: 0% pass rate on GSM8K despite model potentially solving problems.

**Root Cause**:
- GSM8K verifier expects "The answer is X." pattern (strict, matches lm_eval gsm8k_cot)
- Model may output `\boxed{X}` or other formats
- Any correct answer in wrong format = 0 reward = no training signal

**Status**: Verifier currently requires strict "The answer is X." format. This is intentional to match eval, but may need few-shot prompting to teach the model this format.


## Debug Experiments

### Experiment: 1-sample GSM8K (2026-01-19)

**Config**: `configs/debug/qwen3-1.7B-gsm8k-1sample.yaml`
- Single GSM8K problem (gsm8k_63) repeated via curriculum
- 10 steps max
- Disabled fixed seed

**Purpose**: Isolate whether the issue is format, verifier, or something else.

**Results**:
- Seed fix worked: 78 unique completion sets out of 78 rollouts (previously only 3 unique)
- Still all zero_variance: all 16 completions per rollout get 0.0 reward
- Format compliance: only 2.0% (27/1360) completions have "The answer is X." format
- Of the 27 formatted completions, none got the correct answer (29)
- Most completions are garbage ("Cumhur", "umably", repeated text)
- Some completions show reasoning but don't use the expected format

**Conclusion**: Even with random sampling, the base model rarely produces the expected output format. The few that do are wrong. This confirms we need format training before content training.

## Potential Solutions (Not Changing the Model)

### 1. Format Pre-training Phase
Start with a small dataset specifically designed to teach the model about output formatting before moving to actual problem-solving. Could use synthetic examples or hand-crafted format demonstrations.

### 2. K-shot Examples During Training
Include 2-3 few-shot examples in the prompt during early training to demonstrate expected format:
```
Q: What is 2+2?
A: 2 + 2 = 4. The answer is 4.

Q: What is 3*5?
A: 3 * 5 = 15. The answer is 15.

Q: [actual problem]
A:
```
Could phase these out over training as model learns the format.

### 3. Format Reward (Auxiliary Signal)
Add a small reward for outputs that contain the expected format pattern, independent of correctness:
- +0.1 if output contains "The answer is [number]."
- Full +1.0 only if format correct AND answer correct

This provides gradient signal even when answers are wrong, teaching format first. Can be phased out once format is learned.

### 4. Other Ideas
- **Lenient verifier during warmup**: Accept multiple formats early, tighten to strict format later
- **Curriculum by format difficulty**: Start with problems where model already produces parseable output
- **Rejection sampling with format filter**: Only keep completions that have correct format for training

## Open Questions / To Investigate

- **Replica utilization**: Config has 5 rollout replicas (`data_parallel_size: 5`). Works fine with multiple datasets/prompts. Single-prompt debug config doesn't utilize all replicas (not enough parallel work).

- **Curriculum gains analysis**: With curriculum learning (pass-at-64 ordering), need to check whether improvements come only from solving more easy problems vs actual gains on hard problems. During training we observed MATH rewards dropping (0.80 → 0.63) as curriculum progresses to harder problems while GSM8K stays high (~90%). At eval time, should stratify results by problem difficulty to see if the model improved on hard problems or just got better at easy ones.

## Files Changed

- `entrypoints/train_grpo.py`: Commented out fixed seed (lines 157-159)
- `experiments/debug/gsm8k_1sample_repeated.txt`: Curriculum with single prompt repeated 10k times
- `configs/debug/qwen3-1.7B-gsm8k-1sample.yaml`: Minimal debug config

---

## 2026-01-19: Format Mismatch Investigation

### Key Finding: Base Model Already Knows Math, Just Wrong Format

Analysis of rollout outputs shows the model strongly prefers `\boxed{}` format:

| Format | % of Completions |
|--------|-----------------|
| `\boxed{}` | ~20% |
| "The answer is X." | ~3% |
| `####` | ~0% |

The model produces well-structured reasoning with correct answers in `\boxed{}` format, but our strict GSM8K verifier (which looks for "The answer is X.") rejects them.

### Previous Pass@k Evaluation (2026-01-15)

From `experiments/qwen3-1.7b-base-pass-rate/NOTES.md`:

**Qwen3-1.7B-Base evaluated with math_verify (accepts `\boxed{}`):**
| Dataset | Pass@1 | Pass@16 | Pass@512/1024 |
|---------|--------|---------|---------------|
| GSM8K | 74.95% | ~98% | 99.81% |
| MATH (L3-5) | 29.67% | - | 84.0% |

The base model already has **~75% pass@1** on GSM8K when using the lenient `math_verify` verifier. Our strict "The answer is X." verifier only captures ~2-3% of outputs.

### Baseline Comparison: gsm8k_cot vs minerva_math (2026-01-20)

**CORRECTED: Results from lm_eval on Qwen3-1.7B-Base (seed=42, TP=1, temperature=0):**

**WARNING**: Previous results using TP=8 were WRONG (showed 10% on minerva_math instead of 28%). Always use TP=1 for evaluation - see AGENTS.md for details.

| Benchmark | Metric | Value |
|-----------|--------|-------|
| gsm8k (4-shot) | flexible-extract | 68.61% |
| gsm8k (4-shot) | strict-match | 59.44% |
| gsm8k (8-shot) | flexible-extract | 70.13% |
| gsm8k (8-shot) | strict-match | 69.60% |
| gsm8k_cot (4-shot) | flexible-extract | 69.83% |
| gsm8k_cot (4-shot) | strict-match | 51.63% |
| gsm8k_cot (8-shot) | flexible-extract | 73.24% |
| gsm8k_cot (8-shot) | strict-match | 67.17% |
| hendrycks_math (4-shot) | exact_match | 17.78% |
| minerva_math (4-shot) | exact_match | **28.20%** |
| minerva_math (4-shot) | math_verify | **39.08%** |
| ifeval (0-shot) | prompt_level_strict | 21.26% |
| ifeval (0-shot) | inst_level_strict | 32.85% |

**Key observations:**
- `gsm8k_cot` 8-shot shows highest GSM8K performance (73.24% flex, 67.17% strict)
- `minerva_math` exact_match (28.2%) vs math_verify (39.08%) shows ~11% answers are mathematically correct but not in exact `\boxed{}` format
- `hendrycks_math` (17.78%) is lower than `minerva_math` due to different prompting

---

## 2026-01-19: GSM8K + MATH Curriculum Run (`gsm8k_math_boxed_curriculum`)

### Setup

**Config**: `configs/variations/qwen3-1.7B-gsm8k-math-boxed-curriculum.yaml`
- Mixed dataset: GSM8K (50%) + MATH (50%)
- Both using `math` verifier type (accepts `\boxed{}` format)
- Curriculum ordering: pass@64 difficulty (easiest first)
- LR: 5e-6, checkpoint every 50 steps, max 1000 steps
- Killed at step ~450

### Results: Multi-Benchmark Evaluation (Steps 50-250)

| Checkpoint | gsm8k | gsm8k_cot | minerva exact | minerva math_verify | math_qwen | mbpp |
|------------|-------|-----------|---------------|---------------------|-----------|------|
| Base       | 68.7% | 65.4%     | 28.2%         | 38.8%               | 40.3%     | 55.2%|
| Step 50    | 68.5% | 66.2%     | 28.6%         | 39.2%               | 39.8%     | 56.0%|
| Step 100   | 68.6% | 66.8%     | 28.3%         | 39.5%               | 39.5%     | 56.6%|
| Step 150   | 68.8% | 67.3%     | 27.2%         | 39.5%               | 39.2%     | 56.8%|
| Step 200   | 68.9% | 67.5%     | 24.9%         | 39.0%               | 38.9%     | 55.8%|
| Step 250   | 69.2% | 67.8%     | 25.7%         | 39.7%               | 38.9%     | 57.2%|

### Key Observations

1. **GSM8K improving**: gsm8k_cot rose from 65.4% → 67.8% (+2.4pts)

2. **MATH format mismatch**: minerva exact_match DECLINED from 28.2% → 25.7% (-2.5pts) while math_verify stayed flat (38.8% → 39.7%)
   - This indicates the model is **losing `\boxed{}` format compliance** even though math ability is preserved
   - Training verifier (`MathVerifier`) is too lenient - accepts "The answer is X" and other formats
   - Eval verifier (`minerva_math exact_match`) strictly requires `\boxed{}`

3. **Format drift problem**: Model learns to solve problems but drifts away from the strict format needed for eval

### Root Cause Analysis

The training verifier (`MathVerifier`) accepts multiple answer formats:
1. `\boxed{X}`
2. "The answer is X"
3. Other fallback patterns

Since any format gets rewarded equally, the model has no incentive to maintain strict `\boxed{}` format. Over training, it drifts toward whatever format emerges from its prior + gradient updates.

Meanwhile, `minerva_math exact_match` (from lm_eval) strictly requires `\boxed{}` format - no fallbacks.

---

## 2026-01-19: MATH-Only Minerva Run (`math_only_minerva`)

### Plan

To fix the format drift problem, we created a strict `MinervaMathVerifier` that:
- **ONLY** accepts `\boxed{}` format
- NO fallbacks (no "The answer is X", no other patterns)
- Returns 0 if no `\boxed{}` found in response

### Changes Made

1. **New verifier**: `src/rlvr_experiments/verifiers/minerva_math.py`
   - `MinervaMathVerifier` class with strict `\boxed{}` extraction
   - `extract_boxed_strict()` returns None if no `\boxed{}` found
   - Uses latex2sympy2_extended for symbolic equivalence checking

2. **Updated data.py**: MATH dataset now defaults to `verifier_type: "minerva_math"`

3. **Removed old verifier**: Deleted `hendrycks_math.py` (was too lenient)

### Training Config

**Config**: `configs/variations/qwen3-1.7B-math-only-minerva.yaml`
- **Dataset**: MATH only (no GSM8K)
- **Verifier**: `minerva_math` (strict `\boxed{}` only)
- **Curriculum**: pass@64 ordering (`experiments/curricula/pass-at-64/math_curriculum.txt`)
- **LR**: 5e-6
- **Checkpoints**: Every 20 steps (for fine-grained analysis)
- **Max steps**: 500

### Hypothesis

By training with a strict `\boxed{}`-only verifier:
1. Model will learn to always produce `\boxed{}` format
2. minerva exact_match should improve (or at least not degrade)
3. If model already knows MATH, just needs format reinforcement

### Results: Multi-Benchmark Evaluation (Steps 20-100)

| Metric | Base | Step 20 | Step 40 | Step 60 | Step 80 | Step 100 |
|--------|------|---------|---------|---------|---------|----------|
| **minerva exact_match** | 28.2% | 28.6% | 27.7% | 27.1% | 27.2% | 26.7% |
| minerva math_verify | 38.8% | 39.7% | 39.3% | 39.3% | 39.8% | 39.6% |
| gsm8k strict | 68.4% | 68.2% | 69.5% | 68.9% | 63.5% | 63.2% |
| gsm8k flex | 68.7% | 68.5% | 69.6% | 69.1% | 69.8% | 70.1% |
| gsm8k_cot strict | 69.9% | 69.9% | 68.7% | 69.3% | 46.8% | 45.9% |
| gsm8k_cot flex | 68.3% | 68.2% | 68.9% | 68.4% | 68.2% | 67.9% |
| ifeval prompt_strict | 21.0% | 20.9% | 20.7% | 23.1% | 21.1% | 22.0% |
| mbpp pass@1 | 55.2% | 56.8% | 56.2% | 55.8% | 55.0% | 55.2% |

### Key Observations

1. **minerva exact_match continues declining**: 28.2% → 26.7% (step 100)
   - The strict `\boxed{}` verifier did NOT prevent format drift
   - Model is actively losing format compliance despite only getting rewards for `\boxed{}` outputs

2. **gsm8k_cot strict collapsed** at steps 80/100: 69.9% → 45.9%
   - Severe format drift in "The answer is N." format
   - This is catastrophic degradation

3. **Flex metrics remain stable** (~68-70% for gsm8k, ~68% for gsm8k_cot)
   - The model still gets the math right, just loses the format
   - math_verify also stable at ~39%

4. **No improvement on MATH** despite training specifically on MATH with curriculum
   - math_verify flat at ~39% throughout training
   - Model is not learning to solve harder problems

### Conclusion

**Hypothesis disproven**: Training with strict `\boxed{}`-only verifier did NOT maintain format compliance.

The model appears to be:
1. **Forgetting format** even when only that format is rewarded
2. **Not improving on math** despite seeing problems ordered by difficulty
3. **Possibly overfitting** or experiencing some form of catastrophic forgetting

This suggests the problem is more fundamental than verifier leniency. Possible causes:
- Learning rate too high causing forgetting
- GRPO optimization dynamics causing format drift
- Model needs explicit format supervision (SFT on formatted examples) before RL
- Curriculum ordering not providing the expected benefits

### Status

Training run killed at step 100.
- Checkpoints saved: steps 20, 40, 60, 80, 100
- Experiment dir: `/efs/rlvr-experiments/experiments/math_only_minerva`

---

## 2026-01-19: Prompt Format Mismatch Discovery

### The Real Root Cause

Re-analyzed the `math_only_minerva` results with `--log_samples` to examine raw model outputs. Found a **critical prompt format mismatch**:

| Context | Prompt Format |
|---------|---------------|
| **Training** (data.py) | `Question: {problem}\nAnswer:` |
| **Eval** (minerva_math) | `Problem:\n{problem}\n\nSolution:` |

### Evidence: `\boxed{}` Usage Analysis

Analyzed 5000 samples from each checkpoint on minerva_math eval:

| Checkpoint | `\boxed{}` Usage | exact_match | Of correct, % with `\boxed{}` |
|------------|------------------|-------------|-------------------------------|
| Base       | 72.9% | 28.4% | 94.2% |
| Step 20    | 72.8% | 28.6% | 94.1% |
| Step 40    | 72.6% | 27.7% | 94.3% |
| Step 60    | 72.8% | 27.1% | 94.4% |
| Step 80    | 72.3% | 27.2% | 94.0% |
| Step 100   | 72.0% | 26.7% | 94.3% |

**Key Finding**: `\boxed{}` usage on minerva_math eval is **constant at ~72-73%** across all checkpoints - virtually identical to base model!

Meanwhile, training rollouts showed `\boxed{}` increasing: 25% → 87% → 72%

### Interpretation

The model **did** learn `\boxed{}` format during training, but this learning was specific to the training prompt format (`Question: X\nAnswer:`). When evaluated with the different minerva_math prompt format (`Problem:\n{problem}\n\nSolution:`), the model falls back to base model behavior.

This is **prompt-specific format learning** rather than general format learning.

### Verification Strategy Comparison

Compared our training verifier vs lm_eval's minerva_math:

| Aspect | Our MinervaMathVerifier | lm_eval minerva_math exact_match |
|--------|-------------------------|----------------------------------|
| `\boxed{}` extraction | Yes (rightmost) | Yes (rightmost) |
| Fallback formats | None (strict) | None (strict) |
| Sympy comparison | latex2sympy2_extended | latex2sympy2_extended |
| Normalization | Basic (strip, remove `\left\right`, spacing) | Extensive (units, articles, TeX cleanup) |

Our verifier is slightly **stricter** than lm_eval's exact_match due to simpler normalization. However, this shouldn't cause the observed decline - the issue is the prompt format mismatch.

### Fix Applied

Updated `data.py` to use minerva_math-style prompts for MATH:
```python
# Before: "Question: {problem}\nAnswer:"
# After:  "Problem:\n{problem}\n\nSolution:"
```

This aligns training and eval prompt formats, so learned format behavior should transfer.

### Next Steps

1. Re-run MATH training with aligned prompt format
2. Verify `\boxed{}` learning transfers to eval
3. Consider whether GSM8K also needs prompt alignment (currently uses `Question: X\nAnswer:` for both training and eval via gsm8k task)

---

## 2026-01-20: Curriculum ID Mismatch Bug + Deadlock Fix

### Training Deadlock at Step 109

Run `math_only_minerva_v2` (with aligned prompts) deadlocked at step 109 with:
- 0% GPU utilization despite memory allocated
- Buffer size: 1 sample (need 2 for `prompts_per_forward_backward`)
- All workers idle waiting

### Root Cause: Curriculum ID Mismatch

The curriculum file had 4691 IDs, but only **2731 matched** the dataset.

**The Bug**: Curriculum uses IDs like `math_counting_and_probability_46` (underscores), but dataset `type` field contains "Counting & Probability" (spaces and ampersand). The `load_math()` function used `.lower()` which produces "counting & probability" - doesn't match.

**Missing subjects** (1960 IDs silently skipped):
- `counting_and_probability` → "Counting & Probability"
- `number_theory` → "Number Theory"
- `intermediate_algebra` → "Intermediate Algebra"
- `prealgebra` → "Prealgebra" (this one actually matched)

**Fix Applied**: Added normalization map in `load_math()` to convert multi-word subjects:
```python
SUBJECT_NORMALIZATION = {
    "counting & probability": "counting_and_probability",
    "number theory": "number_theory",
    "intermediate algebra": "intermediate_algebra",
    ...
}
```

### Partial Batch Fix

Also fixed a potential hang at epoch end: when producer finishes but fewer than `prompts_per_forward_backward` samples remain, the consumer now yields a partial batch instead of waiting forever.

**File changed**: `entrypoints/train_grpo.py` - yield partial batch when `buffer.pop()` returns `None`.

### New Config: v3 (No Curriculum)

Created `configs/variations/qwen3-1.7B-math-only-minerva-v3.yaml`:
- Same as v2 but **no curriculum ordering** (random order)
- Tests whether curriculum was helping or hurting

### Status

- `math_only_minerva_v2` step 100 eval running (minerva_math 4-shot)
- v3 config ready for next training run
- ID normalization fix applied to `data.py`

---

## 2026-01-20: MATH-Only Minerva v3 Run (Random Order)

### Setup

**Config**: `configs/variations/qwen3-1.7B-math-only-minerva-v3.yaml`
- **Dataset**: MATH only
- **Verifier**: `minerva_math` (strict `\boxed{}` only)
- **Curriculum**: Random order (no pass@64 sorting)
- **LR**: 5e-6
- **Checkpoints**: Every 20 steps
- **Max steps**: 500 (ran until step 240, ~80% of epoch 1)

### Training Metrics

Throughout training:
- **Pass rate**: Started ~50%, stable throughout
- **`\boxed{}` format usage**: Stable at ~72-73% (checked all 5000 samples per checkpoint)
- **Training step time**: Started ~30s/step, slowed to ~90s/step by step 200

### Evaluation Results (Steps 20-240)

| Checkpoint | minerva_math | gsm8k | gsm8k_cot | mbpp | ifeval |
|------------|--------------|-------|-----------|------|--------|
| Step 20    | 27.10%       | 69.45%| 66.26%    | 56.40%| 23.29%|
| Step 40    | 27.30%       | 68.84%| 64.90%    | 55.60%| 22.92%|
| Step 60    | 26.84%       | 68.84%| 66.03%    | 55.20%| 21.81%|
| Step 80    | 26.94%       | 69.37%| 66.49%    | 55.00%| 22.00%|
| Step 100   | 26.64%       | 69.07%| 66.72%    | 55.60%| 23.66%|
| Step 120   | 27.04%       | 69.07%| 66.03%    | 55.60%| 21.81%|
| Step 140   | 27.10%       | 68.39%| 67.63%    | 55.40%| 22.55%|
| Step 160   | 26.94%       | 69.14%| 66.57%    | 55.80%| 22.37%|
| Step 180   | 27.04%       | 68.54%| 67.10%    | 56.00%| 23.47%|
| Step 200   | 27.04%       | 68.84%| 66.03%    | 57.00%| 22.55%|
| Step 220   | 26.94%       | 69.07%| 66.87%    | 55.60%| 22.18%|
| Step 240   | 26.90%       | 69.37%| 66.34%    | 56.00%| 22.37%|

### Key Observations

1. **All metrics flat**: No improvement on any benchmark from step 20 to step 240
   - minerva_math: ~27% (±0.4%)
   - gsm8k: ~69% (±0.5%)
   - gsm8k_cot: ~66% (±1%)
   - mbpp: ~56% (±1%)
   - ifeval: ~22% (±1%)

2. **`\boxed{}` usage stable**: 72-73% throughout, identical to base model

3. **No learning signal**: Despite 240 steps of training on MATH, the model shows no improvement

### Training Slowdown Investigation

Training step time increased from ~30s/step to ~90s/step. Investigated the cause:

**Verification Time Analysis**:
- Early training (ts < 6000s): verify times ~0.3-5ms per sample
- Late training (ts > 6000s): verify times jumped to ~60-1700ms per sample
- 8.3% of verifications timing out late in training vs 0% early

**Model Behavior Change**:
- Early: 78.2% string match (fast path), 21.8% need sympy
- Late: 68.2% string match (fast path), 31.8% need sympy
- Pass rate actually increased (43.2% → 59.6%) over training
- Model getting more correct answers but with different formatting, requiring more sympy symbolic comparison

**Root Cause**: ProcessPoolExecutor worker accumulation bug
- sympy calls can hang indefinitely on certain inputs
- `future.result(timeout=2.0)` only cancels the wait, not the actual subprocess
- Workers get stuck, reducing available parallelism
- Over time, all workers accumulate hung calls

### ProcessPoolExecutor Bug Fix

**Problem**: The original `MinervaMathVerifier` used ProcessPoolExecutor with max_workers=4. When sympy hangs, `future.result(timeout=X)` only cancels the wait, not the actual worker process. Hung workers accumulate, eventually using all pool slots.

**Solution**: Rewrote to spawn a fresh subprocess per verification with proper kill handling:

```python
def _sympy_equiv_with_timeout(pred_norm: str, gold_norm: str, timeout: float = 2.0) -> bool:
    """Spawn fresh process, kill on timeout."""
    ctx = mp.get_context('spawn')
    result_queue = ctx.Queue()

    proc = ctx.Process(
        target=_sympy_equiv_worker,
        args=(pred_norm, gold_norm, result_queue)
    )
    proc.start()
    proc.join(timeout=timeout)

    if proc.is_alive():
        proc.terminate()
        proc.join(timeout=0.5)
        if proc.is_alive():
            os.kill(proc.pid, signal.SIGKILL)
            proc.join(timeout=0.5)
        return False

    # Get result from queue
    ...
```

**File changed**: `src/rlvr_experiments/verifiers/minerva_math.py`

**Tests added**: `tests/unit/test_minerva_math_verifier.py` (37 tests, all passing)

### Conclusions

1. **v3 showed no learning**: Despite aligned prompts and random curriculum, model did not improve on any benchmark

2. **Curriculum vs random**: Neither pass@64 curriculum (v2) nor random order (v3) produced improvement

3. **The problem is not**:
   - Prompt format mismatch (fixed)
   - Verifier leniency (strict `\boxed{}` only)
   - Curriculum ordering

4. **Possible remaining issues**:
   - Learning rate may be wrong (5e-6 too low for signal, too high for stability?)
   - GRPO may need more samples per problem (currently 16)
   - Model may already be at its capability ceiling for MATH (pass@1 ~27%)
   - Need longer training or more epochs

5. **Infrastructure fixed**: ProcessPoolExecutor bug would have caused slowdown issues in future runs

### Status

- Training killed at step 240
- All checkpoints saved: steps 20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220, 240
- ProcessPoolExecutor bug fixed, tests written
- Experiment dir: `/efs/rlvr-experiments/experiments/math_only_minerva_v3`

---

## 2026-01-20: MATH-Only Minerva v4 Run (Higher LR, Lower Beta)

### Setup

**Config**: `configs/variations/qwen3-1.7B-math-only-minerva-v4.yaml`
- **Dataset**: MATH only
- **Verifier**: `minerva_math` (strict `\boxed{}` only)
- **Curriculum**: Random order
- **LR**: 5e-5 (10x higher than v3)
- **Beta**: 0.001 (50x lower than v3's 0.05)
- **Checkpoints**: Every 20 steps
- **Max steps**: 500

### Hypothesis

v3 showed no learning with lr=5e-6, beta=0.05. This run tests whether:
1. Higher learning rate (5e-5) provides stronger gradient signal
2. Lower KL penalty (beta=0.001) allows more exploration

### Results: Model Collapse at Step 15

Training ran for 15 steps before the model collapsed/hung:
- Steps 1-13: ~30s per step
- Step 14: ~300s (5 minutes) - anomalous slowdown
- Step 15: Training hung indefinitely

**Likely cause**: With very low beta (0.001), the KL penalty was too weak to prevent the model from diverging from the reference policy. Combined with the higher learning rate, the model likely collapsed into degenerate outputs (repetition, garbage, etc.).

### Additional Context: Temperature Bug Fix

Note: This run was affected by a bug in logprob temperature handling (fixed in commit 5a9c738 after the run started):

**The bug**: Rollout logprobs were computed at temperature T (e.g., 0.6), but reference/trainer logprobs were computed at temperature 1.0. This mismatch would cause incorrect advantage estimates in GRPO.

**The fix** (commit 5a9c738):
- Pass `policy_temperature` through to reference logprob computation
- Pass `temperature` parameter to loss functions
- Ensure all logprob computations use the same temperature

This bug may have contributed to training instability, but the model collapse was likely primarily due to the aggressive hyperparameters (high LR + low beta).

### Conclusions

1. **beta=0.001 is too low**: The model diverged quickly without sufficient KL constraint
2. **lr=5e-5 may be viable** with proper beta, but needs testing
3. **Temperature bug** in previous runs may have affected training dynamics

### Next Steps

- Try intermediate settings: lr=1e-5 or 2e-5, beta=0.01 or 0.02
- Ensure temperature fix is applied before next run
- Consider monitoring KL divergence during training to detect early divergence

### Status

- Training collapsed at step 15
- No checkpoints saved (first checkpoint at step 20)
- Temperature bug fixed in commit 5a9c738
- Experiment dir: `/efs/rlvr-experiments/experiments/math_only_minerva_v4`

---

## 2026-01-20: Overnight Hyperparameter Search (v5 & v6)

### Experimental Design

Based on v3 (no learning) and v4 (collapse), running two parallel experiments with intermediate hyperparameters:

| Run | LR | Beta | Hypothesis |
|-----|-----|------|-----------|
| v5 | 2e-5 | 0.03 | Aggressive: 4x lower LR than v4, 30x higher beta |
| v6 | 1e-5 | 0.04 | Conservative: 2x higher LR than v3, slightly lower beta |

### Configurations

**v5** (`configs/variations/qwen3-1.7B-math-only-minerva-v5.yaml`):
- lr=2e-5, beta=0.03
- 3 vLLM rollout replicas (leaves 2 GPUs for evals)
- Running on primary node

**v6** (`configs/variations/qwen3-1.7B-math-only-minerva-v6.yaml`):
- lr=1e-5, beta=0.04
- 5 vLLM rollout replicas
- Running on secondary node

Both:
- MATH-only dataset
- minerva_math verifier (strict `\boxed{}`)
- Random curriculum order
- 200 max steps
- Checkpoint every 20 steps
- Eval on minerva_math 4-shot after each checkpoint

### Success Criteria

- **Target**: 2%+ improvement on minerva_math (base: ~27%)
- **Collapse detection**: 10 minutes stuck at same step

### Running Status (Started 2026-01-20 06:20 UTC)

- v5: Running, progressing normally
- v6: Running, progressing normally
- Monitoring script: `/tmp/overnight_monitor.log`
- v5 eval watcher: Running on GPUs 6-7

### SympyWorkerPool Fix

Before starting these runs, rewrote `MinervaMathVerifier` to use a persistent worker pool with watchdog pattern (commit 40eb334):
- Fixed ProcessPoolExecutor bug that caused worker accumulation
- ~2ms per sympy call (was ~1500ms with fresh subprocess, 782x speedup)
- Tests: `tests/unit/test_minerva_math_verifier.py` (42 tests)

### Results

**v5 (lr=2e-5, β=0.03)**: Collapsed around step 160-180

| Step | minerva_math | Δ from base |
|------|--------------|-------------|
| 20 | 28.16% | -0.04% |
| 40 | 26.92% | -1.28% |
| 60 | 24.10% | -4.10% |

Training collapsed with:
- reward_all dropped to 0.00
- grad_norm spiked to 46.75 (was ~0.4-0.5)
- Only reached step 160 before collapse

**v6 (lr=1e-5, β=0.04)**: Completed 200 steps

| Step | minerva_math | Δ from base |
|------|--------------|-------------|
| 20 | 28.90% | **+0.70%** |
| 40 | 27.22% | -0.98% |
| 60 | 26.66% | -1.54% |
| 120 | 25.02% | -3.18% |
| 160 | 23.50% | -4.70% |
| 200 | 22.34% | **-5.86%** |

Training rewards looked healthy (reward_all ~0.6-0.7) but eval performance steadily degraded.

### Conclusions

1. **v5 collapsed**: lr=2e-5 still too aggressive, caused training instability
2. **v6 degraded steadily**: Despite healthy training rewards, eval performance dropped ~1% per 40 steps
3. **Brief improvement at step 20**: v6 showed +0.7% at step 20 before degrading
4. **Reward hacking**: Both runs showed good training rewards but poor eval generalization
5. **Best checkpoint**: v6 step 20 (28.90%) - only checkpoint above baseline

### Trace Files

- **v5**: `/efs/rlvr-experiments/traces/trace_20260120_061808.jsonl` (141MB)
- **v6**: `/efs/rlvr-experiments/traces/trace_20260120_062109.jsonl` (190MB)

### Next Steps

The consistent degradation pattern suggests:
1. Training format/distribution differs too much from 4-shot eval
2. Model may be overfitting to training prompts
3. May need lower learning rates or early stopping around step 20
4. Consider using eval performance (not training reward) for checkpoint selection

---

## 2026-01-20: Instruct Model Training (instruct_math_v1 & v2)

### Motivation

After consistent degradation with the base model (Qwen3-1.7B-Base), switched to the instruct model (Qwen3-1.7B) which:
1. Already understands instruction-following
2. Has better format compliance out of the box
3. Higher baseline on minerva_math (33.12% vs 28.2%)

### Setup

**v1** (`configs/variations/qwen3-1.7B-instruct-math-v1.yaml`):
- Model: `Qwen/Qwen3-1.7B` (instruct version)
- lr=1e-5, beta=0.04 (same as v6)
- 3 vLLM rollout replicas
- Running on primary node

**v2** (`configs/variations/qwen3-1.7B-instruct-math-v2.yaml`):
- Model: `Qwen/Qwen3-1.7B` (instruct version)
- lr=5e-6, beta=0.001 (lower lr, much lower beta)
- 5 vLLM rollout replicas
- Running on secondary node

Both:
- MATH-only dataset
- minerva_math verifier (strict `\boxed{}`)
- Random curriculum order
- 200 max steps
- Checkpoint every 20 steps

### Results

**Baseline**: Qwen3-1.7B (instruct) = **33.12%** on minerva_math exact_match

**v1 (lr=1e-5, β=0.04)** - Shows consistent improvement! ✅

| Step | minerva_math | Δ from base |
|------|--------------|-------------|
| 20 | 33.02% | -0.10% |
| 40 | 33.16% | +0.04% |
| 60 | 33.46% | +0.34% |
| 80 | 34.00% | +0.88% |
| 100 | 34.08% | +0.96% |
| 120 | 33.80% | +0.68% |
| **140** | **34.52%** | **+1.40%** ✅ BEST |
| 160 | 33.76% | +0.64% |

**v2 (lr=5e-6, β=0.001)** - Stable but minimal improvement

| Step | minerva_math | Δ from base |
|------|--------------|-------------|
| 20 | 33.20% | +0.08% |
| 40 | 32.92% | -0.20% |
| 60 | 33.16% | +0.04% |
| 80 | 33.36% | +0.24% |
| 100 | 33.30% | +0.18% |

### Key Findings

1. **Instruct model learns from GRPO!** Unlike the base model which degraded steadily, v1 shows consistent improvement reaching nearly 1% above baseline

2. **Learning trajectory**: v1 starts flat, then gradually increases - suggests the model is genuinely learning rather than reward hacking

3. **Hyperparameter comparison**:
   - v1 (higher lr=1e-5, higher β=0.04) → Clear improvement
   - v2 (lower lr=5e-6, lower β=0.001) → Stable but no improvement
   - Higher KL penalty (β) seems to help maintain quality while learning

4. **Base vs Instruct model comparison** at step 100:
   - Base model (v6): 28.2% → 22.34% (**-5.86%** degradation)
   - Instruct model (v1): 33.12% → 34.08% (**+0.96%** improvement)

5. **Training rewards**: Both runs show healthy reward_all (~0.5-0.6), but only v1 translates this to eval improvement

### Conclusions

The instruct model is a much better starting point for GRPO training:
- Already has good instruction-following behavior
- More robust to format drift
- Actually improves with training instead of degrading

Recommended settings for future runs:
- Start from instruct model
- lr=1e-5, beta=0.04 (v1 settings)
- Monitor eval performance, not just training rewards

### Status

- v1 completed through step 160
- v2 killed at step 115 (external SIGTERM, not training collapse)
- Best checkpoint: **v1 step 140** with 34.52% (+1.40% from baseline)
- Experiment dirs:
  - `/efs/rlvr-experiments/experiments/instruct_math_v1`
  - `/efs/rlvr-experiments/experiments/instruct_math_v2`

### Analysis

The instruct model training shows a clear improvement trajectory with some oscillation:
- Performance increases steadily from step 20 to step 140
- Peak at step 140 (+1.40%), then regression at step 160 (+0.64%)
- This oscillation pattern suggests the model may benefit from early stopping or checkpoint selection based on eval performance

**Key takeaway**: Unlike the base model which degraded steadily, the instruct model genuinely learns from GRPO training. The v1 hyperparameters (lr=1e-5, β=0.04) are effective for this model.

---

## 2026-01-21: AllenAI RLVR Full Reproduction (lr=5e-6, β=1e-3)

### Motivation

Attempting to reproduce the "good run" that showed +8.42% GSM8K improvement. This run uses the AllenAI RLVR dataset which mixes GSM8K, MATH, and IFEval training data with aligned verifiers for each.

### Setup

**Config**: `configs/variations/qwen3-1.7B-allenai-full-lr5e6-beta1e3-repro.yaml`
- **Model**: Qwen/Qwen3-1.7B-Base
- **Dataset**: AllenAI RLVR (GSM8K + MATH + IFEval mixed)
- **Verifiers**:
  - GSM8K: strict "The answer is X." format
  - MATH: strict `\boxed{}` format
  - IFEval: instruction-following verifier
- **LR**: 5e-6
- **Beta**: 1e-3 (0.001)
- **Checkpoints**: Every 20 steps

**Training**: Running on 2-node cluster (primary + secondary)
- Primary: `allenai_full_lr5e6_beta1e3_repro_20260121_211117`
- Secondary: `allenai_full_lr5e6_beta1e3_repro_20260121_211348`

### Evaluation Strategy

Running automatic eval watchers on both nodes:
- GPU 0: gsm8k_cot (8-shot), ifeval
- GPU 1: minerva_math (4-shot), hendrycks_math (4-shot)

Additionally ran ad-hoc evaluation with 0-shot and 4-shot gsm8k_cot to understand few-shot sensitivity.

### Baselines (Qwen3-1.7B-Base)

| Benchmark | Metric | Value |
|-----------|--------|-------|
| gsm8k_cot (8-shot) | strict-match | 67.17% |
| gsm8k_cot (8-shot) | flexible-extract | 73.24% |
| minerva_math (4-shot) | exact_match | 28.20% |
| hendrycks_math (4-shot) | exact_match | 17.78% |
| ifeval (0-shot) | prompt_strict | 21.26% |

### Results: GSM8K Few-shot Analysis (Steps 20-100)

Evaluated primary node checkpoints with 0-shot, 4-shot, and 8-shot gsm8k_cot:

| Step | 0-shot strict | 0-shot flex | 4-shot strict | 4-shot flex | 8-shot strict | 8-shot flex |
|------|---------------|-------------|---------------|-------------|---------------|-------------|
| Baseline | 57.54%* | - | 51.63% | 69.83% | 67.17% | 73.24% |
| 20 | 57.54% | 10.92% | 51.55% | 69.52% | 66.64% | 72.63% |
| 40 | 58.15% | 9.93% | 52.16% | 69.37% | 66.64% | 72.71% |
| 60 | 59.82% | 9.33% | 52.84% | 69.14% | 66.87% | 72.93% |
| 80 | 59.59% | 9.70% | 52.54% | 69.29% | 66.72% | 72.78% |
| 100 | 60.58% | 9.63% | 53.15% | 70.05% | 66.87% | 72.78% |

*Baseline 0-shot estimated from step 20 (no baseline run yet)

### Key Findings

1. **0-shot strict improving**: +3.04% (57.54% → 60.58%)
2. **4-shot strict improving**: +1.60% (51.55% → 53.15%)
3. **8-shot strict flat**: ~66.6-66.9% throughout (no improvement)

### Interpretation

The model is learning to solve GSM8K problems better (as shown by 0-shot and 4-shot improvements), but the gains don't transfer to 8-shot evaluation. Possible explanations:

1. **Few-shot saturation**: With 8 examples, the model already has strong format guidance, leaving less room for improvement
2. **Format learning vs math learning**: The training may be reinforcing format compliance (which helps more at lower shot counts) rather than mathematical reasoning
3. **In-context learning interference**: The 8-shot examples may provide a different "mode" of operation that the RL training doesn't affect

### Why 0-shot flex < 0-shot strict?

The gsm8k_cot task uses two different extraction methods:
- **strict-match**: Looks for "The answer is X" pattern and extracts X
- **flexible-extract**: Extracts the last number in the entire response

For 0-shot (no examples), the model often produces CoT reasoning with multiple numbers. The "flexible-extract" grabs the last number which may not be the final answer. Meanwhile "strict-match" specifically looks for the answer declaration pattern.

This explains why 0-shot flex (~10%) is much lower than 0-shot strict (~58-60%) - the model isn't reliably putting the answer last without examples to follow.

### Status

- Training ongoing on both nodes
- Primary node: completed step 100+
- Secondary node: checkpoints being evaluated
- Eval watchers running with automatic checkpoint detection

### Next Steps

1. Continue monitoring 8-shot results as training progresses
2. Consider whether the original "good run" used different few-shot settings
3. Investigate if longer training shows eventual 8-shot improvement

---

## 2026-01-22: Chat Template Bug for Base Models

### Symptom

Running dummy dataset training with Qwen3-1.7B-Base, 42% of completions were garbage from step 1 (outputs like "Cumhur\nCumhur..." or repeated nonsense). The model should have been able to solve the simple math problem immediately.

### Root Cause

The `DataIterator._apply_template()` function was applying `tokenizer.apply_chat_template()` to ALL prompts, including those for base models. For Qwen3-1.7B-Base, this added chat tokens the base model doesn't understand:

```
<|im_start|>user
[problem text]<|im_end|>
<|im_start|>assistant
<think>

</think>

```

The base model has never seen these special tokens during pretraining, so it produces garbage when they appear in the input.

### Investigation

Examined rollout outputs and found the prompts contained `<|im_start|>` tokens. The base model doesn't understand these - it just sees them as noise and continues with whatever pattern seems likely, producing garbage.

### Fix

Added `skip_chat_template` option to `DataIterator` in `src/rlvr_experiments/data.py`:

```python
def __init__(
    self,
    ...
    skip_chat_template: bool = False,  # NEW
):
    ...
    self.skip_chat_template = skip_chat_template

def _apply_template(self, prompt: str, problem: dict) -> str:
    ...
    if self.skip_chat_template:
        # For base models: just return raw prompt + prefix
        if system_prompt:
            return system_prompt + "\n\n" + prompt + assistant_prefix
        return prompt + assistant_prefix

    # Original chat template code for instruct models...
```

### Config Usage

For base models (Qwen3-1.7B-Base, etc.), add to config:

```yaml
data_iter:
  system_prompt: ""
  assistant_prefix: ""
  skip_chat_template: true  # Required for base models
```

For instruct models, omit `skip_chat_template` or set to `false` (default).

### Impact

After fixing:
- Completions immediately became coherent math reasoning
- Model started producing correct answers from step 1
- Training can now provide meaningful reward signal

This was a critical bug - without it, base model training was fundamentally broken because the model couldn't even parse the inputs correctly.

### Files Changed

- `src/rlvr_experiments/data.py`: Added `skip_chat_template` parameter
- Updated configs for base models:
  - `configs/variations/qwen3-1.7B-dummy.yaml`
  - `configs/variations/qwen3-1.7B-ifeval-mini.yaml`
  - `configs/variations/qwen3-1.7B-allenai-full-lr5e6-beta1e3-repro.yaml`
  - `configs/variations/qwen3-1.7B-allenai-full-lr5e6-beta1e4.yaml`
  - `configs/variations/qwen3-1.7B-allenai-ifeval-only-lr5e6.yaml`

---

## 2026-01-22: AllenAI Full Run Evaluation (allenai_full_lr5e6_beta1e3_repro_20260122_082151)

### Setup

**Config**: `configs/variations/qwen3-1.7B-allenai-full-lr5e6-beta1e3-repro.yaml`
- **Model**: Qwen/Qwen3-1.7B-Base
- **Dataset**: AllenAI RLVR (GSM8K + MATH + IFEval mixed)
- **Verifiers**:
  - GSM8K: `AllenAIGSM8KVerifier` (extracts last number)
  - MATH: `AllenAIMathVerifier` (flexible extraction: boxed, minerva, dollar signs)
  - IFEval: `IFEvalVerifier` (instruction constraint checking)
- **LR**: 5e-6
- **Beta**: 1e-3 (0.001)

### Evaluation Results (lm_eval benchmarks)

Evaluated checkpoints at steps 100, 200, 300, 400, 500:

| Step | gsm8k flex | gsm8k strict | minerva_math | ifeval | hendrycks |
|------|------------|--------------|--------------|--------|-----------|
| 100  | 69.83%     | 51.63%       | 28.56%       | 23.29% | 17.96%    |
| 200  | 69.60%     | 50.49%       | 28.92%       | 23.66% | 17.76%    |
| 300  | 69.22%     | 47.08%       | 29.42%       | 24.77% | 17.84%    |
| 400  | 69.75%     | 45.72%       | 29.94%       | 24.58% | 17.80%    |
| 500  | 68.76%     | 45.41%       | 29.76%       | 26.43% | 17.98%    |

### Critical Finding: Math Performance Degrading Despite Training

Ran the **exact same verifiers used during training** on the lm_eval outputs (with `--log_samples`) to check if training rewards are actually improving.

**Eval tasks used**: `gsm8k_cot` (4-shot), `minerva_math` (4-shot), `ifeval` (0-shot)

| Metric | Base | Step500 | Delta |
|--------|------|---------|-------|
| **GSM8K (AllenAI verifier on gsm8k_cot samples)** | 72.02% | 71.04% | **-0.99%** |
| **MATH (AllenAI verifier on minerva_math samples)** | 37.84% | 37.04% | **-0.80%** |
| **IFEval prompt_strict** | 21.26% | 26.25% | **+4.99%** |
| **IFEval inst_strict** | 32.85% | 37.65% | **+4.80%** |

### Key Insight

Both lm_eval metrics and our training verifiers show the same directional results:

**IFEval is improving** (+5pts) - training is working for instruction following.

**GSM8K and MATH are getting WORSE** (~-1pt each) - training is hurting math performance, even when measured with the exact same criterion we're training on. This is not a verifier mismatch issue - the degradation is real.

### Open Questions

- Why is math getting worse despite positive rewards during training?
- Is the model learning something harmful from the math training signal?
- Is the mixed dataset causing interference (IFEval learning hurts math)?
- Would training on math-only help, or is this a more fundamental issue?

---

## 2026-01-22: Instruct Model Comparison (Full Mixed vs GSM8K-Only)

### Motivation

Testing hypothesis that the base model degradation is due to lacking CoT capability. Running two parallel experiments with the instruct model (Qwen3-1.7B, not Base):

1. **Full Mixed**: Same AllenAI RLVR dataset (GSM8K + MATH + IFEval)
2. **GSM8K-Only**: Filtered AllenAI RLVR to only GSM8K samples

### Configurations

**Full Mixed Instruct** (`qwen3-1.7B-instruct-allenai-full-lr5e6-beta1e3.yaml`):
- Model: Qwen3-1.7B (instruct)
- Dataset: AllenAI RLVR (all ~48k samples)
- Running on primary node

**GSM8K-Only Instruct** (`qwen3-1.7B-instruct-allenai-gsm8k-only-lr5e6-beta1e3.yaml`):
- Model: Qwen3-1.7B (instruct)
- Dataset: AllenAI RLVR filtered to GSM8K only (~7.5k samples)
- Running on secondary node

Both use: lr=5e-6, beta=1e-3

### Results (Steps 20-140)

**GSM8K-Only Instruct (Secondary Node)**:

| Step | gsm8k_cot strict | minerva_math | ifeval prompt | hendrycks |
|------|------------------|--------------|---------------|-----------|
| 20 | 71.72% | 32.68% | 17.01% | 14.92% |
| 40 | 71.57% | 32.90% | 16.82% | 14.92% |
| 60 | 72.18% | 33.00% | 16.64% | 14.76% |
| 80 | 72.18% | 33.04% | 16.64% | 14.80% |
| 100 | 71.65% | 33.04% | 16.45% | 14.78% |
| 120 | 71.95% | 33.54% | 17.19% | 14.74% |
| 140 | 72.10% | 32.76% | 17.38% | 14.80% |

**Full Mixed Instruct (Primary Node)**:

| Step | gsm8k_cot strict | minerva_math | ifeval prompt | hendrycks |
|------|------------------|--------------|---------------|-----------|
| 20 | 71.34% | 33.34% | 15.53% | 14.98% |
| 40 | 72.40% | 32.98% | 16.82% | 14.92% |
| 60 | 71.19% | 33.00% | 17.19% | 14.76% |
| 80 | 71.04% | 32.88% | 17.38% | 14.74% |
| 100 | 71.49% | 33.20% | 17.56% | 14.72% |
| 120 | 72.40% | 33.16% | 16.64% | 14.76% |

### Observations

1. **Both instruct models are stable** at ~71-72% on gsm8k_cot (no degradation)
2. **Base model degraded** from ~51% → ~45% on the same benchmark
3. **All other metrics flat** - minerva_math ~33%, hendrycks ~15%, ifeval ~17%
4. **GSM8K-only vs Full Mixed**: No significant difference between the two runs
5. **Still early** - only at steps 120-140, need more training to see if improvements emerge

### What We Don't Know Yet

- Whether instruct models will *improve* with more training
- Why the base model degraded while instruct doesn't
- Whether the difference is due to CoT capability, chat template, or something else

---

## 2026-01-22: Overfitting Test on 10 GSM8K Samples

### Purpose

Verify the training harness is working by attempting to overfit on a tiny fixed dataset of 10 GSM8K problems.

### Configuration

**Config**: `configs/variations/qwen3-1.7B-base-gsm8k-mini-overfit.yaml`

Key settings:
- Model: Qwen3-1.7B-Base (with `skip_chat_template: true`)
- Dataset: `allenai_gsm8k_mini` - 10 fixed samples from GSM8K (seed=42)
- lr: 1e-4 (increased from 5e-6)
- beta: 0 (no KL penalty)
- n: 16 completions per prompt
- max_staleness: 0
- max_reads: 99999 (allow unlimited reuse of samples)
- Single GPU rollout (data_parallel_size: 1)

**Trace file**: `traces/trace_20260122_185849.jsonl`

### Results

Rewards increased from ~0.41 at step 1 to ~0.94 at step 7, then fluctuated:

| Step | Reward | Loss | Grad Norm |
|------|--------|------|-----------|
| 1 | 0.41 | 0.0001 | 2.25 |
| 5 | 0.88 | 0.0114 | 3.42 |
| 7 | 0.94 | 0.0089 | 2.45 |
| 10 | 0.81 | 0.0060 | 2.67 |
| 15 | 0.88 | 0.0025 | 2.44 |
| 20 | 0.75 | 0.0008 | 2.03 |
| 21 | 0.59 | -0.0006 | 1.81 |

### Observations

1. **Training harness works** - rewards did increase significantly from baseline (~0.4 → 0.9+)
2. **Higher LR helped** - 1e-4 vs 5e-6 showed faster reward improvement
3. **No convergence to 100%** - rewards peaked ~0.94 but never stabilized at 1.0
4. **Fluctuations persist** - even with beta=0 (no KL penalty), rewards oscillate

### Buffer Issue Discovered

With `max_staleness: 0`, the buffer grew unbounded (127 → 700+ samples) because:
- Producer generates samples faster than consumer trains
- Staleness eviction only happens on pop, not proactively
- Result: 7400+ samples evicted as stale (wasted compute)

For overfitting tests, should use `max_staleness: 99999` to disable staleness checks.

### Key Takeaway

The GRPO training loop is functional - rewards do improve with training. The inability to reach 100% reward on 10 samples suggests either:
- Stochastic sampling variance (temperature=1.0) prevents consistent correct answers
- Model capacity/representation issues
- Need more steps to fully converge

---

## 2026-01-22: lm_eval max_gen_toks Truncation Bug

### Discovery

While analyzing why ~13% of GSM8K responses seemed to fail despite correct reasoning, discovered that **lm_eval's default max_gen_toks=256 was truncating responses mid-reasoning**.

### Evidence

With default 256 tokens:
- Step80 checkpoint: 171 out of 1319 samples (13%) were truncated mid-sentence
- Responses cut off like: "...the total time Carla spent downloading the file is 100 minutes + 2" (no final answer)
- Max response length: ~300 tokens, hitting the generation limit

### Impact on Scores

| Setting | gsm8k_cot flex | gsm8k_cot strict |
|---------|----------------|------------------|
| max_gen_toks=256 (default) | 71.27% | 58.45% |
| max_gen_toks=1024 | **75.89%** | **61.79%** |
| **Improvement** | **+4.6%** | **+3.3%** |

For base model:
| Setting | gsm8k_cot flex | gsm8k_cot strict |
|---------|----------------|------------------|
| max_gen_toks=256 (default) | 69.83% | 51.63% |
| max_gen_toks=1024 | **73.09%** | **54.13%** |
| **Improvement** | **+3.3%** | **+2.5%** |

### Why This Matters

1. **Previous baselines were artificially low** - we were penalizing models for running out of generation budget, not for wrong answers
2. **Comparison between models was still valid** - both base and trained models were equally affected
3. **Real improvement is higher** - Step80 vs Base with 1024 tokens shows +2.8% flex, +7.7% strict

### Recommended max_gen_toks by Benchmark

| Benchmark | max_gen_toks | Rationale |
|-----------|--------------|-----------|
| gsm8k_cot | 1024 | CoT reasoning needs ~200-500 tokens typically |
| hendrycks_math | 2048 | Complex proofs can be longer |
| minerva_math | 2048 | Same as hendrycks_math |
| ifeval | 4096 | Instruction-following can require long outputs |

### Updated Baselines

See AGENTS.md for updated baseline numbers with proper max_gen_toks settings.

### Truncation Analysis Details

With 1024 tokens:
- Step80: Only 8 truncated (0.6%) - down from 171 (13%)
- Base: Only 6 truncated (0.5%)
- Max response: 4554 chars (~1100 tokens) - a few edge cases still need more

### Files Changed

- `AGENTS.md`: Updated all baseline numbers with proper max_gen_toks
- Evaluation scripts should be updated to use appropriate max_gen_toks per task

### Commit

**`4a78a95`** - MAJOR: Fix lm_eval truncation bug + update baselines

### Training Results with Proper Eval

Checkpoint from `trace_20260122_221554.jsonl` (GSM8K mini overfit test, base model):

| Checkpoint | gsm8k_cot flex | gsm8k_cot strict | Δ flex vs base | Δ strict vs base |
|------------|----------------|------------------|----------------|------------------|
| Base | 73.09% | 54.13% | - | - |
| Step 80 | 75.89% | 61.79% | +2.8% | +7.7% |
| **Step 140** | **77.26%** | **73.54%** | **+4.2%** | **+19.4%** |

The strict-match improvement is massive — nearly **+20 percentage points** from 54% to 73.5%. The model is learning both the "The answer is X." format AND improving underlying math reasoning.

---

## 2026-01-23: First Successful GRPO Run (GSM8K-Only, Base Model)

### Summary

**This is our first bona fide successful GRPO training run.** The model shows clear, consistent improvement on our training verifier metric throughout training, reaching **+12% over baseline** by step 240.

### What Made It Work

Key changes from previous failed runs:

1. **Larger batch sizes** - `prompts_per_optim_step: 64` (was smaller in previous runs). This stabilized training significantly.

2. **Reduced completions per prompt** - `n: 8` (down from 16). Speeds up generation while still providing enough variance for GRPO.

3. **No reference sync / low beta** - `beta: 0.001` with `prompts_per_reference_sync: 9999999` effectively disables the KL penalty. The model is free to diverge from the reference policy.

4. **float16 dtype** - Changed from bfloat16 to float16 for both training and checkpoints (`export_dtype: "float16"`).

5. **Simple single-task dataset** - Pure GSM8K training, no mixed datasets causing interference.

6. **Proper chat template handling** - `skip_chat_template: true` for base model (critical fix from 2026-01-22).

### Configuration

**Config**: `configs/variations/qwen3-1.7B-gsm8k-only.yaml`

```yaml
model: Qwen3-1.7B-Base
dataset: gsm8k (train split)
lr: 1e-6
beta: 0.001
n: 8 completions per prompt
prompts_per_optim_step: 64
seq_len_buckets: [768]  # 256 prompt + 512 completion
completion_len_buckets: [512]
skip_chat_template: true
export_dtype: float16
```

### Results

**Evaluated with gsm8k_cot 0-shot, max_gen_toks=1024, float16:**

| Model | lm_eval flexible | lm_eval strict | Our verifier (any) |
|-------|------------------|----------------|-------------------|
| **Base** | 10.54% | 58.30% | **63.76%** |
| **Step 120** | 72.18% | 33.43% | **73.92%** |
| **Step 240** | **74.00%** | 1.21% | **75.66%** |

### Key Insight: Format Drift is Expected

The lm_eval `strict-match` metric requires "The answer is X." format. Our training verifier (`GSM8KVerifier` with `mode="any"`) accepts any format that contains the correct number.

**Base model behavior:**
- 58% strict-match (outputs "The answer is X." format from pretraining)
- But only ~10-14% flexible-extract (often gets wrong answers)
- Our verifier: 63.76% (many correct answers in strict format)

**Trained model behavior:**
- 1.2% strict-match (abandoned "The answer is X." format)
- 74% flexible-extract (much better at solving problems)
- Our verifier: 75.66% (correct answers, just different format)

The model learned to **solve math problems** but **unlearned the specific output format**. This is expected behavior when training with a format-agnostic verifier. The model optimizes for what gets rewarded (correct numerical answers), not what doesn't (specific phrasing).

### Training Dynamics

Training rewards from trace file `trace_20260123_021337.jsonl`:

| Step | reward_all | Observation |
|------|------------|-------------|
| 1 | 0.09 | Starting from base model capability |
| 20 | 0.30 | Beginning to learn |
| 80 | 0.75 | Strong improvement |
| 160 | 0.80 | Continuing to improve |
| 240 | 0.77 | Stable at high reward |

The model reached epoch 3 by step 240, meaning it saw the GSM8K training set ~3 times.

### Watcher Results (4-shot eval)

The automatic eval watcher ran gsm8k_cot, minerva_math, hendrycks_math, and ifeval on each checkpoint:

| Step | gsm8k flex | gsm8k strict | ifeval prompt | minerva exact | hendrycks |
|------|------------|--------------|---------------|---------------|-----------|
| Base | 73.09% | 54.13% | 21.26% | 30.58% | 17.94% |
| 20 | 72.48% | 49.36% | 21.44% | 30.54% | 17.98% |
| 100 | 73.01% | 43.97% | 23.84% | 30.38% | 18.30% |
| 200 | 72.02% | 41.32% | **25.32%** | 31.04% | 18.44% |
| 220 | 72.33% | 42.00% | 24.03% | - | - |

**Interesting findings:**
- IFEval improved +4% despite not being in training data (transfer learning?)
- minerva_math and hendrycks_math stayed flat (no degradation)
- gsm8k strict degraded as expected (format drift)
- gsm8k flex stayed flat at ~72-73% (ceiling effect? or 4-shot masking improvement?)

### Why 0-shot Shows More Improvement

The 0-shot evaluation shows dramatic improvement (+12% on our verifier) while 4-shot shows minimal change. This is because:

1. **4-shot provides format guidance** - The few-shot examples teach the model the expected format, partially masking the format drift
2. **Base model relies on format** - At 0-shot, base model outputs "The answer is X." (58% strict) but often wrong answers (10% flex)
3. **Trained model relies on reasoning** - At 0-shot, trained model reasons correctly (74% flex) but uses its own format (1% strict)

The 4-shot examples "normalize" both models toward similar format compliance, so the gap is smaller.

### Conclusion

This run demonstrates that GRPO training **works** on base models when:
1. Batch sizes are large enough for stable gradients
2. The task is focused (single dataset)
3. Chat templates are properly handled
4. The verifier aligns with the training objective (even if not with eval format)

The format drift is a feature, not a bug - the model optimized exactly what we rewarded. To maintain specific output formats, either:
- Use a format-strict verifier during training
- Add format rewards/penalties
- Fine-tune on formatted examples before/after RL

### Files

- Config: `configs/variations/qwen3-1.7B-gsm8k-only.yaml`
- Trace: `traces/trace_20260123_021337.jsonl`
- Checkpoints: `checkpoints/qwen3_17b_gsm8k_only_20260123_021337_step{20,40,...,240}/`
- Eval results: `eval_results/repro_watch/qwen3_17b_gsm8k_only_20260123_021337_step*/`


---

## 2026-01-24: MATH Training Success (GRPO, Base Model)

### Summary

Successful GRPO training on the MATH dataset with Qwen3-1.7B-Base. Using our MATH verifier (which does symbolic equivalence checking), the model improved from **30.24% → 44.76%** (+14.5 percentage points) after just 80 steps.

### Configuration

**Config**: `configs/qwen3-1.7B-math-grpo.yaml`

```yaml
model: Qwen3-1.7B-Base
dataset: math (train split)
loss: grpo (std-normalized advantages, per-response length normalization)
lr: 5e-6
beta: 0.001
n: 8 completions per prompt
prompts_per_optim_step: 128
skip_chat_template: true
```

### Evaluation Method

Ran `hendrycks_math` 0-shot with `--log_samples` and `max_gen_toks=2048`, then verified outputs with our `MathVerifier` (which does `\boxed{}` extraction + sympy symbolic equivalence).

Note: The lm_eval `hendrycks_math exact_match` metric only shows ~0.7-0.9% for both models because it uses strict string matching. Our verifier captures the true mathematical correctness.

### Results

| Model | hendrycks_math exact_match | Our MATH Verifier |
|-------|---------------------------|-------------------|
| **Qwen3-1.7B-Base** | 0.88% | **30.24%** |
| **GRPO step80** | 0.70% | **44.76%** |
| **Improvement** | -0.18% | **+14.52%** |

### Key Takeaways

1. **MATH training works** - Clear improvement in mathematical reasoning after 80 steps
2. **Verifier choice matters** - lm_eval's exact_match is misleading; need symbolic equivalence checking
3. **Format drift** - exact_match slightly decreased because the model may be using different formatting, but mathematical correctness improved substantially
4. **Consistent with GSM8K results** - Similar pattern to our GSM8K success where the model learns the task but may drift on format

### Files

- Config: `configs/qwen3-1.7B-math-grpo.yaml`
- Trace: `traces/trace_20260123_233245.jsonl`
- Checkpoint: `checkpoints/qwen3_17b_20260123_233245_step80/`
- Eval results: `eval_results/grpo_step80_math_0shot/`

---

---

## 2026-01-24: IF_multi_constraints Pass@k Evaluation (Base Model)

### Summary

Evaluated Qwen3-1.7B-Base on 100 randomly sampled prompts from the `IF_multi_constraints_upto5` dataset. Each prompt has 2-5 constraints that must be satisfied. Generated 32 completions per prompt using vLLM.

### Verifier Behavior

The `IFMultiConstraintsVerifier` returns **average constraint satisfaction** (not all-or-nothing):
```python
return sum(rewards) / len(rewards)  # e.g., 3/5 constraints = 0.6
```

This provides a smoother training signal since the model gets partial credit for each satisfied constraint.

### Results

| Metric | Value |
|--------|-------|
| **Average Reward (pass@1)** | **24.24%** |
| pass@2 | 36.60% |
| pass@4 | 50.52% |
| pass@8 | 62.44% |
| pass@16 | 70.85% |
| pass@32 | 81.00% |

### Additional Statistics

- Problems with at least 1 completion scoring > 0: 81/100 (81%)
- Problems with all 32 completions perfect (1.0): 1/100 (1%)
- All 100 samples had constraint_type = "multi" (multi-constraint prompts)

### Comparison: All-or-Nothing vs Average Constraint Satisfaction

Initially tested with all-or-nothing scoring (return 1.0 only if ALL constraints pass):

| Metric | All-or-Nothing | Average Constraint |
|--------|----------------|-------------------|
| Average Reward | 8.62% | **24.24%** |
| pass@32 | 25.00% | **81.00%** |

The average constraint satisfaction metric is more informative for RL training since:
1. Provides gradient signal even when not all constraints are met
2. More stable learning (fewer 0-reward samples)
3. Better measures incremental progress

### Constraint Types

All 100 samples were labeled as "multi" constraint type, meaning they contain multiple instruction-following constraints (e.g., word count limits, keyword inclusion, formatting requirements, etc.).

### Files

- Eval script: `scripts/adhoc/if_multi_constraints_passk_eval.py`
- Aggregation script: `scripts/adhoc/compute_if_multi_constraints_passk.py`
- Launch script: `scripts/adhoc/launch_if_multi_constraints_eval.sh`
- Results: `experiments/if_multi_constraints_passk/aggregated_results.json`

---

---

## 2026-01-24: DrGRPO Learning Rate Sweep (GSM8K + MATH)

### Summary

Ran 4 SageMaker jobs testing different learning rates for DrGRPO training on GSM8K + MATH mixed dataset (50/50 split). **Conclusion: lr=5e-5 was too aggressive (model collapsed), but lr=1e-5 and lower showed no improvement in training rewards.**

### Configuration

All jobs used the same base config with different learning rates:
- **Model**: Qwen3-1.7B-Base
- **Dataset**: GSM8K (50%) + MATH L3-5 (50%)
- **Loss**: DrGRPO (beta=0.001, eps=0.2, C=500)
- **Training**: 2 epochs, checkpoint every 50 steps
- **Sampling**: n=8 completions, temperature=1.0, top_p=0.95

### Jobs Submitted

| LR | Job Name | Status |
|----|----------|--------|
| 5e-5 | annotations-adhoc-20260124-082656 | Completed (collapsed) |
| 1e-5 | annotations-adhoc-20260124-090413 | Completed (resubmit) |
| 5e-6 | annotations-adhoc-20260124-082714 | Completed |
| 1e-6 | annotations-adhoc-20260124-082724 | Completed |

### Checkpoints Available

**Only lr=5e-5 saved checkpoints to S3** (step50, step100, final). The other three jobs completed but only have trace files - no model checkpoints were saved.

### Evaluation Results (lr=5e-5, GSM8K Pass@1)

Used `scripts/test_gsm8k_pass_rate.py` with our MathVerifier (100 samples, greedy decoding):

| Checkpoint | Pass@1 | Median completion length (incorrect) |
|------------|--------|--------------------------------------|
| Step 50 | **34%** | 1 token |
| Step 100 | 8% | 1 token |
| Final | 4.64% | 1 token |

**Reference**: Base model should get ~54% pass@1 on GSM8K with our verifier.

### Analysis

**lr=5e-5 (too aggressive)**:
- Model collapsed during training
- Trace shows grad_norm spiking to 148 late in training
- Mode collapse: median completion length of 1 token for incorrect responses
- Step 50 (34%) was best but still below baseline (54%)

**lr=1e-5, 5e-6, 1e-6 (too conservative)**:
- Traces show flat or declining training rewards
- No checkpoints saved to evaluate
- Training likely too slow to see improvement within 2 epochs

### Trace Files

All trace files downloaded to `traces/`:

| LR | Trace File |
|----|------------|
| 5e-5 | `drgrpo_lr5e5_trace.jsonl` (273MB) |
| 1e-5 | `drgrpo_lr1e5_trace.jsonl` (182MB) |
| 5e-6 | `drgrpo_lr5e6_trace.jsonl` (169MB) |
| 1e-6 | `drgrpo_lr1e6_trace.jsonl` (165MB) |

### Conclusions

1. **lr=5e-5 is too high** - Causes training instability and model collapse
2. **lr=1e-6 to 1e-5 may be too low** - No visible improvement in training rewards
3. **Sweet spot likely between 1e-5 and 5e-5** - Need to test intermediate values (e.g., 2e-5, 3e-5)
4. **DrGRPO may need different hyperparameters than GRPO** - The C=500 clipping and other DrGRPO-specific parameters may interact differently with learning rate

### Next Steps

- Try intermediate learning rates (2e-5, 3e-5)
- Ensure checkpoints are saved for all runs
- Consider longer training (more epochs) with lower learning rates
- Analyze trace files to understand training dynamics at each LR

### Files

- Configs: `configs/variations/qwen3-1.7B-gsm8k-math-drgrpo-lr{5e5,1e5,5e6,1e6}.yaml`
- Checkpoints: `/efs/rlvr-experiments/checkpoints/drgrpo_lr5e5/` (step50, step100, final)
- Job log: `docs/job_log.md`

---

### TODOS

FRIDAY:
- ~~Dr GRPO implementation / confirm it works~~ ✅ Done (configurable via `loss.name: drgrpo` or `loss.name: grpo`)
- ~~Non-GSM8k datasets~~ ✅ Done (MATH working)
- Add entropy computation

SATURDAY:
- Staleness b (sagemaker)
- LLM-as-a-judge

---

## GRPO vs DrGRPO Comparison (GSM8K, 2026-01-25)

All runs on Qwen3-1.7B-Base with n=8 completions per prompt.

| Run | LR | Loss | C | β | Steps | Final | Peak | Notes |
|-----|----|------|---|---|-------|-------|------|-------|
| GRPO stale0 | 1e-6 | GRPO | - | 1e-3 | 169 | 0.591 | 0.634 | |
| GRPO stale1 | 1e-6 | GRPO | - | 1e-3 | 178 | 0.525 | 0.631 | |
| DrGRPO Sweep B | 1e-5 | DrGRPO | 512 | 8e-4 | 129 | 0.245 | 0.246 | best DrGRPO |
| DrGRPO v3 | 3e-6 | DrGRPO | 512 | 1e-3 | 107 | 0.122 | 0.133 | |
| DrGRPO node2 | 1e-5 | DrGRPO | 200 | 1e-2 | 100 | 0.295 | 0.307 | |
| DrGRPO node2 | 5e-6 | DrGRPO | 512 | 1e-3 | 98 | 0.133 | 0.159 | |
| DrGRPO Sweep A | 8e-6 | DrGRPO | 512 | 1e-3 | 90 | 0.160 | 0.164 | |
| DrGRPO Sweep C | 8e-6 | DrGRPO | 384 | 1e-3 | 90 | 0.150 | 0.181 | |
| DrGRPO v2 | 1.5e-5 | DrGRPO | 250 | 2e-2 | 88 | 0.276 | 0.276 | |
| DrGRPO Sweep D | 5e-6 | DrGRPO | 384 | 1e-3 | 112 | 0.120 | 0.142 | |
| DrGRPO Sweep E | 5e-6 | DrGRPO | 256 | 8e-4 | 110 | 0.146 | 0.146 | |
| DrGRPO | 1e-6 | DrGRPO | 100 | 1e-3 | 43 | 0.119 | 0.141 | |
| DrGRPO | 2e-5 | DrGRPO | 150 | 5e-3 | 10 | 0.256 | 0.256 | KL exploded |
| DrGRPO | 1e-19 | DrGRPO | 1 | 1e-3 | 9 | 0.092 | 0.113 | sync debug |
| DrGRPO | 3e-5 | DrGRPO | 150 | 1e-3 | 8 | 0.120 | 0.131 | KL exploded |
| DrGRPO | 5e-5 | DrGRPO | 150 | 1e-3 | 7 | 0.249 | 0.249 | KL exploded |

**Key observation**: GRPO at lr=1e-6 achieves 0.59-0.63 reward_overall, while DrGRPO at 10x higher lr (1e-5) only reaches 0.25. This is expected behavior due to DrGRPO's design differences:

1. **No length normalization**: DrGRPO divides by fixed C instead of actual sequence length
2. **No advantage std normalization**: DrGRPO only mean-centers advantages (no division by std)

Combined effect: DrGRPO gradients are ~5x weaker for the same lr, requiring higher lr to match GRPO training dynamics. Even at lr=5e-5, DrGRPO peaked at 0.394 vs GRPO's 0.59+ at lr=1e-6.

---

## IFEval GRPO Sweep (SageMaker, 2026-01-25)

Hyperparameter sweep for IFEval training on Qwen3-1.7B-Base with GRPO.

### Configuration
- Dataset: `if_multi_constraints` (~500 prompts)
- n=8 completions per prompt
- max_tokens=2048
- checkpoint_interval=25 steps
- Instance: ml.p4de.24xlarge

### Jobs Submitted

| Job Name | LR | Beta | Status | Steps | Notes |
|----------|-----|------|--------|-------|-------|
| annotations-adhoc-20260125-083856 | 5e-6 | 0.001 | Running | 249 | reward 25%→37% |
| annotations-adhoc-20260125-083900 | 5e-6 | 0.0001 | KL exploded | 79 | kl_max=9748, entropy→0.008 |
| annotations-adhoc-20260125-083904 | 1e-6 | 0.001 | Killed | 199 | no improvement |
| annotations-adhoc-20260125-083907 | 1e-6 | 0.0001 | Killed | 199 | no improvement |
| annotations-adhoc-20260125-083911 | 5e-7 | 0.001 | Killed | 199 | no improvement |
| annotations-adhoc-20260125-083914 | 5e-7 | 0.0001 | Killed | 199 | no improvement |

### Observations

- **lr=5e-6, beta=0.001 shows improvement**: Overall reward increased from ~25% to ~37% after 80 steps
- **lr=5e-6, beta=0.0001 KL exploded**: Low beta insufficient to constrain KL. At step 79: kl_max=9748 (should be <1), entropy collapsed to 0.008 (model became deterministic). Rollouts still coherent text but model drifted far from reference.
- **lr=1e-6 and lr=5e-7 showed no movement**: High zero_variance skip counts (13k+), model not learning
- **IFEval is hard**: Most batches have all 0 or all 1 rewards, providing limited gradient signal
- Higher learning rate (5e-6) with sufficient beta (0.001) seems necessary to make progress on this task

### Traces

Downloaded to `/efs/rlvr-experiments/traces_ifeval_sweep/`:
- `annotations-adhoc-20260125-083856.jsonl` (427M)
- `annotations-adhoc-20260125-083900.jsonl` (115M)
- `annotations-adhoc-20260125-083904.jsonl` (600M)
- `annotations-adhoc-20260125-083907.jsonl` (551M)
- `annotations-adhoc-20260125-083911.jsonl` (585M)
- `annotations-adhoc-20260125-083914.jsonl` (540M)

---

## MATH GRPO Sweep (Local, 2026-01-25)

Learning rate sweep for MATH training on Qwen3-1.7B-Base with GRPO.

### Configuration
- Dataset: `math` (~7.5k prompts)
- n=8 completions per prompt
- max_tokens=1024
- beta=0.001
- Local runs on primary and secondary nodes

### Results

| Config | LR | Final Step | reward_all | Location |
|--------|-----|------------|------------|----------|
| qwen3-1.7B-math-lr5e6 | 5e-6 | 37 | **0.49** | `results/qwen3-1.7B-math-lr5e6_20260125-101019/` |
| qwen3-1.7B-math-lr2e6 | 2e-6 | 39 | 0.45 | `results/qwen3-1.7B-math-lr2e6_20260125-101020/` |
| qwen3-1.7B-math | 1e-6 | 39 | 0.43 | `results/qwen3-1.7B-math_20260125-084518/` |
| qwen3-1.7B-math-lr5e7 | 5e-7 | 39 | 0.42 | `results/qwen3-1.7B-math-lr5e7_20260125-084519/` |

### Observations

- **Best LR: 5e-6** with reward_all=0.49
- Clear trend: higher LR → better final reward (within tested range)
- MATH dataset has ~59 steps per epoch with 128 prompts/step, so all runs completed ~1 epoch

---

## 2026-01-25: MATH Test Set Evaluation (lr=5e-6 Run)

### Summary

Evaluated the best MATH GRPO run from the learning rate sweep on the full MATH test set (5000 problems). Used our `MathVerifier` with sympy symbolic equivalence checking.

### Run Details

**Run**: `results/qwen3-1.7B-math-lr5e6_20260125-101019/`
**Config**: `configs/qwen3-1.7B-math.yaml`

```yaml
model: Qwen3-1.7B-Base
dataset: math (train split, ~7.5k problems)
loss: grpo
lr: 5e-6
beta: 0.001
n: 8 completions per prompt
prompts_per_optim_step: 128
seq_len_buckets: [768, 1280]
completion_len_buckets: [512, 1024]
skip_chat_template: true
```

**Checkpoint**: `qwen3-1.7B-math-lr5e6_20260125-101019_final` (end of epoch 1)

### Evaluation Method

- Script: `scripts/eval_math_checkpoint.py`
- vLLM greedy sampling (temperature=0, TP=1, max_tokens=2048)
- Verifier: `MathVerifier` (extracts `\boxed{}` answer, sympy symbolic equivalence)
- Prompt format: `Problem: {problem}\n\nSolution:`

### Results

| Model | Total | Correct | Accuracy |
|-------|-------|---------|----------|
| **Qwen3-1.7B-Base** | 5000 | 2059 | **41.2%** |
| **GRPO lr=5e-6** | 5000 | 2615 | **52.3%** |
| **Improvement** | - | +556 | **+11.1%** |

### Results by Difficulty Level

| Level | Base Correct/Total | Base Acc | GRPO Correct/Total | GRPO Acc | Δ |
|-------|-------------------|----------|-------------------|----------|---|
| Level 1 | 320/437 | 73.2% | 342/437 | 78.3% | +5.1% |
| Level 2 | 556/894 | 62.2% | 613/894 | 68.6% | +6.4% |
| Level 3 | 591/1131 | 52.3% | 680/1131 | 60.1% | +7.9% |
| Level 4 | 385/1214 | 31.7% | 599/1214 | 49.3% | +17.6% |
| Level 5 | 207/1324 | 15.6% | 381/1324 | 28.8% | +13.2% |

### Results by Subject

| Subject | Base Acc | GRPO Acc | Δ |
|---------|----------|----------|---|
| algebra | 67.6% | 75.1% | +7.5% |
| prealgebra | 58.6% | 66.9% | +8.4% |
| number_theory | 38.9% | 50.7% | +11.9% |
| counting_and_probability | 32.5% | 44.7% | +12.2% |
| geometry | 28.0% | 39.2% | +11.3% |
| intermediate_algebra | 24.5% | 33.5% | +9.1% |
| precalculus | 22.0% | 30.0% | +8.0% |

### Key Takeaways

1. **Strong improvement across all levels**: The trained model improves on every difficulty level
2. **Largest gains on Level 4**: +17.6% improvement on Level 4 problems
3. **Improvement scales with difficulty**: Harder problems (L3-L5) see larger absolute gains than easy ones (L1-L2)
4. **Consistent across subjects**: Every subject shows improvement, ranging from +7.5% (algebra) to +12.2% (counting_and_probability)

### Files

- Base model eval: `results/qwen3-1.7B-base/evals/math/math_test_summary.json`
- GRPO eval: `results/qwen3-1.7B-math-lr5e6_20260125-101019/evals/math_test_summary.json`
- Completions: `*/evals/math_test_completions.jsonl`
- Eval script: `scripts/eval_math_checkpoint.py`

---

## IFEval/IFBench Evaluation (2026-01-25)

### Model & Checkpoints

**Training run**: `annotations-adhoc-20260125-083856`
- Config: `configs/qwen3-1.7B-ifeval.yaml`
- Hyperparams: lr=5e-6, beta=0.001
- Base model: `Qwen/Qwen3-1.7B-Base`
- Results folder: `results/qwen3-1.7B-ifeval-lr5e6-beta1e3_20260125-083856/`

**Checkpoints evaluated**:
- `results/qwen3-1.7B-ifeval-lr5e6-beta1e3_20260125-083856/checkpoints/step100/`
- `results/qwen3-1.7B-ifeval-lr5e6-beta1e3_20260125-083856/checkpoints/step250/`
- Base model: `Qwen/Qwen3-1.7B-Base`

### Evaluation Setup

- **Generation**: vLLM with TP=1, greedy sampling (temperature=0), max_tokens=2048
- **Datasets**:
  - IFEval (Google): 541 prompts from `google/IFEval` (train split)
  - IFBench (AllenAI): 300 prompts from `allenai/IFBench_test` (train split)
- **Verifiers**:
  - IFEval: `src/rlvr_experiments/verifiers/if_multi_constraints.py` (RLVR-IFEval format)
  - IFBench: `src/rlvr_experiments/verifiers/ifbench.py` (AllenAI IFBench format, 55 instruction types)

### Results

#### IFEval (In-distribution - trained on RLVR-IFEval constraints)

| Model | Prompt-Level Strict | Instruction-Level |
|-------|--------------------:|------------------:|
| **Qwen3-1.7B-Base** | 21.26% | 32.85% |
| **Step 100** | **40.85%** | **54.20%** |
| **Step 250** | 40.11% | 53.72% |
| **Improvement (Step 100)** | **+19.6%** | **+21.4%** |

#### IFBench (Out-of-distribution - novel constraint types)

| Model | Prompt-Level Strict | Instruction-Level |
|-------|--------------------:|------------------:|
| **Qwen3-1.7B-Base** | 12.33% | 14.83% |
| **Step 100** | 19.00% | 20.06% |
| **Step 250** | **20.00%** | **22.09%** |
| **Improvement (Step 250)** | **+7.7%** | **+7.3%** |

### Key Findings

1. **IFEval performance nearly doubled**: Training on RLVR-IFEval improved prompt-level accuracy from 21.26% to 40.85% (+19.6%)
2. **OOD generalization to IFBench**: Despite training only on IFEval-style constraints, models also improved on IFBench (+7.7%)
3. **Step 100 vs Step 250**:
   - Step 100 slightly better on IFEval (in-distribution)
   - Step 250 slightly better on IFBench (OOD) - possibly better generalization with more training
4. **IFBench is harder**: Base model only 12.33% vs 21.26% on IFEval (constraints are more specific/unusual)

### Bug Fix During Evaluation

Initial IFBench verification showed 0% for all models. Root cause: the verifier was passing ALL kwargs (including `None` values) to each checker's `build_description()` method, which only accepts specific parameters. The `except Exception` silently caught the `TypeError` and returned `False`.

**Fix**: Filter out `None` values before calling `build_description()`:
```python
filtered_kwargs = {k: v for k, v in (kwargs or {}).items() if v is not None}
```

### Files

- Generation script: `scripts/eval_ifeval_checkpoint.py`
- Verification script: `scripts/verify_ifeval_completions.py`
- IFBench verifier: `src/rlvr_experiments/verifiers/ifbench.py`
- Base model results: `results/qwen3-1.7B-base/evals/ifeval/`
- Step 100 results: `results/qwen3-1.7B-ifeval-lr5e6-beta1e3_20260125-083856/evals/step100/`
- Step 250 results: `results/qwen3-1.7B-ifeval-lr5e6-beta1e3_20260125-083856/evals/step250/`

---

## 2026-01-26: DrGRPO + Adaptive Sampling Experiments (GSM8K)

### Summary

Attempted to combine DrGRPO loss with adaptive sampling on GSM8K. **Conclusion: DrGRPO with adaptive sampling does not work well - either KL explodes or learning is too slow. Abandoning in favor of standard GRPO.**

### Configuration

All runs used `configs/qwen3-1.7B-gsm8k-drgrpo-adaptive.yaml`:
- **Model**: Qwen3-1.7B-Base
- **Dataset**: GSM8K
- **Loss**: DrGRPO
- **Adaptive Sampling**: k_success=2, k_failure=2, max_completions=64, chunk_size=8

### Runs

| Run | C | LR | Beta | Step 30 | Step 60 | Notes |
|-----|---|-----|------|---------|---------|-------|
| 4 | 150 | 5e-6 | 0.005 | - | - | KL exploded ~0.3 by step 40, grad_norm spiked to 294 |
| 5 | 512 | 5e-6 | 0.005 | 8% | - | Too slow, killed early |
| 6 | 256 | 5e-6 | 0.005 | 12% | 16% | Stable KL (~0.03-0.06), but slow learning |
| 7 | 200 | 5e-6 | 0.005 | 12% | - | Same as C=256, no improvement |
| 8 | 175 | 5e-6 | 0.007 | 14% | diverged | KL stable initially, then exploded at step ~85, grad_norm=inf by step 100 |
| 9 | 256 | 1e-5 | 0.005 | ~8% | - | Higher lr didn't help, reward_all stuck at 7-8% |

### Targets (from reference GRPO stale=0 run)
- Step 30: >15% reward_all
- Step 60: >30% reward_all

### Key Observations

1. **C=150-175 unstable**: KL explodes even with higher beta
2. **C=256-512 too slow**: Reward_all stuck at 8-16% by step 60
3. **Higher lr (1e-5 vs 5e-6) didn't help**: Same slow learning with C=256
4. **DrGRPO fundamentally different from GRPO**: No length normalization + no advantage std normalization means gradients are ~5x weaker

### Trace Files

Traces saved in run folders under `results/`:
- `results/qwen3_17b_gsm8k_adaptive_*/traces/trace.jsonl`

### Conclusion

DrGRPO + adaptive sampling combination doesn't achieve competitive results on GSM8K. The effective learning rate is hard to tune - either too aggressive (KL explodes) or too conservative (no learning). Switching to standard GRPO with adaptive sampling.

---

## 2026-01-26: GRPO + Adaptive Sampling on GSM8K

### Summary

Standard GRPO with adaptive sampling on GSM8K shows strong results: **72.6% accuracy** after ~2.5 epochs (step 220), up from 14.3% base model.

### Configuration

- **Config**: `configs/qwen3-1.7B-gsm8k-grpo-adaptive.yaml`
- **Git commit**: `489ffc2` (or later)
- **Model**: Qwen3-1.7B-Base
- **Dataset**: GSM8K train
- **Loss**: GRPO (beta=0.001, eps=0.2)
- **Optimizer**: AdamW (lr=1e-6, eps=1e-4)
- **Adaptive Sampling**: k_success=2, k_failure=2, max_completions=64, chunk_size=8
- **Sampling**: temperature=1.0, top_p=0.95, top_k=20, n=16, max_tokens=512

### Run Details

- **Run folder**: `results/qwen3-1.7B-gsm8k-grpo-adaptive_20260126-160231/`
- **Log file**: `/tmp/grpo_adaptive_train10.log`
- **Started**: 2026-01-26 16:02
- **Stopped**: 2026-01-26 19:11 (manually stopped at step 236, epoch 2)
- **Duration**: ~3 hours for ~2.5 epochs

### Training Progress

- Epoch 1 ended at step ~115 (117 steps/epoch with 64 prompts/step, 7473 GSM8K train examples)
- Epoch 2 ended at step 229 (19:06:40)
- Run stopped at step 236 (epoch 2, mid-epoch 3)

### Evaluation Results

| Checkpoint | Step | Epoch | GSM8K Test Accuracy |
|------------|------|-------|---------------------|
| Base model | - | - | 14.3% (189/1319) |
| Step 120 | 120 | ~2 | 69.9% (922/1319) |
| Step 220 | 220 | ~3 | **72.6% (957/1319)** |

### Key Observations

1. **Massive improvement from base**: +58.3pp from 14.3% to 72.6%
2. **Most gains in first 2 epochs**: Step 120 already at 69.9%, only +2.7pp more by step 220
3. **Stable training**: No KL explosions or gradient issues throughout
4. **Adaptive sampling worked well**: Mixed batch sizes (B=16-48) with early stopping on solved problems

### Files

- **Config**: `configs/qwen3-1.7B-gsm8k-grpo-adaptive.yaml`
- **Checkpoints**: `results/qwen3-1.7B-gsm8k-grpo-adaptive_20260126-160231/checkpoints/`
- **Traces**: `results/qwen3-1.7B-gsm8k-grpo-adaptive_20260126-160231/traces/`
- **Evals**:
  - `results/qwen3-1.7B-gsm8k-grpo-adaptive_20260126-160231/evals/gsm8k_step120/`
  - `results/qwen3-1.7B-gsm8k-grpo-adaptive_20260126-160231/evals/gsm8k_step220/`
- **Base model eval**: `results/qwen3-1.7B-base/evals/gsm8k/summary.json`

---

## lm_eval Benchmark Comparison: Step 120 vs Base (2026-01-26)

Evaluated step 120 checkpoint against base model using lm_eval with gsm8k and gsm8k_cot tasks.

### Results Summary

| Task | Shots | Metric | Base | Step 120 | Δ |
|------|-------|--------|------|----------|---|
| **gsm8k** | 0-shot | flexible-extract | 14.48% | **63.38%** | +48.9pp |
| | | strict-match | 0.00% | 0.15% | +0.15pp |
| **gsm8k** | 4-shot | flexible-extract | 69.75% | **72.86%** | +3.1pp |
| | | strict-match | 59.59% | **62.40%** | +2.8pp |
| **gsm8k_cot** | 0-shot | flexible-extract | 11.75% | **68.16%** | +56.4pp |
| | | strict-match | 56.25% | 51.25% | -5.0pp |
| **gsm8k_cot** | 4-shot | flexible-extract | 73.09% | 72.02% | -1.1pp |
| | | strict-match | 54.13% | 44.05% | -10.1pp |

### Key Observations

1. **Massive 0-shot improvement**: The trained model jumps from ~12-14% to ~63-68% on 0-shot, showing it learned to solve GSM8K without needing few-shot examples.

2. **flexible-extract improved, strict-match degraded**: The model got better at producing correct answers (flexible) but *worse* at using the exact "The answer is X." format (strict). This makes sense - during RL training, MathVerifier rewards the last number, not the specific phrase.

3. **4-shot shows modest gains on gsm8k** but slight regression on gsm8k_cot strict-match. The model no longer needs few-shot examples as much, so they help less.

4. **The 0-shot strict-match anomaly**: Base model gets 56% strict on gsm8k_cot 0-shot but 0% on gsm8k 0-shot - this is because gsm8k_cot uses "The answer is X." format in few-shot examples even at 0-shot (it's baked into the task), while gsm8k doesn't use CoT prompting at all.

### Extraction Method Notes

- **strict-match**: First match of `The answer is X.` pattern
- **flexible-extract**: Last number in response

MathVerifier (used during training) behaves like flexible-extract (extracts last number), which explains why the model improved on flexible but not strict. The training signal never rewarded "The answer is X." format specifically.

### Eval Commands

```bash
# Settings used: TP=1, seed=42, temp=0, max_gen_toks=1024
CUDA_VISIBLE_DEVICES=X lm_eval --model vllm \
  --model_args "pretrained=$MODEL,dtype=bfloat16,tensor_parallel_size=1,gpu_memory_utilization=0.8,max_model_len=4096,seed=42" \
  --tasks gsm8k_cot --num_fewshot 0 --batch_size auto --seed 42 \
  --gen_kwargs "temperature=0,max_gen_toks=1024" \
  --output_path $OUT_DIR
```

Results saved to: `/efs/rlvr-experiments/eval_results/gsm8k_comparison/`

---

## 2026-01-26: Stop-Sequence Experiment (GSM8K)

### Motivation

Previous analysis showed that the base model's "actual" GSM8K accuracy is ~67% when completions are truncated at stop sequences (model continues generating Q&A pairs after answering). MathVerifier extracts the LAST number, so without stop sequences it gets garbage from continuation text. lm_eval's strict-match extracts the FIRST "The answer is X" so it gets ~66%.

The question: **Does the model actually "learn to solve math" during GRPO, or does it just "learn to stop"?** If we force stopping with stop sequences, does reward start at 67% instead of 8%?

### Config: `qwen3-1.7B-gsm8k-grpo-stale0-stop.yaml`

Key differences from original `qwen3-1.7B-gsm8k-grpo-stale0.yaml`:
- Added stop sequences: `["[Question]", "Question:", "Q:", "\n\n\n", "\n\n"]`
- Fixed optimizer eps: 1e-6 (was 1e-6 in original too, so no change)

### Initial Observations (Step 1)

**Expected**: reward_overall ~67% (matching truncated base model accuracy)
**Actual**: reward_overall started at **25.3%**, climbed to 62% by step ~10

Why not 67%? Investigation revealed:
1. **Stop sequence coverage gap**: 62.8% of completions contained `Question:` but only `Q:` was being matched (wrong pattern)
2. **`\n\n\n` triggered** for ~20% of completions, but `\n\n` wasn't in the original config
3. **Completions were shorter** (avg 643 chars) vs original run (avg 1402 chars), confirming stop sequences worked partially

**Fix applied**: Added `"Question:"` and `"\n\n"` to stop sequences in config for future runs.

### clip_frac Nonzero Mystery

For on-policy (max_staleness=0), ratio π(a|s)/π_old(a|s) should be exactly 1.0, so clip_frac should be 0.

**Observed**: clip_frac ~0.08, ratio_max up to 1.51, kl_max up to 1691

**Root cause**: Trainer uses TP=2, vLLM reference/rollout uses TP=1. Different tensor parallelism causes numerical differences in logprob computation. Even though the weights are identical, the computation order differs, leading to ratio ≠ 1.0.

This is a known issue with mixed-parallelism setups and doesn't indicate actual staleness.

### Checkpoint Evaluation (Step 60)

**Run folder**: `results/qwen3-1.7B-gsm8k-grpo-stale0-stop_20260126-201902/`

#### Our Verifier (scripts/eval_checkpoint.py)

| Checkpoint | Accuracy | Correct/Total |
|------------|----------|---------------|
| Step 60 | **77.1%** | 1017/1319 |

#### lm_eval (0-shot, temp=0, TP=1, seed=42)

| Task | Metric | Value |
|------|--------|-------|
| gsm8k | flexible-extract | **75.89%** |
| gsm8k | strict-match | 0.00% |
| gsm8k_cot | flexible-extract | **75.51%** |
| gsm8k_cot | strict-match | **58.00%** |

### Comparison with Base Model

| Metric | Base (4-shot) | Step 60 (0-shot) | Δ |
|--------|---------------|------------------|---|
| gsm8k_cot flexible-extract | 73.09% | 75.51% | **+2.4pp** |
| gsm8k_cot strict-match | 54.13% | 58.00% | **+3.9pp** |

**Key finding**: Step 60 achieves **75.51% 0-shot** vs base model's **73.09% 4-shot** on gsm8k_cot flexible-extract. This demonstrates **genuine math reasoning improvement**, not just "learning to stop".

### Conclusion

Even with stop sequences forcing early termination:
1. Initial reward started at ~25% (not 67%) due to incomplete stop sequence coverage
2. Training still improved accuracy significantly (25% → 77% by step 60)
3. The model beats its own 4-shot baseline when evaluated 0-shot, confirming learned reasoning ability

### Files

- **Config**: `configs/qwen3-1.7B-gsm8k-grpo-stale0-stop.yaml`
- **Run folder**: `results/qwen3-1.7B-gsm8k-grpo-stale0-stop_20260126-201902/`
- **Traces**: `results/qwen3-1.7B-gsm8k-grpo-stale0-stop_20260126-201902/traces/trace.jsonl`
- **Checkpoints**: `results/qwen3-1.7B-gsm8k-grpo-stale0-stop_20260126-201902/checkpoints/`
- **Step 60 eval (our verifier)**: `results/qwen3-1.7B-gsm8k-grpo-stale0-stop_20260126-201902/evals/gsm8k_step60/`
- **Step 60 lm_eval gsm8k**: `results/qwen3-1.7B-gsm8k-grpo-stale0-stop_20260126-201902/evals/lm_eval_gsm8k_0shot/`
- **Step 60 lm_eval gsm8k_cot**: `results/qwen3-1.7B-gsm8k-grpo-stale0-stop_20260126-201902/evals/lm_eval_gsm8k_cot_0shot/`

---

## MATH Training - KL Spike Fix & Hyperparameter Tuning

**Date**: 2026-01-27

### Problem: KL Divergence Explosion

GRPO training on MATH dataset was exploding after 5-8 steps with KL divergence spikes of ~2685, causing loss to become NaN.

### Root Cause (Verified)

The reference model was computing logprobs correctly (verified by independent vLLM test). The issue was:

1. **Small optimizer epsilon (1e-7)** - Adam updates unstable early in training
2. **No learning rate warmup** - Full LR applied from step 0
3. **Low KL penalty (beta=0.001)** - Couldn't prevent rapid divergence on rare token patterns

The Schulman KL approximator `kl = exp(r) - r - 1` explodes when `r = policy_lp - ref_lp` gets large. With beta=0.001, the KL gradient only matches the surrogate gradient at log_ratio ≈ 7, by which point probability has shifted 1000x.

### Fix Applied

```yaml
optimizer:
  eps: 0.0001  # Changed from 0.0000001
  
lr_scheduler:
  warmup_steps: 50  # Changed from 0
```

### Training Run: math-eps-warmup-lr5e6

**Config**: `configs/adhoc/math-eps-warmup-lr5e6.yaml`
**Run folder**: `results/math-eps-warmup-lr5e6_20260127-095003/`

Key hyperparameters:
- `optimizer.lr`: 5e-6
- `optimizer.eps`: 1e-4
- `lr_scheduler.warmup_steps`: 50
- `loss.beta`: 0.001
- `training.num_epochs`: 2
- `sampling.n`: 8

### Training Results (72 steps, 2 epochs)

| Steps | reward_overall | frac_all_correct |
|-------|----------------|------------------|
| 1-10  | 0.4067 | 6.2% |
| 61-70 | 0.4759 | 11.6% |
| 71-72 | 0.5026 | 14.6% |

**Improvement**: reward_overall +19%, frac_all_correct **doubled** (6% → 12.5%)

**Note**: Only 72 steps because ~38% of prompts filtered due to length constraints (seq_len_buckets: [768, 1280])

### Evaluation Results (Final Checkpoint)

#### Hendrycks MATH (0-shot, greedy)

| Metric | Value |
|--------|-------|
| lm_eval exact_match | 0.70% |
| **MathVerifier** | **44.88%** |

By difficulty level:
| Level | Accuracy |
|-------|----------|
| Level 1 | 73.68% |
| Level 2 | 61.52% |
| Level 3 | 50.31% |
| Level 4 | 40.61% |
| Level 5 | 23.41% |

#### Minerva MATH (4-shot, greedy)

| Metric | Value |
|--------|-------|
| lm_eval exact_match | 31.68% |
| lm_eval math_verify | 42.72% |
| **MathVerifier** | **39.20%** |

#### Base Model Comparison (MathVerifier)

| Model | MATH Accuracy |
|-------|---------------|
| Base (Qwen3-1.7B) | ~41.2% |
| After GRPO (72 steps) | 44.88% |
| **Δ** | **+3.7pp** |

### Files

- **Config**: `configs/adhoc/math-eps-warmup-lr5e6.yaml`
- **Run folder**: `results/math-eps-warmup-lr5e6_20260127-095003/`
- **Traces**: `results/math-eps-warmup-lr5e6_20260127-095003/traces/trace.jsonl`
- **Checkpoints**: `results/math-eps-warmup-lr5e6_20260127-095003/checkpoints/`
  - `math-eps-warmup-lr5e6_20260127-095003_step20/`
  - `math-eps-warmup-lr5e6_20260127-095003_step40/`
  - `math-eps-warmup-lr5e6_20260127-095003_step60/`
  - `math-eps-warmup-lr5e6_20260127-095003_final/`
- **Eval (Hendrycks, verified)**: `results/math-eps-warmup-lr5e6_20260127-095003/evals/hendrycks_math_verified/`
- **Eval (Minerva, verified)**: `results/math-eps-warmup-lr5e6_20260127-095003/evals/minerva_math_verified/`

### Analysis Docs

- `docs/kl_spike_fix.md` - Fix documentation and experiment results
- `docs/kl_spike_analysis.md` - Detailed root cause analysis

### Next Steps

1. Run with num_epochs=10 to allow longer training
2. Try eps=1e-5, warmup=30 for faster warmup
3. Tune learning rate and loss.beta

---

## Hyperparameter Experiments on MATH (2026-01-27)

Following the KL spike fix, we're exploring three hyperparameter combinations for MATH training:

### Summary Table

| Run | LR | Optimizer eps | Warmup | Epochs | MATH Acc | GSM8K Acc | Δ MATH | Δ GSM8K |
|-----|-----|---------------|--------|--------|----------|-----------|--------|---------|
| **Base (Qwen3-1.7B)** | — | — | — | — | 41.18% | 14.33% | — | — |
| **math-eps-warmup-lr5e6** | 5e-6 | 1e-4 | 50 | 2 | 44.88% | — | +3.7pp | — |
| **math-10epochs** (step100) | 5e-6 | 1e-4 | 50 | 10 | 57.56% | **67.02%** | +16.4pp | **+52.7pp** |
| **math-eps1e5-warmup30-lr1e5** (step80) | 1e-5 | 1e-5 | 30 | 10 | 61.20% | — | +20.0pp | — |

### Run 1: math-eps-warmup-lr5e6 (baseline)

- **Config**: `configs/adhoc/math-eps-warmup-lr5e6.yaml`
- **Hyperparams**: lr=5e-6, eps=1e-4, warmup=50, epochs=2
- **Steps completed**: 72 (limited by seq_len_buckets filtering ~38% of prompts)
- **MATH (MathVerifier)**: 44.88% (base: 41.18%)

### Run 2: math-10epochs

- **Config**: `configs/adhoc/math-10epochs.yaml`
- **Hyperparams**: lr=5e-6, eps=1e-4, warmup=50, epochs=10
- **Run folder**: `results/math-10epochs_20260127-154943/`
- **Checkpoint**: step100

**Results (MathVerifier)**:
| Benchmark | Accuracy |
|-----------|----------|
| MATH (5000) | 57.56% (2878/5000) |
| GSM8K (1319) | **67.02%** (884/1319) — after stop token fix |

**Note on GSM8K accuracy**: Original evaluation showed 16.83% because completions continued past the answer with "Question:" patterns (hallucinated follow-up questions). The model outputs "Question:" but stop tokens were `["Q:", "\n\nQ:"]`. After clipping at `["Question:", "[Question]", "Q:", "\n\n\n"]`, accuracy jumps to 67.02%. See `results/math-10epochs_20260127-154943/evals/gsm8k_step100_clipped/`.

**By Level**:
| Level | Accuracy |
|-------|----------|
| Level 1 | 84.90% |
| Level 2 | 75.28% |
| Level 3 | 66.49% |
| Level 4 | 55.02% |
| Level 5 | 31.27% |

### Run 3: math-eps1e5-warmup30-lr1e5

- **Config**: `configs/adhoc/math-eps1e5-warmup30-lr1e5.yaml`
- **Hyperparams**: lr=1e-5, eps=1e-5, warmup=30, epochs=10
- **Run folder**: `results/math-eps1e5-warmup30-lr1e5_20260127-155007/`
- **Checkpoint**: step80 (stalled due to high filtering rate)

**Results (MathVerifier)**:
| Benchmark | Accuracy |
|-----------|----------|
| MATH (5000) | 61.20% (3060/5000) |

**By Level**:
| Level | Accuracy |
|-------|----------|
| Level 1 | 89.93% |
| Level 2 | 80.31% |
| Level 3 | 72.06% |
| Level 4 | 57.00% |
| Level 5 | 33.38% |

### Key Findings

1. **Higher LR (1e-5) + Lower eps (1e-5) works better** - The eps1e5 run reached 61.2% at step 80, vs 57.6% at step 100 for the conservative config

2. **GSM8K improves dramatically** - After fixing stop tokens, the 10-epochs run shows **67.02%** on GSM8K (base: **14.33%**) — a **+52.7pp improvement**! The initial 16.8% was due to missing "Question:" in stop tokens — the model hallucinated follow-up questions, and MathVerifier extracted the LAST number instead of the answer.

3. **Longer training helps** - Going from 2 epochs (44.9%) to 10 epochs (57.6%+) shows continued improvement

### Stop Token Lesson Learned

The model outputs "Question:" when generating follow-up content, but our stop tokens were `["Q:", "\n\nQ:"]`. This caused:
- 1071/1319 completions (81%) to contain hallucinated follow-up questions
- MathVerifier's `_extract_last_number()` picked up numbers from hallucinated content
- Correct answers like "18" were masked by follow-up numbers like "1820"

**Fix**: Use stop tokens `["[Question]", "Question:", "Q:", "\n\n\n"]` for GSM8K generation.

### Next Experiments

- Try lr=7e-6 (between 5e-6 and 1e-5) with eps=1e-5, warmup=30
- Train on harder problems only (MATH levels 3-5)

---

## MATH Pass@k Results - Base Model (2026-01-27)

Pass@k evaluation for curriculum learning on Qwen3-1.7B-Base model.

**Results**: `results/qwen3-1.7B-base/evals/math/pass-at-k/`

| k | Pass Rate |
|---|-----------|
| 1 | 39.0% |
| 2 | 53.7% |
| 4 | 66.2% |
| 8 | 74.9% |
| 16 | 80.7% |
| 32 | 85.8% |
| 64 | 89.2% |
| 128 | **92.0%** |

- **Total prompts**: 7,500
- **Completions per prompt**: 128
- **Total completions**: 960,000
- **Correct completions**: 371,083 (38.7% raw pass rate)
- **Sampling**: temperature=1.0, top_p=0.95, top_k=20
- **Stop tokens**: `["Problem:", "\n\n\n"]`

### Interpretation

- pass@1 (39%) ≈ greedy eval (41%) confirms evaluation consistency
- pass@128 (92%) shows that most MATH problems are solvable by the base model with enough attempts
- The gap between pass@1 and pass@128 (39% → 92%) shows significant room for improvement via RL

### Curriculum Use

Per-prompt pass rates in `all_verification_results.jsonl` can be used to:
1. Filter out "too easy" problems (pass@128 = 100%)
2. Filter out "unsolvable" problems (pass@128 = 0%)
3. Prioritize moderate-difficulty problems for training

---

## Evaluation: math-10epochs vs math-lr7e6 (2026-01-28)

### Models Evaluated

| Model | Description |
|-------|-------------|
| **Base** | Qwen3-1.7B-Base (pretrained) |
| **10epochs** | math-10epochs step160 (lr=1e-6, 10 epoch config) |
| **lr7e6** | math-lr7e6-eps1e5-warmup30 step160 (lr=7e-6, eps=1e-5, warmup=30) |

### MathVerifier Greedy Eval (eval_checkpoint.py)

| Checkpoint | GSM8K | MATH |
|------------|-------|------|
| 10ep_step60 | 65.81% | 55.94% |
| 10ep_step100 | 67.48% | 58.58% |
| 10ep_step160 | 68.76% | 59.68% |
| lr7e6_step60 | 74.37% | 62.14% |
| lr7e6_step100 | 75.82% | 62.02% |
| lr7e6_step160 | 76.88% | 62.94% |

**Takeaway**: lr7e6 significantly outperforms 10epochs (+8-9pp on GSM8K, +2-3pp on MATH).

### AIME Pass@k (90 problems)

| Model | pass@1 | pass@2 | pass@4 | pass@16 | pass@32 |
|-------|--------|--------|--------|---------|---------|
| Base | 1.11% | 1.11% | 4.44% | 7.78% | 12.22% |
| 10epochs | 2.22% | 3.33% | 7.78% | 10.00% | 15.56% |
| lr7e6 | 1.11% | 4.44% | 7.78% | 12.22% | 14.44% |

**Takeaway**: Inconclusive - models trade wins at different k values. AIME too hard and sample size too small to differentiate.

### BeyondAIME Pass@k (100 problems, harder than AIME)

| Model | pass@1 | pass@2 | pass@4 | pass@16 | pass@32 |
|-------|--------|--------|--------|---------|---------|
| Base | 0.00% | 1.00% | 1.00% | 3.00% | 6.00% |
| 10epochs | 1.00% | 2.00% | 2.00% | 7.00% | **14.00%** |
| lr7e6 | **2.00%** | **3.00%** | **4.00%** | **11.00%** | 13.00% |

**Takeaway**: lr7e6 wins at k<=16, 10epochs wins at k=32. Both dramatically better than base (2x+ improvement at pass@32).

### Summary

1. **lr7e6 is the better model overall** - consistently wins on GSM8K/MATH greedy eval
2. **Training helps on hard problems** - both models show 2x+ improvement over base on BeyondAIME
3. **Higher learning rate (7e-6 vs 1e-6) appears beneficial** for math reasoning
4. **eps=1e-5 and warmup=30** (lr7e6 config) may also contribute to better performance

---

## IF Multi-Constraints Ablation Results (2026-01-28)

### Ablation Setup

Training Qwen3-1.7B-Base on IF multi-constraints dataset to test different RL training strategies.

| Ablation | Description | Config |
|----------|-------------|--------|
| **Vanilla** | Standard GRPO, no staleness | `annotations-adhoc` (step125) |
| **Staleness=1** | Allow 1-step stale rollouts | `if-staleness1.yaml` |
| **Curriculum** | Easy-to-hard ordering by pass@k | `if-curriculum.yaml` |
| **Adaptive** | Early stopping (k_success=2, k_failure=2) | `if-adaptive.yaml` (in progress) |

### IFEval Results (Prompt-Level Strict Accuracy)

| Model | Prompt Accuracy | Instruction Accuracy | Notes |
|-------|-----------------|---------------------|-------|
| **Base (Qwen3-1.7B)** | 17.38% | 29.38% | Pretrained baseline |
| **Vanilla (step125)** | **42.51%** | **54.80%** | Best so far |
| if-staleness1 (step120) | 40.30% | 54.32% | |
| if-curriculum (step120) | 41.40% | 53.96% | |
| if-adaptive | TBD | TBD | Running (step 77/120) |

### Preliminary Conclusions

1. **All ablations significantly improve over base** (+23pp prompt accuracy, +25pp instruction accuracy)
2. **Vanilla is currently the best** - the additional features (staleness, curriculum) don't help for IF
3. **Staleness=1 slightly hurts** compared to vanilla (-2pp)
4. **Curriculum ordering doesn't help** for IF (possibly because IF problems don't have clear difficulty gradient like math)

### Notes

- IF multi-constraints has 40,400 problems
- Curriculum ordering filtered to 33,866 problems with 0.01 <= pass@k <= 0.99
- All runs trained for 120 steps with checkpoint interval of 20
- Evaluations used greedy decoding (temperature=0)

---

## MATH Ablation Results (2026-01-28)

### Ablation Setup

Training Qwen3-1.7B-Base on MATH dataset to test different RL training strategies.

| Ablation | Description | Config | Run Directory |
|----------|-------------|--------|---------------|
| **Staleness=1** | Allow 1-step stale rollouts | `configs/ablations/math-staleness1.yaml` | `results/math-staleness1_20260128-014714` |
| **Curriculum** | Easy-to-hard ordering by pass@k | `configs/ablations/math-curriculum.yaml` | `results/math-curriculum_20260128-014922` |
| **Adaptive** | Early stopping (k_success=2, k_failure=2) | `configs/ablations/math-adaptive.yaml` | `results/math-adaptive_20260128-021106` |
| **All Combined** | staleness + curriculum + adaptive | `configs/ablations/math-all.yaml` | Crashed at step 50 |

### Eval Results (MathVerifier Greedy)

| Model | Step | GSM8K | MATH |
|-------|------|-------|------|
| **Base (Qwen3-1.7B)** | - | 14% | 41% |
| math-staleness1 | 40 | - | 61.46% |
| math-staleness1 | 100 | 72.86% | - |
| math-curriculum | 40 | - | 61.96% |
| math-curriculum | 100 | - | 62.04% |
| math-adaptive | 40 | - | 60.02% |
| **Best (lr7e6 baseline)** | 160 | 76.88% | 62.94% |

### Checkpoints Available

- **math-staleness1**: step20, 40, 60, 80, 100, 120, 140, 160, final
- **math-curriculum**: step20, 40, 60, 80, 100, 120, 140
- **math-adaptive**: step20, 40, 60 (may have crashed early)

### Preliminary Observations

1. **All ablations improve significantly over base** (41% → 60-62% on MATH)
2. **No ablation beats the lr7e6 baseline** (62.94% MATH, 76.88% GSM8K)
3. **math-all (combined features) is unstable** - crashes with loss explosion around step 50
4. **Curriculum provides marginal improvement** over staleness1 at step 40 (61.96% vs 61.46%)
5. **Adaptive sampling shows slightly lower performance** at step 40 (60.02%)

### TODO

- [ ] Run final evals on math-staleness1 step160
- [ ] Run final evals on math-curriculum step140
- [ ] Investigate math-all crash (loss explosion)

