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
