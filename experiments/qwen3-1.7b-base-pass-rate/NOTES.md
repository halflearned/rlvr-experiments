# Qwen3-1.7B-Base Pass@k Evaluation

**Date:** 2026-01-15
**Model:** Qwen3-1.7B-Base (`/efs/rlvr-experiments/assets/hf/Qwen3-1.7B-Base`)

## Summary

Evaluated pass@k rates on TRAIN splits of GSM8k, MATH, and MBPP datasets to measure the model's coverage - how many problems can be solved with enough sampling attempts.

| Dataset | Prompts | Pass@1 | Pass@k | k |
|---------|---------|--------|--------|---|
| GSM8k | 7,473 | 74.95% | 99.81% | 1024 |
| MATH (L3-5) | 5,584 | 29.67% | 84.0% | 512 |
| MBPP | 374 | 26.58% | 89.6% | 512 |

## Sampling Configuration

All evaluations used the following sampling parameters (vLLM SamplingParams):

| Parameter | GSM8k | MATH | MBPP |
|-----------|-------|------|------|
| temperature | 0.6 | 0.6 | 0.6 |
| top_p | 0.95 | 0.95 | 0.95 |
| top_k | 20 | 20 | 20 |
| max_tokens | 512 | 2048 | 512 |
| n (completions/prompt) | 1024 | 512 | 512 |

**Generation setup:**
- 8x data parallelism (8 vLLM replicas, 1 GPU each)
- bfloat16 precision
- prefix caching enabled

## Dataset Details

### GSM8k (Grade School Math)
- **Split:** train
- **Prompts:** 7,473
- **Total completions:** 7,652,352 (1024 per prompt)
- **Prompt format:**
  ```
  Solve the following math problem. Put the final answer in \boxed{}.

  Problem: {question}

  Let's think step by step.
  ```
- **Verifier:** math_verify (sympy-based symbolic comparison)

### MATH (Hendrycks MATH)
- **Split:** train
- **Levels:** 3, 4, 5 only
- **Prompts:** 5,584
- **Total completions:** 2,859,008 (512 per prompt)
- **Level breakdown:**
  - Level 3: 41.7% pass@1, 92% pass@512 (1510 problems)
  - Level 4: 28.5% pass@1, 89% pass@512 (1770 problems)
  - Level 5: 16.4% pass@1, 79% pass@512 (2304 problems)
- **Prompt format:** Same as GSM8k
- **Verifier:** math_verify (sympy-based symbolic comparison)

### MBPP (Mostly Basic Python Programs)
- **Split:** train
- **Prompts:** 374
- **Total completions:** 191,488 (512 per prompt)
- **Prompt format:**
  ```
  Write a Python function to solve the following problem.

  {problem_description}

  {first_test_case}

  ```python
  ```
- **Verifier:** Docker-isolated Python execution with test assertions

## Result Files

| Dataset | Results File | Completions File |
|---------|-------------|------------------|
| GSM8k | `gsm8k_Qwen3-1.7B-Base_20260114_131659.json` (893MB) | (inline) |
| MATH | `math_L3-4-5_Qwen3-1.7B-Base_20260115_133255.json` (348MB) | `math_L3-4-5_Qwen3-1.7B-Base_completions_20260115_083136.json` (3.0GB) |
| MBPP | `mbpp_Qwen3-1.7B-Base_20260115_160349.json` (23MB) | `mbpp_Qwen3-1.7B-Base_completions_20260115_084337.json` (120MB) |

## Key Observations

1. **GSM8k is nearly saturated:** 99.81% of problems solvable with 1024 samples - only 14 prompts had 0 correct completions.

2. **MATH difficulty scales as expected:** Harder levels have lower pass rates, but even Level 5 problems are 79% solvable with 512 samples.

3. **MBPP has harder tail:** 10.4% of problems (39) had 0/512 correct completions, suggesting some problems may require different approaches or have issues.

4. **Truncation rate:** GSM8k/MBPP had <3% truncated completions, MATH had ~10% (longer reasoning needed for harder problems).

## Reproduction

```bash
# GSM8k
python scripts/eval_pass_rate.py \
    --dataset gsm8k --split train --num-prompts 0 \
    --n 1024 --temperature 0.6 --top-p 0.95 --top-k 20 \
    --max-tokens 512

# MATH (Levels 3-5)
python scripts/eval_pass_rate.py \
    --dataset math --split train --levels 3,4,5 --num-prompts 0 \
    --n 512 --temperature 0.6 --top-p 0.95 --top-k 20 \
    --max-tokens 2048

# MBPP
python scripts/eval_pass_rate.py \
    --dataset mbpp --split train --num-prompts 0 \
    --n 512 --temperature 0.6 --top-p 0.95 --top-k 20 \
    --max-tokens 512
```
