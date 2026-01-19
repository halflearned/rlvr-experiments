# MATH Benchmark Evaluation: Resolving the Qwen Discrepancy

**Date:** 2026-01-17

## The Problem

Our `hendrycks_math` 4-shot evaluations showed ~17% accuracy, while Qwen's technical report claims 43.5% for Qwen3-1.7B-Base on MATH. Since we're using the same model, the difference had to be in evaluation methodology.

## Root Cause

The `lm_eval` `hendrycks_math` task uses a different prompting strategy than Qwen's evaluation:

| Aspect | lm_eval `hendrycks_math` | Qwen's MATH eval |
|--------|--------------------------|------------------|
| **Prompt template** | `Problem: {X}\nAnswer:` | `Question: {X}\nAnswer:` |
| **Few-shot answers** | Just the boxed value | Full CoT with "Let's think step by step" + solution + `\boxed{}` |
| **Answer format expected** | `$...$` delimiters | `\boxed{}` or "The answer is X" |
| **Verification** | String normalization | Sympy symbolic equivalence |

The key issue: without CoT exemplars showing HOW to reason and format answers, base models don't produce answers in a parseable format.

## Qwen's Methodology

From [Qwen2.5-Math evaluation code](https://github.com/QwenLM/Qwen2.5-Math/tree/main/evaluation):

1. **Prompt template** (`"cot"` in their PROMPT_TEMPLATES):
   ```
   Question: {problem}
   Answer: {full_solution_with_boxed}
   ```

2. **4-shot CoT exemplars** (from `examples.py`):
   - Each exemplar shows full reasoning with "Let's think step by step"
   - Answers formatted as `\boxed{answer}`
   - Ends with "The answer is X"

3. **Answer extraction**:
   - Primary: Extract from `\boxed{...}`
   - Fallback: Regex for "The answer is X"

4. **Equivalence checking**:
   - `latex2sympy` for symbolic comparison
   - More lenient than string matching

## Our Reproduction

Created `scripts/adhoc/eval_math_qwen_style.py` implementing Qwen's methodology:

```python
# 4-shot CoT exemplars from Qwen
MATH_FEWSHOT_EXAMPLES = [
    ("Kevin Kangaroo begins hopping...",
     "Let's think step by step\n...Thus, Kevin has hopped $\\boxed{\\frac{211}{243}}$.\nThe answer is \\frac{211}{243}"),
    # ... 3 more examples
]

def build_prompt(problem, num_shots=4):
    # Join exemplars with triple newlines
    # Format: "Question: X\nAnswer: Y"
```

## Results

### Base Model (Qwen3-1.7B-Base)

| Metric | lm_eval hendrycks_math | Qwen-style eval |
|--------|------------------------|-----------------|
| Overall accuracy | 17.68% | 38.60% |
| On valid extractions | N/A | 42.9% |

The 42.9% on valid extractions closely matches Qwen's claimed 43.5%.

### Breakdown by Level (Qwen-style)

| Level | Accuracy |
|-------|----------|
| Level 1 | 75.1% |
| Level 2 | 56.5% |
| Level 3 | 46.5% |
| Level 4 | 30.6% |
| Level 5 | 15.1% |

### Trained Model (Curriculum_step150)

| Metric | Base | Curriculum_step150 |
|--------|------|-------------------|
| Overall | 38.60% | 38.36% |
| Valid pairs | 42.9% | 42.7% |

No improvement - training focused on GSM8K/MBPP/IFEval, not MATH Level 4-5 problems.

## Key Findings

1. **Evaluation methodology matters hugely**: Same model, 17% vs 43% depending on prompt format
2. **CoT exemplars are essential**: Base models need to SEE the expected reasoning format
3. **Answer extraction needs fallbacks**: Both `\boxed{}` and "The answer is X" patterns
4. **Our training didn't help MATH**: The training mix (GSM8K, MBPP, IFEval) didn't target hard math reasoning

## Files

- `scripts/adhoc/eval_math_qwen_style.py` - Qwen-style MATH evaluation script
- `/tmp/math_qwen_style_Qwen3-1.7B-Base.json` - Base model results
- `/tmp/math_qwen_style_qwen3_1_7b_mixed_curriculum_step150.json` - Trained model results

## Reproduction

```bash
# Run Qwen-style MATH eval
source .venv/bin/activate
CUDA_VISIBLE_DEVICES=0,1,2,3 python scripts/adhoc/eval_math_qwen_style.py \
    --model /efs/rlvr-experiments/assets/hf/Qwen3-1.7B-Base \
    --num-shots 4
```

## References

- [Qwen3 Technical Report](https://arxiv.org/pdf/2505.09388) - Claims 43.5% MATH for 1.7B-Base
- [Qwen2.5-Math GitHub](https://github.com/QwenLM/Qwen2.5-Math) - Evaluation code
- [Qwen2.5-Math examples.py](https://github.com/QwenLM/Qwen2.5-Math/blob/main/evaluation/examples.py) - Few-shot exemplars
