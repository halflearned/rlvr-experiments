# Math Dataset Training vs Evaluation Formats

This document summarizes the prompt formats and extraction patterns used across our training and the lm_eval evaluation tasks.

## Summary Table

| | GSM8K (our training) | MATH (our training) | MATH (our eval) | gsm8k_cot (lm_eval) | gsm8k (lm_eval) |
|---|---|---|---|---|---|
| **Prompt** | `Q: X\nA:` | `Question: X\nAnswer:` | `Question: X\nAnswer:` | `Q: X\nA:` | `Question: X\nAnswer:` |
| **Source** | [data.py:90](../../src/rlvr_experiments/data.py#L90) | [data.py:328](../../src/rlvr_experiments/data.py#L328) | [eval_math_qwen_style.py:50](eval_math_qwen_style.py#L50) | gsm8k-cot.yaml:4-6 | gsm8k.yaml:10 |
| **Extraction** | `The answer is X.` (strict) | `\boxed{}` then `The answer is X` | `\boxed{}` then `The answer is X` | `The answer is (-?[0-9.,]+).` | `#### (-?[0-9.,]+)` |
| **Source** | [gsm8k.py:22-27](../../src/rlvr_experiments/verifiers/gsm8k.py#L22-L27) | [hendrycks_math.py:24-48](../../src/rlvr_experiments/verifiers/hendrycks_math.py#L24-L48) | [eval_math_qwen_style.py:57-85](eval_math_qwen_style.py#L57-L85) | gsm8k-cot.yaml:49 | gsm8k.yaml:36 |
| **Flex extraction** | N/A | N/A | N/A | `(-?[$0-9.,]{2,})\|(-?[0-9]+)` | `(-?[$0-9.,]{2,})\|(-?[0-9]+)` |
| **Source** | | | | gsm8k-cot.yaml:55 | gsm8k.yaml:42 |

## Key Findings

### GSM8K is now aligned

After adding `GSM8KVerifier`:
- Training prompt: `Q: X\nA:` matches `gsm8k_cot` eval
- Training extraction: `The answer is X.` matches `gsm8k_cot` strict extraction
- Verifier type: `gsm8k`

### MATH is aligned

After adding `HendrycksMathVerifier`:
- Training prompt: `Question: X\nAnswer:` matches eval
- Training extraction: `\boxed{}` then `The answer is X` matches eval
- Verifier type: `hendrycks_math`

## Verifier Types

| verifier_type | Used by | Extraction method |
|---------------|---------|-------------------|
| `gsm8k` | GSM8K | `The answer is X.` pattern - matches gsm8k_cot strict |
| `hendrycks_math` | MATH | `\boxed{}` first, then `The answer is X` fallback - matches Qwen eval |
| `math` | (deprecated) | `math_verify` package (LatexExtractionConfig + ExprExtractionConfig) - lenient |

## lm_eval Task Details

### gsm8k_cot
- Prompt: `Q: X\nA:`
- Few-shot examples show CoT reasoning ending with `The answer is X.`
- Strict: `The answer is (-?[0-9.,]+).`
- Flex: Last number in response

### gsm8k
- Prompt: `Question: X\nAnswer:`
- Few-shot examples from dataset ending with `#### X`
- Strict: `#### (-?[0-9.,]+)`
- Flex: Last number in response
