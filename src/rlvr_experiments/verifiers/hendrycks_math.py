"""
Verifier for Hendrycks MATH benchmark using Qwen's methodology.

This verifier matches the extraction and verification logic used in Qwen's
MATH evaluation, which achieves ~43% on the base model (vs ~17% with lm_eval's
hendrycks_math task).

Key differences from the generic MathVerifier:
1. Extraction: \boxed{} primary, "The answer is X" fallback (exact Qwen patterns)
2. Verification: String normalization + sympy symbolic equivalence
"""

import re
import time
from typing import Optional


def extract_boxed(text: str) -> Optional[str]:
    """Extract content from \\boxed{...}, handling nested braces.

    Takes the LAST \\boxed{} occurrence (rightmost), which is typically
    the final answer after chain-of-thought reasoning.
    """
    idx = text.rfind("\\boxed")
    if idx >= 0:
        i = idx
        while i < len(text) and text[i] != '{':
            i += 1
        if i < len(text):
            # Find matching closing brace
            brace_count = 0
            start = i
            while i < len(text):
                if text[i] == '{':
                    brace_count += 1
                elif text[i] == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        return text[start + 1:i]
                i += 1

    # Fallback: try "The answer is X" pattern (exact Qwen pattern)
    match = re.search(r"[Tt]he answer is[:\s]*\$?([^\$\n]+)\$?", text)
    if match:
        answer = match.group(1).strip()
        # Clean up common suffixes
        answer = re.sub(r"\.?\s*$", "", answer)
        return answer

    return None


def normalize_answer(answer: str) -> str:
    """Normalize answer for comparison."""
    if answer is None:
        return ""
    # Basic normalization
    answer = answer.strip()
    answer = answer.replace(" ", "")
    answer = answer.replace("\\left", "")
    answer = answer.replace("\\right", "")
    answer = answer.replace("\\!", "")
    answer = answer.replace("\\,", "")
    answer = answer.replace("\\;", "")
    answer = answer.replace("\\quad", "")
    answer = answer.replace("\\qquad", "")
    answer = answer.replace("dfrac", "frac")
    answer = answer.replace("tfrac", "frac")
    return answer


def is_equiv(pred: str, gold: str) -> bool:
    """Check if two answers are equivalent.

    Uses string normalization first, then falls back to sympy
    symbolic comparison via latex2sympy2_extended (same lib as math_verify).
    """
    if pred is None or gold is None:
        return False

    pred_norm = normalize_answer(pred)
    gold_norm = normalize_answer(gold)

    # Direct string match
    if pred_norm == gold_norm:
        return True

    # Try sympy comparison using latex2sympy2_extended (same lib as math_verify)
    try:
        from latex2sympy2_extended import latex2sympy
        from sympy import simplify

        pred_expr = latex2sympy(pred_norm)
        gold_expr = latex2sympy(gold_norm)

        diff = simplify(pred_expr - gold_expr)
        if diff == 0:
            return True
    except Exception:
        pass

    return False


class HendrycksMathVerifier:
    """
    Verifier for Hendrycks MATH benchmark using Qwen's methodology.

    This matches the exact extraction and verification used in our
    eval_math_qwen_style.py script, ensuring training rewards align
    with evaluation metrics.
    """

    def __init__(self):
        pass

    def verify(self, response: str, target: str) -> float:
        """Return 1.0 if response matches target, 0.0 otherwise.

        Args:
            response: Model's generated response
            target: Gold solution (we extract \boxed{} from this too)
        """
        try:
            # Extract answers from both response and target
            pred = extract_boxed(response)
            gold = extract_boxed(target)

            if pred is None or gold is None:
                return 0.0

            return 1.0 if is_equiv(pred, gold) else 0.0
        except Exception:
            return 0.0

    async def verify_completions(self, problem: dict, completions: list[str], **kwargs) -> list[float]:
        """Verify N completions for one problem. Returns list of scores."""
        # Target is the gold solution containing \boxed{answer}
        target = problem.get("answer", "")
        scores = [self.verify(c, target) for c in completions]
        return scores

    async def verify_batch(self, problems: list[dict], completions: list[str]) -> tuple[list[float], list[float]]:
        """Verify a batch. Returns (scores, durations_ms)."""
        scores = []
        durations = []
        for p, c in zip(problems, completions):
            t0 = time.perf_counter()
            score = self.verify(c, p["answer"])
            dur = (time.perf_counter() - t0) * 1000
            scores.append(score)
            durations.append(dur)
        return scores, durations

    async def verify_batch_with_timing(self, problems: list[dict], completions: list[str]) -> tuple[list[float], list[float], list[tuple[float, float]]]:
        """Verify a batch with timing spans. Returns (scores, durations_ms, timing_spans)."""
        scores = []
        durations = []
        timing_spans = []
        offset = 0.0
        for p, c in zip(problems, completions):
            t0 = time.perf_counter()
            score = self.verify(c, p["answer"])
            dur_ms = (time.perf_counter() - t0) * 1000
            scores.append(score)
            durations.append(dur_ms)
            timing_spans.append((offset, dur_ms))
            offset += dur_ms
        return scores, durations, timing_spans
