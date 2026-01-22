"""
Verifier for GSM8K with lm_eval-compatible extraction modes.

Supports both strict-match and flexible-extract patterns from lm_eval's gsm8k_cot task,
plus an AllenAI-style "last number" extraction for maximum recall.
"""

import re
import time
from typing import Optional


# =============================================================================
# Extraction Functions (matching lm_eval's gsm8k_cot filters)
# =============================================================================

def extract_strict(text: str) -> Optional[str]:
    """Extract answer using lm_eval's strict-match pattern.

    Pattern: The answer is (\-?[0-9\.\,]+).
    Requires the exact format "The answer is X."
    """
    match = re.search(r"The answer is (\-?[0-9\.\,]+)\.", text)
    if match:
        return match.group(1)
    return None


def extract_flexible(text: str) -> Optional[str]:
    """Extract answer using lm_eval's flexible-extract pattern.

    Pattern: (-?[$0-9.,]{2,})|(-?[0-9]+) with group_select=-1 (last match)
    Matches dollar amounts, decimals, or plain integers.
    """
    # lm_eval's pattern: (-?[$0-9.,]{2,})|(-?[0-9]+)
    matches = re.findall(r"(-?[$0-9.,]{2,})|(-?[0-9]+)", text)
    if matches:
        # Each match is a tuple (group1, group2), take the non-empty one from last match
        last_match = matches[-1]
        return last_match[0] if last_match[0] else last_match[1]
    return None


def extract_last_number(text: str) -> Optional[str]:
    """Extract the last number from the response (AllenAI style).

    Most lenient extraction - finds any number in the response.
    Handles negative numbers and decimals.
    """
    # Remove commas between digits (e.g., "1,234" -> "1234")
    text = re.sub(r"(\d),(\d)", r"\1\2", text)
    # Match integers and decimals, with optional sign
    numbers = re.findall(r"[-+]?(?:\d+\.\d+|\d+)", text)
    if numbers:
        return numbers[-1]
    return None


# =============================================================================
# Normalization and Comparison
# =============================================================================

def normalize_number(s: str) -> Optional[str]:
    """Normalize a number string for comparison.

    Matches lm_eval's regexes_to_ignore: [',', '\$', '(?s).*#### ', '\.$']
    """
    if s is None:
        return None
    # Remove commas and dollar signs
    s = s.replace(",", "").replace("$", "")
    # Remove trailing period if present
    s = s.rstrip(".")
    # Strip whitespace
    s = s.strip()
    return s


def is_equiv(pred: str, gold: str) -> bool:
    """Check if two GSM8K answers are equivalent.

    GSM8K answers are integers, but we handle decimals like "6.0" == "6".
    """
    if pred is None or gold is None:
        return False

    pred_norm = normalize_number(pred)
    gold_norm = normalize_number(gold)

    if pred_norm is None or gold_norm is None:
        return False

    # Direct string match after normalization
    if pred_norm == gold_norm:
        return True

    # Try numeric comparison (handles "6.0" == "6", "18.00" == "18")
    try:
        pred_num = float(pred_norm)
        gold_num = float(gold_norm)
        # Compare as integers if both are whole numbers
        if pred_num == int(pred_num) and gold_num == int(gold_num):
            return int(pred_num) == int(gold_num)
        return pred_num == gold_num
    except (ValueError, TypeError):
        pass

    return False


# =============================================================================
# Verifier Class
# =============================================================================

class GSM8KVerifier:
    """
    Verifier for GSM8K with configurable extraction modes.

    Args:
        mode: Extraction mode - one of:
            - "strict": lm_eval's strict-match (requires "The answer is X.")
            - "flexible": lm_eval's flexible-extract (dollar amounts, decimals, integers)
            - "last_number": AllenAI-style last number extraction (most lenient)
            - "any": Try strict first, then flexible, then last_number (default)
        format_weight: Partial credit for format match but wrong answer (default 0.0)
    """

    def __init__(self, mode: str = "any", format_weight: float = 0.0):
        self.mode = mode
        self.format_weight = format_weight

    def extract(self, response: str) -> Optional[str]:
        """Extract answer from response using configured mode."""
        if self.mode == "strict":
            return extract_strict(response)
        elif self.mode == "flexible":
            return extract_flexible(response)
        elif self.mode == "last_number":
            return extract_last_number(response)
        elif self.mode == "any":
            # Try each method in order of strictness
            result = extract_strict(response)
            if result is not None:
                return result
            result = extract_flexible(response)
            if result is not None:
                return result
            return extract_last_number(response)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

    def verify(self, response: str, target: str) -> float:
        """Return score based on extraction and correctness.

        Args:
            response: Model's generated response
            target: Gold answer (just the number, e.g., "42")

        Returns:
            - 0.0 if no extraction possible
            - format_weight if extracted but wrong answer
            - 1.0 if extracted and correct answer
        """
        try:
            pred = self.extract(response)
            if pred is None:
                return 0.0

            if is_equiv(pred, target):
                return 1.0
            return self.format_weight
        except Exception:
            return 0.0

    def verify_detailed(self, response: str, target: str) -> dict:
        """Return detailed verification results for analysis.

        Returns dict with:
            - correct: bool
            - extracted: str or None
            - target: str
            - strict_match: str or None
            - flexible_match: str or None
            - last_number_match: str or None
        """
        strict = extract_strict(response)
        flexible = extract_flexible(response)
        last_num = extract_last_number(response)

        pred = self.extract(response)
        correct = is_equiv(pred, target) if pred else False

        return {
            "correct": correct,
            "extracted": pred,
            "target": target,
            "strict_match": strict,
            "flexible_match": flexible,
            "last_number_match": last_num,
        }

    async def verify_completions(self, problem: dict, completions: list[str], **kwargs) -> list[float]:
        """Verify N completions for one problem. Returns list of scores."""
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
