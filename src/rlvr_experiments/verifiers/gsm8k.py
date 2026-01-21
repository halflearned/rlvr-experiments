"""
Verifier for GSM8K using AllenAI-style extraction.

This verifier extracts the last number in the response (after removing commas),
matching the AllenAI RLVR GSM8K reward strategy used in the good run.
"""

import re
import time
from typing import Optional


def extract_answer(text: str) -> Optional[str]:
    """Extract the last number from the response (AllenAI GSM8K style)."""
    # Remove commas between digits (e.g., "1,234" -> "1234")
    text = re.sub(r"(\\d),(\\d)", r"\\1\\2", text)
    numbers = re.findall(r"[-+]?\\d*\\.\\d+|\\d+", text)
    if numbers:
        return numbers[-1]
    return None


def normalize_number(s: str) -> Optional[str]:
    """Normalize a number string for comparison."""
    if s is None:
        return None
    # Remove commas and dollar signs (gsm8k_cot regexes_to_ignore)
    s = s.replace(",", "").replace("$", "")
    # Remove trailing period if present
    s = s.rstrip(".")
    # Strip whitespace
    s = s.strip()
    return s


def is_equiv(pred: str, gold: str) -> bool:
    """Check if two GSM8K answers are equivalent.

    GSM8K answers are always integers, so we compare normalized strings.
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

    # Try numeric comparison (handles "6.0" == "6")
    try:
        pred_num = float(pred_norm)
        gold_num = float(gold_norm)
        # GSM8K answers are integers, so check if they're equal as integers
        if pred_num == int(pred_num) and gold_num == int(gold_num):
            return int(pred_num) == int(gold_num)
        return pred_num == gold_num
    except (ValueError, TypeError):
        pass

    return False


class GSM8KVerifier:
    """
    Verifier for GSM8K using AllenAI last-number extraction.

    Args:
        format_weight: Optional format reward (0.0 to 1.0). If > 0, gives partial
            credit for having any numeric output even with wrong answer.
            Default 0.0 means strict correctness only (matches good run).
    """

    def __init__(self, format_weight: float = 0.0):
        self.format_weight = format_weight

    def verify(self, response: str, target: str) -> float:
        """Return score based on format and correctness.

        Args:
            response: Model's generated response
            target: Gold answer (just the number, e.g., "42")

        Returns:
            - 0.0 if no format match
            - format_weight if format match but wrong answer
            - 1.0 if format match and correct answer
        """
        try:
            pred = extract_answer(response)
            if pred is None:
                return 0.0

            if is_equiv(pred, target):
                return 1.0
            return self.format_weight
        except Exception:
            return 0.0

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
