"""
Verifier for MATH using AllenAI flex extraction.

Matches the good run reward strategy:
- boxed answers
- Minerva-style "Final Answer" extraction
- last-$ fallback
with Hendrycks-style equivalence (no sympy by default).
"""

import time

from .allenai import AllenAIMathVerifier


class MinervaMathVerifier:
    """
    MATH verifier aligned to AllenAI's flex extraction.

    Args:
        use_sympy: If True, enable sympy equivalence checks (disabled by default
            to match the good run's behavior).
    """

    def __init__(self, use_sympy: bool = False):
        self._verifier = AllenAIMathVerifier(use_sympy=use_sympy)

    def verify(self, response: str, target: str) -> float:
        """Return 1.0 if response matches target, 0.0 otherwise."""
        return self._verifier.verify(response, target)

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
