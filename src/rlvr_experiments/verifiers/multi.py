"""Multi-verifier that dispatches based on problem['verifier_type'].

For mixed dataset training where different samples need different verifiers.
"""

import time
from typing import Any

from .math import MathVerifier
from .gsm8k import GSM8KVerifier
from .minerva_math import MinervaMathVerifier
from .ifeval import IFEvalVerifier
from .if_multi_constraints import IFMultiConstraintsVerifier
from .code import HumanEvalVerifier, MBPPVerifier, APPSVerifier
from .allenai import AllenAIGSM8KVerifier, AllenAIMathVerifier


class MultiVerifier:
    """
    Verifier that dispatches to specialized verifiers based on problem['verifier_type'].

    Supports mixed dataset training where different samples require different
    verification logic (math, code, ifeval, etc.).

    Usage:
        verifier = MultiVerifier()
        # Each problem dict must have 'verifier_type' key
        scores = await verifier.verify_completions(problem, completions)
    """

    def __init__(
        self,
        math_timeout: float = 5.0,
        math_max_workers: int = 4,
        ifeval_timeout: float = 1.0,
        gsm8k_format_weight: float = 0.0,
    ):
        """Initialize all sub-verifiers.

        Args:
            math_timeout: Timeout for math verification (sympy can hang)
            math_max_workers: Number of workers for parallel math verification
            ifeval_timeout: Timeout for ifeval verification
            gsm8k_format_weight: Weight for GSM8K format reward (0.0 to 1.0).
                If > 0, gives partial credit for correct format even with wrong answer.
        """
        self._verifiers: dict[str, Any] = {
            "math": MathVerifier(timeout=math_timeout, max_workers=math_max_workers),
            # Match gsm8k-only runs (MathVerifier via math_verify) for identical reward routing.
            "gsm8k": MathVerifier(timeout=math_timeout, max_workers=math_max_workers),
            "minerva_math": MinervaMathVerifier(),
            "ifeval": IFEvalVerifier(timeout=ifeval_timeout),
            "if_multi_constraints": IFMultiConstraintsVerifier(timeout=ifeval_timeout),
            "humaneval": HumanEvalVerifier(),
            "mbpp": MBPPVerifier(),
            "apps": APPSVerifier(),
            # AllenAI verifiers (for allenai/RLVR-GSM-MATH-IF-Mixed-Constraints dataset)
            "allenai_gsm8k": AllenAIGSM8KVerifier(),
            "allenai_math": AllenAIMathVerifier(use_sympy=False),  # Disable sympy for speed
        }

    def _get_verifier(self, verifier_type: str) -> Any:
        """Get verifier by type, raising if unknown."""
        if verifier_type not in self._verifiers:
            raise ValueError(
                f"Unknown verifier_type: {verifier_type}. "
                f"Available: {list(self._verifiers.keys())}"
            )
        return self._verifiers[verifier_type]

    async def verify_completions(
        self, problem: dict, completions: list[str], **kwargs
    ) -> list[float]:
        """Verify N completions for one problem.

        Dispatches to the appropriate verifier based on problem['verifier_type'].
        """
        verifier_type = problem.get("verifier_type")
        if not verifier_type:
            raise ValueError("problem must have 'verifier_type' key for MultiVerifier")

        verifier = self._get_verifier(verifier_type)
        return await verifier.verify_completions(problem, completions, **kwargs)

    async def verify_batch(
        self, problems: list[dict], completions: list[str]
    ) -> tuple[list[float], list[float]]:
        """Verify a batch. Returns (scores, durations_ms).

        Each problem is dispatched to its appropriate verifier.
        """
        scores = []
        durations = []

        for p, c in zip(problems, completions):
            verifier_type = p.get("verifier_type")
            if not verifier_type:
                scores.append(0.0)
                durations.append(0.0)
                continue

            verifier = self._get_verifier(verifier_type)
            t0 = time.perf_counter()

            # Use single-item verify for math/ifeval/AllenAI, verify_completions for code
            if verifier_type in ("math", "hendrycks_math"):
                score = verifier.verify(c, p["answer"])
            elif verifier_type in ("ifeval", "if_multi_constraints", "allenai_gsm8k", "allenai_math"):
                score = verifier.verify(c, p.get("ground_truth", ""))
            elif verifier_type in ("gsm8k", "minerva_math"):
                # These verifiers return list[float], not tuple
                scores_result = await verifier.verify_completions(p, [c])
                score = scores_result[0] if scores_result else 0.0
            else:
                # Code verifiers - verify_completions returns (scores, durations)
                scores_result, _ = await verifier.verify_completions(p, [c])
                score = scores_result[0] if scores_result else 0.0

            dur_ms = (time.perf_counter() - t0) * 1000
            scores.append(score)
            durations.append(dur_ms)

        return scores, durations

    async def verify_batch_with_timing(
        self, problems: list[dict], completions: list[str]
    ) -> tuple[list[float], list[float], list[tuple[float, float]]]:
        """Verify a batch with timing spans. Returns (scores, durations_ms, timing_spans)."""
        scores = []
        durations = []
        timing_spans = []
        offset = 0.0

        for p, c in zip(problems, completions):
            verifier_type = p.get("verifier_type")
            if not verifier_type:
                scores.append(0.0)
                durations.append(0.0)
                timing_spans.append((offset, 0.0))
                continue

            verifier = self._get_verifier(verifier_type)
            t0 = time.perf_counter()

            # Use single-item verify for math/ifeval/AllenAI, verify_completions for code
            if verifier_type in ("math", "hendrycks_math"):
                score = verifier.verify(c, p["answer"])
            elif verifier_type in ("ifeval", "if_multi_constraints", "allenai_gsm8k", "allenai_math"):
                score = verifier.verify(c, p.get("ground_truth", ""))
            elif verifier_type in ("gsm8k", "minerva_math"):
                # These verifiers return list[float], not tuple
                scores_result = await verifier.verify_completions(p, [c])
                score = scores_result[0] if scores_result else 0.0
            else:
                # Code verifiers - verify_completions returns (scores, durations)
                scores_result, _ = await verifier.verify_completions(p, [c])
                score = scores_result[0] if scores_result else 0.0

            dur_ms = (time.perf_counter() - t0) * 1000
            scores.append(score)
            durations.append(dur_ms)
            timing_spans.append((offset, dur_ms))
            offset += dur_ms

        return scores, durations, timing_spans
