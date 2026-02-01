"""Multi-verifier that dispatches based on problem['verifier_type'].

For mixed dataset training where different samples need different verifiers.
"""

import asyncio
import time
from collections import defaultdict
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
        max_concurrent: int = 4,
    ):
        """Initialize all sub-verifiers.

        Args:
            math_timeout: Timeout for math verification (sympy can hang)
            math_max_workers: Number of workers for parallel math verification
            ifeval_timeout: Timeout for ifeval verification
            gsm8k_format_weight: Weight for GSM8K format reward (0.0 to 1.0).
                If > 0, gives partial credit for correct format even with wrong answer.
            max_concurrent: Max concurrent Docker containers for code verifiers.
        """
        self._verifiers: dict[str, Any] = {
            "math": MathVerifier(timeout=math_timeout, max_workers=math_max_workers),
            # Match gsm8k-only runs (MathVerifier via math_verify) for identical reward routing.
            "gsm8k": MathVerifier(timeout=math_timeout, max_workers=math_max_workers),
            "minerva_math": MinervaMathVerifier(),
            "ifeval": IFEvalVerifier(timeout=ifeval_timeout),
            "if_multi_constraints": IFMultiConstraintsVerifier(timeout=ifeval_timeout),
            "humaneval": HumanEvalVerifier(max_concurrent=max_concurrent),
            "mbpp": MBPPVerifier(max_concurrent=max_concurrent),
            "apps": APPSVerifier(max_concurrent=max_concurrent),
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

        Groups items by verifier_type and delegates to each sub-verifier's
        verify_batch_with_timing, which runs completions in parallel.
        """
        scores, durations, _ = await self.verify_batch_with_timing(problems, completions)
        return scores, durations

    async def verify_batch_with_timing(
        self, problems: list[dict], completions: list[str]
    ) -> tuple[list[float], list[float], list[tuple[float, float]]]:
        """Verify a batch with timing spans. Returns (scores, durations_ms, timing_spans).

        Groups items by verifier_type and delegates each group to its sub-verifier's
        verify_batch_with_timing, enabling parallel execution (e.g., concurrent Docker
        containers for code verifiers).
        """
        n = len(problems)
        scores = [0.0] * n
        durations = [0.0] * n
        timing_spans = [(0.0, 0.0)] * n

        # Group indices by verifier_type
        groups: dict[str, list[int]] = defaultdict(list)
        for i, p in enumerate(problems):
            vtype = p.get("verifier_type")
            if vtype:
                groups[vtype].append(i)
            # Items without verifier_type keep default 0.0 scores

        # Run each group through its sub-verifier's verify_batch_with_timing
        # (which uses asyncio.gather internally for code verifiers)
        async def run_group(vtype: str, indices: list[int]):
            verifier = self._get_verifier(vtype)
            group_problems = [problems[i] for i in indices]
            group_completions = [completions[i] for i in indices]
            group_scores, group_durations, group_timing = await verifier.verify_batch_with_timing(
                group_problems, group_completions
            )
            for j, idx in enumerate(indices):
                scores[idx] = group_scores[j]
                durations[idx] = group_durations[j]
                timing_spans[idx] = group_timing[j]

        # Run all groups concurrently (e.g., if batch has both APPS and MBPP)
        await asyncio.gather(*(run_group(vtype, indices) for vtype, indices in groups.items()))

        return scores, durations, timing_spans
