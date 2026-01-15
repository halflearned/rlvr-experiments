"""Verifiers for RL training.

This module provides:
- Code verification using Docker containers (isolated, secure execution)
- Math verification (numeric answer matching)
- Distributed verification via Ray worker pools

Usage:
    from rlvr_experiments.verifiers import CodeExecutor, HumanEvalVerifier

    # Direct code execution
    executor = CodeExecutor()
    result = await executor.execute("print(1 + 1)")

    # Local code verification (single node)
    verifier = HumanEvalVerifier()
    scores = await verifier.verify_completions(problem, completions)

    # Math verification
    from rlvr_experiments.verifiers import MathVerifier
    verifier = MathVerifier()
    score = verifier.verify(response, target="42")

    # Distributed verification (multi-node)
    from rlvr_experiments.verifiers import VerifierPool
    verifier = VerifierPool(HumanEvalVerifier, num_workers=4)
    scores = await verifier.verify_completions(problem, completions)
"""

from .code_executor import CodeExecutor, ExecutorConfig, ExecutionResult
from .code import (
    APPSVerifier,
    CodeVerifier,
    HumanEvalVerifier,
    MBPPVerifier,
    SimpleCodeVerifier,
    TestResult,
)
from .math import MathVerifier
from .ifeval import IFEvalVerifier
from .multi import MultiVerifier
from .distributed import VerifierPool

__all__ = [
    # Code executor
    "CodeExecutor",
    "ExecutorConfig",
    "ExecutionResult",
    # Code verifiers
    "APPSVerifier",
    "CodeVerifier",
    "HumanEvalVerifier",
    "MBPPVerifier",
    "SimpleCodeVerifier",
    "TestResult",
    # Math verifier
    "MathVerifier",
    # IFEval verifier
    "IFEvalVerifier",
    # Multi-verifier (for mixed datasets)
    "MultiVerifier",
    # Distributed
    "VerifierPool",
]
