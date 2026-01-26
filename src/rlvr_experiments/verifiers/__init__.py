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
from .gsm8k import GSM8KVerifier
from .minerva_math import MinervaMathVerifier
from .ifeval import IFEvalVerifier
from .if_multi_constraints import IFMultiConstraintsVerifier
from .allenai import AllenAIGSM8KVerifier, AllenAIMathVerifier
from .multi import MultiVerifier
from .distributed import VerifierPool

# Lazy import for LLMJudgeVerifier to avoid vLLM GPU init on non-GPU workers
def __getattr__(name: str):
    if name == "LLMJudgeVerifier":
        from .llm_judge import LLMJudgeVerifier
        return LLMJudgeVerifier
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

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
    # Math verifiers
    "MathVerifier",
    "GSM8KVerifier",
    "MinervaMathVerifier",
    # AllenAI verifiers
    "AllenAIGSM8KVerifier",
    "AllenAIMathVerifier",
    # IFEval verifier
    "IFEvalVerifier",
    "IFMultiConstraintsVerifier",
    # Multi-verifier (for mixed datasets)
    "MultiVerifier",
    # Distributed
    "VerifierPool",
    # LLM judge
    "LLMJudgeVerifier",
]
