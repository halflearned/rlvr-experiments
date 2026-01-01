"""Code execution sandbox for RL training with untrusted code.

This module provides isolated code execution using Docker containers with:
- Network isolation (--network=none)
- Filesystem isolation (--read-only + tmpfs)
- Resource limits (memory, CPU, PIDs)
- Security hardening (dropped capabilities, no-new-privileges)

Design: Fresh container per execution for perfect determinism.
The ~200ms container overhead is negligible when running thousands
of executions in parallel across Ray workers.

Usage:
    from rlvr_experiments.sandbox import CodeExecutor, HumanEvalVerifier

    # Direct execution
    executor = CodeExecutor()
    result = await executor.execute("print(1 + 1)")

    # Local verification (single node)
    verifier = HumanEvalVerifier()
    scores = await verifier.verify_completions(problem, completions)

    # Distributed verification (multi-node via Ray)
    from rlvr_experiments.sandbox import RayCodeVerifier
    verifier = RayCodeVerifier(HumanEvalVerifier, num_workers=4)
    scores = await verifier.verify_completions(problem, completions)
"""

from .executor import CodeExecutor, ExecutorConfig, ExecutionResult
from .verifier import (
    CodeVerifier,
    HumanEvalVerifier,
    MBPPVerifier,
    SimpleCodeVerifier,
    TestResult,
)
from .distributed import RayCodeVerifier

__all__ = [
    # Executor
    "CodeExecutor",
    "ExecutorConfig",
    "ExecutionResult",
    # Verifiers
    "CodeVerifier",
    "HumanEvalVerifier",
    "MBPPVerifier",
    "SimpleCodeVerifier",
    "TestResult",
    # Distributed
    "RayCodeVerifier",
]
