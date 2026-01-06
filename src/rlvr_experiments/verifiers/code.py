"""Code verifiers for HumanEval/MBPP-style problems."""

import asyncio
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass

from .code_executor import CodeExecutor, ExecutorConfig, ExecutionResult


def extract_code_from_markdown(text: str) -> str:
    """Extract Python code from ```python blocks, or return text as-is."""
    for pattern in [r'```(?:python|py)\s*\n(.*?)```', r'```\s*\n(.*?)```']:
        blocks = re.findall(pattern, text, re.DOTALL)
        if blocks:
            return '\n\n'.join(b.strip() for b in blocks)
    return text


@dataclass
class TestResult:
    """Result of running test cases against generated code."""

    passed: int
    total: int
    execution_result: ExecutionResult

    @property
    def score(self) -> float:
        """Return pass rate as a score between 0 and 1."""
        if self.total == 0:
            return 0.0
        return self.passed / self.total

    @property
    def all_passed(self) -> bool:
        return self.passed == self.total and self.total > 0


class CodeVerifier(ABC):
    """Base class for code verifiers."""

    def __init__(self, executor: CodeExecutor | None = None, timeout: float = 10.0, max_concurrent: int = 4):
        self.executor = executor or CodeExecutor(ExecutorConfig(timeout=timeout), max_concurrent=max_concurrent)

    @abstractmethod
    def assemble_code(self, problem: dict, completion: str) -> str:
        """Combine problem and completion into executable code."""
        pass

    def count_tests(self, problem: dict) -> int:
        """Count test cases. Override if needed."""
        return 1

    async def verify(self, problem: dict, completion: str) -> TestResult:
        """Verify a single completion."""
        code = self.assemble_code(problem, completion)
        total = self.count_tests(problem)
        result = await self.executor.execute(code)
        passed = total if result.success else 0
        return TestResult(passed=passed, total=total, execution_result=result)

    async def verify_batch(self, problems: list[dict], completions: list[str]) -> tuple[list[float], list[float]]:
        """Verify pairs of (problem, completion) in parallel. Returns (scores, durations_ms)."""
        scores, durations, _ = await self.verify_batch_with_timing(problems, completions)
        return scores, durations

    async def verify_batch_with_timing(self, problems: list[dict], completions: list[str]) -> tuple[list[float], list[float], list[tuple[float, float]]]:
        """Verify with timing spans. Returns (scores, durations_ms, timing_spans).

        timing_spans is list of (start_offset_ms, duration_ms) for each completion.
        These represent actual execution time (after semaphore acquired), not queue wait time.
        """
        import time
        batch_start = time.perf_counter()

        tasks = [self.verify(p, c) for p, c in zip(problems, completions)]
        results = await asyncio.gather(*tasks)

        scores = [r.score for r in results]
        durations = [r.execution_result.duration_ms for r in results]
        # Use actual_start_time from executor (after semaphore acquired) for accurate spans
        timing_spans = [
            ((r.execution_result.actual_start_time - batch_start) * 1000, r.execution_result.duration_ms)
            if r.execution_result.actual_start_time else (0.0, r.execution_result.duration_ms)
            for r in results
        ]
        return scores, durations, timing_spans

    async def verify_completions(self, problem: dict, completions: list[str], **kwargs) -> tuple[list[float], list[float]]:
        """Verify N completions for one problem. Returns (scores, durations_ms)."""
        return await self.verify_batch([problem] * len(completions), completions)


class HumanEvalVerifier(CodeVerifier):
    """HumanEval: prompt has signature+docstring, model generates body, test calls check(entry_point)."""

    def assemble_code(self, problem: dict, completion: str) -> str:
        prompt, test, entry_point = problem["prompt"], problem["test"], problem["entry_point"]
        completion = self._clean_completion(completion, entry_point)
        return f"{prompt}{completion}\n\n{test}\n\ncheck({entry_point})\n"

    def _clean_completion(self, completion: str, entry_point: str) -> str:
        """If model re-outputs function signature, extract just the body."""
        pattern = rf"^\s*def\s+{re.escape(entry_point)}\s*\([^)]*\)[^:]*:"
        match = re.match(pattern, completion)
        if not match:
            return completion

        after_sig = completion[match.end():]

        # Skip optional docstring
        for doc_pattern in [r'\n[ \t]*""".*?"""[ \t]*\n', r"\n[ \t]*'''.*?'''[ \t]*\n"]:
            doc_match = re.match(doc_pattern, after_sig, re.DOTALL)
            if doc_match:
                return after_sig[doc_match.end():]

        return after_sig[1:] if after_sig.startswith('\n') else after_sig

    def count_tests(self, problem: dict) -> int:
        return max(len(re.findall(r"^\s*assert\s+", problem.get("test", ""), re.MULTILINE)), 1)


class MBPPVerifier(CodeVerifier):
    """MBPP: model generates complete function, test_list has assertions."""

    def assemble_code(self, problem: dict, completion: str) -> str:
        code = extract_code_from_markdown(completion)
        tests = "\n".join(problem["test_list"])
        return f"{code.strip()}\n\n{tests}\n"

    def count_tests(self, problem: dict) -> int:
        return len(problem.get("test_list", []))


class SimpleCodeVerifier(CodeVerifier):
    """Generic verifier: problem has 'tests' string with assertions."""

    def assemble_code(self, problem: dict, completion: str) -> str:
        return f"{completion.strip()}\n\n{problem.get('tests', '')}\n"

    def count_tests(self, problem: dict) -> int:
        return max(len(re.findall(r"^\s*assert\s+", problem.get("tests", ""), re.MULTILINE)), 1)
