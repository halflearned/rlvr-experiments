"""Code verifiers for HumanEval/MBPP-style problems."""

import asyncio
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass

from .code_executor import CodeExecutor, ExecutorConfig, ExecutionResult


def truncate_to_first_block(text: str) -> str:
    """Truncate code at the second top-level def/class.

    Keeps only the first function/class definition, discarding any
    subsequent top-level definitions the model may have generated.
    This ensures the model is only rewarded for the quality of its
    first answer, not for learning to stop generating.
    """
    lines = text.split('\n')
    def_count = 0
    for i, line in enumerate(lines):
        # Top-level def or class (no indentation)
        if re.match(r'^(def |class )\w', line):
            def_count += 1
            if def_count >= 2:
                return '\n'.join(lines[:i]).rstrip()
    return text


def extract_code_from_markdown(text: str) -> str:
    """Extract Python code from ```python blocks, or return text as-is.

    Also handles:
    - Trailing ``` (model closing a code block started by assistant_prefix)
    - ``` followed by text (model adding explanation after code)
    - [DONE] markers (lm_eval MBPP format)
    """
    for pattern in [r'```(?:python|py)\s*\n(.*?)```', r'```\s*\n(.*?)```']:
        blocks = re.findall(pattern, text, re.DOTALL)
        if blocks:
            return '\n\n'.join(b.strip() for b in blocks)
    # Strip ``` and anything after it (model closing code block, possibly with explanation)
    text = re.sub(r'\n```[\s\S]*$', '', text)
    # Strip trailing ``` if present (model closing a code block started by assistant_prefix)
    text = re.sub(r'\s*```\s*$', '', text)
    # Strip [DONE] marker (lm_eval MBPP format)
    text = re.sub(r'\s*\[DONE\]\s*$', '', text)
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
        # Strip markdown artifacts (``` etc) before processing
        completion = re.sub(r'\n```[\s\S]*$', '', completion)
        completion = re.sub(r'\s*```\s*$', '', completion)
        completion = self._clean_completion(completion, entry_point)
        # Truncate at first top-level def/class — completion is a function body
        # (indented), so any top-level def/class is extraneous generation
        lines = completion.split('\n')
        for i, line in enumerate(lines):
            if re.match(r'^(def |class )\w', line):
                completion = '\n'.join(lines[:i]).rstrip()
                break
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

    def __init__(self, truncate: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.truncate = truncate

    def assemble_code(self, problem: dict, completion: str) -> str:
        code = extract_code_from_markdown(completion)
        if self.truncate:
            code = truncate_to_first_block(code)
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


class APPSStdinVerifier(CodeVerifier):
    """APPS verifier for stdin/stdout format problems.

    Runs all test cases in a single container invocation using internal subprocesses.
    This avoids the N-containers-per-problem bottleneck that causes progressive slowdown.
    """

    # Per-test subprocess timeout inside the container (seconds)
    PER_TEST_TIMEOUT = 3

    def __init__(self, executor: CodeExecutor | None = None, timeout: float = 10.0, max_concurrent: int = 4):
        # Don't pass timeout to super — we override execute with a dynamic timeout
        self._base_timeout = timeout
        self._max_concurrent = max_concurrent
        self.executor = executor or CodeExecutor(ExecutorConfig(timeout=timeout), max_concurrent=max_concurrent)

    def assemble_code(self, problem: dict, completion: str) -> str:
        return extract_code_from_markdown(completion)

    def count_tests(self, problem: dict) -> int:
        inputs = list(problem.get("inputs", []))
        return max(len(inputs), 1)

    async def verify(self, problem: dict, completion: str) -> TestResult:
        """Verify completion against all input/output test cases in a single container."""
        code = self.assemble_code(problem, completion)
        inputs = list(problem.get("inputs", []))
        outputs = list(problem.get("outputs", []))

        if len(inputs) == 0 or len(outputs) == 0:
            return TestResult(
                passed=0,
                total=1,
                execution_result=ExecutionResult(
                    stdout="", stderr="No test cases available",
                    exit_code=-1, timed_out=False, duration_ms=0.0
                )
            )

        total = len(inputs)

        # Scale container timeout with test count: base + per_test * n_tests
        # Most tests complete in <1s; the per-test timeout handles hangs
        container_timeout = self._base_timeout + self.PER_TEST_TIMEOUT * total
        old_timeout = self.executor.config.timeout
        self.executor.config.timeout = container_timeout
        try:
            bundled = self._build_bundled_test(code, inputs, outputs)
            result = await self.executor.execute(bundled)
        finally:
            self.executor.config.timeout = old_timeout

        # Parse "RLVR_RESULT: X/Y" from stdout
        passed = 0
        if not result.timed_out:
            for line in result.stdout.strip().split('\n'):
                if line.startswith('RLVR_RESULT:'):
                    try:
                        parts = line.split(':')[1].strip().split('/')
                        passed = int(parts[0])
                    except (ValueError, IndexError):
                        pass

        return TestResult(
            passed=passed,
            total=total,
            execution_result=result,
        )

    def _build_bundled_test(self, code: str, inputs: list[str], outputs: list[str]) -> str:
        """Build a single script that runs the solution against all test cases."""
        import json as _json
        test_cases_json = _json.dumps([
            {"input": inp, "expected": out}
            for inp, out in zip(inputs, outputs)
        ])

        return f'''import subprocess, sys, json, tempfile, os

_SOLUTION = {repr(code)}
_TEST_CASES = json.loads({repr(test_cases_json)})
_TIMEOUT = {self.PER_TEST_TIMEOUT}
_MAX_CONSECUTIVE_FAILURES = 3  # bail early if code is broken/hanging

# Write solution to a temp file once
_tmpfile = tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, dir='/tmp')
_tmpfile.write(_SOLUTION)
_tmpfile.close()

_passed = 0
_total = len(_TEST_CASES)
_consecutive_failures = 0

for _tc in _TEST_CASES:
    try:
        _proc = subprocess.run(
            [sys.executable, _tmpfile.name],
            input=_tc["input"],
            capture_output=True,
            text=True,
            timeout=_TIMEOUT,
        )
        if _proc.returncode == 0:
            _actual = _proc.stdout.strip().split()
            _expected = _tc["expected"].strip().split()
            if _actual == _expected:
                _passed += 1
                _consecutive_failures = 0
                continue
        _consecutive_failures += 1
    except subprocess.TimeoutExpired:
        _consecutive_failures += 1
    except Exception:
        _consecutive_failures += 1
    if _consecutive_failures >= _MAX_CONSECUTIVE_FAILURES:
        break

os.unlink(_tmpfile.name)
print(f"RLVR_RESULT:{{_passed}}/{{_total}}")
'''


class APPSFunctionVerifier(CodeVerifier):
    """APPS verifier for function-call format problems (LeetCode style).

    These are LeetCode-style problems where:
    - inputs are lists of arguments (e.g., [[1, 2, 3], 3] for two arguments)
    - outputs are expected return values
    - fn_name specifies the method name to call
    - starter_code contains a class Solution with method signature

    The model generates code that should include the class definition.
    """

    def assemble_code(self, problem: dict, completion: str) -> str:
        """Extract and clean code from completion."""
        return extract_code_from_markdown(completion)

    def count_tests(self, problem: dict) -> int:
        """Count number of test cases."""
        inputs = list(problem.get("inputs", []))
        return max(len(inputs), 1)

    async def verify(self, problem: dict, completion: str) -> TestResult:
        """Verify completion by calling the function with test inputs."""
        code = self.assemble_code(problem, completion)
        inputs = list(problem.get("inputs", []))
        outputs = list(problem.get("outputs", []))
        fn_name = problem.get("fn_name", "")

        if len(inputs) == 0 or len(outputs) == 0 or not fn_name:
            return TestResult(
                passed=0,
                total=1,
                execution_result=ExecutionResult(
                    stdout="", stderr="Missing inputs, outputs, or fn_name",
                    exit_code=-1, timed_out=False, duration_ms=0.0
                )
            )

        # Build test code that calls the function
        test_code = self._build_test_code(code, inputs, outputs, fn_name)
        result = await self.executor.execute(test_code)

        # Parse result - we print passed/total at the end
        if result.success:
            # Extract pass count from stdout (we print "PASSED: X/Y")
            try:
                for line in result.stdout.strip().split('\n'):
                    if line.startswith('PASSED:'):
                        parts = line.split(':')[1].strip().split('/')
                        passed = int(parts[0])
                        total = int(parts[1])
                        return TestResult(
                            passed=passed,
                            total=total,
                            execution_result=result
                        )
            except (ValueError, IndexError):
                pass
            # If we can't parse, assume all passed if no error
            return TestResult(passed=len(inputs), total=len(inputs), execution_result=result)
        else:
            return TestResult(passed=0, total=len(inputs), execution_result=result)

    def _build_test_code(self, code: str, inputs: list, outputs: list, fn_name: str) -> str:
        """Build executable test code that calls the function."""
        # APPS function-call inputs are wrapped in an extra list:
        # inputs = [[[1, 2, 3], 3]] means test case 0 has args [1,2,3] and 3
        test_cases = []
        for i, (inp, expected) in enumerate(zip(inputs, outputs)):
            # inp is a list of arguments, expected is the return value
            test_cases.append({
                "args": inp,  # This is already a list of arguments
                "expected": expected
            })

        test_cases_repr = repr(test_cases)

        return f'''{code}

# Test runner
_test_cases = {test_cases_repr}
_passed = 0
_total = len(_test_cases)

_sol = Solution()
for _i, _tc in enumerate(_test_cases):
    _args = _tc["args"]
    _expected = _tc["expected"]
    try:
        _result = _sol.{fn_name}(*_args)
        # Flexible comparison: handle lists, floats, etc.
        if _result == _expected:
            _passed += 1
        elif isinstance(_result, float) and isinstance(_expected, float):
            if abs(_result - _expected) < 1e-6:
                _passed += 1
        elif isinstance(_result, list) and isinstance(_expected, list):
            if sorted(_result) == sorted(_expected):
                _passed += 1
    except Exception as e:
        pass  # Test failed

print(f"PASSED: {{_passed}}/{{_total}}")
'''


# Backwards compatibility alias
APPSVerifier = APPSStdinVerifier
