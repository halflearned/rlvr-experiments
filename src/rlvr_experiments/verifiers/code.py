"""Code verifiers for HumanEval/MBPP-style problems."""

import asyncio
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass

from .code_executor import CodeExecutor, ExecutorConfig, ExecutionResult


def extract_code_from_markdown(text: str) -> str:
    """Extract Python code from ```python blocks, or return text as-is.

    Also handles:
    - Trailing ``` (model closing a code block started by assistant_prefix)
    - [DONE] markers (lm_eval MBPP format)
    """
    for pattern in [r'```(?:python|py)\s*\n(.*?)```', r'```\s*\n(.*?)```']:
        blocks = re.findall(pattern, text, re.DOTALL)
        if blocks:
            return '\n\n'.join(b.strip() for b in blocks)
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


class APPSStdinVerifier(CodeVerifier):
    """APPS verifier for stdin/stdout format problems.

    These are competitive programming style problems where:
    - inputs are strings (simulating stdin)
    - outputs are strings (expected stdout)
    - The model generates a complete program that reads from stdin and writes to stdout.
    """

    def assemble_code(self, problem: dict, completion: str) -> str:
        """Extract and clean code from completion."""
        return extract_code_from_markdown(completion)

    def count_tests(self, problem: dict) -> int:
        """Count number of test cases."""
        inputs = problem.get("inputs", [])
        return max(len(inputs), 1)

    async def verify(self, problem: dict, completion: str) -> TestResult:
        """Verify completion against all input/output test cases."""
        code = self.assemble_code(problem, completion)
        inputs = problem.get("inputs", [])
        outputs = problem.get("outputs", [])

        if not inputs or not outputs:
            # No test cases - can't verify, return failure
            return TestResult(
                passed=0,
                total=1,
                execution_result=ExecutionResult(
                    stdout="", stderr="No test cases available",
                    exit_code=-1, timed_out=False, duration_ms=0.0
                )
            )

        total = len(inputs)
        passed = 0
        all_stdout = []
        all_stderr = []
        total_duration = 0.0
        any_timeout = False
        last_result = None

        for inp, expected_out in zip(inputs, outputs):
            # Create code that reads from a string as if it were stdin
            # This wraps the solution to feed it input
            wrapped_code = self._wrap_with_input(code, inp)
            result = await self.executor.execute(wrapped_code)
            last_result = result
            total_duration += result.duration_ms

            if result.timed_out:
                any_timeout = True
                continue

            if result.exit_code != 0:
                all_stderr.append(result.stderr)
                continue

            # Compare output (normalize whitespace)
            actual = result.stdout.strip()
            expected = expected_out.strip()

            if self._outputs_match(actual, expected):
                passed += 1
            else:
                all_stderr.append(f"Expected:\n{expected}\nGot:\n{actual}")

            all_stdout.append(result.stdout)

        return TestResult(
            passed=passed,
            total=total,
            execution_result=ExecutionResult(
                stdout="\n---\n".join(all_stdout),
                stderr="\n---\n".join(all_stderr),
                exit_code=0 if passed == total else 1,
                timed_out=any_timeout,
                duration_ms=total_duration,
                actual_start_time=last_result.actual_start_time if last_result else None,
            )
        )

    def _wrap_with_input(self, code: str, stdin_input: str) -> str:
        """Wrap code to provide stdin input."""
        # Use StringIO to mock stdin
        escaped_input = stdin_input.replace("\\", "\\\\").replace('"""', '\\"\\"\\"')
        return f'''import sys
from io import StringIO

_INPUT = """{escaped_input}"""
sys.stdin = StringIO(_INPUT)

{code}
'''

    def _outputs_match(self, actual: str, expected: str) -> bool:
        """Compare outputs with flexible whitespace handling."""
        # Normalize: strip, split by whitespace, compare tokens
        actual_tokens = actual.split()
        expected_tokens = expected.split()
        return actual_tokens == expected_tokens


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
        inputs = problem.get("inputs", [])
        return max(len(inputs), 1)

    async def verify(self, problem: dict, completion: str) -> TestResult:
        """Verify completion by calling the function with test inputs."""
        code = self.assemble_code(problem, completion)
        inputs = problem.get("inputs", [])
        outputs = problem.get("outputs", [])
        fn_name = problem.get("fn_name", "")

        if not inputs or not outputs or not fn_name:
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
