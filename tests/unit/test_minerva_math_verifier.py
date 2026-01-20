"""Tests for MinervaMathVerifier with persistent worker pool."""

import time
import pytest

from rlvr_experiments.verifiers.minerva_math import (
    MinervaMathVerifier,
    SympyWorkerPool,
    extract_boxed_strict,
    normalize_answer,
    is_equiv,
    _sympy_equiv_with_timeout,
)


class TestExtractBoxedStrict:
    """Tests for extract_boxed_strict function."""

    def test_simple_boxed(self):
        """Extract simple boxed answer."""
        assert extract_boxed_strict("\\boxed{42}") == "42"

    def test_boxed_with_text(self):
        """Extract boxed from text with surrounding content."""
        text = "The answer is \\boxed{7} as shown above."
        assert extract_boxed_strict(text) == "7"

    def test_nested_braces(self):
        """Handle nested braces in boxed content."""
        text = "\\boxed{\\frac{1}{2}}"
        assert extract_boxed_strict(text) == "\\frac{1}{2}"

    def test_multiple_boxed_takes_last(self):
        """When multiple \\boxed{}, take the last one."""
        text = "First: \\boxed{3}. Correction: \\boxed{4}."
        assert extract_boxed_strict(text) == "4"

    def test_no_boxed_returns_none(self):
        """Return None when no \\boxed{} found."""
        assert extract_boxed_strict("The answer is 42") is None
        assert extract_boxed_strict("") is None

    def test_unclosed_boxed(self):
        """Unclosed boxed should return None."""
        assert extract_boxed_strict("\\boxed{42") is None

    def test_deeply_nested_braces(self):
        """Handle deeply nested braces."""
        text = "\\boxed{\\frac{\\sqrt{3}}{2}}"
        assert extract_boxed_strict(text) == "\\frac{\\sqrt{3}}{2}"


class TestNormalizeAnswer:
    """Tests for normalize_answer function."""

    def test_removes_spaces(self):
        """Spaces should be removed."""
        assert normalize_answer("1 + 2") == "1+2"

    def test_removes_latex_spacing(self):
        """LaTeX spacing commands removed."""
        assert normalize_answer("\\frac{1}{2}\\,") == "\\frac{1}{2}"
        assert normalize_answer("\\quad x") == "x"

    def test_removes_left_right(self):
        """\\left and \\right removed."""
        assert normalize_answer("\\left(x\\right)") == "(x)"

    def test_normalizes_frac_variants(self):
        """dfrac and tfrac normalized to frac."""
        assert normalize_answer("\\dfrac{1}{2}") == "\\frac{1}{2}"
        assert normalize_answer("\\tfrac{1}{2}") == "\\frac{1}{2}"

    def test_handles_none(self):
        """None input returns empty string."""
        assert normalize_answer(None) == ""


class TestIsEquiv:
    """Tests for is_equiv function."""

    def test_string_match(self):
        """Direct string match - fast path."""
        assert is_equiv("42", "42") is True
        assert is_equiv("\\frac{1}{2}", "\\frac{1}{2}") is True

    def test_string_mismatch(self):
        """Non-equivalent answers."""
        assert is_equiv("41", "42") is False

    def test_none_inputs(self):
        """None inputs return False."""
        assert is_equiv(None, "42") is False
        assert is_equiv("42", None) is False
        assert is_equiv(None, None) is False

    def test_sympy_equivalence(self):
        """Symbolic equivalence via sympy."""
        # These require sympy to verify equivalence
        assert is_equiv("2", "\\frac{4}{2}") is True
        assert is_equiv("0.5", "\\frac{1}{2}") is True

    def test_timeout_returns_false(self):
        """Timeout should return False, not hang."""
        # Use very short timeout with complex expression
        result = is_equiv("x", "y", timeout=0.001)
        # Should return False (either due to mismatch or timeout), not hang
        assert result is False


class TestSympyWorkerPool:
    """Tests for the SympyWorkerPool class."""

    def test_basic_equivalence(self):
        """Basic equivalence check through pool."""
        pool = SympyWorkerPool(timeout=5.0)
        try:
            assert pool.check_equiv("2", "2") is True
            assert pool.check_equiv("3", "4") is False
        finally:
            pool.shutdown()

    def test_fraction_equivalence(self):
        """Fraction to decimal equivalence."""
        pool = SympyWorkerPool(timeout=5.0)
        try:
            assert pool.check_equiv("0.5", "\\frac{1}{2}") is True
        finally:
            pool.shutdown()

    def test_worker_reuse(self):
        """Same worker should handle multiple requests efficiently."""
        pool = SympyWorkerPool(timeout=5.0)
        try:
            # First call starts worker (includes sympy import)
            start1 = time.perf_counter()
            pool.check_equiv("1", "1")
            first_call = time.perf_counter() - start1

            # Subsequent calls should be faster (worker already warm)
            times = []
            for _ in range(5):
                start = time.perf_counter()
                pool.check_equiv("2", "2")
                times.append(time.perf_counter() - start)

            avg_subsequent = sum(times) / len(times)
            # Subsequent calls should be at least 10x faster than first
            # (first includes 1.5s import, subsequent should be ~5ms)
            assert avg_subsequent < first_call / 5
        finally:
            pool.shutdown()

    def test_timeout_kills_worker(self):
        """Timeout should kill hung worker and return False."""
        pool = SympyWorkerPool(timeout=0.001)  # Very short timeout
        try:
            start = time.perf_counter()
            # This will likely timeout because worker needs to start + import
            result = pool.check_equiv("1", "1")
            elapsed = time.perf_counter() - start

            # Should return within reasonable time (not hang)
            assert elapsed < 5.0
            # Result is False (timed out) or True (fast enough)
            assert isinstance(result, bool)
        finally:
            pool.shutdown()

    def test_worker_restart_after_timeout(self):
        """Worker should restart after timeout and work again."""
        pool = SympyWorkerPool(timeout=3.0)
        try:
            # First call - should work
            result1 = pool.check_equiv("1", "1")
            assert result1 is True

            # Force a timeout by using very short timeout temporarily
            pool.timeout = 0.001
            pool.check_equiv("2", "2")  # Will likely timeout

            # Restore timeout and verify worker recovers
            pool.timeout = 5.0
            result2 = pool.check_equiv("3", "3")
            assert result2 is True
        finally:
            pool.shutdown()

    def test_shutdown_cleans_up(self):
        """Shutdown should clean up worker process."""
        pool = SympyWorkerPool(timeout=5.0)
        pool.check_equiv("1", "1")  # Start worker

        assert pool._worker is not None
        assert pool._worker.is_alive()

        pool.shutdown()

        assert pool._worker is None
        assert pool._started is False


class TestSympyEquivWithTimeout:
    """Tests for _sympy_equiv_with_timeout using the singleton pool."""

    def test_simple_equivalence(self):
        """Simple numeric equivalence."""
        assert _sympy_equiv_with_timeout("2", "2", timeout=5.0) is True

    def test_fraction_equivalence(self):
        """Fraction to decimal equivalence."""
        assert _sympy_equiv_with_timeout("0.5", "\\frac{1}{2}", timeout=5.0) is True

    def test_non_equivalent(self):
        """Non-equivalent values return False."""
        assert _sympy_equiv_with_timeout("3", "4", timeout=5.0) is False


class TestMinervaMathVerifier:
    """Tests for the MinervaMathVerifier class."""

    @pytest.fixture
    def verifier(self):
        """Create a verifier instance."""
        return MinervaMathVerifier()

    def test_verify_correct_boxed(self, verifier):
        """Correct answer with \\boxed{} format."""
        response = "The answer is \\boxed{42}"
        target = "\\boxed{42}"
        assert verifier.verify(response, target) == 1.0

    def test_verify_incorrect_boxed(self, verifier):
        """Incorrect answer."""
        response = "The answer is \\boxed{41}"
        target = "\\boxed{42}"
        assert verifier.verify(response, target) == 0.0

    def test_verify_no_boxed_in_response(self, verifier):
        """Response without \\boxed{} returns 0."""
        response = "The answer is 42"
        target = "\\boxed{42}"
        assert verifier.verify(response, target) == 0.0

    def test_verify_no_boxed_in_target(self, verifier):
        """Target without \\boxed{} returns 0."""
        response = "\\boxed{42}"
        target = "42"
        assert verifier.verify(response, target) == 0.0

    def test_verify_equivalent_expressions(self, verifier):
        """Mathematically equivalent expressions."""
        response = "\\boxed{\\frac{4}{2}}"
        target = "\\boxed{2}"
        assert verifier.verify(response, target) == 1.0

    def test_verify_handles_exception(self, verifier):
        """Exceptions should return 0.0, not propagate."""
        result = verifier.verify("", "")
        assert result == 0.0

    @pytest.mark.asyncio
    async def test_verify_completions(self, verifier):
        """Test batch verification of completions."""
        problem = {"answer": "\\boxed{7}"}
        completions = [
            "\\boxed{7}",
            "The answer is 7",  # No boxed - should fail
            "\\boxed{8}",
            "I think it's \\boxed{7}",
        ]
        scores = await verifier.verify_completions(problem, completions)
        assert len(scores) == 4
        assert scores[0] == 1.0  # boxed 7
        assert scores[1] == 0.0  # no boxed - strict mode fails
        assert scores[2] == 0.0  # wrong answer
        assert scores[3] == 1.0  # boxed 7 with text

    @pytest.mark.asyncio
    async def test_verify_batch(self, verifier):
        """Test verify_batch returns scores and durations."""
        problems = [
            {"answer": "\\boxed{1}"},
            {"answer": "\\boxed{2}"},
            {"answer": "\\boxed{3}"},
        ]
        completions = ["\\boxed{1}", "\\boxed{2}", "\\boxed{999}"]
        scores, durations = await verifier.verify_batch(problems, completions)

        assert len(scores) == 3
        assert len(durations) == 3
        assert scores[0] == 1.0
        assert scores[1] == 1.0
        assert scores[2] == 0.0
        # All durations should be positive
        assert all(d > 0 for d in durations)

    @pytest.mark.asyncio
    async def test_verify_batch_with_timing(self, verifier):
        """Test verify_batch_with_timing returns timing spans."""
        problems = [{"answer": "\\boxed{1}"}, {"answer": "\\boxed{2}"}]
        completions = ["\\boxed{1}", "\\boxed{2}"]
        scores, durations, timing_spans = await verifier.verify_batch_with_timing(
            problems, completions
        )

        assert len(scores) == 2
        assert len(durations) == 2
        assert len(timing_spans) == 2

        # First span should start at 0
        assert timing_spans[0][0] == 0.0
        # Second span should start after first duration
        assert timing_spans[1][0] == pytest.approx(durations[0], rel=0.01)


class TestMinervaMathVerifierTimeout:
    """Tests specifically for timeout behavior - ensures no process accumulation."""

    def test_repeated_verifications_no_slowdown(self):
        """Multiple verifications should not slow down due to stuck processes."""
        verifier = MinervaMathVerifier()

        # Time first batch
        start1 = time.perf_counter()
        for _ in range(10):
            verifier.verify("\\boxed{1}", "\\boxed{1}")
        elapsed1 = time.perf_counter() - start1

        # Time second batch (should be similar, not slower)
        start2 = time.perf_counter()
        for _ in range(10):
            verifier.verify("\\boxed{2}", "\\boxed{2}")
        elapsed2 = time.perf_counter() - start2

        # Second batch should not be significantly slower
        # (If processes accumulated, second batch would be much slower)
        assert elapsed2 < elapsed1 * 3  # Allow some variance but not 3x slowdown

    def test_verification_completes_under_timeout(self):
        """Normal verifications should complete well under timeout."""
        verifier = MinervaMathVerifier()

        start = time.perf_counter()
        result = verifier.verify("\\boxed{42}", "\\boxed{42}")
        elapsed = time.perf_counter() - start

        assert result == 1.0
        # String match should be very fast (< 100ms)
        # Sympy path would be slower but still under 2s timeout
        assert elapsed < 2.0

    def test_many_sympy_calls_stay_fast(self):
        """Many sympy calls should remain fast (worker stays warm)."""
        verifier = MinervaMathVerifier()

        # Warm up the pool
        verifier.verify("\\boxed{\\frac{2}{1}}", "\\boxed{2}")

        # Now many sympy calls should be fast
        times = []
        for i in range(10):
            start = time.perf_counter()
            verifier.verify(f"\\boxed{{\\frac{{{i+1}}}{{1}}}}", f"\\boxed{{{i+1}}}")
            times.append(time.perf_counter() - start)

        avg_time = sum(times) / len(times)
        # Should average under 100ms per call with warm worker
        # (vs 1.5s with cold subprocess spawn)
        assert avg_time < 0.5


class TestMinervaMathVerifierEdgeCases:
    """Edge case tests for MinervaMathVerifier."""

    @pytest.fixture
    def verifier(self):
        return MinervaMathVerifier()

    def test_complex_latex(self, verifier):
        """Complex LaTeX expressions."""
        response = "\\boxed{\\frac{\\sqrt{3}}{2}}"
        target = "\\boxed{\\frac{\\sqrt{3}}{2}}"
        assert verifier.verify(response, target) == 1.0

    def test_multiline_response(self, verifier):
        """Response with multiple lines."""
        response = """Let me solve this step by step.
        First, we calculate...
        Therefore, the answer is \\boxed{4}."""
        target = "\\boxed{4}"
        assert verifier.verify(response, target) == 1.0

    def test_multiple_boxed_takes_last(self, verifier):
        """Multiple \\boxed{} should use the last one."""
        response = "First: \\boxed{3}. Wait, correction: \\boxed{4}."
        target = "\\boxed{4}"
        assert verifier.verify(response, target) == 1.0

    def test_empty_boxed(self, verifier):
        """Empty \\boxed{} should return 0."""
        response = "\\boxed{}"
        target = "\\boxed{42}"
        assert verifier.verify(response, target) == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
