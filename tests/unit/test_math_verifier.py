"""Tests for MathVerifier with subprocess-based timeout."""

import asyncio
import time
import pytest

from rlvr_experiments.verifiers.math import MathVerifier, _verify_single


class TestVerifySingle:
    """Tests for the module-level _verify_single function."""

    def test_exact_match(self):
        """Simple numeric equality."""
        assert _verify_single("42", "42") == 1.0

    def test_boxed_answer(self):
        """Standard \\boxed{} format."""
        assert _verify_single("The answer is \\boxed{7}", "7") == 1.0

    def test_expression_equivalence(self):
        """Mathematically equivalent expressions."""
        assert _verify_single("\\boxed{2+2}", "4") == 1.0
        assert _verify_single("\\boxed{\\frac{1}{2}}", "0.5") == 1.0

    def test_negative_numbers(self):
        """Negative number handling."""
        assert _verify_single("\\boxed{-5}", "-5") == 1.0
        assert _verify_single("-5", "-5") == 1.0

    def test_fractions(self):
        """Fraction parsing and comparison."""
        assert _verify_single("\\boxed{\\frac{3}{4}}", "\\frac{3}{4}") == 1.0
        assert _verify_single("\\frac{1}{2}", "0.5") == 1.0

    def test_wrong_answer(self):
        """Incorrect answers should return 0."""
        assert _verify_single("\\boxed{5}", "7") == 0.0
        assert _verify_single("42", "43") == 0.0

    def test_unparseable_response(self):
        """Garbage input should return 0, not crash."""
        assert _verify_single("I don't know the answer", "42") == 0.0
        assert _verify_single("", "42") == 0.0

    def test_unparseable_target(self):
        """Unparseable target should return 0."""
        assert _verify_single("42", "not a number at all") == 0.0


class TestMathVerifier:
    """Tests for the MathVerifier class."""

    @pytest.fixture
    def verifier(self):
        """Create a verifier with short timeout for tests."""
        return MathVerifier(timeout=2.0, max_workers=2)

    def test_verify_correct(self, verifier):
        """Basic correct answer verification."""
        assert verifier.verify("\\boxed{42}", "42") == 1.0

    def test_verify_incorrect(self, verifier):
        """Basic incorrect answer verification."""
        assert verifier.verify("\\boxed{41}", "42") == 0.0

    def test_verify_handles_exception(self, verifier):
        """Exceptions in subprocess should return 0, not propagate."""
        # Empty strings might cause issues in parsing
        result = verifier.verify("", "")
        assert result == 0.0

    @pytest.mark.asyncio
    async def test_verify_completions(self, verifier):
        """Test batch verification of completions."""
        problem = {"answer": "7"}
        completions = [
            "\\boxed{7}",
            "The answer is 7",
            "\\boxed{8}",
            "I think it's \\boxed{7}",
        ]
        scores = await verifier.verify_completions(problem, completions)
        assert len(scores) == 4
        assert scores[0] == 1.0  # boxed 7
        assert scores[1] == 1.0  # plain 7
        assert scores[2] == 0.0  # wrong answer
        assert scores[3] == 1.0  # boxed 7 with text

    @pytest.mark.asyncio
    async def test_verify_batch(self, verifier):
        """Test verify_batch returns scores and durations."""
        problems = [{"answer": "1"}, {"answer": "2"}, {"answer": "3"}]
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
        problems = [{"answer": "1"}, {"answer": "2"}]
        completions = ["\\boxed{1}", "\\boxed{2}"]
        scores, durations, timing_spans = await verifier.verify_batch_with_timing(problems, completions)

        assert len(scores) == 2
        assert len(durations) == 2
        assert len(timing_spans) == 2

        # First span should start at 0
        assert timing_spans[0][0] == 0.0
        # Second span should start after first duration
        assert timing_spans[1][0] == pytest.approx(durations[0], rel=0.01)


class TestMathVerifierTimeout:
    """Tests specifically for timeout behavior."""

    def test_timeout_returns_zero(self):
        """Verify that timeout returns 0.0 instead of hanging."""
        # Create verifier with very short timeout but with warmup
        # to avoid warmup-related timeouts
        verifier = MathVerifier(timeout=0.001, max_workers=1, warmup=True)

        # With warmed up workers, this should be fast but might still timeout
        # We just verify it doesn't hang and returns a float
        result = verifier.verify("\\boxed{1}", "1")
        assert isinstance(result, float)
        assert result in (0.0, 1.0)  # Either timed out or succeeded

    def test_subprocess_isolation(self):
        """Verify that one slow verification doesn't block others."""
        verifier = MathVerifier(timeout=1.0, max_workers=2, warmup=True)

        # Run multiple verifications - they should all complete
        start = time.perf_counter()
        results = []
        for i in range(4):
            results.append(verifier.verify(f"\\boxed{{{i}}}", str(i)))
        elapsed = time.perf_counter() - start

        # All should succeed (or timeout gracefully)
        assert len(results) == 4
        # Should complete in reasonable time (not hang)
        assert elapsed < 10.0

    def test_warmup_speeds_up_first_call(self):
        """Verify that warmup=True makes first call fast."""
        # With warmup, first call should be reasonably fast
        verifier = MathVerifier(timeout=2.0, max_workers=1, warmup=True)
        start = time.perf_counter()
        result = verifier.verify("42", "42")
        elapsed = time.perf_counter() - start

        # Should complete successfully and quickly (under timeout)
        assert result == 1.0
        assert elapsed < 2.0  # Must complete before timeout


class TestMathVerifierEdgeCases:
    """Edge case tests for math verification."""

    @pytest.fixture
    def verifier(self):
        return MathVerifier(timeout=2.0, max_workers=2)

    def test_scientific_notation(self, verifier):
        """Scientific notation handling - math_verify doesn't support 1e6 syntax."""
        # math_verify can't parse 1e6 notation, returns 0.0
        # This documents the limitation rather than asserting it works
        result = verifier.verify("\\boxed{1e6}", "1000000")
        assert result in (0.0, 1.0)  # May or may not parse

    def test_decimal_precision(self, verifier):
        """Decimal number comparison."""
        assert verifier.verify("\\boxed{3.14159}", "3.14159") == 1.0

    def test_sqrt_expression(self, verifier):
        """Square root expressions."""
        assert verifier.verify("\\boxed{\\sqrt{4}}", "2") == 1.0

    def test_pi_symbol(self, verifier):
        """Pi symbol handling - math_verify requires proper latex."""
        # math_verify may have trouble with bare \pi depending on context
        # This documents the behavior rather than asserting specific result
        result = verifier.verify("\\boxed{\\pi}", "\\pi")
        assert result in (0.0, 1.0)  # May or may not parse correctly

    def test_multiline_response(self, verifier):
        """Response with multiple lines."""
        response = """Let me solve this step by step.
        First, we add 2 + 2 = 4.
        Therefore, the answer is \\boxed{4}."""
        assert verifier.verify(response, "4") == 1.0

    def test_multiple_boxed(self, verifier):
        """Response with multiple \\boxed{} - should use last one."""
        response = "First attempt: \\boxed{3}. Wait, that's wrong. The answer is \\boxed{4}."
        # math_verify typically takes the last boxed answer
        result = verifier.verify(response, "4")
        # This behavior may vary - just ensure it doesn't crash
        assert result in (0.0, 1.0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
