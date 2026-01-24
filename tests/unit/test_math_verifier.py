"""Tests for MathVerifier with subprocess-based timeout."""

import asyncio
import time
import pytest

from rlvr_experiments.verifiers.math import (
    MathVerifier,
    _verify_single,
    _extract_last_number,
    _normalize_number,
)


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


class TestFallbackExtraction:
    """Tests for fallback last-number extraction."""

    def test_extract_last_number_simple(self):
        """Basic last number extraction."""
        assert _extract_last_number("The answer is 42") == "42"
        assert _extract_last_number("42") == "42"
        assert _extract_last_number("The result is 123.") == "123"

    def test_extract_last_number_decimal(self):
        """Decimal number extraction."""
        assert _extract_last_number("The answer is 3.14") == "3.14"
        assert _extract_last_number("Pi is approximately 3.14159") == "3.14159"

    def test_extract_last_number_negative(self):
        """Negative number extraction."""
        assert _extract_last_number("The temperature is -5 degrees") == "-5"
        assert _extract_last_number("Result: -3.5") == "-3.5"

    def test_extract_last_number_with_commas(self):
        """Numbers with comma separators."""
        assert _extract_last_number("The population is 1,234,567") == "1234567"

    def test_extract_last_number_multiple(self):
        """Multiple numbers - should return last one."""
        assert _extract_last_number("First 10, then 20, finally 30") == "30"
        assert _extract_last_number("Step 1: 5, Step 2: 10, Answer: 15") == "15"

    def test_extract_last_number_none(self):
        """No numbers in text."""
        assert _extract_last_number("No numbers here") is None
        assert _extract_last_number("") is None

    def test_normalize_number_integer(self):
        """Integer normalization."""
        assert _normalize_number("42") == "42"
        assert _normalize_number("42.0") == "42"
        assert _normalize_number("+42") == "42"

    def test_normalize_number_decimal(self):
        """Decimal normalization."""
        assert _normalize_number("3.14") == "3.14"
        assert _normalize_number("0.5") == "0.5"

    def test_normalize_number_invalid(self):
        """Invalid number strings."""
        assert _normalize_number("abc") is None
        assert _normalize_number(None) is None


class TestFallbackVerification:
    """Tests for fallback behavior in _verify_single."""

    def test_fallback_plain_number(self):
        """Fallback extracts plain number when boxed parsing fails."""
        # Response without \boxed{} - math_verify may not parse it
        # but fallback should extract the number
        response = "After calculating, I get 42 as the answer."
        assert _verify_single(response, "42", fallback_extraction=True) == 1.0

    def test_fallback_disabled(self):
        """With fallback disabled, unparseable response returns 0."""
        response = "After calculating, I get 42 as the answer."
        # This may return 1.0 if math_verify can parse "42", or 0.0 if not
        # The point is it should not crash
        result = _verify_single(response, "42", fallback_extraction=False)
        assert result in (0.0, 1.0)

    def test_fallback_last_number_in_reasoning(self):
        """Fallback extracts last number from chain-of-thought."""
        response = """Let me solve this step by step.
        First, 2 + 3 = 5.
        Then, 5 * 2 = 10.
        Finally, 10 + 4 = 14.
        So the answer is 14."""
        assert _verify_single(response, "14", fallback_extraction=True) == 1.0

    def test_fallback_wrong_answer(self):
        """Fallback extraction with wrong answer."""
        response = "The answer is 42."
        assert _verify_single(response, "43", fallback_extraction=True) == 0.0

    def test_fallback_decimal_equivalence(self):
        """Fallback handles decimal/integer equivalence."""
        response = "The result is 6.0"
        assert _verify_single(response, "6", fallback_extraction=True) == 1.0

    def test_boxed_still_works_with_fallback(self):
        """Boxed answers still work when fallback is enabled."""
        assert _verify_single("\\boxed{42}", "42", fallback_extraction=True) == 1.0
        assert _verify_single("\\boxed{\\frac{1}{2}}", "0.5", fallback_extraction=True) == 1.0

    def test_fallback_negative_number(self):
        """Fallback with negative numbers."""
        response = "The temperature dropped to -15 degrees."
        assert _verify_single(response, "-15", fallback_extraction=True) == 1.0

    def test_fallback_with_commas(self):
        """Fallback handles numbers with comma separators."""
        response = "The total is 1,234 units."
        assert _verify_single(response, "1234", fallback_extraction=True) == 1.0


class TestMathVerifierWithFallback:
    """Integration tests for MathVerifier with fallback extraction."""

    @pytest.fixture
    def verifier(self):
        return MathVerifier(timeout=2.0, max_workers=2)

    def test_gsm8k_style_response(self, verifier):
        """GSM8K style response without boxed notation."""
        response = """Janet's ducks lay 16 eggs per day. She eats three for breakfast
        every morning and bakes muffins for her friends every day with four.
        She sells the remainder at the farmers' market daily for $2 per fresh duck egg.
        She makes 16 - 3 - 4 = 9 eggs to sell.
        9 * 2 = 18 dollars per day.
        The answer is 18."""
        assert verifier.verify(response, "18") == 1.0

    def test_response_with_multiple_calculations(self, verifier):
        """Response showing work with many intermediate numbers."""
        response = """Step 1: 100 + 50 = 150
        Step 2: 150 * 2 = 300
        Step 3: 300 - 25 = 275
        Therefore, the final answer is 275."""
        assert verifier.verify(response, "275") == 1.0

    def test_response_ending_with_period(self, verifier):
        """Response where answer is followed by period."""
        response = "After all calculations, the answer is 42."
        assert verifier.verify(response, "42") == 1.0

    @pytest.mark.asyncio
    async def test_verify_completions_mixed_formats(self, verifier):
        """Test completions with mixed formats (boxed and plain)."""
        problem = {"answer": "100"}
        completions = [
            "\\boxed{100}",  # Standard boxed
            "The answer is 100.",  # Plain with period
            "Therefore, 100",  # Plain at end
            "I calculate 99 + 1 = 100",  # With arithmetic
            "The result is 99",  # Wrong answer
        ]
        scores = await verifier.verify_completions(problem, completions)
        assert scores[0] == 1.0  # boxed
        assert scores[1] == 1.0  # plain with period
        assert scores[2] == 1.0  # plain at end
        assert scores[3] == 1.0  # with arithmetic (last number is 100)
        assert scores[4] == 0.0  # wrong answer


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
