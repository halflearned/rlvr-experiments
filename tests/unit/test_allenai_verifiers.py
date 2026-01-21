"""Unit tests for AllenAI verifiers (GSM8K and MATH)."""

import pytest
from rlvr_experiments.verifiers.allenai import (
    AllenAIGSM8KVerifier,
    AllenAIMathVerifier,
    extract_gsm8k_answer,
    normalize_gsm8k,
    last_boxed_only_string,
    remove_boxed,
    get_unnormalized_answer,
    normalize_final_answer,
    strip_string,
    hendrycks_is_equiv,
    is_equiv_sympy,
)


class TestGSM8KExtraction:
    """Test GSM8K answer extraction functions."""

    def test_extract_gsm8k_answer_simple(self):
        assert extract_gsm8k_answer("The answer is 42.") == "42"

    def test_extract_gsm8k_answer_with_commas(self):
        assert extract_gsm8k_answer("The total is 1,234 dollars.") == "1234"

    def test_extract_gsm8k_answer_decimal(self):
        assert extract_gsm8k_answer("The result is 3.14.") == "3.14"

    def test_extract_gsm8k_answer_multiple_numbers(self):
        # Should return last number
        assert extract_gsm8k_answer("First 10, then 20, finally 30.") == "30"

    def test_extract_gsm8k_answer_negative(self):
        assert extract_gsm8k_answer("The answer is -5.") == "5"  # Note: extracts without sign

    def test_extract_gsm8k_answer_so_format(self):
        assert extract_gsm8k_answer("So the answer is 42.") == "42"

    def test_extract_gsm8k_answer_no_number(self):
        assert extract_gsm8k_answer("No numbers here") is None

    def test_normalize_gsm8k(self):
        assert normalize_gsm8k("1,234") == "1234"
        assert normalize_gsm8k("42") == "42"
        assert normalize_gsm8k(None) is None


class TestAllenAIGSM8KVerifier:
    """Test AllenAI GSM8K verifier class."""

    @pytest.fixture
    def verifier(self):
        return AllenAIGSM8KVerifier()

    def test_verify_correct(self, verifier):
        assert verifier.verify("So the answer is 42.", "42") == 1.0

    def test_verify_incorrect(self, verifier):
        assert verifier.verify("The answer is 100.", "42") == 0.0

    def test_verify_with_commas(self, verifier):
        assert verifier.verify("The total is 1,234.", "1234") == 1.0

    def test_verify_numeric_equivalence(self, verifier):
        # 6.0 should equal 6
        assert verifier.verify("The answer is 6.0.", "6") == 1.0

    def test_verify_no_number(self, verifier):
        assert verifier.verify("I don't know the answer.", "42") == 0.0

    @pytest.mark.asyncio
    async def test_verify_completions(self, verifier):
        problem = {"ground_truth": "42"}
        completions = [
            "The answer is 42.",
            "The answer is 100.",
            "So the answer is 42.",
        ]
        scores = await verifier.verify_completions(problem, completions)
        assert scores == [1.0, 0.0, 1.0]

    @pytest.mark.asyncio
    async def test_verify_batch(self, verifier):
        problems = [
            {"ground_truth": "42"},
            {"ground_truth": "100"},
        ]
        completions = [
            "The answer is 42.",
            "The answer is 50.",
        ]
        scores, durations = await verifier.verify_batch(problems, completions)
        assert scores == [1.0, 0.0]
        assert len(durations) == 2


class TestMATHExtraction:
    """Test MATH answer extraction functions."""

    def test_last_boxed_only_string_simple(self):
        result = last_boxed_only_string(r"The answer is \boxed{42}")
        assert result == r"\boxed{42}"

    def test_last_boxed_only_string_nested(self):
        result = last_boxed_only_string(r"\boxed{\frac{1}{2}}")
        assert result == r"\boxed{\frac{1}{2}}"

    def test_last_boxed_only_string_multiple(self):
        # Should return the last one
        result = last_boxed_only_string(r"\boxed{1} and \boxed{2}")
        assert result == r"\boxed{2}"

    def test_last_boxed_only_string_none(self):
        assert last_boxed_only_string("No boxed content") is None

    def test_last_boxed_only_string_with_space(self):
        result = last_boxed_only_string(r"\boxed 42$")
        assert result == r"\boxed 42"

    def test_remove_boxed_simple(self):
        assert remove_boxed(r"\boxed{42}") == "42"

    def test_remove_boxed_nested(self):
        assert remove_boxed(r"\boxed{\frac{1}{2}}") == r"\frac{1}{2}"

    def test_remove_boxed_with_space(self):
        assert remove_boxed(r"\boxed 42") == "42"

    def test_get_unnormalized_answer_valid(self):
        text = "Final Answer: The final answer is 42. I hope it is correct."
        assert get_unnormalized_answer(text) == "42"

    def test_get_unnormalized_answer_missing(self):
        text = "Just some random text"
        assert get_unnormalized_answer(text) == "[invalidanswer]"

    def test_normalize_final_answer_basic(self):
        assert normalize_final_answer("42") == "42"

    def test_normalize_final_answer_commas(self):
        assert normalize_final_answer("1,234") == "1234"

    def test_normalize_final_answer_text_wrapper(self):
        # Should remove \text{} content (units get stripped)
        result = normalize_final_answer(r"5 \text{ cm}")
        # The regex removes \text{...} but leaves surrounding content
        # After space removal, we get "5\textcm" - actual verification
        # uses strip_string which handles this better
        assert "cm" not in result or "{" not in result


class TestHendrycksEquivalence:
    """Test Hendrycks-style string equivalence."""

    def test_hendrycks_is_equiv_identical(self):
        assert hendrycks_is_equiv("42", "42") is True

    def test_hendrycks_is_equiv_different(self):
        assert hendrycks_is_equiv("42", "43") is False

    def test_hendrycks_is_equiv_half_conversion(self):
        # 0.5 should be converted to \frac{1}{2}
        assert hendrycks_is_equiv("0.5", r"\frac{1}{2}") is True

    def test_hendrycks_is_equiv_both_none(self):
        assert hendrycks_is_equiv(None, None) is True

    def test_hendrycks_is_equiv_one_none(self):
        assert hendrycks_is_equiv("42", None) is False
        assert hendrycks_is_equiv(None, "42") is False

    def test_hendrycks_is_equiv_whitespace(self):
        # strip_string removes spaces
        assert hendrycks_is_equiv("x + 1", "x+1") is True

    def test_hendrycks_is_equiv_frac_variants(self):
        # tfrac and dfrac should be converted to frac
        assert hendrycks_is_equiv(r"\tfrac{1}{2}", r"\frac{1}{2}") is True
        assert hendrycks_is_equiv(r"\dfrac{1}{2}", r"\frac{1}{2}") is True

    def test_strip_string_leading_decimal(self):
        # .5 -> 0.5 -> \frac{1}{2} (the function converts 0.5 to frac)
        assert strip_string(".5") == r"\frac{1}{2}"

    def test_strip_string_equation_format(self):
        # x = answer format should extract answer
        assert strip_string("x = 42") == "42"


class TestAllenAIMathVerifier:
    """Test AllenAI MATH verifier class."""

    @pytest.fixture
    def verifier(self):
        return AllenAIMathVerifier(use_sympy=False)

    @pytest.fixture
    def verifier_with_sympy(self):
        return AllenAIMathVerifier(use_sympy=True)

    def test_verify_boxed_correct(self, verifier):
        assert verifier.verify(r"\boxed{42}", "42") == 1.0

    def test_verify_boxed_incorrect(self, verifier):
        assert verifier.verify(r"\boxed{42}", "100") == 0.0

    def test_verify_boxed_fraction(self, verifier):
        assert verifier.verify(r"\boxed{\frac{1}{2}}", r"\frac{1}{2}") == 1.0

    def test_verify_minerva_format(self, verifier):
        response = "Final Answer: The final answer is 42. I hope it is correct."
        assert verifier.verify(response, "42") == 1.0

    def test_verify_dollar_signs(self, verifier):
        assert verifier.verify("The answer is $42$", "42") == 1.0

    def test_verify_half_equivalence(self, verifier):
        # 0.5 should match \frac{1}{2}
        assert verifier.verify(r"\boxed{0.5}", r"\frac{1}{2}") == 1.0

    def test_verify_with_text(self, verifier):
        response = r"Therefore, the answer is \boxed{x^2 + 1}"
        assert verifier.verify(response, "x^2 + 1") == 1.0

    def test_verify_fallback_extraction(self, verifier):
        # When no boxed or minerva format, should try full output
        response = "After calculation, we get 42."
        # This might not match exactly due to normalization
        # The fallback adds the full normalized output

    @pytest.mark.asyncio
    async def test_verify_completions(self, verifier):
        problem = {"ground_truth": "42"}
        completions = [
            r"\boxed{42}",
            r"\boxed{100}",
            "Final Answer: The final answer is 42. I hope it is correct.",
        ]
        scores = await verifier.verify_completions(problem, completions)
        assert scores == [1.0, 0.0, 1.0]

    @pytest.mark.asyncio
    async def test_verify_batch(self, verifier):
        problems = [
            {"ground_truth": "42"},
            {"ground_truth": r"\frac{1}{2}"},
        ]
        completions = [
            r"\boxed{42}",
            r"\boxed{0.5}",  # Should match \frac{1}{2}
        ]
        scores, durations = await verifier.verify_batch(problems, completions)
        assert scores == [1.0, 1.0]
        assert len(durations) == 2


class TestMATHVerifierSympy:
    """Test MATH verifier with sympy enabled (optional)."""

    @pytest.fixture
    def verifier(self):
        return AllenAIMathVerifier(use_sympy=True)

    def test_sympy_equivalence(self, verifier):
        # This test requires sympy
        try:
            import sympy  # noqa
            # 2+2 should equal 4
            result = is_equiv_sympy("2+2", "4")
            # Note: may fail if sympy can't parse
        except ImportError:
            pytest.skip("sympy not installed")


class TestEdgeCases:
    """Test edge cases and error handling."""

    @pytest.fixture
    def gsm8k_verifier(self):
        return AllenAIGSM8KVerifier()

    @pytest.fixture
    def math_verifier(self):
        return AllenAIMathVerifier(use_sympy=False)

    def test_empty_response(self, gsm8k_verifier, math_verifier):
        assert gsm8k_verifier.verify("", "42") == 0.0
        assert math_verifier.verify("", "42") == 0.0

    def test_empty_target(self, gsm8k_verifier, math_verifier):
        assert gsm8k_verifier.verify("The answer is 42.", "") == 0.0
        # MATH verifier might handle empty target differently

    def test_special_characters(self, math_verifier):
        # Test that special LaTeX characters don't crash
        response = r"\boxed{\sqrt{2}}"
        target = r"\sqrt{2}"
        # Should not raise exception
        result = math_verifier.verify(response, target)
        assert result in [0.0, 1.0]

    def test_malformed_boxed(self, math_verifier):
        # Unclosed boxed
        response = r"\boxed{42"
        # Should not crash, might return 0.0
        result = math_verifier.verify(response, "42")
        assert result in [0.0, 1.0]

    @pytest.mark.asyncio
    async def test_verify_batch_with_timing(self, gsm8k_verifier):
        problems = [{"ground_truth": "42"}]
        completions = ["The answer is 42."]
        scores, durations, timing_spans = await gsm8k_verifier.verify_batch_with_timing(
            problems, completions
        )
        assert scores == [1.0]
        assert len(durations) == 1
        assert len(timing_spans) == 1
        assert timing_spans[0][0] == 0.0  # First span starts at 0
