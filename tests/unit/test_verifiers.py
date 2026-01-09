"""Unit tests for verifiers."""

import pytest
from rlvr_experiments.verifiers.math import (
    MathVerifier,
    _extract_boxed,
    _extract_answer,
    _parse_number,
)


class TestExtractBoxed:
    def test_simple_boxed(self):
        assert _extract_boxed(r"The answer is \boxed{42}") == "42"

    def test_boxed_with_fraction(self):
        assert _extract_boxed(r"\boxed{\frac{1}{2}}") == r"\frac{1}{2}"

    def test_nested_braces(self):
        assert _extract_boxed(r"\boxed{{a}{b}}") == "{a}{b}"

    def test_no_boxed(self):
        assert _extract_boxed("The answer is 42") is None

    def test_unclosed_boxed(self):
        # Unclosed brace should return None
        assert _extract_boxed(r"\boxed{42") is None

    def test_boxed_with_spaces(self):
        assert _extract_boxed(r"\boxed{  42  }") == "42"


class TestExtractAnswer:
    def test_answer_tags(self):
        assert _extract_answer("<answer>42</answer>") == "42"

    def test_answer_tags_case_insensitive(self):
        assert _extract_answer("<ANSWER>42</ANSWER>") == "42"

    def test_answer_tags_with_spaces(self):
        assert _extract_answer("<answer>  42  </answer>") == "42"

    def test_answer_tags_multiline(self):
        assert _extract_answer("<answer>\n42\n</answer>") == "42"

    def test_falls_back_to_boxed(self):
        assert _extract_answer(r"So the answer is \boxed{42}") == "42"

    def test_prefers_answer_tags_over_boxed(self):
        # If both present, answer tags take priority
        text = r"<answer>1</answer> and also \boxed{2}"
        assert _extract_answer(text) == "1"

    def test_no_answer(self):
        assert _extract_answer("Just some text without an answer") is None


class TestParseNumber:
    def test_integer(self):
        assert _parse_number("42") == 42.0

    def test_float(self):
        assert _parse_number("3.14") == pytest.approx(3.14)

    def test_negative(self):
        assert _parse_number("-7") == -7.0

    def test_fraction(self):
        assert _parse_number("1/2") == pytest.approx(0.5)

    def test_fraction_with_spaces(self):
        assert _parse_number(" 3 / 4 ".replace(" ", "")) == pytest.approx(0.75)

    def test_latex_frac(self):
        assert _parse_number(r"\frac{1}{4}") == pytest.approx(0.25)

    def test_latex_dfrac(self):
        assert _parse_number(r"\dfrac{3}{4}") == pytest.approx(0.75)

    def test_with_commas(self):
        assert _parse_number("1,000") == 1000.0

    def test_invalid_raises(self):
        with pytest.raises(ValueError):
            _parse_number("abc")

    def test_division_by_zero_raises(self):
        with pytest.raises(ZeroDivisionError):
            _parse_number("1/0")


class TestMathVerifier:
    @pytest.fixture
    def verifier(self):
        return MathVerifier(tolerance=1e-3)

    def test_correct_integer(self, verifier):
        assert verifier.verify("<answer>42</answer>", "42") == 1.0

    def test_correct_fraction(self, verifier):
        assert verifier.verify("<answer>0.5</answer>", "1/2") == 1.0

    def test_correct_latex_fraction(self, verifier):
        assert verifier.verify(r"<answer>\frac{1}{2}</answer>", "0.5") == 1.0

    def test_incorrect_answer(self, verifier):
        assert verifier.verify("<answer>41</answer>", "42") == 0.0

    def test_no_answer_in_response(self, verifier):
        assert verifier.verify("I don't know", "42") == 0.0

    def test_invalid_target(self, verifier):
        assert verifier.verify("<answer>42</answer>", "invalid") == 0.0

    def test_invalid_answer(self, verifier):
        assert verifier.verify("<answer>xyz</answer>", "42") == 0.0

    def test_within_tolerance(self, verifier):
        assert verifier.verify("<answer>42.0001</answer>", "42") == 1.0

    def test_outside_tolerance(self, verifier):
        assert verifier.verify("<answer>42.01</answer>", "42") == 0.0

    def test_boxed_answer(self, verifier):
        assert verifier.verify(r"Therefore \boxed{42}", "42") == 1.0

    def test_complex_response(self, verifier):
        response = """
        Let me solve this step by step.
        First, I calculate 7/12 + 5/18 = 21/36 + 10/36 = 31/36
        Then I divide by 31/36: (31/36) / (31/36) = 1
        <answer>1</answer>
        """
        assert verifier.verify(response, "1") == 1.0


class TestMathVerifierAsync:
    @pytest.fixture
    def verifier(self):
        return MathVerifier()

    @pytest.mark.asyncio
    async def test_verify_completions(self, verifier):
        problem = {"answer": "42"}
        completions = [
            "<answer>42</answer>",
            "<answer>41</answer>",
            "<answer>42.0</answer>",
        ]
        scores = await verifier.verify_completions(problem, completions)
        assert scores == [1.0, 0.0, 1.0]

    @pytest.mark.asyncio
    async def test_verify_batch(self, verifier):
        problems = [{"answer": "1"}, {"answer": "2"}, {"answer": "3"}]
        completions = ["<answer>1</answer>", "<answer>3</answer>", "<answer>3</answer>"]
        scores, durations = await verifier.verify_batch(problems, completions)
        assert scores == [1.0, 0.0, 1.0]
        assert all(d == 0.0 for d in durations)

    @pytest.mark.asyncio
    async def test_verify_batch_with_timing(self, verifier):
        problems = [{"answer": "1"}]
        completions = ["<answer>1</answer>"]
        scores, durations, timing_spans = await verifier.verify_batch_with_timing(
            problems, completions
        )
        assert scores == [1.0]
        assert durations == [0.0]
        assert timing_spans == [(0.0, 0.0)]
