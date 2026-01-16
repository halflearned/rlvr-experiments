"""Unit tests for verifiers."""

import pytest
from rlvr_experiments.verifiers.math import MathVerifier


class TestMathVerifier:
    @pytest.fixture
    def verifier(self):
        return MathVerifier()

    # Basic numeric verification
    def test_correct_integer(self, verifier):
        assert verifier.verify("$42$", "$42$") == 1.0

    def test_correct_integer_boxed(self, verifier):
        assert verifier.verify(r"\boxed{42}", r"\boxed{42}") == 1.0

    def test_incorrect_answer(self, verifier):
        assert verifier.verify(r"\boxed{41}", r"\boxed{42}") == 0.0

    # Symbolic equivalence - fractions
    def test_fraction_equivalence(self, verifier):
        assert verifier.verify(r"\boxed{\frac{1}{2}}", r"\boxed{0.5}") == 1.0

    def test_fraction_equivalence_reversed(self, verifier):
        assert verifier.verify(r"\boxed{0.5}", r"\boxed{\frac{1}{2}}") == 1.0

    def test_equivalent_fractions(self, verifier):
        assert verifier.verify(r"\boxed{\frac{2}{4}}", r"\boxed{\frac{1}{2}}") == 1.0

    def test_dfrac_equivalence(self, verifier):
        assert verifier.verify(r"\boxed{\dfrac{3}{4}}", r"\boxed{0.75}") == 1.0

    # Symbolic equivalence - expressions
    def test_sqrt_equivalence(self, verifier):
        assert verifier.verify(r"\boxed{\sqrt{4}}", r"\boxed{2}") == 1.0

    def test_expression_equivalence(self, verifier):
        assert verifier.verify(r"\boxed{2+3}", r"\boxed{5}") == 1.0

    def test_power_equivalence(self, verifier):
        assert verifier.verify(r"\boxed{2^3}", r"\boxed{8}") == 1.0

    # Set equivalence
    def test_set_equivalence(self, verifier):
        assert verifier.verify(r"$\{1, 2, 3\}$", r"$\{3, 2, 1\}$") == 1.0

    # Negative numbers
    def test_negative_number(self, verifier):
        assert verifier.verify(r"\boxed{-7}", r"\boxed{-7}") == 1.0

    def test_negative_fraction(self, verifier):
        assert verifier.verify(r"\boxed{-\frac{1}{2}}", r"\boxed{-0.5}") == 1.0

    # Edge cases
    def test_no_answer_in_response(self, verifier):
        assert verifier.verify("I don't know", r"\boxed{42}") == 0.0

    def test_malformed_latex(self, verifier):
        # Should not crash, just return 0
        assert verifier.verify(r"\boxed{", r"\boxed{42}") == 0.0

    def test_empty_response(self, verifier):
        assert verifier.verify("", r"\boxed{42}") == 0.0

    # Real MATH dataset style answers
    def test_complex_solution_with_boxed(self, verifier):
        response = r"""
        Let me solve this step by step.
        First, we have $\frac{7}{12} + \frac{5}{18} = \frac{21}{36} + \frac{10}{36} = \frac{31}{36}$
        Then dividing by $\frac{31}{36}$: $\frac{31/36}{31/36} = 1$
        Therefore, the answer is $\boxed{1}$
        """
        target = r"The answer is $\boxed{1}$"
        assert verifier.verify(response, target) == 1.0

    def test_latex_fraction_in_solution(self, verifier):
        response = r"After simplification, we get $\boxed{\frac{1}{4}}$"
        target = r"$\boxed{\frac{1}{4}}$"
        assert verifier.verify(response, target) == 1.0

    def test_decimal_vs_fraction(self, verifier):
        response = r"\boxed{0.25}"
        target = r"\boxed{\frac{1}{4}}"
        assert verifier.verify(response, target) == 1.0


class TestMathVerifierAsync:
    @pytest.fixture
    def verifier(self):
        return MathVerifier()

    @pytest.mark.asyncio
    async def test_verify_completions(self, verifier):
        problem = {"answer": r"\boxed{42}"}
        completions = [
            r"\boxed{42}",
            r"\boxed{41}",
            r"\boxed{42.0}",
        ]
        scores = await verifier.verify_completions(problem, completions)
        assert scores == [1.0, 0.0, 1.0]

    @pytest.mark.asyncio
    async def test_verify_completions_symbolic(self, verifier):
        problem = {"answer": r"\boxed{\frac{1}{2}}"}
        completions = [
            r"\boxed{0.5}",
            r"\boxed{\frac{2}{4}}",
            r"\boxed{0.25}",
        ]
        scores = await verifier.verify_completions(problem, completions)
        assert scores == [1.0, 1.0, 0.0]

    @pytest.mark.asyncio
    async def test_verify_batch(self, verifier):
        problems = [
            {"answer": r"\boxed{1}"},
            {"answer": r"\boxed{2}"},
            {"answer": r"\boxed{3}"},
        ]
        completions = [
            r"\boxed{1}",
            r"\boxed{3}",
            r"\boxed{3}",
        ]
        scores, durations = await verifier.verify_batch(problems, completions)
        assert scores == [1.0, 0.0, 1.0]
        # Durations should be non-negative (actual timing values)
        assert all(d >= 0.0 for d in durations)
        assert len(durations) == 3

    @pytest.mark.asyncio
    async def test_verify_batch_with_timing(self, verifier):
        problems = [{"answer": r"\boxed{1}"}]
        completions = [r"\boxed{1}"]
        scores, durations, timing_spans = await verifier.verify_batch_with_timing(
            problems, completions
        )
        assert scores == [1.0]
        # Durations should be non-negative (actual timing values)
        assert len(durations) == 1
        assert durations[0] >= 0.0
        # Timing spans should be valid (start <= end)
        assert len(timing_spans) == 1
        assert timing_spans[0][0] <= timing_spans[0][1]
