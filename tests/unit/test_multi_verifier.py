"""Unit tests for MultiVerifier."""

import json
import pytest
from rlvr_experiments.verifiers.multi import MultiVerifier


class TestMultiVerifierInit:
    def test_init(self):
        v = MultiVerifier()
        assert v is not None

    def test_has_all_verifiers(self):
        v = MultiVerifier()
        assert "math" in v._verifiers
        assert "ifeval" in v._verifiers
        assert "humaneval" in v._verifiers
        assert "mbpp" in v._verifiers
        assert "apps" in v._verifiers


class TestMultiVerifierMath:
    @pytest.fixture
    def verifier(self):
        return MultiVerifier()

    @pytest.mark.asyncio
    async def test_verify_math_correct(self, verifier):
        problem = {
            "verifier_type": "math",
            "answer": r"\boxed{42}",
        }
        scores = await verifier.verify_completions(problem, [r"\boxed{42}"])
        assert scores == [1.0]

    @pytest.mark.asyncio
    async def test_verify_math_incorrect(self, verifier):
        problem = {
            "verifier_type": "math",
            "answer": r"\boxed{42}",
        }
        scores = await verifier.verify_completions(problem, [r"\boxed{41}"])
        assert scores == [0.0]

    @pytest.mark.asyncio
    async def test_verify_math_batch(self, verifier):
        problem = {
            "verifier_type": "math",
            "answer": r"\boxed{42}",
        }
        scores = await verifier.verify_completions(
            problem, [r"\boxed{42}", r"\boxed{41}", r"\boxed{42.0}"]
        )
        assert scores == [1.0, 0.0, 1.0]


class TestMultiVerifierIFEval:
    @pytest.fixture
    def verifier(self):
        return MultiVerifier()

    @pytest.mark.asyncio
    async def test_verify_ifeval_lowercase_pass(self, verifier):
        problem = {
            "verifier_type": "ifeval",
            "ground_truth": json.dumps({"func_name": "validate_lowercase"}),
        }
        scores = await verifier.verify_completions(problem, ["hello world"])
        assert scores == [1.0]

    @pytest.mark.asyncio
    async def test_verify_ifeval_lowercase_fail(self, verifier):
        problem = {
            "verifier_type": "ifeval",
            "ground_truth": json.dumps({"func_name": "validate_lowercase"}),
        }
        scores = await verifier.verify_completions(problem, ["Hello World"])
        assert scores == [0.0]

    @pytest.mark.asyncio
    async def test_verify_ifeval_batch(self, verifier):
        problem = {
            "verifier_type": "ifeval",
            "ground_truth": json.dumps({"func_name": "validate_no_commas"}),
        }
        scores = await verifier.verify_completions(
            problem, ["hello world", "hello, world", "test"]
        )
        assert scores == [1.0, 0.0, 1.0]


class TestMultiVerifierDispatch:
    @pytest.fixture
    def verifier(self):
        return MultiVerifier()

    @pytest.mark.asyncio
    async def test_missing_verifier_type_raises(self, verifier):
        problem = {"answer": "42"}  # No verifier_type
        with pytest.raises(ValueError, match="must have 'verifier_type' key"):
            await verifier.verify_completions(problem, ["42"])

    @pytest.mark.asyncio
    async def test_unknown_verifier_type_raises(self, verifier):
        problem = {"verifier_type": "unknown", "answer": "42"}
        with pytest.raises(ValueError, match="Unknown verifier_type"):
            await verifier.verify_completions(problem, ["42"])


class TestMultiVerifierBatch:
    @pytest.fixture
    def verifier(self):
        return MultiVerifier()

    @pytest.mark.asyncio
    async def test_verify_batch_mixed(self, verifier):
        """Test batch verification with mixed verifier types."""
        problems = [
            {"verifier_type": "math", "answer": r"\boxed{1}"},
            {"verifier_type": "ifeval", "ground_truth": json.dumps({"func_name": "validate_lowercase"})},
            {"verifier_type": "math", "answer": r"\boxed{2}"},
        ]
        completions = [
            r"\boxed{1}",  # correct math
            "hello world",  # correct lowercase
            r"\boxed{3}",  # incorrect math
        ]

        scores, durations = await verifier.verify_batch(problems, completions)

        assert scores == [1.0, 1.0, 0.0]
        assert len(durations) == 3
        assert all(d >= 0 for d in durations)

    @pytest.mark.asyncio
    async def test_verify_batch_with_timing(self, verifier):
        """Test batch verification with timing spans."""
        problems = [
            {"verifier_type": "math", "answer": r"\boxed{1}"},
            {"verifier_type": "ifeval", "ground_truth": json.dumps({"func_name": "validate_lowercase"})},
        ]
        completions = [r"\boxed{1}", "hello"]

        scores, durations, timing_spans = await verifier.verify_batch_with_timing(
            problems, completions
        )

        assert scores == [1.0, 1.0]
        assert len(durations) == 2
        assert len(timing_spans) == 2
        assert timing_spans[0][0] == 0.0  # First span starts at 0
