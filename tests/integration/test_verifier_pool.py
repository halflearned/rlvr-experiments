"""Integration tests for VerifierPool with actual Ray workers."""

import pytest
import asyncio

from rlvr_experiments.verifiers.math import MathVerifier
from rlvr_experiments.verifiers.distributed import VerifierPool


@pytest.mark.integration
class TestVerifierPoolWithMath:
    """Test VerifierPool with MathVerifier (no Docker needed)."""

    @pytest.fixture
    def pool(self, ray_context):
        pool = VerifierPool(MathVerifier, num_workers=2)
        yield pool
        pool.shutdown()

    @pytest.mark.asyncio
    async def test_verify_completions_correct(self, pool):
        problem = {"answer": "42"}
        completions = ["<answer>42</answer>"]

        scores = await pool.verify_completions(problem, completions)

        assert scores == [1.0]

    @pytest.mark.asyncio
    async def test_verify_completions_incorrect(self, pool):
        problem = {"answer": "42"}
        completions = ["<answer>41</answer>"]

        scores = await pool.verify_completions(problem, completions)

        assert scores == [0.0]

    @pytest.mark.asyncio
    async def test_verify_completions_mixed(self, pool):
        problem = {"answer": "10"}
        completions = [
            "<answer>10</answer>",
            "<answer>20</answer>",
            "<answer>10.0</answer>",
            "no answer here",
        ]

        scores = await pool.verify_completions(problem, completions)

        assert scores == [1.0, 0.0, 1.0, 0.0]

    @pytest.mark.asyncio
    async def test_round_robin_distribution(self, pool):
        problem = {"answer": "1"}
        completions = ["<answer>1</answer>"]

        # Make multiple calls and verify round-robin
        for _ in range(4):
            scores = await pool.verify_completions(problem, completions)
            assert scores == [1.0]

        # Check internal index advanced
        assert pool._idx == 4

    @pytest.mark.asyncio
    async def test_concurrent_verifications(self, pool):
        problem = {"answer": "5"}
        completions = ["<answer>5</answer>", "<answer>6</answer>"]

        # Run multiple verifications concurrently
        tasks = [
            pool.verify_completions(problem, completions)
            for _ in range(10)
        ]
        results = await asyncio.gather(*tasks)

        # All should return same results
        for scores in results:
            assert scores == [1.0, 0.0]

    @pytest.mark.asyncio
    async def test_explicit_worker_id(self, pool):
        problem = {"answer": "1"}
        completions = ["<answer>1</answer>"]

        # Force specific worker
        scores = await pool.verify_completions(problem, completions, worker_id=0)
        assert scores == [1.0]

        scores = await pool.verify_completions(problem, completions, worker_id=1)
        assert scores == [1.0]


@pytest.mark.integration
class TestVerifierPoolLifecycle:
    def test_workers_created(self, ray_context):
        pool = VerifierPool(MathVerifier, num_workers=3)
        try:
            assert len(pool.workers) == 3
            assert pool.num_workers == 3
        finally:
            pool.shutdown()

    def test_shutdown_kills_workers(self, ray_context):
        pool = VerifierPool(MathVerifier, num_workers=2)
        workers = pool.workers.copy()

        pool.shutdown()

        assert pool.workers == []
        # Workers should be dead
        for worker in workers:
            with pytest.raises(Exception):  # RayActorError
                ray_context.get(worker.ready.remote())

    def test_verifier_kwargs_passed(self, ray_context):
        # MathVerifier accepts tolerance
        pool = VerifierPool(MathVerifier, num_workers=1, tolerance=0.1)
        try:
            assert pool.verifier_kwargs == {"tolerance": 0.1}
        finally:
            pool.shutdown()


@pytest.mark.integration
class TestVerifierPoolErrorHandling:
    def test_bad_verifier_class_raises(self, ray_context):
        class BadVerifier:
            def __init__(self):
                raise ValueError("Init failed!")

        with pytest.raises(RuntimeError, match="worker failed to initialize"):
            VerifierPool(BadVerifier, num_workers=1)

    @pytest.mark.asyncio
    async def test_verifier_handles_edge_cases(self, ray_context):
        pool = VerifierPool(MathVerifier, num_workers=1)
        try:
            # Empty completions list
            problem = {"answer": "1"}
            scores = await pool.verify_completions(problem, [])
            assert scores == []

            # Invalid answer format in problem
            problem = {"answer": "not_a_number"}
            scores = await pool.verify_completions(problem, ["<answer>42</answer>"])
            assert scores == [0.0]
        finally:
            pool.shutdown()
