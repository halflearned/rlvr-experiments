"""Distributed verification pool."""

from __future__ import annotations

import atexit
import asyncio
import logging
import time

import ray

from rlvr_experiments.tracer import get_tracer

logger = logging.getLogger(__name__)


@ray.remote(num_cpus=1, max_concurrency=1)
class _VerifierWorker:
    """Ray actor that runs a CodeVerifier instance."""

    def __init__(self, verifier_cls: type, worker_id: int, verifier_kwargs: dict | None = None):
        self.verifier = verifier_cls(**(verifier_kwargs or {}))
        self.worker_id = worker_id

    def ready(self) -> bool:
        """Check that worker initialized successfully."""
        return True

    async def verify_batch(self, problems: list[dict], completions: list[str]) -> tuple[list[float], list[float], list[tuple[float, float]], int, float]:
        """Returns (scores, durations_ms, timing_spans, worker_id, total_duration_ms).

        timing_spans is list of (start_offset_ms, duration_ms) for each completion.
        """
        batch_start = time.perf_counter()
        scores, durations, timing_spans = await self.verifier.verify_batch_with_timing(problems, completions)
        return scores, durations, timing_spans, self.worker_id, (time.perf_counter() - batch_start) * 1000


class VerifierPool:
    """Distributed verifier pool.

    Args:
        verifier_cls: The verifier class to instantiate on each worker (e.g., MBPPVerifier)
        num_workers: Number of parallel workers
        **verifier_kwargs: Passed to verifier_cls constructor (e.g., timeout=10.0)
    """

    def __init__(self, verifier_cls: type, num_workers: int = 4, **verifier_kwargs):
        self.num_workers = num_workers
        self.verifier_kwargs = verifier_kwargs
        self.workers = [_VerifierWorker.remote(verifier_cls, i, verifier_kwargs) for i in range(num_workers)]
        self._idx = 0

        # Wait for all workers to initialize - surfaces errors immediately
        try:
            ray.get([w.ready.remote() for w in self.workers])
        except ray.exceptions.RayActorError as e:
            raise RuntimeError(f"VerifierPool worker failed to initialize: {e}") from e

        atexit.register(self.shutdown)
        self._register_tracer_metadata()
        logger.info(f"Created VerifierPool with {num_workers} {verifier_cls.__name__} workers")

    def _register_tracer_metadata(self):
        """Register worker metadata with tracer."""
        tracer = get_tracer()
        if not tracer:
            return
        # Get max_concurrent from verifier_kwargs (default 2)
        max_concurrent = self.verifier_kwargs.get('max_concurrent', 2)
        tracer.meta(verifier_workers=self.num_workers, verifier_max_concurrent=max_concurrent)

    async def verify_completions(self, problem: dict, completions: list[str], worker_id: int | None = None, version: int | None = None) -> list[float]:
        """Verify N completions for one problem. Returns list of scores (0.0 or 1.0)."""
        from rlvr_experiments.tracer import get_tracer

        if worker_id is not None:
            wid = worker_id % len(self.workers)
        else:
            wid = self._idx % len(self.workers)
            self._idx += 1

        worker = self.workers[wid]
        tracer = get_tracer()

        future = worker.verify_batch.remote([problem] * len(completions), completions)
        scores, durations, timing_spans, _, worker_duration_ms = await asyncio.wrap_future(future.future())

        # Emit per-slot verification spans for visualization
        # timing_spans contains (start_offset_ms, duration_ms) for each completion
        if tracer:
            now_s = tracer._now_s()
            batch_dur_s = worker_duration_ms / 1000.0
            batch_start_s = now_s - batch_dur_s
            max_concurrent = self.verifier_kwargs.get('max_concurrent', 2)

            # Assign each completion to a slot based on timing order
            # Sort by start time to assign slots in order of execution
            indexed_spans = [(i, timing_spans[i]) for i in range(len(timing_spans))]
            indexed_spans.sort(key=lambda x: x[1][0])  # Sort by start_offset_ms

            # Track when each slot becomes free
            slot_free_times = [0.0] * max_concurrent

            for idx, (start_offset_ms, dur_ms) in indexed_spans:
                # Find the slot that becomes free earliest
                slot = min(range(max_concurrent), key=lambda s: slot_free_times[s])
                slot_free_times[slot] = start_offset_ms + dur_ms

                tracer._emit({
                    "type": "span",
                    "name": "verify",
                    "ts": batch_start_s + start_offset_ms / 1000.0,
                    "dur": dur_ms / 1000.0,
                    "worker": wid,
                    "slot": slot,
                    "passed": 1 if scores[idx] > 0 else 0,
                    "prompt_id": problem.get("prompt_id", "unknown"),
                    "version": version,
                })

        return scores

    def shutdown(self):
        """Shutdown all workers."""
        for worker in self.workers:
            ray.kill(worker)
        self.workers = []
