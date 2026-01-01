"""Distributed verification pool."""

from __future__ import annotations

import atexit
import asyncio
import logging
import os
import time

import ray

from .code import CodeVerifier
from rlvr_experiments.tracer import get_tracer

logger = logging.getLogger(__name__)


@ray.remote(num_cpus=1, max_concurrency=1)
class _VerifierWorker:
    """Ray actor that runs a CodeVerifier instance."""

    def __init__(self, verifier_cls: type[CodeVerifier], worker_id: int, verifier_kwargs: dict | None = None):
        self.verifier = verifier_cls(**(verifier_kwargs or {}))
        self.worker_id = worker_id

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

    def __init__(self, verifier_cls: type[CodeVerifier], num_workers: int = 4, **verifier_kwargs):
        self.num_workers = num_workers
        self.verifier_kwargs = verifier_kwargs
        self._worker_tid_base = 1000  # Base tid for Perfetto worker threads
        self.workers = [_VerifierWorker.remote(verifier_cls, i, verifier_kwargs) for i in range(num_workers)]
        self._idx = 0
        # Track last span end time per worker to prevent overlaps in trace
        self._worker_last_end_us: dict[int, float] = {}
        atexit.register(self.shutdown)
        self._register_tracer_threads()
        logger.info(f"Created VerifierPool with {num_workers} {verifier_cls.__name__} workers")

    def _register_tracer_threads(self):
        """Register worker thread names with Perfetto tracer."""
        tracer = get_tracer()
        if not tracer:
            return
        for i in range(self.num_workers):
            base_tid = self._worker_tid_base + i * 3
            tracer._emit({"name": "thread_name", "ph": "M", "pid": os.getpid(),
                          "tid": base_tid, "args": {"name": f"verifier_{i:02d}"}})
            tracer._emit({"name": "thread_name", "ph": "M", "pid": os.getpid(),
                          "tid": base_tid + 1, "args": {"name": f"verifier_{i:02d}.0"}})
            tracer._emit({"name": "thread_name", "ph": "M", "pid": os.getpid(),
                          "tid": base_tid + 2, "args": {"name": f"verifier_{i:02d}.1"}})

    async def verify_completions(self, problem: dict, completions: list[str], worker_id: int | None = None) -> list[float]:
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

        # Emit nested spans: parent for the batch, children for each completion
        if tracer:
            now_us = tracer._now_us()
            dur_us = worker_duration_ms * 1000
            # Use adjacent tids: base (parent), base+1 (lane 0), base+2 (lane 1)
            base_tid = self._worker_tid_base + wid * 3

            # Place span ending at now, but ensure no overlap with previous span on this worker
            # (clock drift between worker and main process can cause calculated start to be before previous end)
            last_end = self._worker_last_end_us.get(wid, 0)
            batch_start_us = max(now_us - dur_us, last_end)
            batch_end_us = batch_start_us + dur_us
            self._worker_last_end_us[wid] = batch_end_us

            passed_count = sum(1 for s in scores if s > 0)

            # Parent span for the whole verification batch
            tracer._emit({
                "name": "verify",
                "cat": "verify",
                "ph": "X",
                "ts": batch_start_us,
                "dur": dur_us,
                "pid": os.getpid(),
                "tid": base_tid,
                "args": {
                    "n": len(completions),
                    "passed": passed_count,
                    "failed": len(completions) - passed_count,
                },
            })

            # Child spans for each completion
            # timing_spans are relative to worker's batch_start, scale to fit parent
            # Use sub-tids to show concurrent executions on separate rows
            # Sort by start time and assign to alternating lanes
            if timing_spans:
                worker_total_ms = max(start + dur for start, dur in timing_spans)
                scale = worker_duration_ms / worker_total_ms if worker_total_ms > 0 else 1.0

                # Sort by start time and assign lane based on execution order
                indexed_spans = list(enumerate(timing_spans))
                indexed_spans.sort(key=lambda x: x[1][0])  # Sort by start_offset_ms

                for lane, (i, (start_offset_ms, dur_ms)) in enumerate(indexed_spans):
                    passed = scores[i] > 0
                    # Alternate between lane 0 (base_tid+1) and lane 1 (base_tid+2)
                    sub_tid = base_tid + 1 + (lane % 2)
                    tracer._emit({
                        "name": "pass" if passed else "fail",
                        "cat": "exec",
                        "ph": "X",
                        "ts": batch_start_us + start_offset_ms * scale * 1000,
                        "dur": dur_ms * scale * 1000,
                        "pid": os.getpid(),
                        "tid": sub_tid,
                        "args": {"idx": i, "ms": round(dur_ms, 1)},
                    })

        return scores

    def shutdown(self):
        """Shutdown all workers."""
        for worker in self.workers:
            ray.kill(worker)
        self.workers = []
