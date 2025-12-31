"""Ray-distributed code verification."""

from __future__ import annotations

import atexit
import asyncio
import logging
import os
import time

import ray

from .verifier import CodeVerifier
from rlvr_experiments.tracer import get_tracer

logger = logging.getLogger(__name__)


@ray.remote(num_cpus=1)
class _VerifierWorker:
    """Ray actor that runs a CodeVerifier instance."""

    def __init__(self, verifier_cls: type[CodeVerifier], worker_id: int, verifier_kwargs: dict | None = None):
        self.verifier = verifier_cls(**(verifier_kwargs or {}))
        self.worker_id = worker_id

    async def verify_batch(self, problems: list[dict], completions: list[str]) -> tuple[list[float], int, float]:
        """Returns (scores, worker_id, duration_ms)."""
        start = time.perf_counter()
        scores = await self.verifier.verify_batch(problems, completions)
        return scores, self.worker_id, (time.perf_counter() - start) * 1000


class RayCodeVerifier:
    """Distributed code verifier using Ray actor pool."""

    def __init__(self, verifier_cls: type[CodeVerifier], num_workers: int = 4, verifier_kwargs: dict | None = None):
        self.num_workers = num_workers
        self.verifier_kwargs = verifier_kwargs or {}
        self._worker_tid_base = 1000  # Base tid for Perfetto worker threads
        self.workers = [_VerifierWorker.remote(verifier_cls, i, self.verifier_kwargs) for i in range(num_workers)]
        self._idx = 0
        atexit.register(self.shutdown)
        logger.info(f"Created RayCodeVerifier with {num_workers} {verifier_cls.__name__} workers")

    def emit_worker_thread_names(self):
        """Emit Perfetto thread name metadata for workers."""
        tracer = get_tracer()
        if tracer:
            for i in range(self.num_workers):
                tracer._emit({"name": "thread_name", "ph": "M", "pid": os.getpid(),
                              "tid": self._worker_tid_base + i, "args": {"name": f"worker_{i}"}})

    def _next_worker(self):
        worker = self.workers[self._idx % len(self.workers)]
        self._idx += 1
        return worker

    async def verify_completions(self, problem: dict, completions: list[str]) -> list[float]:
        """Verify N completions for one problem. Returns list of scores."""
        tracer = get_tracer()

        # Distribute across workers
        chunk_size = max(1, len(completions) // len(self.workers))
        futures = [
            self._next_worker().verify_batch.remote([problem] * len(completions[i:i + chunk_size]), completions[i:i + chunk_size])
            for i in range(0, len(completions), chunk_size)
        ]

        results = await asyncio.gather(*[asyncio.wrap_future(f.future()) for f in futures])

        # Flatten scores, emit worker spans
        scores = []
        for chunk_scores, worker_id, duration_ms in results:
            scores.extend(chunk_scores)
            if tracer:
                self._emit_worker_span(tracer, worker_id, duration_ms, len(chunk_scores))

        return scores

    def _emit_worker_span(self, tracer, worker_id: int, duration_ms: float, num_verified: int):
        """Emit a Perfetto span for a worker's verification work."""
        tracer._emit({
            "name": "verify", "cat": "worker", "ph": "X",
            "ts": tracer._now_us() - duration_ms * 1000, "dur": duration_ms * 1000,
            "pid": os.getpid(), "tid": self._worker_tid_base + worker_id,
            "args": {"worker_id": worker_id, "num_verified": num_verified},
        })

    def shutdown(self):
        """Shutdown all workers."""
        for worker in self.workers:
            ray.kill(worker)
        self.workers = []
