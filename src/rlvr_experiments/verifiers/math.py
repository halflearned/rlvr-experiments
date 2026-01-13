import multiprocessing as mp
import time
from concurrent.futures import ProcessPoolExecutor, TimeoutError as FuturesTimeoutError


# Module-level function for subprocess execution (must be picklable)
def _verify_single(response: str, target: str) -> float:
    """Verify a single response against target. Runs in subprocess."""
    from math_verify import parse, verify, LatexExtractionConfig, ExprExtractionConfig
    try:
        extraction_config = [LatexExtractionConfig(), ExprExtractionConfig()]
        gold = parse(target, extraction_config=extraction_config)
        answer = parse(response, extraction_config=extraction_config)
        if gold is None or answer is None:
            return 0.0
        return 1.0 if verify(gold, answer) else 0.0
    except Exception:
        return 0.0


def _warmup() -> bool:
    """Import math_verify in subprocess to warm up. Returns True."""
    from math_verify import parse, verify  # noqa: F401
    return True


class MathVerifier:
    """
    Verifier for mathematical answers using math_verify package.

    Uses subprocess with hard timeout to prevent sympy hangs from blocking.
    """

    def __init__(self, timeout: float = 5.0, max_workers: int = 4, warmup: bool = True):
        self.timeout = timeout
        self.max_workers = max_workers
        # Use spawn to avoid issues with forking in threaded environments
        self._executor = ProcessPoolExecutor(
            max_workers=max_workers,
            mp_context=mp.get_context('spawn')
        )
        if warmup:
            self._warmup_workers()

    def _warmup_workers(self) -> None:
        """Pre-import math_verify in all worker processes.

        This avoids the ~1.5s import time on first verification call.
        """
        futures = [self._executor.submit(_warmup) for _ in range(self.max_workers)]
        for f in futures:
            f.result(timeout=30.0)  # Long timeout for initial import

    def verify(self, response: str, target: str) -> float:
        """Return 1.0 if response matches target, 0.0 otherwise. Uses subprocess with timeout."""
        try:
            future = self._executor.submit(_verify_single, response, target)
            return future.result(timeout=self.timeout)
        except FuturesTimeoutError:
            return 0.0
        except Exception:
            return 0.0

    async def verify_completions(self, problem: dict, completions: list[str], **kwargs) -> list[float]:
        """Verify N completions for one problem. Returns list of scores."""
        target = problem["answer"]
        scores = [self.verify(c, target) for c in completions]
        return scores

    async def verify_batch(self, problems: list[dict], completions: list[str]) -> tuple[list[float], list[float]]:
        """Verify a batch. Returns (scores, durations_ms)."""
        scores = []
        durations = []
        for p, c in zip(problems, completions):
            t0 = time.perf_counter()
            score = self.verify(c, p["answer"])
            dur = (time.perf_counter() - t0) * 1000
            scores.append(score)
            durations.append(dur)
        return scores, durations

    async def verify_batch_with_timing(self, problems: list[dict], completions: list[str]) -> tuple[list[float], list[float], list[tuple[float, float]]]:
        """Verify a batch with timing spans. Returns (scores, durations_ms, timing_spans)."""
        scores = []
        durations = []
        timing_spans = []
        offset = 0.0
        for p, c in zip(problems, completions):
            t0 = time.perf_counter()
            score = self.verify(c, p["answer"])
            dur_ms = (time.perf_counter() - t0) * 1000
            scores.append(score)
            durations.append(dur_ms)
            timing_spans.append((offset, dur_ms))
            offset += dur_ms
        return scores, durations, timing_spans
