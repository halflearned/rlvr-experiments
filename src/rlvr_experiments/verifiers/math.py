import multiprocessing as mp
import re
import time
from concurrent.futures import ProcessPoolExecutor, TimeoutError as FuturesTimeoutError


def _extract_last_number(text: str) -> str | None:
    """Extract the last number from text (fallback extraction)."""
    # Remove commas between digits (e.g., "1,234" -> "1234")
    text = re.sub(r"(\d),(\d)", r"\1\2", text)
    # Match integers and decimals, with optional sign
    numbers = re.findall(r"[-+]?(?:\d+\.\d+|\d+)", text)
    if numbers:
        return numbers[-1]
    return None


def _normalize_number(s: str) -> str | None:
    """Normalize a number string for comparison."""
    if s is None:
        return None
    s = s.strip().lstrip("+")
    try:
        f = float(s)
        if f == int(f):
            return str(int(f))
        return s
    except ValueError:
        return None


# Module-level function for subprocess execution (must be picklable)
def _verify_single(response: str, target: str, fallback_extraction: bool = True) -> float:
    """Verify a single response against target. Runs in subprocess.

    Args:
        response: Model's generated response
        target: Gold answer
        fallback_extraction: If True, fall back to last-number extraction when
                            math_verify.parse() returns None for the response
    """
    from math_verify import parse, verify, LatexExtractionConfig, ExprExtractionConfig
    try:
        extraction_config = [LatexExtractionConfig(), ExprExtractionConfig()]
        gold = parse(target, extraction_config=extraction_config)
        answer = parse(response, extraction_config=extraction_config)

        # If math_verify couldn't parse response, try fallback extraction
        if answer is None and fallback_extraction:
            pred_num = _extract_last_number(response)
            if pred_num is not None:
                # Try parsing the extracted number
                answer = parse(pred_num, extraction_config=extraction_config)
                # If still can't parse, try direct numeric comparison
                if answer is None and gold is not None:
                    gold_num = _extract_last_number(target)
                    if gold_num is not None:
                        pred_norm = _normalize_number(pred_num)
                        gold_norm = _normalize_number(gold_num)
                        if pred_norm is not None and gold_norm is not None and pred_norm == gold_norm:
                            return 1.0

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
        Warms up in batches to avoid overwhelming the system.
        """
        # Warm up in batches of 16 to avoid timeout issues with many workers
        batch_size = min(16, self.max_workers)
        for i in range(0, self.max_workers, batch_size):
            batch_count = min(batch_size, self.max_workers - i)
            futures = [self._executor.submit(_warmup) for _ in range(batch_count)]
            for f in futures:
                try:
                    f.result(timeout=60.0)
                except Exception:
                    pass  # Continue even if some workers fail to warm up

    def verify(self, response: str, target: str) -> float:
        """Return 1.0 if response matches target, 0.0 otherwise. Uses subprocess with timeout."""
        try:
            future = self._executor.submit(_verify_single, response, target)
            return future.result(timeout=self.timeout)
        except FuturesTimeoutError:
            return 0.0
        except Exception:
            return 0.0

    def verify_batch_parallel(self, completions: list[str], target: str) -> list[float]:
        """Verify multiple completions against a single target in parallel.

        Submits all verification tasks at once and collects results.
        """
        futures = [
            self._executor.submit(_verify_single, completion, target)
            for completion in completions
        ]

        scores = []
        for f in futures:
            try:
                score = f.result(timeout=self.timeout)
            except FuturesTimeoutError:
                score = 0.0
            except Exception:
                score = 0.0
            scores.append(score)

        return scores

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
