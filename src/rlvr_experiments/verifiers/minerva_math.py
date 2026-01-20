"""
Verifier for MATH benchmark requiring strict \\boxed{} format.

This verifier ONLY accepts \\boxed{} format - no fallbacks.
Aligns with lm_eval's minerva_math exact_match metric.
"""

import atexit
import multiprocessing as mp
import os
import queue
import signal
import time
from typing import Optional


def extract_boxed_strict(text: str) -> Optional[str]:
    """Extract content from \\boxed{...}, handling nested braces.

    Takes the LAST \\boxed{} occurrence (rightmost), which is typically
    the final answer after chain-of-thought reasoning.

    Returns None if no \\boxed{} found - NO FALLBACKS.
    """
    idx = text.rfind("\\boxed")
    if idx < 0:
        return None

    i = idx
    while i < len(text) and text[i] != '{':
        i += 1
    if i >= len(text):
        return None

    # Find matching closing brace
    brace_count = 0
    start = i
    while i < len(text):
        if text[i] == '{':
            brace_count += 1
        elif text[i] == '}':
            brace_count -= 1
            if brace_count == 0:
                return text[start + 1:i]
        i += 1

    return None


def normalize_answer(answer: str) -> str:
    """Normalize answer for comparison."""
    if answer is None:
        return ""
    # Basic normalization
    answer = answer.strip()
    answer = answer.replace(" ", "")
    answer = answer.replace("\\left", "")
    answer = answer.replace("\\right", "")
    answer = answer.replace("\\!", "")
    answer = answer.replace("\\,", "")
    answer = answer.replace("\\;", "")
    answer = answer.replace("\\quad", "")
    answer = answer.replace("\\qquad", "")
    answer = answer.replace("dfrac", "frac")
    answer = answer.replace("tfrac", "frac")
    return answer


def _worker_loop(request_queue: mp.Queue, response_queue: mp.Queue) -> None:
    """Long-lived worker process that keeps sympy loaded.

    Processes requests from request_queue and puts results in response_queue.
    Each request is (pred_norm, gold_norm) tuple.
    Response is True/False for equivalence.
    """
    # Import once at startup - this is the expensive part
    from latex2sympy2_extended import latex2sympy
    from sympy import simplify

    while True:
        try:
            pred_norm, gold_norm = request_queue.get()
        except (EOFError, OSError):
            # Parent died or queue closed
            break

        try:
            pred_expr = latex2sympy(pred_norm)
            gold_expr = latex2sympy(gold_norm)
            diff = simplify(pred_expr - gold_expr)
            result = diff == 0
        except Exception:
            result = False

        try:
            response_queue.put(result)
        except (EOFError, OSError):
            # Parent died or queue closed
            break


class SympyWorkerPool:
    """Persistent sympy worker with watchdog-based restart on hang.

    Keeps a single worker process alive with sympy pre-loaded for fast (~5ms)
    equivalence checks. If a check hangs (exceeds timeout), the watchdog kills
    and restarts the worker automatically.

    This avoids both:
    - ProcessPoolExecutor worker accumulation (hung workers never freed)
    - Fresh subprocess per call overhead (1.5s import time)
    """

    def __init__(self, timeout: float = 2.0):
        self.timeout = timeout
        self._worker: Optional[mp.Process] = None
        self._request_queue: Optional[mp.Queue] = None
        self._response_queue: Optional[mp.Queue] = None
        self._ctx = mp.get_context('spawn')
        self._lock = self._ctx.Lock()  # Thread-safe access
        self._started = False

    def _start_worker(self) -> None:
        """Start a fresh worker process."""
        self._request_queue = self._ctx.Queue()
        self._response_queue = self._ctx.Queue()
        self._worker = self._ctx.Process(
            target=_worker_loop,
            args=(self._request_queue, self._response_queue),
            daemon=True
        )
        self._worker.start()
        self._started = True

    def _kill_worker(self) -> None:
        """Force kill the worker process."""
        if self._worker is None:
            return

        if self._worker.is_alive():
            self._worker.terminate()
            self._worker.join(timeout=0.5)
            if self._worker.is_alive():
                try:
                    os.kill(self._worker.pid, signal.SIGKILL)
                except (ProcessLookupError, OSError):
                    pass
                self._worker.join(timeout=0.5)

        # Clean up queues
        self._worker = None
        self._request_queue = None
        self._response_queue = None
        self._started = False

    def _ensure_worker(self) -> None:
        """Ensure worker is running, start if needed."""
        if not self._started or self._worker is None or not self._worker.is_alive():
            self._kill_worker()  # Clean up any dead worker
            self._start_worker()

    def check_equiv(self, pred_norm: str, gold_norm: str) -> bool:
        """Check if two normalized answers are symbolically equivalent.

        Returns False on timeout (hung sympy) - worker is auto-restarted.
        """
        with self._lock:
            self._ensure_worker()

            # Send request
            try:
                self._request_queue.put((pred_norm, gold_norm))
            except Exception:
                # Queue broken, restart worker
                self._kill_worker()
                return False

            # Wait for response with timeout
            try:
                result = self._response_queue.get(timeout=self.timeout)
                return result
            except queue.Empty:
                # Timeout - worker hung, kill and restart
                self._kill_worker()
                return False
            except Exception:
                # Queue broken
                self._kill_worker()
                return False

    def shutdown(self) -> None:
        """Cleanly shut down the worker pool."""
        with self._lock:
            self._kill_worker()


# Module-level singleton pool (lazy initialization)
_POOL: Optional[SympyWorkerPool] = None
_POOL_LOCK = mp.get_context('spawn').Lock()


def _get_pool() -> SympyWorkerPool:
    """Get or create the singleton pool."""
    global _POOL
    with _POOL_LOCK:
        if _POOL is None:
            _POOL = SympyWorkerPool(timeout=2.0)
            # Register cleanup on exit
            atexit.register(_shutdown_pool)
        return _POOL


def _shutdown_pool() -> None:
    """Shutdown the pool on exit."""
    global _POOL
    if _POOL is not None:
        _POOL.shutdown()
        _POOL = None


def _sympy_equiv_with_timeout(pred_norm: str, gold_norm: str, timeout: float = 2.0) -> bool:
    """Check sympy equivalence using the persistent worker pool.

    The timeout parameter is used for one-off calls but the pool uses its
    own configured timeout (2.0s by default).
    """
    pool = _get_pool()
    return pool.check_equiv(pred_norm, gold_norm)


def is_equiv(pred: str, gold: str, timeout: float = 2.0) -> bool:
    """Check if two answers are equivalent.

    Uses string normalization first, then falls back to sympy
    symbolic comparison via the persistent worker pool.

    Fast path (~0.005ms): normalized strings match
    Slow path (~5ms): sympy symbolic comparison
    """
    if pred is None or gold is None:
        return False

    pred_norm = normalize_answer(pred)
    gold_norm = normalize_answer(gold)

    # Direct string match - fast path
    if pred_norm == gold_norm:
        return True

    # Try sympy comparison via persistent worker
    try:
        return _sympy_equiv_with_timeout(pred_norm, gold_norm, timeout)
    except Exception:
        return False


class MinervaMathVerifier:
    """
    Strict verifier for MATH benchmark requiring \\boxed{} format.

    Only accepts answers in \\boxed{} format - no "The answer is X" fallback.
    This aligns with lm_eval's minerva_math exact_match metric.
    """

    def __init__(self):
        pass

    def verify(self, response: str, target: str) -> float:
        """Return 1.0 if response matches target, 0.0 otherwise.

        Args:
            response: Model's generated response (must contain \\boxed{})
            target: Gold solution (we extract \\boxed{} from this too)
        """
        try:
            # Extract answers - strict boxed only
            pred = extract_boxed_strict(response)
            gold = extract_boxed_strict(target)

            if pred is None or gold is None:
                return 0.0

            return 1.0 if is_equiv(pred, gold) else 0.0
        except Exception:
            return 0.0

    async def verify_completions(self, problem: dict, completions: list[str], **kwargs) -> list[float]:
        """Verify N completions for one problem. Returns list of scores."""
        target = problem.get("answer", "")
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
