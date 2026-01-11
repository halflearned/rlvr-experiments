from math_verify import parse as _parse_original, verify as _verify_original, LatexExtractionConfig, ExprExtractionConfig


def parse(*args, **kwargs):
    """Wrapper around math_verify.parse that handles signal errors in non-main threads.

    math_verify uses SIGALRM for timeouts, which fails in non-main threads (e.g., Ray workers).
    We disable the timeout by setting it to 0 (which disables the signal-based timeout).
    """
    kwargs.setdefault('parsing_timeout', 0)
    return _parse_original(*args, **kwargs)


def verify(*args, **kwargs):
    """Wrapper around math_verify.verify that handles signal errors in non-main threads.

    math_verify uses SIGALRM for timeouts, which fails in non-main threads (e.g., Ray workers).
    We disable the timeout by setting timeout_seconds=None.
    """
    kwargs.setdefault('timeout_seconds', None)
    return _verify_original(*args, **kwargs)


class MathVerifier:
    """
    Verifier for mathematical answers using math_verify package.

    Supports symbolic comparison of LaTeX expressions, fractions, sets,
    matrices, equations, and more via SymPy.
    """

    def __init__(self):
        # Use both LaTeX and expression extraction for flexibility
        self._extraction_config = [LatexExtractionConfig(), ExprExtractionConfig()]

    def verify(self, response: str, target: str) -> float:
        """Return 1.0 if response matches target, 0.0 otherwise."""
        try:
            # NOTE: Removed \boxed{} requirement - base models don't output it consistently
            # and need to learn to solve problems first before learning format
            gold = parse(target, extraction_config=self._extraction_config)
            answer = parse(response, extraction_config=self._extraction_config)
            if gold is None or answer is None:
                return 0.0
            return 1.0 if verify(gold, answer) else 0.0
        except Exception as e:
            # Log first few exceptions to help debug
            import traceback
            print(f"[VERIFY EXCEPTION] {type(e).__name__}: {e}", flush=True)
            traceback.print_exc()
            return 0.0

    async def verify_completions(self, problem: dict, completions: list[str], **kwargs) -> list[float]:
        """Verify N completions for one problem. Returns list of scores."""
        target = problem["answer"]
        scores = [self.verify(c, target) for c in completions]
        # Debug: log first completion verification details
        if sum(scores) == 0:
            gold = parse(target, extraction_config=self._extraction_config)
            answer = parse(completions[0], extraction_config=self._extraction_config)
            print(f"[VERIFY DEBUG] All 0 scores! gold={gold}, answer={answer}, target_len={len(target)}", flush=True)
        return scores

    async def verify_batch(self, problems: list[dict], completions: list[str]) -> tuple[list[float], list[float]]:
        """Verify a batch. Returns (scores, durations_ms)."""
        scores = [self.verify(c, p["answer"]) for p, c in zip(problems, completions)]
        durations = [0.0] * len(completions)  # instant, no timing needed
        return scores, durations

    async def verify_batch_with_timing(self, problems: list[dict], completions: list[str]) -> tuple[list[float], list[float], list[tuple[float, float]]]:
        """Verify a batch with timing spans. Returns (scores, durations_ms, timing_spans)."""
        # Debug: print what we're getting
        if problems:
            p0 = problems[0]
            c0 = completions[0] if completions else ""
            print(f"[VERIFY_BATCH] problems[0] keys={list(p0.keys())}, answer_len={len(p0.get('answer',''))}", flush=True)
            print(f"[VERIFY_BATCH] answer[:100]={p0.get('answer','')[:100]}", flush=True)
            print(f"[VERIFY_BATCH] completion[:100]={c0[:100]}", flush=True)
        scores = [self.verify(c, p["answer"]) for p, c in zip(problems, completions)]
        if sum(scores) == 0 and problems:
            print(f"[VERIFY_BATCH] ALL ZERO! Checking parse...", flush=True)
            gold = parse(problems[0]["answer"], extraction_config=self._extraction_config)
            answer = parse(completions[0], extraction_config=self._extraction_config)
            print(f"[VERIFY_BATCH] gold={gold}, answer={answer}", flush=True)
        durations = [0.0] * len(completions)
        timing_spans = [(0.0, 0.0)] * len(completions)  # no meaningful timing for math
        return scores, durations, timing_spans
