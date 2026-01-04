from __future__ import annotations

import asyncio
import atexit
import inspect
import json
import os
import signal
import threading
import time
from contextlib import contextmanager, nullcontext
from functools import wraps
from typing import Any, Callable, Dict, Optional, ParamSpec, TypeVar


class TraceRecorder:
    """
    RLVR Trace Recorder - simple JSONL format optimized for streaming.

    Event types:
      - span: {type: "span", name, ts, dur, ...extra}
      - counter: {type: "counter", name, ts, ...values}
      - meta: {type: "meta", ...data}

    All timestamps are in seconds (float) relative to trace start.
    """

    def __init__(self, path: str, *, use_task_ids: bool = True) -> None:
        self.path = path
        self._lock = threading.Lock()
        self._use_task_ids = use_task_ids
        self._file: Optional[Any] = None
        self._start_ns = time.perf_counter_ns()
        self._open_file()

    def _open_file(self) -> None:
        out_dir = os.path.dirname(self.path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        self._file = open(self.path, "w")

    def _now_s(self) -> float:
        """Current time in seconds since trace start."""
        return (time.perf_counter_ns() - self._start_ns) / 1e9

    def _get_task_id(self) -> int | None:
        """Get asyncio task ID if available."""
        if self._use_task_ids:
            try:
                task = asyncio.current_task()
                if task is not None:
                    return id(task) % 1_000_000
            except RuntimeError:
                pass
        return None

    def _emit(self, event: Dict[str, object]) -> None:
        with self._lock:
            if self._file is not None:
                self._file.write(json.dumps(event) + "\n")
                self._file.flush()

    @contextmanager
    def span(
        self,
        name: str,
        *,
        cat: str | None = None,
        tid: int | None = None,
        args: Dict[str, object] | None = None,
    ):
        """Emit a duration span event."""
        ts = self._now_s()
        try:
            yield
        finally:
            dur = self._now_s() - ts
            event = {"type": "span", "name": name, "ts": ts, "dur": dur}
            if cat:
                event["cat"] = cat
            if args:
                event.update(args)
            self._emit(event)

    def instant(self, name: str, **kwargs) -> None:
        """Emit an instant event (point in time)."""
        event = {"type": "instant", "name": name, "ts": self._now_s()}
        event.update(kwargs)
        self._emit(event)

    def counter(self, name: str, values: Dict[str, Any]) -> None:
        """Emit a counter/metrics event. Values are flattened into the event."""
        event = {"type": "counter", "name": name, "ts": self._now_s()}
        event.update(values)
        self._emit(event)

    def buffer(
        self,
        size: int,
        by_version: Dict[int, int] | None = None,
        fates: Dict[str, Dict[int, int]] | None = None,
    ) -> None:
        """Emit buffer state with item fate tracking.

        Args:
            size: Current buffer size
            by_version: Per-version item counts currently in buffer
            fates: Per-version cumulative fate counts:
                - used: Items fully consumed (good)
                - wasted: Items evicted without any reads (bad)
                - partial: Items evicted with some reads (in between)
        """
        event = {"type": "buffer", "ts": self._now_s(), "size": size}
        if by_version:
            event["by_version"] = by_version
        if fates:
            event["fates"] = fates
        self._emit(event)

    def meta(self, **kwargs) -> None:
        """Emit metadata event."""
        event = {"type": "meta", "ts": self._now_s()}
        event.update(kwargs)
        self._emit(event)

    def close(self) -> None:
        with self._lock:
            if self._file is not None:
                self._file.close()
                self._file = None
                print(f"[tracer] Wrote trace to {self.path}")


_TRACE_ENV = "RLVR_TRACE_PATH"
_TRACE_WORKERS_ENV = "RLVR_TRACE_WORKERS"
_GLOBAL_TRACER: Optional[TraceRecorder] = None
_SIGNAL_HANDLER_ACTIVE = False  # Guard against reentrant signal handling


def _in_ray_worker() -> bool:
    return bool(os.environ.get("RAY_WORKER_ID"))


def _path_with_pid(path: str) -> str:
    root, ext = os.path.splitext(path)
    return f"{root}.{os.getpid()}{ext}"


def _signal_handler(signum, frame):
    """Handle SIGINT/SIGTERM to ensure trace is written, then show traceback."""
    global _SIGNAL_HANDLER_ACTIVE
    # Prevent reentrancy - if we're already handling a signal, just exit
    if _SIGNAL_HANDLER_ACTIVE:
        os._exit(1)
    _SIGNAL_HANDLER_ACTIVE = True

    import sys
    import traceback
    print("\n--- Interrupted ---", file=sys.stderr)
    traceback.print_stack(frame, file=sys.stderr)
    close_tracer()
    os._exit(1)  # Force exit, skip atexit handlers that might hang


def init_global_tracer(path: str, *, use_task_ids: bool = True) -> Optional[TraceRecorder]:
    global _GLOBAL_TRACER
    if not path:
        return None
    _GLOBAL_TRACER = TraceRecorder(path, use_task_ids=use_task_ids)
    atexit.register(_GLOBAL_TRACER.close)
    # Also handle Ctrl+C and termination signals
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)
    return _GLOBAL_TRACER


def get_tracer() -> Optional[TraceRecorder]:
    return _GLOBAL_TRACER


def close_tracer() -> None:
    tracer = get_tracer()
    if tracer is not None:
        tracer.close()


def set_current_task_name(name: str) -> None:
    tracer = get_tracer()
    if tracer is not None:
        tracer.set_current_task_name(name)


def _init_from_env() -> None:
    path = os.environ.get(_TRACE_ENV)
    if not path:
        return
    if _in_ray_worker():
        if os.environ.get(_TRACE_WORKERS_ENV) != "1":
            return
        path = _path_with_pid(path)
    init_global_tracer(path)


_init_from_env()


def trace_span(name: str, **kwargs):
    tracer = get_tracer()
    if tracer is None:
        return nullcontext()
    return tracer.span(name, **kwargs)


P = ParamSpec("P")
R = TypeVar("R")


def traced(name: Optional[str] = None) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Decorator to wrap a function in a trace span."""
    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        span_name = name or func.__name__

        if inspect.iscoroutinefunction(func):
            @wraps(func)
            async def async_wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
                with trace_span(span_name):
                    return await func(*args, **kwargs)

            return async_wrapper

        @wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            with trace_span(span_name):
                return func(*args, **kwargs)

        return wrapper

    return decorator
