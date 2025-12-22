from __future__ import annotations

import asyncio
import atexit
import inspect
import json
import os
import threading
import time
from contextlib import contextmanager, nullcontext
from functools import wraps
from typing import Any, Callable, Dict, Optional, ParamSpec, TypeVar


class TraceRecorder:
    def __init__(self, path: str, *, use_task_ids: bool = True) -> None:
        self.path = path
        self._events: list[Dict[str, object]] = []
        self._lock = threading.Lock()
        self._pid = os.getpid()
        self._use_task_ids = use_task_ids
        self._task_tids: dict[asyncio.Task, int] = {}
        self._next_tid = 1

    def _now_us(self) -> float:
        return time.perf_counter_ns() / 1000.0

    def _get_tid(self) -> int:
        if self._use_task_ids:
            try:
                task = asyncio.current_task()
            except RuntimeError:
                task = None
            if task is not None:
                tid = self._task_tids.get(task)
                if tid is None:
                    tid = self._next_tid
                    self._next_tid += 1
                    self._task_tids[task] = tid
                return tid
        return threading.get_ident()

    def _emit(self, event: Dict[str, object]) -> None:
        with self._lock:
            self._events.append(event)

    def set_current_task_name(self, name: str) -> None:
        tid = self._get_tid()
        self._emit(
            {
                "name": "thread_name",
                "ph": "M",
                "pid": self._pid,
                "tid": tid,
                "args": {"name": name},
            }
        )

    @contextmanager
    def span(
        self,
        name: str,
        *,
        cat: str = "trace",
        args: Optional[Dict[str, object]] = None,
    ):
        ts = self._now_us()
        tid = self._get_tid()
        try:
            yield
        finally:
            dur = self._now_us() - ts
            event = {
                "name": name,
                "cat": cat,
                "ph": "X",
                "ts": ts,
                "dur": dur,
                "pid": self._pid,
                "tid": tid,
            }
            if args:
                event["args"] = args
            self._emit(event)

    def dump(self) -> None:
        out_dir = os.path.dirname(self.path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        payload = {"traceEvents": self._events, "displayTimeUnit": "ms"}
        with open(self.path, "w") as f:
            json.dump(payload, f)


_TRACE_ENV = "RLVR_TRACE_PATH"
_TRACE_WORKERS_ENV = "RLVR_TRACE_WORKERS"
_GLOBAL_TRACER: Optional[TraceRecorder] = None


def _in_ray_worker() -> bool:
    return bool(os.environ.get("RAY_WORKER_ID"))


def _path_with_pid(path: str) -> str:
    root, ext = os.path.splitext(path)
    return f"{root}.{os.getpid()}{ext}"


def init_global_tracer(path: str) -> Optional[TraceRecorder]:
    global _GLOBAL_TRACER
    if not path:
        return None
    _GLOBAL_TRACER = TraceRecorder(path)
    atexit.register(_GLOBAL_TRACER.dump)
    return _GLOBAL_TRACER


def get_tracer() -> Optional[TraceRecorder]:
    return _GLOBAL_TRACER


def dump_traces() -> None:
    tracer = get_tracer()
    if tracer is not None:
        tracer.dump()


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


def trace_span(tracer: Optional[TraceRecorder], name: str, **kwargs):
    tracer = tracer or get_tracer()
    if tracer is None:
        return nullcontext()
    return tracer.span(name, **kwargs)


P = ParamSpec("P")
R = TypeVar("R")


def _resolve_tracer(
    sig: inspect.Signature,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    tracer_arg: str,
) -> Optional[TraceRecorder]:
    tracer = None
    if tracer_arg in sig.parameters:
        try:
            bound = sig.bind_partial(*args, **kwargs)
        except TypeError:
            bound = None
        if bound is not None:
            tracer = bound.arguments.get(tracer_arg)
    return tracer or get_tracer()


def traced(
    name: Optional[str] = None,
    *,
    tracer_arg: str = "tracer",
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        span_name = name or func.__name__
        sig = inspect.signature(func)

        if inspect.iscoroutinefunction(func):
            @wraps(func)
            async def async_wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
                tracer = _resolve_tracer(sig, args, kwargs, tracer_arg)
                with trace_span(tracer, span_name):
                    return await func(*args, **kwargs)

            return async_wrapper

        @wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            tracer = _resolve_tracer(sig, args, kwargs, tracer_arg)
            with trace_span(tracer, span_name):
                return func(*args, **kwargs)

        return wrapper

    return decorator
