"""Lightweight sample-level logging for RL training.

Logs the journey of each sample through the pipeline as JSONL.
Easy to query with jq or load into pandas.

Usage:
    from rlvr_experiments.sample_logger import init_sample_logger, log_sample, close_sample_logger

    init_sample_logger("samples.jsonl")

    # Log anything - just pass event type and data
    log_sample("verification", problem_id="123", passed=3, failed=5)
    log_sample("training", problem_id="123", step=42, included=True)

    close_sample_logger()
"""

import atexit
import json
import os
import threading
import time
from typing import Any, Optional


class SampleLogger:
    """Thread-safe JSONL logger."""

    def __init__(self, path: str):
        self.path = path
        self._lock = threading.Lock()
        self._file = None
        self._start_time = time.time()
        self._open_file()

    def _open_file(self):
        out_dir = os.path.dirname(self.path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        self._file = open(self.path, "w")

    def log(self, event: str, **data: Any):
        """Log an event with arbitrary data."""
        record = {"event": event, "ts": time.time() - self._start_time, **data}
        with self._lock:
            if self._file:
                self._file.write(json.dumps(record, default=self._json_default) + "\n")
                self._file.flush()

    @staticmethod
    def _json_default(obj):
        """Handle non-JSON-serializable types."""
        if hasattr(obj, 'tolist'):  # PyTorch tensors, numpy arrays
            return obj.tolist()
        if hasattr(obj, 'item'):  # numpy scalars
            return obj.item()
        raise TypeError(f'Object of type {type(obj).__name__} is not JSON serializable')

    def close(self):
        with self._lock:
            if self._file:
                self._file.close()
                self._file = None


# Global instance
_LOGGER: Optional[SampleLogger] = None


def init_sample_logger(path: str) -> SampleLogger:
    """Initialize the global sample logger."""
    global _LOGGER
    _LOGGER = SampleLogger(path)
    atexit.register(close_sample_logger)
    return _LOGGER


def log_sample(event: str, **data: Any):
    """Log a sample event. No-op if logger not initialized."""
    if _LOGGER:
        _LOGGER.log(event, **data)


def close_sample_logger():
    """Close the global sample logger."""
    global _LOGGER
    if _LOGGER:
        _LOGGER.close()
        _LOGGER = None
