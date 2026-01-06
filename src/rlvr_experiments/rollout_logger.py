"""Rollout logger for tracking completions and rewards.

Logs each generation+verification result to a JSONL file for debugging and analysis.
"""

import atexit
import json
import os
import threading
from typing import Any, Optional


class RolloutLogger:
    """Thread-safe JSONL logger for rollout data."""

    def __init__(self, path: str):
        self.path = path
        self._lock = threading.Lock()
        self._file = None
        self._open_file()

    def _open_file(self):
        out_dir = os.path.dirname(self.path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        self._file = open(self.path, "w")

    def log(
        self,
        prompt_id: str,
        prompt: str,
        completions: list[str],
        rewards: list[float],
        version: int,
        epoch: int | None = None,
        **extra: Any,
    ):
        """Log a single rollout (one prompt with N completions).

        Args:
            prompt_id: Unique identifier for the prompt
            prompt: The prompt text (without chat template)
            completions: List of generated completions
            rewards: List of rewards (same length as completions)
            version: Model version that generated these completions
            epoch: Training epoch (optional)
            **extra: Additional fields to include
        """
        record = {
            "prompt_id": prompt_id,
            "prompt": prompt,
            "completions": completions,
            "rewards": rewards,
            "version": version,
        }
        if epoch is not None:
            record["epoch"] = epoch
        record.update(extra)

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
_LOGGER: Optional[RolloutLogger] = None


def init_rollout_logger(path: str) -> RolloutLogger:
    """Initialize the global rollout logger."""
    global _LOGGER
    _LOGGER = RolloutLogger(path)
    atexit.register(close_rollout_logger)
    return _LOGGER


def log_rollout(
    prompt_id: str,
    prompt: str,
    completions: list[str],
    rewards: list[float],
    version: int,
    **extra: Any,
):
    """Log a rollout. No-op if logger not initialized."""
    if _LOGGER:
        _LOGGER.log(prompt_id, prompt, completions, rewards, version, **extra)


def get_rollout_logger() -> Optional[RolloutLogger]:
    """Get the global rollout logger instance."""
    return _LOGGER


def close_rollout_logger():
    """Close the global rollout logger."""
    global _LOGGER
    if _LOGGER:
        _LOGGER.close()
        _LOGGER = None
