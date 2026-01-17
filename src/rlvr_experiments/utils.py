"""Utility functions."""

import json
import os
import random
import shutil
import subprocess
import tempfile

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =============================================================================
# SageMaker utilities
# =============================================================================

S3_BUCKET = "sagemaker-us-west-2-503561457547"
S3_CHECKPOINT_PREFIX = "rlvr-experiments/checkpoints"


def is_sagemaker() -> bool:
    """Check if running on SageMaker."""
    return os.environ.get("SM_MODEL_DIR") is not None


def get_sagemaker_job_name() -> str:
    """Get SageMaker job name from environment."""
    training_env = os.environ.get("SM_TRAINING_ENV", "{}")
    try:
        env = json.loads(training_env)
        return env.get("job_name", "unknown")
    except json.JSONDecodeError:
        return "unknown"


def upload_checkpoint_to_s3(
    local_path: str,
    run_name: str,
    step: str,
    trace_dir: str | None = None,
) -> str:
    """Upload checkpoint and optionally trace files to S3.

    Args:
        local_path: Local path to the checkpoint directory
        run_name: Name of the training run
        step: Step identifier (e.g., "step100" or "final")
        trace_dir: Optional path to trace files directory to upload

    Returns:
        S3 path where checkpoint was uploaded
    """
    job_name = get_sagemaker_job_name()
    s3_path = f"s3://{S3_BUCKET}/{S3_CHECKPOINT_PREFIX}/{job_name}/{run_name}_{step}"

    print(f"[checkpoint] Uploading to {s3_path}")
    try:
        subprocess.run(
            ["aws", "s3", "sync", local_path, s3_path, "--quiet"],
            check=True,
            capture_output=True,
        )
        print(f"[checkpoint] Upload complete: {s3_path}")

        # Also upload trace files if provided
        if trace_dir and os.path.isdir(trace_dir):
            s3_traces = f"s3://{S3_BUCKET}/{S3_CHECKPOINT_PREFIX}/{job_name}/traces"
            print(f"[checkpoint] Uploading traces to {s3_traces}")
            subprocess.run(
                ["aws", "s3", "sync", trace_dir, s3_traces, "--quiet"],
                check=True,
                capture_output=True,
            )
            print(f"[checkpoint] Traces upload complete")

        return s3_path
    except subprocess.CalledProcessError as e:
        print(f"[checkpoint] Upload failed: {e}")
        return ""


def get_checkpoint_dir() -> tuple[str, bool]:
    """Get checkpoint directory and whether to use S3 uploads.

    Returns:
        Tuple of (checkpoint_dir, use_s3_checkpoints)
    """
    if is_sagemaker():
        checkpoint_dir = tempfile.mkdtemp(prefix="ckpt_")
        print(f"[checkpoint] SageMaker mode: saving to temp dir {checkpoint_dir}, uploading to S3")
        return checkpoint_dir, True
    else:
        checkpoint_dir = "/efs/rlvr-experiments/checkpoints"
        print(f"[checkpoint] Local mode: saving to {checkpoint_dir}")
        return checkpoint_dir, False


def cleanup_local_checkpoint(path: str) -> None:
    """Remove local checkpoint directory to save disk space."""
    shutil.rmtree(path, ignore_errors=True)
