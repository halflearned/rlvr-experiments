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
    """Upload checkpoint and optionally trace files to S3 using boto3.

    Args:
        local_path: Local path to the checkpoint directory
        run_name: Name of the training run
        step: Step identifier (e.g., "step100" or "final")
        trace_dir: Optional path to trace files directory to upload

    Returns:
        S3 path where checkpoint was uploaded, or empty string on failure
    """
    import time
    import boto3
    from botocore.config import Config

    def log(msg: str) -> None:
        ts = time.strftime("%H:%M:%S")
        print(f"[{ts}] [s3_upload] {msg}", flush=True)

    job_name = get_sagemaker_job_name()
    s3_prefix = f"{S3_CHECKPOINT_PREFIX}/{job_name}/{run_name}_{step}"

    log(f"STARTING upload: {local_path} -> s3://{S3_BUCKET}/{s3_prefix}")

    try:
        # Use boto3 with increased timeout
        config = Config(connect_timeout=30, read_timeout=60, retries={'max_attempts': 3})
        s3 = boto3.client('s3', config=config)

        # List and upload files
        if not os.path.isdir(local_path):
            log(f"  ERROR: {local_path} is not a directory")
            return ""

        files = [f for f in os.listdir(local_path) if os.path.isfile(os.path.join(local_path, f))]
        total_size = sum(os.path.getsize(os.path.join(local_path, f)) for f in files)
        log(f"  Uploading {len(files)} files, total {total_size/1024/1024:.1f}MB")

        t0 = time.time()
        for i, filename in enumerate(files):
            filepath = os.path.join(local_path, filename)
            s3_key = f"{s3_prefix}/{filename}"
            file_size = os.path.getsize(filepath)
            log(f"  [{i+1}/{len(files)}] {filename} ({file_size/1024/1024:.1f}MB)")
            s3.upload_file(filepath, S3_BUCKET, s3_key)

        elapsed = time.time() - t0
        log(f"  Checkpoint upload COMPLETE in {elapsed:.1f}s")

        # Upload traces if provided
        if trace_dir and os.path.isdir(trace_dir):
            trace_prefix = f"{S3_CHECKPOINT_PREFIX}/{job_name}/traces"
            trace_files = [f for f in os.listdir(trace_dir) if os.path.isfile(os.path.join(trace_dir, f))]
            log(f"  Uploading {len(trace_files)} trace files...")
            for filename in trace_files:
                filepath = os.path.join(trace_dir, filename)
                s3.upload_file(filepath, S3_BUCKET, f"{trace_prefix}/{filename}")
            log(f"  Traces upload COMPLETE")

        return f"s3://{S3_BUCKET}/{s3_prefix}"

    except Exception as e:
        log(f"UPLOAD FAILED: {type(e).__name__}: {e}")
        return ""


def upload_file_to_s3(local_path: str, s3_key: str) -> bool:
    """Upload a single file to S3 using boto3.

    Args:
        local_path: Local file path
        s3_key: S3 key (path within bucket)

    Returns:
        True on success, False on failure
    """
    import time
    import boto3
    from botocore.config import Config

    def log(msg: str) -> None:
        ts = time.strftime("%H:%M:%S")
        print(f"[{ts}] [s3_upload] {msg}", flush=True)

    if not os.path.isfile(local_path):
        log(f"ERROR: {local_path} does not exist")
        return False

    try:
        config = Config(connect_timeout=30, read_timeout=60, retries={'max_attempts': 3})
        s3 = boto3.client('s3', config=config)
        file_size = os.path.getsize(local_path)
        log(f"Uploading {local_path} ({file_size/1024:.1f}KB) -> s3://{S3_BUCKET}/{s3_key}")
        s3.upload_file(local_path, S3_BUCKET, s3_key)
        log(f"  Upload complete")
        return True
    except Exception as e:
        log(f"Upload failed: {type(e).__name__}: {e}")
        return False


def upload_dir_to_s3(local_dir: str, s3_prefix: str) -> bool:
    """Upload a directory to S3 using boto3.

    Args:
        local_dir: Local directory path
        s3_prefix: S3 prefix (path within bucket)

    Returns:
        True on success, False on failure
    """
    import time
    import boto3
    from botocore.config import Config

    def log(msg: str) -> None:
        ts = time.strftime("%H:%M:%S")
        print(f"[{ts}] [s3_upload] {msg}", flush=True)

    if not os.path.isdir(local_dir):
        log(f"ERROR: {local_dir} does not exist or is not a directory")
        return False

    try:
        config = Config(connect_timeout=30, read_timeout=60, retries={'max_attempts': 3})
        s3 = boto3.client('s3', config=config)

        # Walk directory and upload all files
        files_uploaded = 0
        for root, dirs, files in os.walk(local_dir):
            for filename in files:
                local_path = os.path.join(root, filename)
                # Compute relative path from local_dir
                rel_path = os.path.relpath(local_path, local_dir)
                s3_key = f"{s3_prefix}/{rel_path}"
                s3.upload_file(local_path, S3_BUCKET, s3_key)
                files_uploaded += 1

        log(f"Uploaded {files_uploaded} files from {local_dir} -> s3://{S3_BUCKET}/{s3_prefix}")
        return True
    except Exception as e:
        log(f"Upload failed: {type(e).__name__}: {e}")
        return False


def get_checkpoint_dir() -> tuple[str, bool]:
    """Get checkpoint directory and whether to use S3 uploads for checkpoints.

    If RLVR_RUN_DIR is set: saves to <RLVR_RUN_DIR>/checkpoints.
    On SageMaker: saves to SM_MODEL_DIR, which is automatically uploaded to S3 at job end.
    Locally: saves to /efs/rlvr-experiments/checkpoints.

    Returns:
        Tuple of (checkpoint_dir, upload_checkpoints_to_s3)
        - upload_checkpoints_to_s3 is always False now (SageMaker handles it at job end)
    """
    run_dir = os.environ.get("RLVR_RUN_DIR")
    if run_dir:
        checkpoint_dir = os.path.join(run_dir, "checkpoints")
        print(f"[checkpoint] Using run dir: saving to {checkpoint_dir}")
        return checkpoint_dir, False

    if is_sagemaker():
        # Use SM_MODEL_DIR - SageMaker will upload this to S3 when job completes
        checkpoint_dir = os.environ.get("SM_MODEL_DIR", "/opt/ml/model")
        print(f"[checkpoint] SageMaker mode: saving to {checkpoint_dir} (uploaded at job end)")
        return checkpoint_dir, False  # Don't upload checkpoints manually
    else:
        checkpoint_dir = "/efs/rlvr-experiments/checkpoints"
        print(f"[checkpoint] Local mode: saving to {checkpoint_dir}")
        return checkpoint_dir, False


def upload_traces_to_s3(trace_dir: str, run_name: str) -> bool:
    """Upload trace and log files to S3 for monitoring.

    Call this periodically during training to upload traces and rollout logs.
    Only uploads on SageMaker.

    Args:
        trace_dir: Directory containing trace files (e.g., runtime.trace_dir)
        run_name: Name of the training run

    Returns:
        True on success, False on failure
    """
    if not is_sagemaker():
        return True  # No-op for local runs

    import time
    import boto3
    from botocore.config import Config

    def log(msg: str) -> None:
        ts = time.strftime("%H:%M:%S")
        print(f"[{ts}] [s3_traces] {msg}", flush=True)

    job_name = get_sagemaker_job_name()
    s3_prefix = f"{S3_CHECKPOINT_PREFIX}/{job_name}/traces"

    try:
        config = Config(connect_timeout=30, read_timeout=60, retries={'max_attempts': 3})
        s3 = boto3.client('s3', config=config)

        files_uploaded = 0
        total_size = 0

        if not os.path.isdir(trace_dir):
            log(f"Trace dir {trace_dir} does not exist")
            return False

        for filename in os.listdir(trace_dir):
            filepath = os.path.join(trace_dir, filename)
            if os.path.isfile(filepath):
                s3_key = f"{s3_prefix}/{filename}"
                file_size = os.path.getsize(filepath)
                s3.upload_file(filepath, S3_BUCKET, s3_key)
                files_uploaded += 1
                total_size += file_size

        log(f"Uploaded {files_uploaded} files ({total_size/1024:.1f}KB) -> s3://{S3_BUCKET}/{s3_prefix}")
        return True

    except Exception as e:
        log(f"Upload failed: {type(e).__name__}: {e}")
        return False


def cleanup_local_checkpoint(path: str) -> None:
    """Remove local checkpoint directory to save disk space."""
    shutil.rmtree(path, ignore_errors=True)
