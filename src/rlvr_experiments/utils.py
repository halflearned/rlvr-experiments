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
    import time
    import sys

    def log(msg: str) -> None:
        ts = time.strftime("%H:%M:%S")
        print(f"[{ts}] [upload_checkpoint_to_s3] {msg}", flush=True)
        sys.stdout.flush()

    job_name = get_sagemaker_job_name()
    s3_path = f"s3://{S3_BUCKET}/{S3_CHECKPOINT_PREFIX}/{job_name}/{run_name}_{step}"

    log(f"STARTING upload: local_path={local_path}, s3_path={s3_path}")

    # List files to upload
    if os.path.isdir(local_path):
        files = os.listdir(local_path)
        total_size = sum(os.path.getsize(os.path.join(local_path, f)) for f in files if os.path.isfile(os.path.join(local_path, f)))
        log(f"  Local dir has {len(files)} files, total size={total_size/1024/1024:.1f}MB")
        for f in files[:10]:  # Show first 10 files
            fpath = os.path.join(local_path, f)
            if os.path.isfile(fpath):
                log(f"    {f}: {os.path.getsize(fpath)/1024/1024:.1f}MB")
    else:
        log(f"  WARNING: local_path {local_path} is not a directory!")

    try:
        log("  Running aws s3 sync (no --quiet, 10min timeout)...")
        t0 = time.time()
        # Run without --quiet to see progress/errors, with 10 minute timeout
        proc = subprocess.Popen(
            ["aws", "s3", "sync", local_path, s3_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        # Stream output while waiting
        output_lines = []
        while True:
            line = proc.stdout.readline()
            if not line and proc.poll() is not None:
                break
            if line:
                line = line.strip()
                output_lines.append(line)
                # Log every 10th line or important messages
                if len(output_lines) <= 10 or len(output_lines) % 50 == 0 or "error" in line.lower():
                    log(f"    [aws s3] {line}")
            # Check timeout
            if time.time() - t0 > 600:  # 10 minute timeout
                proc.kill()
                log(f"  TIMEOUT after 600s, killed process")
                return ""

        rc = proc.returncode
        elapsed = time.time() - t0
        log(f"  aws s3 sync COMPLETE in {elapsed:.2f}s, return code={rc}, {len(output_lines)} lines output")
        if rc != 0:
            log(f"  ERROR: non-zero return code. Last 10 lines:")
            for line in output_lines[-10:]:
                log(f"    {line}")
            return ""

        # Also upload trace files if provided
        if trace_dir and os.path.isdir(trace_dir):
            s3_traces = f"s3://{S3_BUCKET}/{S3_CHECKPOINT_PREFIX}/{job_name}/traces"
            log(f"  Uploading traces to {s3_traces}...")
            t0 = time.time()
            subprocess.run(
                ["aws", "s3", "sync", trace_dir, s3_traces],
                check=True,
                capture_output=True,
                timeout=300,  # 5 minute timeout for traces
            )
            log(f"  Traces upload COMPLETE in {time.time()-t0:.2f}s")

        log(f"UPLOAD COMPLETE: {s3_path}")
        return s3_path
    except subprocess.TimeoutExpired as e:
        log(f"UPLOAD TIMEOUT: {e}")
        return ""
    except subprocess.CalledProcessError as e:
        log(f"UPLOAD FAILED: {e}")
        log(f"  stdout: {e.stdout}")
        log(f"  stderr: {e.stderr}")
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
