#!/usr/bin/env python3
"""SageMaker launcher for running evaluations.

Simpler than the training launcher - no Ray cluster needed, just run lm_eval.
"""

import json
import os
import subprocess
import sys


# SageMaker environment
SM_NUM_GPUS = int(os.environ.get("SM_NUM_GPUS", "8"))


def run(cmd: list[str], check: bool = True, env=None):
    """Run command and stream output."""
    print(f"[launcher] Running: {' '.join(cmd)}", flush=True)
    return subprocess.run(cmd, check=check, env=env)


def setup_environment():
    """Set up environment for lm_eval."""
    env = os.environ.copy()

    # SageMaker copies source to /opt/ml/code
    src_path = "/opt/ml/code/src"
    pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{src_path}:{pythonpath}" if pythonpath else src_path

    # Fix cuDNN version mismatch
    nvidia_libs = "/opt/conda/lib/python3.11/site-packages/nvidia/cudnn/lib:/opt/conda/lib/python3.11/site-packages/nvidia/cublas/lib"
    ld_path = env.get("LD_LIBRARY_PATH", "")
    env["LD_LIBRARY_PATH"] = f"{nvidia_libs}:{ld_path}" if ld_path else nvidia_libs

    # Allow code evaluation
    env["HF_ALLOW_CODE_EVAL"] = "1"

    return env


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Eval config path")
    args, _ = parser.parse_known_args()

    print(f"[launcher] Starting evaluation", flush=True)
    print(f"[launcher] GPUs available: {SM_NUM_GPUS}", flush=True)
    print(f"[launcher] Config: {args.config}", flush=True)

    env = setup_environment()

    # Run the eval script
    script = "/opt/ml/code/entrypoints/eval_benchmarks.py"
    cmd = [sys.executable, script, args.config]

    print(f"[launcher] Running: {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True, env=env)


if __name__ == "__main__":
    main()
