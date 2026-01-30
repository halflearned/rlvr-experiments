#!/usr/bin/env python3
"""SageMaker launcher that sets up Ray cluster before running training.

Handles both single-node and multi-node cases:
- Single node: starts Ray head and runs training
- Multi-node: node 0 starts Ray head, others join as workers, then node 0 runs training
"""

import json
import os
import subprocess
import sys
import time

# SageMaker environment
SM_HOSTS = json.loads(os.environ.get("SM_HOSTS", '["algo-1"]'))
SM_CURRENT_HOST = os.environ.get("SM_CURRENT_HOST", "algo-1")
SM_NUM_GPUS = int(os.environ.get("SM_NUM_GPUS", "8"))

IS_HEAD = SM_CURRENT_HOST == SM_HOSTS[0]
HEAD_HOST = SM_HOSTS[0]
NUM_NODES = len(SM_HOSTS)

RAY_PORT = 6379
RAY_DASHBOARD_PORT = 8265


def run(cmd: list[str], check: bool = True):
    """Run command and stream output."""
    print(f"[launcher] Running: {' '.join(cmd)}", flush=True)
    return subprocess.run(cmd, check=check)


def wait_for_ray_head(host: str, port: int, timeout: int = 300):
    """Wait for Ray head to be reachable."""
    import socket
    start = time.time()
    while time.time() - start < timeout:
        try:
            with socket.create_connection((host, port), timeout=5):
                print(f"[launcher] Ray head at {host}:{port} is reachable", flush=True)
                return True
        except (ConnectionRefusedError, socket.timeout, OSError):
            time.sleep(2)
    raise TimeoutError(f"Ray head at {host}:{port} not reachable after {timeout}s")


def wait_for_workers(expected: int, timeout: int = 300):
    """Wait for all Ray worker nodes to connect."""
    import ray
    start = time.time()
    while time.time() - start < timeout:
        nodes = [n for n in ray.nodes() if n.get("Alive")]
        if len(nodes) >= expected:
            print(f"[launcher] All {expected} nodes connected to Ray cluster", flush=True)
            return True
        print(f"[launcher] Waiting for workers: {len(nodes)}/{expected}", flush=True)
        time.sleep(5)
    raise TimeoutError(f"Only {len(nodes)}/{expected} nodes connected after {timeout}s")


def start_ray_head():
    """Start Ray head node."""
    cmd = [
        "ray", "start", "--head",
        f"--port={RAY_PORT}",
        f"--dashboard-port={RAY_DASHBOARD_PORT}",
        f"--num-gpus={SM_NUM_GPUS}",
    ]
    run(cmd)
    print(f"[launcher] Ray head started on port {RAY_PORT}", flush=True)


def start_ray_worker(head_addr: str):
    """Start Ray worker and connect to head."""
    cmd = [
        "ray", "start",
        f"--address={head_addr}:{RAY_PORT}",
        f"--num-gpus={SM_NUM_GPUS}",
        "--block",  # Worker blocks forever
    ]
    run(cmd)


def download_model_and_rewrite_config(config_path: str) -> str:
    """Download model from S3 and rewrite config with local paths.

    Returns path to the rewritten config file.
    """
    import yaml
    import os
    import re

    with open(config_path, 'r') as f:
        config_text = f.read()
        config = yaml.safe_load(config_text)

    # Check if model path needs rewriting:
    # 1. HF Hub identifier (contains / but not absolute path): "Qwen/Qwen3-1.7B-Base"
    # 2. Local /efs/ path that won't exist on SageMaker: "/efs/rlvr-experiments/assets/hf/Qwen3-1.7B-Base"
    model_path = config.get('model', {}).get('path', '')
    needs_rewrite = False

    if '/' in model_path and not model_path.startswith('/'):
        # HF-style path: "Qwen/Qwen3-1.7B-Base"
        needs_rewrite = True
        model_name = model_path.split('/')[-1]
    elif model_path.startswith('/efs/'):
        # Local EFS path that won't exist on SageMaker
        needs_rewrite = True
        model_name = model_path.rstrip('/').split('/')[-1]

    if needs_rewrite:
        # Map to S3 bucket path
        # e.g., "Qwen3-1.7B-Base" -> "s3://sagemaker-us-west-2-503561457547/rlvr-experiments/models/Qwen3-1.7B-Base/"
        s3_path = f"s3://sagemaker-us-west-2-503561457547/rlvr-experiments/models/{model_name}/"
        local_path = f"/opt/ml/model/model_cache/{model_name}"

        print(f"[launcher] Downloading model from S3: {s3_path}", flush=True)
        os.makedirs(local_path, exist_ok=True)
        run(["aws", "s3", "sync", s3_path, local_path, "--quiet"])
        print(f"[launcher] Model downloaded to: {local_path}", flush=True)

        # Rewrite all occurrences of the original path to local path in the config
        # This handles model.path, tokenizer.pretrained_model_name_or_path, roles[*].config.model, etc.
        rewritten_config = config_text.replace(model_path, local_path)

        # Write to a temporary config file
        rewritten_path = "/opt/ml/model/config_rewritten.yaml"
        with open(rewritten_path, 'w') as f:
            f.write(rewritten_config)
        print(f"[launcher] Rewrote config with local model path: {rewritten_path}", flush=True)
        return rewritten_path

    return config_path


def run_training(config: str):
    """Run the actual training script."""
    import os

    # Download model from S3 if needed and rewrite config with local paths
    config = download_model_and_rewrite_config(config)

    # SageMaker copies source to /opt/ml/code
    # Ensure PYTHONPATH includes src directory for the subprocess
    env = os.environ.copy()
    src_path = "/opt/ml/code/src"
    pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{src_path}:{pythonpath}" if pythonpath else src_path

    # Fix cuDNN version mismatch: PyTorch 2.9+cu128 bundles cuDNN 9.x, but the base
    # AWS DLC image has an older system cuDNN. Prepend the nvidia pip packages to
    # LD_LIBRARY_PATH so PyTorch finds the bundled (compatible) versions.
    nvidia_libs = "/opt/conda/lib/python3.11/site-packages/nvidia/cudnn/lib:/opt/conda/lib/python3.11/site-packages/nvidia/cublas/lib"
    ld_path = env.get("LD_LIBRARY_PATH", "")
    env["LD_LIBRARY_PATH"] = f"{nvidia_libs}:{ld_path}" if ld_path else nvidia_libs

    script = "/opt/ml/code/entrypoints/train_grpo_sft.py"
    cmd = [sys.executable, script, config]
    print(f"[launcher] Running: {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True, env=env)


def main():
    # Get config from hyperparameters (SageMaker passes them as command line args)
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Training config path")
    args, _ = parser.parse_known_args()

    print(f"[launcher] Node: {SM_CURRENT_HOST} ({'HEAD' if IS_HEAD else 'WORKER'})", flush=True)
    print(f"[launcher] Cluster: {NUM_NODES} nodes, {SM_NUM_GPUS} GPUs each", flush=True)
    print(f"[launcher] All hosts: {SM_HOSTS}", flush=True)

    if IS_HEAD:
        # Head node: start Ray, wait for workers, run training
        start_ray_head()

        if NUM_NODES > 1:
            # Multi-node: wait for workers to join
            import ray
            ray.init(address="auto")
            wait_for_workers(expected=NUM_NODES)

        # Run training
        run_training(args.config)
    else:
        # Worker node: connect to head and block
        wait_for_ray_head(HEAD_HOST, RAY_PORT)
        start_ray_worker(HEAD_HOST)
        # start_ray_worker blocks, so we never reach here


if __name__ == "__main__":
    main()
