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
        "--block=false",
    ]
    run(cmd)
    print(f"[launcher] Ray head started on port {RAY_PORT}", flush=True)


def start_ray_worker(head_addr: str):
    """Start Ray worker and connect to head."""
    cmd = [
        "ray", "start",
        f"--address={head_addr}:{RAY_PORT}",
        f"--num-gpus={SM_NUM_GPUS}",
        "--block=true",  # Worker blocks forever
    ]
    run(cmd)


def run_training(config: str):
    """Run the actual training script."""
    # SageMaker copies source to /opt/ml/code
    script = "/opt/ml/code/entrypoints/train_grpo.py"
    cmd = [sys.executable, script, config]
    run(cmd)


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
