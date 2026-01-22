#!/usr/bin/env python3
"""SageMaker launcher that runs training + eval in parallel.

Launches training on GPUs 0-5 and eval watcher on GPUs 6-7.
Eval results stream to S3 as they appear.
"""

import json
import os
import subprocess
import sys
import tempfile
import threading
import time

import yaml

# SageMaker environment
SM_HOSTS = json.loads(os.environ.get("SM_HOSTS", '["algo-1"]'))
SM_CURRENT_HOST = os.environ.get("SM_CURRENT_HOST", "algo-1")
SM_NUM_GPUS = int(os.environ.get("SM_NUM_GPUS", "8"))

IS_HEAD = SM_CURRENT_HOST == SM_HOSTS[0]
HEAD_HOST = SM_HOSTS[0]
NUM_NODES = len(SM_HOSTS)

RAY_PORT = 6379
RAY_DASHBOARD_PORT = 8265

# S3 bucket for checkpoints and eval results
S3_BUCKET = "sagemaker-us-west-2-503561457547"


def run(cmd: list[str], check: bool = True, env=None):
    """Run command and stream output."""
    print(f"[launcher] Running: {' '.join(cmd)}", flush=True)
    return subprocess.run(cmd, check=check, env=env)


def run_async(cmd: list[str], env=None, log_prefix: str = "") -> subprocess.Popen:
    """Run command asynchronously, streaming output with prefix."""
    print(f"[launcher] Starting async: {' '.join(cmd)}", flush=True)
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=env,
        bufsize=1,
        universal_newlines=True,
    )

    def stream_output():
        for line in proc.stdout:
            print(f"{log_prefix}{line}", end="", flush=True)

    thread = threading.Thread(target=stream_output, daemon=True)
    thread.start()
    return proc


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


def start_ray_head(num_gpus: int):
    """Start Ray head node with specific number of GPUs."""
    cmd = [
        "ray", "start", "--head",
        f"--port={RAY_PORT}",
        f"--dashboard-port={RAY_DASHBOARD_PORT}",
        f"--num-gpus={num_gpus}",
    ]
    run(cmd)
    print(f"[launcher] Ray head started on port {RAY_PORT} with {num_gpus} GPUs", flush=True)


def start_ray_worker(head_addr: str, num_gpus: int):
    """Start Ray worker and connect to head."""
    cmd = [
        "ray", "start",
        f"--address={head_addr}:{RAY_PORT}",
        f"--num-gpus={num_gpus}",
        "--block",
    ]
    run(cmd)


def download_model_and_rewrite_config(config_path: str) -> str:
    """Download model from S3 and rewrite config with local paths."""
    with open(config_path, 'r') as f:
        config_text = f.read()
        config = yaml.safe_load(config_text)

    model_path = config.get('model', {}).get('path', '')
    needs_rewrite = False

    if '/' in model_path and not model_path.startswith('/'):
        needs_rewrite = True
        model_name = model_path.split('/')[-1]
    elif model_path.startswith('/efs/'):
        needs_rewrite = True
        model_name = model_path.rstrip('/').split('/')[-1]

    if needs_rewrite:
        s3_path = f"s3://{S3_BUCKET}/rlvr-experiments/models/{model_name}/"
        local_path = f"/opt/ml/model/model_cache/{model_name}"

        print(f"[launcher] Downloading model from S3: {s3_path}", flush=True)
        os.makedirs(local_path, exist_ok=True)
        run(["aws", "s3", "sync", s3_path, local_path, "--quiet"])
        print(f"[launcher] Model downloaded to: {local_path}", flush=True)

        rewritten_config = config_text.replace(model_path, local_path)
        rewritten_path = "/opt/ml/model/config_rewritten.yaml"
        with open(rewritten_path, 'w') as f:
            f.write(rewritten_config)
        print(f"[launcher] Rewrote config with local model path: {rewritten_path}", flush=True)
        return rewritten_path

    return config_path


def get_base_env() -> dict:
    """Get base environment for subprocess."""
    env = os.environ.copy()
    src_path = "/opt/ml/code/src"
    pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{src_path}:{pythonpath}" if pythonpath else src_path

    nvidia_libs = "/opt/conda/lib/python3.11/site-packages/nvidia/cudnn/lib:/opt/conda/lib/python3.11/site-packages/nvidia/cublas/lib"
    ld_path = env.get("LD_LIBRARY_PATH", "")
    env["LD_LIBRARY_PATH"] = f"{nvidia_libs}:{ld_path}" if ld_path else nvidia_libs

    return env


def create_eval_watch_config(
    job_name: str,
    run_name: str,
    train_gpus: str,
    eval_gpus: str,
    output_s3_path: str,
) -> str:
    """Create eval watcher config that polls S3 checkpoints."""
    # Parse GPU lists
    train_gpu_list = [int(g) for g in train_gpus.split(",")]
    eval_gpu_list = [int(g) for g in eval_gpus.split(",")]

    # S3 checkpoint path pattern
    s3_checkpoint_prefix = f"s3://{S3_BUCKET}/rlvr-experiments/checkpoints/{job_name}"

    config = {
        "seed": 42,
        "output_dir": output_s3_path,

        "vllm": {
            "tensor_parallel_size": 1,
            "dtype": "bfloat16",
            "gpu_memory_utilization": 0.8,
            "max_model_len": 4096,
        },

        "benchmarks": [
            {"task": "gsm8k_cot", "num_fewshot": 4},
            {"task": "ifeval", "num_fewshot": 0},
            {"task": "minerva_math", "num_fewshot": 4},
            {"task": "hendrycks_math", "num_fewshot": 4},
        ],

        "watch": {
            "enable": True,
            "s3_prefix": s3_checkpoint_prefix,
            "pattern": f"{run_name}_step*",
            "poll_interval_s": 60,  # Check every minute
            "gpu_mode": "visible",
            "task_groups": [
                {
                    "gpu": 0,
                    "benchmarks": [
                        {"task": "gsm8k_cot", "num_fewshot": 4},
                        {"task": "ifeval", "num_fewshot": 0},
                    ],
                },
                {
                    "gpu": 1,
                    "benchmarks": [
                        {"task": "minerva_math", "num_fewshot": 4},
                        {"task": "hendrycks_math", "num_fewshot": 4},
                    ],
                },
            ]
        }
    }

    config_path = "/opt/ml/model/eval_watch_config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    print(f"[launcher] Created eval watch config: {config_path}", flush=True)
    return config_path


def run_training(config: str, gpus: str) -> subprocess.Popen:
    """Run training script on specific GPUs."""
    config = download_model_and_rewrite_config(config)

    env = get_base_env()
    env["CUDA_VISIBLE_DEVICES"] = gpus

    script = "/opt/ml/code/entrypoints/train_grpo.py"
    cmd = [sys.executable, script, config]

    return run_async(cmd, env=env, log_prefix="[train] ")


def run_eval_watcher(config: str, gpus: str) -> subprocess.Popen:
    """Run eval watcher on specific GPUs."""
    env = get_base_env()
    env["CUDA_VISIBLE_DEVICES"] = gpus
    env["HF_ALLOW_CODE_EVAL"] = "1"

    script = "/opt/ml/code/entrypoints/eval_benchmarks.py"
    cmd = [sys.executable, script, config]

    return run_async(cmd, env=env, log_prefix="[eval] ")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Training config path")
    parser.add_argument("--eval-config", default=None, help="Eval config path (optional)")
    parser.add_argument("--train-gpus", default="0,1,2,3,4,5", help="GPUs for training")
    parser.add_argument("--eval-gpus", default="6,7", help="GPUs for evaluation")
    args, _ = parser.parse_known_args()

    print(f"[launcher] Node: {SM_CURRENT_HOST} ({'HEAD' if IS_HEAD else 'WORKER'})", flush=True)
    print(f"[launcher] Cluster: {NUM_NODES} nodes, {SM_NUM_GPUS} GPUs each", flush=True)
    print(f"[launcher] Train GPUs: {args.train_gpus}", flush=True)
    print(f"[launcher] Eval GPUs: {args.eval_gpus}", flush=True)

    if IS_HEAD:
        # Count training GPUs
        train_gpu_count = len(args.train_gpus.split(","))

        # CRITICAL: Set CUDA_VISIBLE_DEVICES before starting Ray
        # This ensures Ray actors inherit the correct GPU visibility
        # Without this, DTensor collective operations may hang due to NCCL mismatches
        os.environ["CUDA_VISIBLE_DEVICES"] = args.train_gpus
        print(f"[launcher] Set CUDA_VISIBLE_DEVICES={args.train_gpus} before Ray start", flush=True)

        # Start Ray with only training GPUs
        start_ray_head(num_gpus=train_gpu_count)

        if NUM_NODES > 1:
            import ray
            ray.init(address="auto")
            wait_for_workers(expected=NUM_NODES)

        # Get job name from SageMaker environment
        training_env = os.environ.get("SM_TRAINING_ENV", "{}")
        try:
            env_dict = json.loads(training_env)
            job_name = env_dict.get("job_name", "unknown")
        except json.JSONDecodeError:
            job_name = "unknown"
        print(f"[launcher] Job name: {job_name}", flush=True)

        # Parse run name from training config
        with open(args.config, 'r') as f:
            train_config = yaml.safe_load(f)
        run_name = train_config.get("run", {}).get("name", "run")

        # Always create eval config dynamically with proper S3 paths
        # (even if eval_config is passed, we need to set job-specific paths)
        eval_config = create_eval_watch_config(
            job_name=job_name,
            run_name=run_name,
            train_gpus=args.train_gpus,
            eval_gpus=args.eval_gpus,
            output_s3_path=f"s3://{S3_BUCKET}/rlvr-experiments/eval_results/{job_name}",
        )

        # Start eval watcher first (it will wait for checkpoints)
        print("[launcher] Starting eval watcher...", flush=True)
        eval_proc = run_eval_watcher(eval_config, args.eval_gpus)

        # Give eval watcher a moment to start
        time.sleep(5)

        # Start training
        print("[launcher] Starting training...", flush=True)
        train_proc = run_training(args.config, args.train_gpus)

        # Wait for training to complete
        train_exit_code = train_proc.wait()
        print(f"[launcher] Training completed with exit code: {train_exit_code}", flush=True)

        # Give eval some time to catch up on final checkpoint
        print("[launcher] Waiting for eval to process final checkpoint...", flush=True)
        time.sleep(120)  # 2 minutes

        # Terminate eval watcher
        eval_proc.terminate()
        try:
            eval_proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            eval_proc.kill()

        print("[launcher] Combined run complete", flush=True)

        if train_exit_code != 0:
            sys.exit(train_exit_code)
    else:
        # Worker node: connect to head and block
        wait_for_ray_head(HEAD_HOST, RAY_PORT)
        train_gpu_count = len(args.train_gpus.split(","))
        start_ray_worker(HEAD_HOST, num_gpus=train_gpu_count)


if __name__ == "__main__":
    main()
