#!/usr/bin/env python3
"""Run lm_eval benchmarks on model checkpoints.

This script runs a suite of evaluations on one or more model checkpoints
and outputs results to JSON files. Designed to run on SageMaker or locally.

Example config (configs/eval/example.yaml):

    checkpoints:
      - name: base
        path: s3://bucket/models/Qwen3-1.7B-Base
      - name: step100
        path: s3://bucket/checkpoints/run_xyz/step100
      - name: final
        path: s3://bucket/checkpoints/run_xyz/final

    benchmarks:
      - task: gsm8k
        num_fewshot: 8
      - task: hendrycks_math
        num_fewshot: 4
      - task: minerva_math
        num_fewshot: 4
      - task: ifeval
      - task: mbpp
        num_fewshot: 3

    vllm:
      tensor_parallel_size: 8
      dtype: bfloat16
      gpu_memory_utilization: 0.9
      max_model_len: 4096

    output_dir: s3://bucket/eval_results/run_xyz

Usage:
    python entrypoints/eval_benchmarks.py configs/eval/example.yaml
"""

import argparse
import json
import os
import queue
import subprocess
import sys
import tempfile
import threading
import time
from datetime import datetime
from pathlib import Path

import boto3
import yaml


DEFAULT_S3_BUCKET = os.environ.get("RLVR_S3_BUCKET", "sagemaker-us-west-2-503561457547")
DEFAULT_S3_CHECKPOINT_PREFIX = "rlvr-experiments/checkpoints"


def parse_s3_uri(uri: str) -> tuple[str, str]:
    """Parse s3://bucket/prefix into (bucket, prefix)."""
    if not uri.startswith("s3://"):
        raise ValueError(f"Not an S3 URI: {uri}")
    parts = uri[5:].split("/", 1)
    bucket = parts[0]
    prefix = parts[1] if len(parts) > 1 else ""
    if prefix and not prefix.endswith("/"):
        prefix += "/"
    return bucket, prefix


def default_checkpoint_prefix() -> str | None:
    """Build default checkpoint prefix from SageMaker env."""
    training_env = os.environ.get("SM_TRAINING_ENV")
    if not training_env:
        return None
    try:
        job_name = json.loads(training_env).get("job_name")
    except json.JSONDecodeError:
        return None
    if not job_name:
        return None
    return f"s3://{DEFAULT_S3_BUCKET}/{DEFAULT_S3_CHECKPOINT_PREFIX}/{job_name}/"


def list_s3_checkpoint_dirs(prefix: str) -> list[dict]:
    """List checkpoint directories under an S3 prefix."""
    bucket, key_prefix = parse_s3_uri(prefix)
    client = boto3.client("s3")

    checkpoints = []
    paginator = client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=key_prefix, Delimiter="/"):
        for entry in page.get("CommonPrefixes", []):
            path = entry["Prefix"]
            name = path.rstrip("/").split("/")[-1]
            checkpoints.append({
                "name": name,
                "s3_path": f"s3://{bucket}/{path}",
            })
    return checkpoints


def list_local_checkpoint_dirs(prefix: str) -> list[dict]:
    """List checkpoint directories under a local prefix."""
    base = Path(prefix)
    if not base.exists():
        return []
    checkpoints = []
    for entry in base.iterdir():
        if entry.is_dir():
            checkpoints.append({
                "name": entry.name,
                "path": str(entry),
            })
    return checkpoints


def parse_checkpoint_step(name: str, run_name: str | None) -> tuple[bool, int | None]:
    """Return (is_final, step) for checkpoint name."""
    if run_name:
        if not name.startswith(f"{run_name}_"):
            return False, None
        suffix = name[len(run_name) + 1:]
    else:
        suffix = name.rsplit("_", 1)[-1] if "_" in name else name
    if suffix == "final":
        return True, None
    if suffix.startswith("step"):
        try:
            return False, int(suffix[len("step"):])
        except ValueError:
            return False, None
    return False, None


def ensure_safetensors_name(checkpoint_path: str) -> None:
    """Rename model.safetensors if needed for vLLM."""
    import fcntl

    lock_path = os.path.join(checkpoint_path, ".safetensors_rename.lock")
    model_file = os.path.join(checkpoint_path, "model.safetensors")
    target_file = os.path.join(checkpoint_path, "model-00001-of-00001.safetensors")

    if not os.path.exists(model_file) or os.path.exists(target_file):
        return

    with open(lock_path, "w") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        if os.path.exists(model_file) and not os.path.exists(target_file):
            print(f"[eval] Renaming {model_file} -> {target_file}", flush=True)
            os.rename(model_file, target_file)


def join_output_dir(base: str, suffix: str) -> str:
    """Join output paths with a single slash."""
    return f"{base.rstrip('/')}/{suffix}"


def format_args_dict(value: dict | None) -> str | None:
    """Format dict for oe_eval launch args."""
    if not value:
        return None
    return json.dumps(value)


def normalize_list(value) -> list:
    """Normalize a value into a list."""
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def sync_hf_cache(s3_uri: str, local_dir: str) -> None:
    """Sync HF cache from S3 to a local directory."""
    if not s3_uri:
        return
    os.makedirs(local_dir, exist_ok=True)
    print(f"[eval] Syncing HF cache: {s3_uri} -> {local_dir}", flush=True)
    subprocess.run(["aws", "s3", "sync", s3_uri, local_dir, "--quiet"], check=True)


def run_olmes_eval(
    checkpoint_path: str,
    tasks: list[str],
    output_dir: str,
    cached_output_dir: str | None,
    remote_output_dir: str | None,
    model_type: str | None,
    model_args: dict | None,
    batch_size: str | int | None,
    num_workers: int | None,
    num_gpus: int | None,
    num_shots: int | None,
    limit: int | float | None,
    use_chat_format: bool | None,
    task_args: dict | None,
    recompute_metrics: bool,
    vllm_for_mc: bool,
) -> dict:
    """Run oe_eval.launch for a single checkpoint."""
    if not tasks:
        raise ValueError("olmes.tasks or olmes.task_suites must be provided")

    cmd = [sys.executable, "-m", "oe_eval.launch", "--model", checkpoint_path]
    if model_type:
        cmd.extend(["--model-type", model_type])

    for task in tasks:
        cmd.extend(["--task", task])

    cmd.extend(["--output-dir", output_dir])
    if cached_output_dir:
        cmd.extend(["--cached-output-dir", cached_output_dir])
    if remote_output_dir:
        cmd.extend(["--remote-output-dir", remote_output_dir])

    if batch_size is not None:
        cmd.extend(["--batch-size", str(batch_size)])
    if num_workers is not None:
        cmd.extend(["--num-workers", str(num_workers)])
    if num_gpus is not None:
        cmd.extend(["--gpus", str(num_gpus)])
    if num_shots is not None:
        cmd.extend(["--num-shots", str(num_shots)])
    if limit is not None:
        cmd.extend(["--limit", str(limit)])
    if use_chat_format is not None:
        cmd.extend(["--use-chat-format", str(use_chat_format)])

    model_args_str = format_args_dict(model_args)
    if model_args_str:
        cmd.extend(["--model-args", model_args_str])

    task_args_str = format_args_dict(task_args)
    if task_args_str:
        cmd.extend(["--task-args", task_args_str])

    if recompute_metrics:
        cmd.append("--recompute-metrics")
    if vllm_for_mc:
        cmd.append("--vllm-for-mc")

    env = os.environ.copy()
    env["HF_ALLOW_CODE_EVAL"] = "1"

    print(f"[eval] Running OLMES: {' '.join(cmd)}", flush=True)
    result = subprocess.run(cmd, env=env, capture_output=True, text=True)

    print(result.stdout, flush=True)
    if result.stderr:
        print(result.stderr, file=sys.stderr, flush=True)

    if result.returncode != 0:
        print(f"[eval] WARNING: oe_eval failed with code {result.returncode}", flush=True)
        return {"error": result.stderr}

    metrics_path = os.path.join(output_dir, "metrics.json")
    return {
        "olmes_output_dir": output_dir,
        "olmes_remote_output_dir": remote_output_dir,
        "metrics_path": metrics_path if os.path.exists(metrics_path) else None,
    }


def read_olmes_primary_scores(metrics_path: str | None) -> list[str]:
    """Extract primary score summary from oe_eval metrics.json."""
    if not metrics_path or not os.path.exists(metrics_path):
        return []
    try:
        with open(metrics_path, "r") as f:
            data = json.load(f)
        scores = data.get("all_primary_scores", [])
        return scores if isinstance(scores, list) else []
    except Exception:
        return []


def generate_olmes_summary(results: dict, output_path: str) -> None:
    """Generate a minimal markdown summary for oe_eval outputs."""
    lines = [
        "# OLMES Evaluation Results",
        "",
        f"**Timestamp:** {results['timestamp']}",
        "",
    ]
    for ckpt, info in results.get("results", {}).items():
        lines.append(f"- {ckpt}")
        out_dir = info.get("olmes_output_dir") or "-"
        lines.append(f"  output_dir: {out_dir}")
        remote_dir = info.get("olmes_remote_output_dir")
        if remote_dir:
            lines.append(f"  remote_output_dir: {remote_dir}")
        scores = info.get("primary_scores", [])
        if scores:
            lines.append("  primary_scores:")
            for score in scores:
                lines.append(f"    - {score}")

    with open(output_path, "w") as f:
        f.write("\n".join(lines))

def load_seen(path: str) -> set[str]:
    """Load seen checkpoint names from a JSON file."""
    if not path or not os.path.exists(path):
        return set()
    try:
        with open(path, "r") as f:
            data = json.load(f)
        return set(data or [])
    except Exception:
        return set()


def save_seen(path: str, seen: set[str]) -> None:
    """Persist seen checkpoint names to a JSON file."""
    if not path:
        return
    with open(path, "w") as f:
        json.dump(sorted(seen), f, indent=2)


def get_parallel_gpu_ids(config: dict) -> list[int]:
    """Resolve GPU ids to use for parallel eval."""
    cfg_gpus = config.get("parallel_gpus")
    if cfg_gpus is not None:
        gpus = [int(g) for g in cfg_gpus]
        return gpus or [0]

    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if visible:
        return list(range(len([v for v in visible.split(",") if v.strip() != ""])))

    try:
        import torch
        count = torch.cuda.device_count()
    except Exception:
        count = 0
    return list(range(count or 1))


def run_benchmarks_for_checkpoint(
    ckpt_name: str,
    ckpt_path: str,
    benchmarks: list[dict],
    vllm_config: dict,
    local_output_dir: str,
    timestamp: str,
    seed: int,
    parallel_gpus: list[int],
) -> dict:
    """Run all benchmarks for a checkpoint (optionally in parallel)."""
    ensure_safetensors_name(ckpt_path)

    results: dict[str, dict] = {}
    task_queue: queue.Queue[dict] = queue.Queue()
    for bench in benchmarks:
        task_queue.put(bench)

    lock = threading.Lock()

    def worker(gpu_id: int) -> None:
        while True:
            try:
                bench = task_queue.get_nowait()
            except queue.Empty:
                return

            task = bench["task"]
            num_fewshot = bench.get("num_fewshot")
            fewshot_str = f"{num_fewshot}shot" if num_fewshot is not None else "0shot"
            result_key = f"{task}_{fewshot_str}"

            output_base = os.path.join(local_output_dir, f"{ckpt_name}_{result_key}_{timestamp}")
            if task != "math_qwen":
                os.makedirs(output_base, exist_ok=True)

            print(f"\n[eval] Running {task} ({fewshot_str}) on GPU {gpu_id}...", flush=True)
            result = run_lm_eval(
                checkpoint_path=ckpt_path,
                task=task,
                num_fewshot=num_fewshot,
                vllm_config=vllm_config,
                output_path=output_base,
                seed=seed,
                gpu_id=gpu_id,
            )

            with lock:
                results[result_key] = result

            task_queue.task_done()

    threads = []
    for gpu_id in parallel_gpus:
        t = threading.Thread(target=worker, args=(gpu_id,))
        t.start()
        threads.append(t)

    for t in threads:
        t.join()

    return results

def download_from_s3(s3_path: str, local_dir: str) -> str:
    """Download checkpoint from S3 to local directory."""
    # Extract the checkpoint name from the S3 path
    name = s3_path.rstrip("/").split("/")[-1]
    local_path = os.path.join(local_dir, name)

    print(f"[eval] Downloading {s3_path} -> {local_path}", flush=True)
    os.makedirs(local_path, exist_ok=True)
    subprocess.run(
        ["aws", "s3", "sync", s3_path, local_path, "--quiet"],
        check=True
    )
    return local_path


def upload_to_s3(local_path: str, s3_path: str):
    """Upload file or directory to S3."""
    print(f"[eval] Uploading {local_path} -> {s3_path}", flush=True)
    if os.path.isdir(local_path):
        subprocess.run(
            ["aws", "s3", "sync", local_path, s3_path, "--quiet"],
            check=True
        )
    else:
        subprocess.run(
            ["aws", "s3", "cp", local_path, s3_path, "--quiet"],
            check=True
        )


def resolve_checkpoint_path(path: str, cache_dir: str) -> str:
    """Resolve checkpoint path - download from S3 if needed."""
    if path.startswith("s3://"):
        return download_from_s3(path, cache_dir)
    elif path.startswith("/efs/"):
        # Local EFS path - use directly
        return path
    else:
        # Assume it's a local path
        return path


def run_math_qwen(
    checkpoint_path: str,
    num_fewshot: int,
    output_path: str,
    gpu_id: int | None,
) -> dict:
    """Run math_qwen eval via the custom script."""
    script = "/opt/ml/code/scripts/adhoc/eval_math_qwen_style.py"
    if not os.path.exists(script):
        repo_root = Path(__file__).resolve().parents[1]
        script = os.path.join(repo_root, "scripts", "adhoc", "eval_math_qwen_style.py")
    output_file = f"{output_path}.json"

    cmd = [
        sys.executable,
        script,
        "--model",
        checkpoint_path,
        "--num-shots",
        str(num_fewshot),
        "--tensor-parallel-size",
        "1",
        "--output",
        output_file,
    ]

    env = os.environ.copy()
    if gpu_id is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    print(f"[eval] Running: {' '.join(cmd)}", flush=True)
    result = subprocess.run(cmd, env=env, capture_output=True, text=True)
    print(result.stdout, flush=True)
    if result.stderr:
        print(result.stderr, file=sys.stderr, flush=True)

    if result.returncode != 0:
        print(f"[eval] WARNING: math_qwen failed with code {result.returncode}", flush=True)
        return {"error": result.stderr}

    if os.path.exists(output_file):
        with open(output_file, "r") as f:
            return json.load(f)

    return {"error": "No math_qwen output file found"}


def run_lm_eval(
    checkpoint_path: str,
    task: str,
    num_fewshot: int | None,
    vllm_config: dict,
    output_path: str,
    seed: int = 42,
    gpu_id: int | None = None,
) -> dict:
    """Run lm_eval on a single task and return results."""
    if task == "math_qwen":
        if num_fewshot is None:
            num_fewshot = 4
        return run_math_qwen(checkpoint_path, num_fewshot, output_path, gpu_id)

    # Enforce tensor_parallel_size=1 for deterministic evals
    tp_size = int(vllm_config.get("tensor_parallel_size", 1))
    if tp_size != 1:
        print(f"[eval] WARNING: overriding tensor_parallel_size={tp_size} to 1", flush=True)
        tp_size = 1

    # Build model args string
    model_args = [
        f"pretrained={checkpoint_path}",
        f"tensor_parallel_size={tp_size}",
        f"dtype={vllm_config.get('dtype', 'bfloat16')}",
        f"gpu_memory_utilization={vllm_config.get('gpu_memory_utilization', 0.9)}",
        f"seed={seed}",
    ]
    if "max_model_len" in vllm_config:
        model_args.append(f"max_model_len={vllm_config['max_model_len']}")

    model_args_str = ",".join(model_args)

    # Build command
    cmd = [
        sys.executable, "-m", "lm_eval",
        "--model", "vllm",
        "--model_args", model_args_str,
        "--tasks", task,
        "--batch_size", "auto",
        "--seed", str(seed),
        "--gen_kwargs", "temperature=0",
        "--output_path", output_path,
    ]

    if num_fewshot is not None:
        cmd.extend(["--num_fewshot", str(num_fewshot)])

    # MBPP and code tasks need special flags
    if task in ("mbpp", "humaneval"):
        cmd.append("--confirm_run_unsafe_code")

    # Set environment for code eval
    env = os.environ.copy()
    env["HF_ALLOW_CODE_EVAL"] = "1"
    if gpu_id is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    print(f"[eval] Running: {' '.join(cmd)}", flush=True)

    result = subprocess.run(
        cmd,
        env=env,
        capture_output=True,
        text=True,
    )

    # Print output for logging
    print(result.stdout, flush=True)
    if result.stderr:
        print(result.stderr, file=sys.stderr, flush=True)

    if result.returncode != 0:
        print(f"[eval] WARNING: lm_eval failed with code {result.returncode}", flush=True)
        return {"error": result.stderr}

    # Find and load the results JSON
    # lm_eval outputs to output_path/results_*.json
    results_dir = Path(output_path)
    results_files = list(results_dir.glob("results_*.json"))
    if results_files:
        with open(results_files[-1]) as f:
            return json.load(f)

    return {"error": "No results file found"}


def main():
    parser = argparse.ArgumentParser(description="Run lm_eval benchmarks")
    parser.add_argument("config", help="Eval config YAML file")
    parser.add_argument("--cache-dir", default="/tmp/eval_cache", help="Cache directory for checkpoints")
    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    harness = config.get("harness", "lm_eval")
    checkpoints = config.get("checkpoints", [])
    benchmarks = config.get("benchmarks", [])
    vllm_config = config.get("vllm", {})
    olmes_cfg = config.get("olmes", {}) or {}
    watch_cfg = config.get("watch", {}) or {}
    output_dir = config.get("output_dir", "/tmp/eval_results")
    seed = int(config.get("seed", 42))

    # Create cache and output directories
    os.makedirs(args.cache_dir, exist_ok=True)

    # Determine if output is S3
    output_is_s3 = output_dir.startswith("s3://")
    if output_is_s3:
        local_output_dir = tempfile.mkdtemp(prefix="eval_output_")
    else:
        local_output_dir = output_dir
        os.makedirs(local_output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    parallel_gpus = get_parallel_gpu_ids(config)

    if harness == "olmes":
        hf_cache_s3 = olmes_cfg.get("hf_cache_s3")
        hf_cache_local = olmes_cfg.get("hf_cache_local", "/tmp/hf_cache_olmes")
        if hf_cache_s3:
            sync_hf_cache(hf_cache_s3, hf_cache_local)
        os.environ["HF_HOME"] = hf_cache_local
        os.environ["HF_DATASETS_OFFLINE"] = "1"
        os.environ["HF_HUB_OFFLINE"] = "1"

    # Summary results
    all_results = {
        "timestamp": timestamp,
        "config": config,
        "results": {}
    }

    def write_summaries() -> None:
        summary_path = os.path.join(local_output_dir, f"summary_{timestamp}.json")
        with open(summary_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\n[eval] Summary saved to: {summary_path}", flush=True)

        markdown_path = os.path.join(local_output_dir, f"summary_{timestamp}.md")
        if harness == "olmes":
            generate_olmes_summary(all_results, markdown_path)
        else:
            generate_markdown_summary(all_results, markdown_path)
        print(f"[eval] Markdown summary saved to: {markdown_path}", flush=True)

        if output_is_s3:
            if harness == "olmes":
                if olmes_cfg.get("upload_summary", False):
                    upload_to_s3(summary_path, join_output_dir(output_dir, os.path.basename(summary_path)))
                    upload_to_s3(markdown_path, join_output_dir(output_dir, os.path.basename(markdown_path)))
                    print(f"[eval] Summaries uploaded to: {output_dir}", flush=True)
            else:
                upload_to_s3(local_output_dir, output_dir)
                print(f"[eval] Results uploaded to: {output_dir}", flush=True)

    def evaluate_checkpoint(ckpt: dict) -> None:
        ckpt_name = ckpt["name"]
        ckpt_path = resolve_checkpoint_path(ckpt["path"], args.cache_dir)

        print(f"\n{'='*60}", flush=True)
        print(f"[eval] Evaluating checkpoint: {ckpt_name}", flush=True)
        print(f"[eval] Path: {ckpt_path}", flush=True)
        print(f"{'='*60}\n", flush=True)

        if harness == "olmes":
            ensure_safetensors_name(ckpt_path)
            tasks = []
            tasks.extend(normalize_list(olmes_cfg.get("task_suites")))
            tasks.extend(normalize_list(olmes_cfg.get("tasks")))

            local_out_dir = olmes_cfg.get("output_dir") or local_output_dir
            if local_out_dir.startswith("s3://"):
                local_out_dir = local_output_dir
            local_ckpt_dir = join_output_dir(
                local_out_dir, f"{ckpt_name}_{timestamp}"
            )
            os.makedirs(local_ckpt_dir, exist_ok=True)

            remote_output_dir = olmes_cfg.get("remote_output_dir")
            if remote_output_dir is None and output_is_s3:
                remote_output_dir = output_dir
            if remote_output_dir:
                remote_output_dir = join_output_dir(remote_output_dir, f"{ckpt_name}_{timestamp}")

            num_gpus = olmes_cfg.get("gpus")
            if num_gpus is None:
                num_gpus = len(parallel_gpus)
            num_workers = olmes_cfg.get("num_workers")
            if num_workers is None:
                num_workers = num_gpus
            if num_gpus and num_workers and num_gpus % num_workers != 0:
                raise ValueError("olmes.gpus must be divisible by olmes.num_workers")

            results = run_olmes_eval(
                checkpoint_path=ckpt_path,
                tasks=tasks,
                output_dir=local_ckpt_dir,
                cached_output_dir=olmes_cfg.get("cached_output_dir"),
                remote_output_dir=remote_output_dir,
                model_type=olmes_cfg.get("model_type", "vllm"),
                model_args=olmes_cfg.get("model_args"),
                batch_size=olmes_cfg.get("batch_size"),
                num_workers=num_workers,
                num_gpus=num_gpus,
                num_shots=olmes_cfg.get("num_shots"),
                limit=olmes_cfg.get("limit"),
                use_chat_format=olmes_cfg.get("use_chat_format"),
                task_args=olmes_cfg.get("task_args"),
                recompute_metrics=bool(olmes_cfg.get("recompute_metrics", False)),
                vllm_for_mc=bool(olmes_cfg.get("vllm_for_mc", False)),
            )
            results["primary_scores"] = read_olmes_primary_scores(results.get("metrics_path"))
            all_results["results"][ckpt_name] = results
        else:
            all_results["results"][ckpt_name] = run_benchmarks_for_checkpoint(
                ckpt_name=ckpt_name,
                ckpt_path=ckpt_path,
                benchmarks=benchmarks,
                vllm_config=vllm_config,
                local_output_dir=local_output_dir,
                timestamp=timestamp,
                seed=seed,
                parallel_gpus=parallel_gpus,
            )

            # Print summary
            for task_key, task_results in all_results["results"][ckpt_name].items():
                results_obj = task_results.get("results", {})
                for task_name, metrics in results_obj.items():
                    for metric, value in metrics.items():
                        if isinstance(value, (int, float)):
                            print(f"  {task_name}/{metric}: {value:.4f}", flush=True)

        write_summaries()

    if watch_cfg.get("enable"):
        poll_interval = int(watch_cfg.get("poll_interval_s", 300))
        run_name = watch_cfg.get("run_name") or os.environ.get("RLVR_RUN_NAME")
        checkpoint_prefix = watch_cfg.get("checkpoint_prefix") or default_checkpoint_prefix()
        local_dir = watch_cfg.get("local_dir")
        if local_dir:
            checkpoint_prefix = local_dir
        min_step = watch_cfg.get("min_step")
        max_step = watch_cfg.get("max_step")
        stop_after_final = watch_cfg.get("stop_after_final", True)

        if not checkpoint_prefix:
            raise ValueError("watch.enabled requires checkpoint_prefix or SageMaker env")

        seen_path = (
            os.path.join(local_output_dir, "eval_seen.json")
            if not output_is_s3 else "/tmp/eval_seen.json"
        )
        seen = load_seen(seen_path)

        print(f"[eval] Watch mode enabled", flush=True)
        print(f"[eval] Checkpoint prefix: {checkpoint_prefix}", flush=True)
        if run_name:
            print(f"[eval] Run name filter: {run_name}", flush=True)
        print(f"[eval] Poll interval: {poll_interval}s", flush=True)

        while True:
            if checkpoint_prefix.startswith("s3://"):
                checkpoints = list_s3_checkpoint_dirs(checkpoint_prefix)
            else:
                checkpoints = list_local_checkpoint_dirs(checkpoint_prefix)
            filtered: list[dict] = []
            for ckpt in checkpoints:
                is_final, step = parse_checkpoint_step(ckpt["name"], run_name)
                if step is None and not is_final:
                    continue
                if isinstance(min_step, int) and step is not None and step < min_step:
                    continue
                if isinstance(max_step, int) and step is not None and step > max_step:
                    continue
                ckpt["is_final"] = is_final
                ckpt["step"] = step
                filtered.append(ckpt)

            filtered.sort(key=lambda c: (0 if not c.get("is_final") else 1, c.get("step") or 0))

            for ckpt in filtered:
                name = ckpt["name"]
                if name in seen:
                    continue
                path = ckpt.get("s3_path") or ckpt.get("path")
                if not path:
                    continue
                evaluate_checkpoint({"name": name, "path": path})
                seen.add(name)
                save_seen(seen_path, seen)
                if stop_after_final and ckpt.get("is_final"):
                    print("[eval] Final checkpoint evaluated; stopping watch.", flush=True)
                    print("\n[eval] Done!", flush=True)
                    return

            time.sleep(poll_interval)
    else:
        for ckpt in checkpoints:
            evaluate_checkpoint(ckpt)

        print("\n[eval] Done!", flush=True)


def generate_markdown_summary(results: dict, output_path: str):
    """Generate a markdown table summarizing all results."""
    lines = [
        "# Evaluation Results",
        "",
        f"**Timestamp:** {results['timestamp']}",
        "",
    ]

    # Build table
    checkpoints = list(results["results"].keys())
    if not checkpoints:
        lines.append("No results.")
        with open(output_path, "w") as f:
            f.write("\n".join(lines))
        return

    # Get all tasks
    tasks = set()
    for ckpt_results in results["results"].values():
        tasks.update(ckpt_results.keys())
    tasks = sorted(tasks)

    # Header
    header = "| Task | Metric | " + " | ".join(checkpoints) + " |"
    separator = "|------|--------|" + "|".join(["-------"] * len(checkpoints)) + "|"
    lines.extend([header, separator])

    # Rows - extract key metrics for each task
    # Keys are task names (may include _Nshot suffix)
    key_metrics = {
        "gsm8k": ["exact_match"],
        "gsm8k_cot": ["exact_match"],
        "gsm8k_cot_zeroshot": ["flexible-extract", "strict-match"],
        "hendrycks_math": ["exact_match"],
        "minerva_math": ["exact_match", "math_verify"],
        "math_qwen": ["accuracy"],
        "ifeval": ["prompt_level_strict_acc", "inst_level_strict_acc"],
        "mbpp": ["pass_at_1"],
        "humaneval": ["pass_at_1"],
    }

    for task in tasks:
        # Strip _Nshot suffix to find metrics
        base_task = task.rsplit("_", 1)[0] if task.endswith("shot") else task
        metrics_to_show = key_metrics.get(base_task, ["exact_match"])
        for metric in metrics_to_show:
            row = f"| {task} | {metric} |"
            for ckpt in checkpoints:
                task_results = results["results"].get(ckpt, {}).get(task, {})
                value = extract_metric(task_results, task, metric)
                if value is not None:
                    row += f" {value:.2%} |"
                else:
                    row += " - |"
            lines.append(row)

    with open(output_path, "w") as f:
        f.write("\n".join(lines))


def extract_metric(task_results: dict, task: str, metric: str) -> float | None:
    """Extract a specific metric from lm_eval results."""
    if "error" in task_results:
        return None

    # math_qwen outputs a flat JSON with "accuracy" as a percentage
    if metric == "accuracy" and "accuracy" in task_results:
        return task_results["accuracy"] / 100.0

    results = task_results.get("results", {})
    if not isinstance(results, dict):
        return None

    # Try exact task name first
    if task in results:
        task_metrics = results[task]
        if metric in task_metrics:
            return task_metrics[metric]
        # Try with ,none suffix (lm_eval format)
        metric_none = f"{metric},none"
        if metric_none in task_metrics:
            return task_metrics[metric_none]

    # Try finding the task as a prefix (e.g., hendrycks_math for hendrycks_math_algebra)
    for task_name, task_metrics in results.items():
        if task_name.startswith(task) or task in task_name:
            if metric in task_metrics:
                return task_metrics[metric]
            metric_none = f"{metric},none"
            if metric_none in task_metrics:
                return task_metrics[metric_none]

    return None


if __name__ == "__main__":
    main()
