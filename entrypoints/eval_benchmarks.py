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
import csv
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

import yaml


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


def list_local_checkpoint_dirs(base_dir: str, pattern: str) -> list[dict]:
    """List local checkpoint directories matching a glob pattern."""
    base = Path(base_dir)
    if not base.exists():
        return []
    checkpoints = []
    for path in sorted(base.glob(pattern)):
        if path.is_dir():
            checkpoints.append({"name": path.name, "path": str(path)})
    return checkpoints


def ensure_safetensors_name(checkpoint_path: str) -> None:
    """Ensure vLLM-compatible safetensors filename exists."""
    model_file = os.path.join(checkpoint_path, "model.safetensors")
    target_file = os.path.join(checkpoint_path, "model-00001-of-00001.safetensors")
    if os.path.exists(model_file) and not os.path.exists(target_file):
        os.symlink("model.safetensors", target_file)


def parse_gpu_list(value) -> list[int]:
    """Normalize GPU list from config value."""
    if value is None:
        return []
    if isinstance(value, list):
        return [int(v) for v in value]
    if isinstance(value, str):
        return [int(v.strip()) for v in value.split(",") if v.strip()]
    return [int(value)]


def build_task_groups(benchmarks: list[dict], watch_cfg: dict) -> list[dict]:
    """Build per-GPU task groups for watch mode."""
    task_groups = watch_cfg.get("task_groups")
    if task_groups:
        groups = []
        for group in task_groups:
            gpu_id = int(group["gpu"])
            group_benchmarks = group.get("benchmarks", [])
            if not group_benchmarks:
                raise ValueError("task_groups entries must include benchmarks")
            groups.append({"gpu": gpu_id, "benchmarks": group_benchmarks})
        return groups

    gpus = parse_gpu_list(watch_cfg.get("gpus"))
    if not gpus:
        gpus = [0]
    if not benchmarks:
        raise ValueError("benchmarks must be provided when watch mode is enabled")

    groups_by_gpu = {gpu: [] for gpu in gpus}
    for i, bench in enumerate(benchmarks):
        gpu_id = gpus[i % len(gpus)]
        groups_by_gpu[gpu_id].append(bench)
    return [{"gpu": gpu, "benchmarks": group} for gpu, group in groups_by_gpu.items()]


def load_seen(path: str) -> set[str]:
    if os.path.exists(path):
        with open(path) as f:
            return set(json.load(f))
    return set()


def save_seen(path: str, seen: set[str]) -> None:
    with open(path, "w") as f:
        json.dump(sorted(seen), f, indent=2)


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
    # Build model args string
    model_args = [
        f"pretrained={checkpoint_path}",
        f"tensor_parallel_size={vllm_config.get('tensor_parallel_size', 1)}",
        f"dtype={vllm_config.get('dtype', 'bfloat16')}",
        f"gpu_memory_utilization={vllm_config.get('gpu_memory_utilization', 0.9)}",
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
    parser.add_argument(
        "--cache-dir",
        default="/tmp/eval_cache",
        help="Cache directory for checkpoints",
    )
    parser.add_argument(
        "--watch-path",
        help=(
            "Optional watch path glob (e.g., /efs/.../checkpoints/run_step*). "
            "Overrides watch.local_dir and watch.pattern from YAML."
        ),
    )
    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    checkpoints = config.get("checkpoints", [])
    benchmarks = config.get("benchmarks", [])
    vllm_config = config.get("vllm", {})
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
        generate_markdown_summary(all_results, markdown_path)
        print(f"[eval] Markdown summary saved to: {markdown_path}", flush=True)

        csv_path = os.path.join(local_output_dir, f"results.csv")
        generate_csv_summary(all_results, csv_path)
        print(f"[eval] CSV summary saved to: {csv_path}", flush=True)

        if output_is_s3:
            upload_to_s3(local_output_dir, output_dir)
            print(f"[eval] Results uploaded to: {output_dir}", flush=True)

    if args.watch_path:
        watch_path = os.path.expanduser(args.watch_path)
        if any(ch in watch_path for ch in ["*", "?", "["]):
            watch_cfg["local_dir"] = str(Path(watch_path).parent)
            watch_cfg["pattern"] = Path(watch_path).name
        else:
            watch_cfg["local_dir"] = watch_path
            watch_cfg["pattern"] = "*"
        watch_cfg["enable"] = True

    if watch_cfg.get("enable"):
        local_dir = watch_cfg.get("local_dir")
        if not local_dir:
            raise ValueError("watch.local_dir must be set for watch mode")
        pattern = watch_cfg.get("pattern", "*")
        poll_interval = int(watch_cfg.get("poll_interval_s", 300))

        task_groups = build_task_groups(benchmarks, watch_cfg)
        expected_tasks = sum(len(group["benchmarks"]) for group in task_groups)
        if expected_tasks == 0:
            raise ValueError("No tasks configured for watch mode")

        if vllm_config.get("tensor_parallel_size", 1) != 1:
            print("[eval] Forcing tensor_parallel_size=1 for watch mode", flush=True)
            vllm_config["tensor_parallel_size"] = 1

        seen_path = watch_cfg.get("seen_path") or os.path.join(local_output_dir, "eval_seen.json")
        seen = load_seen(seen_path)
        pending_tasks: dict[str, int] = {}

        work_queues: dict[int, queue.Queue] = {}
        results_queue: queue.Queue = queue.Queue()

        visible_env = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        visible_gpus = []
        if visible_env:
            for entry in visible_env.split(","):
                entry = entry.strip()
                if entry:
                    try:
                        visible_gpus.append(int(entry))
                    except ValueError:
                        pass

        gpu_mode = watch_cfg.get("gpu_mode", "auto")

        def resolve_gpu_id(requested_id: int) -> int:
            if not visible_gpus:
                return requested_id
            if gpu_mode == "absolute":
                return requested_id
            if gpu_mode == "visible":
                return visible_gpus[requested_id]
            # auto: treat in-range as visible index, otherwise absolute
            if 0 <= requested_id < len(visible_gpus):
                return visible_gpus[requested_id]
            return requested_id

        def worker(gpu_id: int, worker_benchmarks: list[dict], work_queue: queue.Queue) -> None:
            resolved_gpu_id = resolve_gpu_id(gpu_id)
            while True:
                item = work_queue.get()
                if item is None:
                    return
                ckpt_name, ckpt_path, run_root = item
                ensure_safetensors_name(ckpt_path)

                for bench in worker_benchmarks:
                    task = bench["task"]
                    num_fewshot = bench.get("num_fewshot")
                    fewshot_str = f"{num_fewshot}shot" if num_fewshot is not None else "0shot"
                    result_key = f"{task}_{fewshot_str}"

                    eval_output_dir = os.path.join(run_root, result_key)
                    os.makedirs(eval_output_dir, exist_ok=True)

                    results = run_lm_eval(
                        checkpoint_path=ckpt_path,
                        task=task,
                        num_fewshot=num_fewshot,
                        vllm_config=vllm_config,
                        output_path=eval_output_dir,
                        seed=seed,
                        gpu_id=resolved_gpu_id,
                    )
                    results_queue.put((ckpt_name, result_key, results))

        for group in task_groups:
            gpu_id = group["gpu"]
            q = queue.Queue()
            work_queues[gpu_id] = q
            thread = threading.Thread(
                target=worker,
                args=(gpu_id, group["benchmarks"], q),
                daemon=True,
            )
            thread.start()

        print(f"[eval] Watch mode enabled: {local_dir}/{pattern}", flush=True)
        print(f"[eval] Poll interval: {poll_interval}s", flush=True)
        print(f"[eval] Task groups: {task_groups}", flush=True)

        next_poll = 0.0
        while True:
            now = time.time()
            if now >= next_poll:
                checkpoints = list_local_checkpoint_dirs(local_dir, pattern)
                for ckpt in checkpoints:
                    ckpt_name = ckpt["name"]
                    ckpt_path = ckpt["path"]

                    if ckpt_name in seen:
                        continue
                    if not os.path.exists(os.path.join(ckpt_path, "model.safetensors")) and not os.path.exists(
                        os.path.join(ckpt_path, "model-00001-of-00001.safetensors")
                    ):
                        continue

                    seen.add(ckpt_name)
                    save_seen(seen_path, seen)
                    pending_tasks[ckpt_name] = expected_tasks

                    run_root = os.path.join(local_output_dir, ckpt_name)
                    os.makedirs(run_root, exist_ok=True)

                    for q in work_queues.values():
                        q.put((ckpt_name, ckpt_path, run_root))

                next_poll = now + poll_interval

            try:
                ckpt_name, result_key, results = results_queue.get(timeout=1)
            except queue.Empty:
                continue

            all_results["results"].setdefault(ckpt_name, {})
            all_results["results"][ckpt_name][result_key] = results
            if ckpt_name in pending_tasks:
                pending_tasks[ckpt_name] -= 1
                if pending_tasks[ckpt_name] <= 0:
                    write_summaries()
                    pending_tasks.pop(ckpt_name, None)

    # Run evaluations (non-watch mode)
    for ckpt in checkpoints:
        ckpt_name = ckpt["name"]
        ckpt_path = resolve_checkpoint_path(ckpt["path"], args.cache_dir)

        print(f"\n{'='*60}", flush=True)
        print(f"[eval] Evaluating checkpoint: {ckpt_name}", flush=True)
        print(f"[eval] Path: {ckpt_path}", flush=True)
        print(f"{'='*60}\n", flush=True)

        all_results["results"][ckpt_name] = {}

        for bench in benchmarks:
            task = bench["task"]
            num_fewshot = bench.get("num_fewshot")

            # Create a unique key for task+fewshot combo
            fewshot_str = f"{num_fewshot}shot" if num_fewshot is not None else "0shot"
            result_key = f"{task}_{fewshot_str}"

            print(f"\n[eval] Running {task} ({fewshot_str})...", flush=True)

            # Output path for this specific eval
            eval_output_dir = os.path.join(
                local_output_dir,
                f"{ckpt_name}_{result_key}_{timestamp}"
            )
            os.makedirs(eval_output_dir, exist_ok=True)

            results = run_lm_eval(
                checkpoint_path=ckpt_path,
                task=task,
                num_fewshot=num_fewshot,
                vllm_config=vllm_config,
                output_path=eval_output_dir,
                seed=seed,
            )

            all_results["results"][ckpt_name][result_key] = results

            # Print summary
            if "results" in results:
                task_results = results["results"]
                for task_name, metrics in task_results.items():
                    for metric, value in metrics.items():
                        if isinstance(value, (int, float)):
                            print(f"  {task_name}/{metric}: {value:.4f}", flush=True)

    if checkpoints:
        write_summaries()

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

    results = task_results.get("results", {})

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


def generate_csv_summary(results: dict, output_path: str):
    """Generate a CSV file summarizing all results.

    Format: step,checkpoint,task,metric,value
    Sorted by step number, then task, then metric.
    """
    import re

    rows = []

    key_metrics = {
        "gsm8k": [("exact_match,strict-match", "exact_match_strict")],
        "gsm8k_cot": [("exact_match,strict-match", "exact_match_strict")],
        "hendrycks_math": [("exact_match,none", "exact_match")],
        "minerva_math": [("exact_match,none", "exact_match"), ("math_verify,none", "math_verify")],
        "ifeval": [("prompt_level_strict_acc,none", "prompt_strict"), ("inst_level_strict_acc,none", "inst_strict")],
        "mbpp": [("pass@1,none", "pass_at_1")],
        "humaneval": [("pass@1,none", "pass_at_1")],
    }

    for ckpt_name, ckpt_results in results.get("results", {}).items():
        # Extract step number from checkpoint name
        match = re.search(r"step(\d+)", ckpt_name)
        step_num = int(match.group(1)) if match else 0

        for task_key, task_data in ckpt_results.items():
            if "error" in task_data:
                continue

            task_results = task_data.get("results", {})

            # Determine base task name for metric lookup
            base_task = task_key.rsplit("_", 1)[0] if task_key.endswith("shot") else task_key
            metrics_to_extract = key_metrics.get(base_task, [("exact_match,none", "exact_match")])

            for lm_eval_key, csv_metric_name in metrics_to_extract:
                value = None
                # Search through task results for the metric
                for task_name, metrics in task_results.items():
                    if lm_eval_key in metrics:
                        value = metrics[lm_eval_key]
                        break

                if value is not None:
                    rows.append({
                        "step": step_num,
                        "checkpoint": ckpt_name,
                        "task": task_key,
                        "metric": csv_metric_name,
                        "value": f"{value:.4f}" if isinstance(value, float) else str(value),
                    })

    # Sort by step number, then task, then metric
    rows.sort(key=lambda row: (row["step"], row["task"], row["metric"]))

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["step", "checkpoint", "task", "metric", "value"])
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
