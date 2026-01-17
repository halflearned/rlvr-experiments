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
import subprocess
import sys
import tempfile
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


def run_lm_eval(
    checkpoint_path: str,
    task: str,
    num_fewshot: int | None,
    vllm_config: dict,
    output_path: str,
    seed: int = 42,
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

    checkpoints = config.get("checkpoints", [])
    benchmarks = config.get("benchmarks", [])
    vllm_config = config.get("vllm", {})
    output_dir = config.get("output_dir", "/tmp/eval_results")

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

    # Run evaluations
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
            )

            all_results["results"][ckpt_name][result_key] = results

            # Print summary
            if "results" in results:
                task_results = results["results"]
                for task_name, metrics in task_results.items():
                    for metric, value in metrics.items():
                        if isinstance(value, (int, float)):
                            print(f"  {task_name}/{metric}: {value:.4f}", flush=True)

    # Save summary results
    summary_path = os.path.join(local_output_dir, f"summary_{timestamp}.json")
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[eval] Summary saved to: {summary_path}", flush=True)

    # Generate markdown summary table
    markdown_path = os.path.join(local_output_dir, f"summary_{timestamp}.md")
    generate_markdown_summary(all_results, markdown_path)
    print(f"[eval] Markdown summary saved to: {markdown_path}", flush=True)

    # Upload to S3 if needed
    if output_is_s3:
        upload_to_s3(local_output_dir, output_dir)
        print(f"[eval] Results uploaded to: {output_dir}", flush=True)

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


if __name__ == "__main__":
    main()
