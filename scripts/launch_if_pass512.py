#!/usr/bin/env python3
"""
Launch pass@512 evaluations for all IF models across multiple nodes.
Each job runs on a single GPU with vLLM serving Qwen3-1.7B.
"""
import subprocess
import sys
import json
import os
import time
from pathlib import Path

# All models with their checkpoints
MODELS = {
    "grpo": "/efs/rlvr-experiments/results/annotations-adhoc-20260125-083856/checkpoints/config_rewritten_20260125-084721_step200",
    "dapo": "/efs/rlvr-experiments/results/qwen3-1.7B-if-dapo_20260130-101403/checkpoints/qwen3-1.7B-if-dapo_20260130-101403_step200",
    "dapo-sft": "/efs/rlvr-experiments/results/qwen3-1.7B-if-dapo_sft_20260130-073008/checkpoints/qwen3-1.7B-if-dapo_sft_20260130-073008_step200",
    "grpo-sft": "/efs/rlvr-experiments/results/qwen3-1.7B-if-grpo_sft_20260130-223405/checkpoints/qwen3-1.7B-if-grpo_sft_20260130-223405_step200",
    "sft-only": "/efs/rlvr-experiments/results/qwen3-1.7B-if-sft-hf-lr5e6-10ep/checkpoint-3810",
    "sm-lr1e5-beta1e4": "/efs/rlvr-experiments/results/annotations-adhoc-20260130-181815/checkpoints/config_rewritten_20260130-182707_step200",
    "sm-lr1e5-beta1e3": "/efs/rlvr-experiments/results/vlm-experiment-20260130-183425/checkpoints/config_rewritten_20260130-184339_step200",
    "sm-lr75e5-beta1e4": "/efs/rlvr-experiments/results/vlm-experiment-20260130-183444/checkpoints/config_rewritten_20260130-184334_step200",
    "sm-lr75e5-beta1e3": "/efs/rlvr-experiments/results/vlm-experiment-20260130-183458/checkpoints/config_rewritten_20260130-184336_step200",
    "sm-dapo-sft-lr1e5-beta1e4": "/efs/rlvr-experiments/results/annotations-adhoc-20260201-091135/checkpoints/config_rewritten_20260201-092032_step200",
    "sm-grpo-lr1e5-beta1e4": "/efs/rlvr-experiments/results/annotations-adhoc-20260201-095517/checkpoints/config_rewritten_20260201-100341_step200",
    "sm-grpo-sft-lr1e5-beta1e4": "/efs/rlvr-experiments/results/annotations-adhoc-20260201-095533/checkpoints/config_rewritten_20260201-100358_step200",
    "base": "Qwen/Qwen3-1.7B-Base",
}

# Map model IDs to their result dirs (for output placement)
RESULT_DIRS = {
    "grpo": "/efs/rlvr-experiments/results/annotations-adhoc-20260125-083856",
    "dapo": "/efs/rlvr-experiments/results/qwen3-1.7B-if-dapo_20260130-101403",
    "dapo-sft": "/efs/rlvr-experiments/results/qwen3-1.7B-if-dapo_sft_20260130-073008",
    "grpo-sft": "/efs/rlvr-experiments/results/qwen3-1.7B-if-grpo_sft_20260130-223405",
    "sft-only": "/efs/rlvr-experiments/results/qwen3-1.7B-if-sft-only-hf-lr5e6",
    "sm-lr1e5-beta1e4": "/efs/rlvr-experiments/results/annotations-adhoc-20260130-181815",
    "sm-lr1e5-beta1e3": "/efs/rlvr-experiments/results/vlm-experiment-20260130-183425",
    "sm-lr75e5-beta1e4": "/efs/rlvr-experiments/results/vlm-experiment-20260130-183444",
    "sm-lr75e5-beta1e3": "/efs/rlvr-experiments/results/vlm-experiment-20260130-183458",
    "sm-dapo-sft-lr1e5-beta1e4": "/efs/rlvr-experiments/results/annotations-adhoc-20260201-091135",
    "sm-grpo-lr1e5-beta1e4": "/efs/rlvr-experiments/results/annotations-adhoc-20260201-095517",
    "sm-grpo-sft-lr1e5-beta1e4": "/efs/rlvr-experiments/results/annotations-adhoc-20260201-095533",
    "base": "/efs/rlvr-experiments/results/qwen3-1.7B-base",
}

BENCHMARKS = ["ifeval", "ifbench"]

def build_jobs():
    """Build all (model, benchmark) jobs."""
    jobs = []
    for model_id in MODELS:
        for bench in BENCHMARKS:
            output_dir = f"{RESULT_DIRS[model_id]}/evals/{bench}/pass-at-k-512"
            jobs.append({
                "model_id": model_id,
                "benchmark": bench,
                "model_path": MODELS[model_id],
                "output_dir": output_dir,
            })
    return jobs


def launch_job_local(job, gpu):
    """Launch a job on a local GPU. Returns (job, Popen, log_path)."""
    log_path = f"/tmp/pass512_{job['model_id']}_{job['benchmark']}.log"
    cmd = (
        f"CUDA_VISIBLE_DEVICES={gpu} "
        f"/efs/rlvr-experiments/.venv/bin/python -u "
        f"/efs/rlvr-experiments/scripts/eval_pass_at_k.py "
        f"{job['benchmark']} "
        f"--split train "
        f"--n 512 "
        f"--batch-size 16 "
        f"--max-tokens 2048 "
        f"--max-model-len 4096 "
        f"--temperature 1.0 "
        f"--model-path {job['model_path']} "
        f"--gpus 0 "
        f"--output-dir {job['output_dir']} "
        f"--verifier-workers 8"
    )
    print(f"[GPU {gpu}] Launching {job['model_id']}/{job['benchmark']} -> {log_path}")
    proc = subprocess.Popen(
        cmd, shell=True,
        stdout=open(log_path, 'w'),
        stderr=subprocess.STDOUT,
        cwd="/efs/rlvr-experiments",
    )
    return (job, proc, log_path)


def launch_job_remote(job, gpu, host):
    """Launch a job on a remote GPU. Returns (job, Popen, log_path)."""
    log_path = f"/tmp/pass512_{job['model_id']}_{job['benchmark']}.log"
    remote_cmd = (
        f"CUDA_VISIBLE_DEVICES={gpu} "
        f"/efs/rlvr-experiments/.venv/bin/python -u "
        f"/efs/rlvr-experiments/scripts/eval_pass_at_k.py "
        f"{job['benchmark']} "
        f"--split train "
        f"--n 512 "
        f"--batch-size 16 "
        f"--max-tokens 2048 "
        f"--max-model-len 4096 "
        f"--temperature 1.0 "
        f"--model-path {job['model_path']} "
        f"--gpus 0 "
        f"--output-dir {job['output_dir']} "
        f"--verifier-workers 8"
    )
    # Remote log goes to the shared EFS log path
    cmd = f'ssh ubuntu@{host} "cd /efs/rlvr-experiments && {remote_cmd}" > {log_path} 2>&1'
    print(f"[{host} GPU {gpu}] Launching {job['model_id']}/{job['benchmark']} -> {log_path}")
    proc = subprocess.Popen(cmd, shell=True)
    return (job, proc, log_path)


def main():
    jobs = build_jobs()
    print(f"Total jobs: {len(jobs)}")

    # Assign jobs to GPUs across 3 nodes
    # Primary: GPUs 0-7 (local)
    # Secondary: GPUs 0-7 (172.31.17.116)
    # Tertiary: GPUs 0-7 (172.31.24.124)

    assignments = []
    for i, job in enumerate(jobs):
        if i < 8:
            assignments.append(("local", i, None))
        elif i < 16:
            assignments.append(("remote", i - 8, "172.31.17.116"))
        elif i < 24:
            assignments.append(("remote", i - 16, "172.31.24.124"))
        else:
            # Overflow: will be launched after first batch completes
            assignments.append(None)

    # Launch first 24 jobs
    running = []
    overflow = []
    for i, job in enumerate(jobs):
        if i < 24:
            node_type, gpu, host = assignments[i]
            if node_type == "local":
                running.append(launch_job_local(job, gpu))
            else:
                running.append(launch_job_remote(job, gpu, host))
        else:
            overflow.append(job)

    print(f"\nLaunched {len(running)} jobs. {len(overflow)} overflow jobs waiting.")

    # Write job manifest
    manifest = []
    for job, proc, log_path in running:
        manifest.append({
            "model_id": job["model_id"],
            "benchmark": job["benchmark"],
            "pid": proc.pid,
            "log": log_path,
            "output_dir": job["output_dir"],
        })
    for job in overflow:
        manifest.append({
            "model_id": job["model_id"],
            "benchmark": job["benchmark"],
            "pid": None,
            "log": None,
            "output_dir": job["output_dir"],
        })

    with open("/tmp/pass512_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Manifest written to /tmp/pass512_manifest.json")

    # Wait for first batch, then launch overflow
    finished = set()
    while len(finished) < len(running):
        time.sleep(30)
        for i, (job, proc, log_path) in enumerate(running):
            if i in finished:
                continue
            ret = proc.poll()
            if ret is not None:
                finished.add(i)
                status = "OK" if ret == 0 else f"FAILED (rc={ret})"
                print(f"[{len(finished)}/{len(running)}] {job['model_id']}/{job['benchmark']}: {status}")

                # Launch overflow job on freed GPU if any
                if overflow and ret == 0:
                    oj = overflow.pop(0)
                    node_type, gpu, host = assignments[i]
                    if node_type == "local":
                        running.append(launch_job_local(oj, gpu))
                    else:
                        running.append(launch_job_remote(oj, gpu, host))

    # Final status
    print("\n" + "=" * 60)
    print("All jobs complete!")
    failed = []
    for i, (job, proc, log_path) in enumerate(running):
        if proc.returncode != 0:
            failed.append((job, log_path))
    if failed:
        print(f"\n{len(failed)} FAILED jobs:")
        for job, log_path in failed:
            print(f"  {job['model_id']}/{job['benchmark']}: {log_path}")
    else:
        print("All jobs succeeded!")


if __name__ == "__main__":
    main()
