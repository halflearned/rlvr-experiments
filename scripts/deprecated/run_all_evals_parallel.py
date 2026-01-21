#!/usr/bin/env python3
"""
EVALUATION PLAN - 175 evals total
=================================

35 checkpoints × 5 tasks = 175 evals

CHECKPOINTS (step100, 200, 300, 400, 500, 600 only):
- annotations-adhoc-20260116-021228: step100, 200, 300 (Mixed lr=1e-6)
- annotations-adhoc-20260116-021258: step100 (Mixed, unknown lr)
- annotations-adhoc-20260116-021345: step100, 200, 300 (Sequential lr=1e-6)
- annotations-adhoc-20260117-025110: step100 (Mixed lr=1e-5 random)
- annotations-adhoc-20260117-061547: step100-600 (Mixed lr=5e-6 random)
- annotations-adhoc-20260117-064406: step100-600 (Mixed lr=1e-6 random s1)
- annotations-adhoc-20260117-064436: step100-600 (Mixed lr=5e-6 random s1)
- hadadv-adhoc-20260116-011803: step100-600 (Mixed lr=5e-6)
- hadadv-adhoc-20260116-011826: step100, 200, 300 (Sequential lr=5e-6)

TASKS:
1. gsm8k 8-shot
2. gsm8k 4-shot
3. hendrycks_math 4-shot
4. ifeval 0-shot
5. mbpp 3-shot

EXECUTION:
- 2 nodes: local + 172.31.17.116
- 4 parallel evals per node (GPUs 0,1 / 2,3 / 4,5 / 6,7)
- tensor_parallel_size=2

OUTPUT:
- Results go to /efs/rlvr-experiments/eval_results_batch/
- After completion, update /efs/rlvr-experiments/experiments/all_results.md
"""

import os
import re
import json
import subprocess
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple

@dataclass
class EvalJob:
    checkpoint_path: str
    checkpoint_name: str
    task: str
    num_fewshot: int
    output_name: str

def get_checkpoints() -> List[str]:
    """Get all checkpoints to evaluate (step100, 200, 300, 400, 500, 600 only)"""
    checkpoints_dir = Path("/efs/rlvr-experiments/checkpoints/sagemaker_downloads")
    all_checkpoints = []
    
    for job_dir in checkpoints_dir.iterdir():
        if not job_dir.is_dir() or job_dir.name.endswith('.tar.gz'):
            continue
        for ckpt in job_dir.iterdir():
            if ckpt.is_dir() and 'step' in ckpt.name:
                match = re.search(r'step(\d+)', ckpt.name)
                if match:
                    step = int(match.group(1))
                    if step % 100 == 0 and step <= 600:
                        all_checkpoints.append(str(ckpt))
    
    return sorted(all_checkpoints)

def get_tasks() -> List[Tuple[str, int]]:
    """Tasks to run: (task_name, num_fewshot)"""
    return [
        ("gsm8k", 8),
        ("gsm8k", 4),
        ("hendrycks_math", 4),
        ("ifeval", 0),
        ("mbpp", 3),
    ]

def generate_eval_jobs() -> List[EvalJob]:
    """Generate all evaluation jobs"""
    checkpoints = get_checkpoints()
    tasks = get_tasks()
    jobs = []
    
    for ckpt_path in checkpoints:
        ckpt_name = Path(ckpt_path).name
        for task, fewshot in tasks:
            output_name = f"{ckpt_name}_{task}_{fewshot}shot"
            jobs.append(EvalJob(
                checkpoint_path=ckpt_path,
                checkpoint_name=ckpt_name,
                task=task,
                num_fewshot=fewshot,
                output_name=output_name
            ))
    
    return jobs

def check_already_done(output_dir: Path) -> set:
    """Check which evals are already complete"""
    done = set()
    if output_dir.exists():
        for item in output_dir.iterdir():
            if item.is_dir():
                # Check if results exist
                results = list(item.glob("*/results*.json"))
                if results:
                    done.add(item.name)
    return done

if __name__ == "__main__":
    jobs = generate_eval_jobs()
    output_dir = Path("/efs/rlvr-experiments/eval_results_batch")
    done = check_already_done(output_dir)
    
    pending = [j for j in jobs if j.output_name not in done]
    
    print(f"Total jobs: {len(jobs)}")
    print(f"Already done: {len(done)}")
    print(f"Pending: {len(pending)}")
    
    # Write pending jobs to JSON for the runner scripts
    with open("/efs/rlvr-experiments/scripts/pending_evals.json", "w") as f:
        json.dump([{
            "checkpoint_path": j.checkpoint_path,
            "checkpoint_name": j.checkpoint_name,
            "task": j.task,
            "num_fewshot": j.num_fewshot,
            "output_name": j.output_name
        } for j in pending], f, indent=2)
    
    print(f"\nWrote {len(pending)} pending jobs to /efs/rlvr-experiments/scripts/pending_evals.json")
