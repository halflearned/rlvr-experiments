#!/usr/bin/env python3
"""Generate CSV from batch eval results."""

import json
from pathlib import Path
from datetime import datetime

RESULTS_DIR = Path("/efs/rlvr-experiments/eval_results_batch")

# Map checkpoint name patterns to job info
# For results WITHOUT job prefix (old format)
JOB_INFO = {
    "mixed_lr1e5_random": {
        "job": "annotations-adhoc-20260117-025110",
        "lr": "1e-5", "staleness": 1, "order": "Mixed", "curriculum": "No",
        "prefix": "Random_lr1e5_s1"
    },
    "mixed_lr1e6_random_s1": {
        "job": "annotations-adhoc-20260117-064406",
        "lr": "1e-6", "staleness": 1, "order": "Mixed", "curriculum": "No",
        "prefix": "Random_lr1e6_s1"
    },
    "mixed_lr5e6_random_s1": {
        "job": "annotations-adhoc-20260117-064436",
        "lr": "5e-6", "staleness": 1, "order": "Mixed", "curriculum": "No",
        "prefix": "Random_lr5e6_s1"
    },
    "mixed_lr5e6_random": {
        "job": "annotations-adhoc-20260117-061547",
        "lr": "5e-6", "staleness": 0, "order": "Mixed", "curriculum": "No",
        "prefix": "Random_lr5e6"
    },
    "qwen3_1_7b_mixed": {
        "job": "annotations-adhoc-20260116-021228",
        "lr": "1e-6", "staleness": 0, "order": "Mixed", "curriculum": "No",
        "prefix": "Mixed_lr1e6"
    },
    "qwen3_1_7b_sequential": {
        "job": "annotations-adhoc-20260116-021345",
        "lr": "1e-6", "staleness": 0, "order": "Sequential", "curriculum": "No",
        "prefix": "Seq_lr1e6"
    },
}

# Job info for results WITH job prefix (new format: hadadv-*)
HADADV_JOBS = {
    "hadadv-adhoc-20260116-011803": {
        "lr": "5e-6", "staleness": 0, "order": "Mixed", "curriculum": "No",
        "prefix": "Hadadv_Mixed_lr5e6"
    },
    "hadadv-adhoc-20260116-011826": {
        "lr": "5e-6", "staleness": 0, "order": "Sequential", "curriculum": "No",
        "prefix": "Hadadv_Seq_lr5e6"
    },
}

def get_job_info(checkpoint_name):
    for pattern, info in sorted(JOB_INFO.items(), key=lambda x: -len(x[0])):
        if checkpoint_name.startswith(pattern):
            return info
    return None

def extract_step(checkpoint_name):
    if "_step" in checkpoint_name:
        return int(checkpoint_name.split("_step")[1].split("_")[0])
    return None

def load_results(result_dir):
    """Load results from a result directory."""
    # Results may be nested in a subdirectory
    results_file = list(result_dir.glob("**/results*.json"))
    if not results_file:
        return None
    with open(results_file[0]) as f:
        return json.load(f)

def extract_metrics(results):
    """Extract metrics from lm_eval results."""
    if not results:
        return {}

    metrics = {}
    r = results.get("results", {})

    # GSM8K - try both key formats
    if "gsm8k" in r:
        gsm = r["gsm8k"]
        metrics["gsm8k_flex"] = gsm.get("exact_match,flexible-extract") or gsm.get("flexible-extract,none")
        metrics["gsm8k_strict"] = gsm.get("exact_match,strict-match") or gsm.get("strict-match,none")

    # hendrycks_math
    if "hendrycks_math" in r:
        metrics["hendrycks_math"] = r["hendrycks_math"].get("exact_match,none")

    # IFEval
    if "ifeval" in r:
        ifeval = r["ifeval"]
        metrics["ifeval_prompt"] = ifeval.get("prompt_level_strict_acc,none")
        metrics["ifeval_inst"] = ifeval.get("inst_level_strict_acc,none")

    # MBPP - key can be pass@1 or pass_at_1
    if "mbpp" in r:
        metrics["mbpp"] = r["mbpp"].get("pass@1,none") or r["mbpp"].get("pass_at_1,none")

    return metrics

def parse_result_name(name):
    """Parse result directory name to extract job, checkpoint, task info.

    Returns: (unique_key, task, job, checkpoint_name) or None if can't parse
    """
    # Check for hadadv job-prefixed format: hadadv-adhoc-YYYYMMDD-HHMMSS_checkpoint_task_Nshot
    for job_name, job_info in HADADV_JOBS.items():
        if name.startswith(job_name + "_"):
            rest = name[len(job_name) + 1:]  # Remove job prefix and underscore

            # Parse task suffix
            if "_gsm8k_8shot" in rest:
                ckpt_name = rest.replace("_gsm8k_8shot", "")
                task = "gsm8k_8shot"
            elif "_gsm8k_4shot" in rest:
                ckpt_name = rest.replace("_gsm8k_4shot", "")
                task = "gsm8k_4shot"
            elif "_hendrycks_math_4shot" in rest:
                ckpt_name = rest.replace("_hendrycks_math_4shot", "")
                task = "hendrycks_math"
            elif "_ifeval_0shot" in rest:
                ckpt_name = rest.replace("_ifeval_0shot", "")
                task = "ifeval"
            elif "_mbpp_3shot" in rest:
                ckpt_name = rest.replace("_mbpp_3shot", "")
                task = "mbpp"
            else:
                return None

            unique_key = f"{job_name}_{ckpt_name}"
            return (unique_key, task, job_name, ckpt_name, job_info)

    # Old format without job prefix
    if "_gsm8k_8shot" in name:
        ckpt_name = name.replace("_gsm8k_8shot", "")
        task = "gsm8k_8shot"
    elif "_gsm8k_4shot" in name:
        ckpt_name = name.replace("_gsm8k_4shot", "")
        task = "gsm8k_4shot"
    elif "_hendrycks_math_4shot" in name:
        ckpt_name = name.replace("_hendrycks_math_4shot", "")
        task = "hendrycks_math"
    elif "_ifeval_0shot" in name:
        ckpt_name = name.replace("_ifeval_0shot", "")
        task = "ifeval"
    elif "_mbpp_3shot" in name:
        ckpt_name = name.replace("_mbpp_3shot", "")
        task = "mbpp"
    else:
        return None

    job_info = get_job_info(ckpt_name)
    if not job_info:
        return None

    unique_key = ckpt_name
    return (unique_key, task, job_info["job"], ckpt_name, job_info)

def main():
    # Collect all results by unique checkpoint key
    checkpoint_results = {}

    for result_dir in RESULTS_DIR.iterdir():
        if not result_dir.is_dir():
            continue

        name = result_dir.name
        results = load_results(result_dir)
        if not results:
            continue

        parsed = parse_result_name(name)
        if not parsed:
            continue

        unique_key, task, job, ckpt_name, job_info = parsed

        if unique_key not in checkpoint_results:
            checkpoint_results[unique_key] = {
                "job": job,
                "ckpt_name": ckpt_name,
                "job_info": job_info,
            }

        metrics = extract_metrics(results)

        if task == "gsm8k_8shot":
            checkpoint_results[unique_key]["gsm8k_8shot_flex"] = metrics.get("gsm8k_flex")
            checkpoint_results[unique_key]["gsm8k_8shot_strict"] = metrics.get("gsm8k_strict")
        elif task == "gsm8k_4shot":
            checkpoint_results[unique_key]["gsm8k_4shot_flex"] = metrics.get("gsm8k_flex")
            checkpoint_results[unique_key]["gsm8k_4shot_strict"] = metrics.get("gsm8k_strict")
        elif task == "hendrycks_math":
            checkpoint_results[unique_key]["hendrycks_math"] = metrics.get("hendrycks_math")
        elif task == "ifeval":
            checkpoint_results[unique_key]["ifeval_prompt"] = metrics.get("ifeval_prompt")
            checkpoint_results[unique_key]["ifeval_inst"] = metrics.get("ifeval_inst")
        elif task == "mbpp":
            checkpoint_results[unique_key]["mbpp"] = metrics.get("mbpp")

    # Print CSV header
    print("checkpoint,job,lr,step,staleness,order,curriculum,gsm8k_4shot_flex,gsm8k_4shot_strict,gsm8k_8shot_flex,gsm8k_8shot_strict,hendrycks_math,ifeval_prompt,ifeval_inst,mbpp,run_date")

    # Sort by job info prefix and step
    sorted_checkpoints = []
    for unique_key, data in checkpoint_results.items():
        job_info = data["job_info"]
        ckpt_name = data["ckpt_name"]
        step = extract_step(ckpt_name)
        if step:
            sorted_checkpoints.append((job_info["prefix"], step, unique_key, data, job_info))

    sorted_checkpoints.sort(key=lambda x: (x[0], x[1]))

    for prefix, step, unique_key, data, job_info in sorted_checkpoints:
        row = [
            f"{prefix}_step{step}",
            data["job"],
            job_info["lr"],
            str(step),
            str(job_info["staleness"]),
            job_info["order"],
            job_info["curriculum"],
        ]

        # Add metrics (as decimals, not percentages)
        for key in ["gsm8k_4shot_flex", "gsm8k_4shot_strict", "gsm8k_8shot_flex", "gsm8k_8shot_strict",
                    "hendrycks_math", "ifeval_prompt", "ifeval_inst", "mbpp"]:
            val = data.get(key)
            if val is not None:
                row.append(f"{val:.4f}")
            else:
                row.append("")

        row.append(datetime.now().strftime("%Y-%m-%d"))
        print(",".join(row))

if __name__ == "__main__":
    main()
