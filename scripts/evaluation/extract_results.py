#!/usr/bin/env python3
"""
Extract evaluation results from lm_eval output directories into CSV/Markdown.

=============================================================================
IMPORTANT: DO NOT CREATE NEW SCRIPTS FOR CORNER CASES.
If something doesn't work (new job type, new metric key, etc.), FIX THIS SCRIPT.
This is the ONE script for extracting results. Keep it that way.
=============================================================================

Usage:
    python extract_results.py [--output-dir DIR] [--csv FILE] [--md FILE]

Default output directory: /efs/rlvr-experiments/eval_results_batch
"""

import argparse
import json
from pathlib import Path
from datetime import datetime
from collections import defaultdict

# Default paths
DEFAULT_RESULTS_DIR = Path("/efs/rlvr-experiments/eval_results_batch")
DEFAULT_CSV_OUTPUT = Path("/efs/rlvr-experiments/experiments/all_results.csv")
DEFAULT_MD_OUTPUT = Path("/efs/rlvr-experiments/experiments/all_results.md")

# Checkpoint name to job metadata mapping
# Format: pattern -> {job, lr, staleness, order, curriculum, display_prefix}
JOB_METADATA = {
    # GSM8K + MATH curriculum runs (2026-01-19)
    "gsm8k_math_boxed_curriculum": {
        "job": "gsm8k_math_boxed_curriculum",
        "lr": "1e-5", "staleness": 0, "order": "Curriculum", "curriculum": "Yes",
        "prefix": "GSM8K_MATH_Curriculum"
    },
    # annotations-adhoc jobs (old naming without job prefix)
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

# hadadv jobs (new naming with job prefix in output name)
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


def load_results(result_dir: Path) -> dict | None:
    """Load results from a result directory (handles nested structure)."""
    results_files = list(result_dir.glob("**/results*.json"))
    if not results_files:
        return None
    with open(results_files[0]) as f:
        return json.load(f)


def extract_metrics(results: dict) -> dict:
    """Extract relevant metrics from lm_eval results."""
    if not results:
        return {}

    metrics = {}
    r = results.get("results", {})

    # GSM8K - try both key formats
    if "gsm8k" in r:
        gsm = r["gsm8k"]
        metrics["gsm8k_flex"] = gsm.get("exact_match,flexible-extract") or gsm.get("flexible-extract,none")
        metrics["gsm8k_strict"] = gsm.get("exact_match,strict-match") or gsm.get("strict-match,none")

    # GSM8K CoT - also supports strict-match
    if "gsm8k_cot" in r:
        gsm_cot = r["gsm8k_cot"]
        metrics["gsm8k_cot_flex"] = gsm_cot.get("exact_match,flexible-extract") or gsm_cot.get("flexible-extract,none")
        metrics["gsm8k_cot_strict"] = gsm_cot.get("exact_match,strict-match") or gsm_cot.get("strict-match,none")

    # hendrycks_math
    if "hendrycks_math" in r:
        metrics["hendrycks_math"] = r["hendrycks_math"].get("exact_match,none")

    # minerva_math
    if "minerva_math" in r:
        metrics["minerva_math"] = r["minerva_math"].get("exact_match,none")
        metrics["minerva_math_verify"] = r["minerva_math"].get("math_verify,none")

    # IFEval
    if "ifeval" in r:
        ifeval = r["ifeval"]
        metrics["ifeval_prompt"] = ifeval.get("prompt_level_strict_acc,none")
        metrics["ifeval_inst"] = ifeval.get("inst_level_strict_acc,none")

    # MBPP - key can be pass_at_1 or pass@1 (check pass_at_1 first, it's more common)
    if "mbpp" in r:
        metrics["mbpp"] = r["mbpp"].get("pass_at_1,none") or r["mbpp"].get("pass@1,none")

    return metrics


def extract_step(name: str) -> int | None:
    """Extract step number from checkpoint name."""
    if "_step" in name:
        try:
            return int(name.split("_step")[1].split("_")[0])
        except (ValueError, IndexError):
            return None
    return None


def get_job_info(ckpt_name: str) -> dict | None:
    """Get job metadata for a checkpoint name."""
    # Sort by length descending to match longer patterns first
    for pattern in sorted(JOB_METADATA.keys(), key=len, reverse=True):
        if ckpt_name.startswith(pattern):
            return JOB_METADATA[pattern]
    return None


def parse_result_name(name: str) -> tuple | None:
    """
    Parse result directory name to extract checkpoint info and task.

    Returns: (unique_key, task, job, ckpt_name, job_info) or None
    """
    # Check for hadadv job-prefixed format: hadadv-adhoc-YYYYMMDD-HHMMSS_checkpoint_task_Nshot
    for job_name, job_info in HADADV_JOBS.items():
        if name.startswith(job_name + "_"):
            rest = name[len(job_name) + 1:]

            # Parse task suffix
            task_suffixes = [
                ("_gsm8k_cot_8shot", "gsm8k_cot_8shot"),
                ("_gsm8k_cot_4shot", "gsm8k_cot_4shot"),
                ("_gsm8k_8shot", "gsm8k_8shot"),
                ("_gsm8k_4shot", "gsm8k_4shot"),
                ("_hendrycks_math_4shot", "hendrycks_math"),
                ("_minerva_math_4shot", "minerva_math"),
                ("_math_qwen_4shot", "math_qwen"),
                ("_ifeval_0shot", "ifeval"),
                ("_mbpp_3shot", "mbpp"),
            ]

            for suffix, task in task_suffixes:
                if suffix in rest:
                    ckpt_name = rest.replace(suffix, "")
                    unique_key = f"{job_name}_{ckpt_name}"
                    full_info = {"job": job_name, **job_info}
                    return (unique_key, task, job_name, ckpt_name, full_info)
            return None

    # Old format without job prefix
    task_suffixes = [
        ("_gsm8k_cot_8shot", "gsm8k_cot_8shot"),
        ("_gsm8k_cot_4shot", "gsm8k_cot_4shot"),
        ("_gsm8k_8shot", "gsm8k_8shot"),
        ("_gsm8k_4shot", "gsm8k_4shot"),
        ("_hendrycks_math_4shot", "hendrycks_math"),
        ("_minerva_math_4shot", "minerva_math"),
        ("_math_qwen_4shot", "math_qwen"),
        ("_ifeval_0shot", "ifeval"),
        ("_mbpp_3shot", "mbpp"),
    ]

    for suffix, task in task_suffixes:
        if suffix in name:
            ckpt_name = name.replace(suffix, "")
            job_info = get_job_info(ckpt_name)
            if job_info:
                return (ckpt_name, task, job_info["job"], ckpt_name, job_info)
            return None

    return None


def collect_results(results_dir: Path) -> dict:
    """Collect all results into a structured dictionary."""
    checkpoint_results = {}

    # First, collect lm_eval results from directories
    for result_dir in results_dir.iterdir():
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
        elif task == "gsm8k_cot_8shot":
            checkpoint_results[unique_key]["gsm8k_cot_8shot_flex"] = metrics.get("gsm8k_cot_flex")
            checkpoint_results[unique_key]["gsm8k_cot_8shot_strict"] = metrics.get("gsm8k_cot_strict")
        elif task == "gsm8k_cot_4shot":
            checkpoint_results[unique_key]["gsm8k_cot_4shot_flex"] = metrics.get("gsm8k_cot_flex")
            checkpoint_results[unique_key]["gsm8k_cot_4shot_strict"] = metrics.get("gsm8k_cot_strict")
        elif task == "hendrycks_math":
            checkpoint_results[unique_key]["hendrycks_math"] = metrics.get("hendrycks_math")
        elif task == "minerva_math":
            checkpoint_results[unique_key]["minerva_math"] = metrics.get("minerva_math")
            checkpoint_results[unique_key]["minerva_math_verify"] = metrics.get("minerva_math_verify")
        elif task == "ifeval":
            checkpoint_results[unique_key]["ifeval_prompt"] = metrics.get("ifeval_prompt")
            checkpoint_results[unique_key]["ifeval_inst"] = metrics.get("ifeval_inst")
        elif task == "mbpp":
            checkpoint_results[unique_key]["mbpp"] = metrics.get("mbpp")

    # Second, collect math_qwen results from JSON files
    for json_file in results_dir.glob("*_math_qwen_4shot.json"):
        name = json_file.stem  # filename without .json
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

        # Load math_qwen JSON and extract accuracy
        with open(json_file) as f:
            data = json.load(f)
            # math_qwen stores correct count and total, compute ratio
            correct = data.get("correct", 0)
            total = data.get("total", 1)
            checkpoint_results[unique_key]["math_qwen"] = correct / total if total else 0

    return checkpoint_results


def write_csv(checkpoint_results: dict, output_path: Path):
    """Write results to CSV file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Sort by prefix and step
    sorted_checkpoints = []
    for unique_key, data in checkpoint_results.items():
        job_info = data["job_info"]
        step = extract_step(data["ckpt_name"])
        if step:
            sorted_checkpoints.append((job_info["prefix"], step, unique_key, data, job_info))

    sorted_checkpoints.sort(key=lambda x: (x[0], x[1]))

    with open(output_path, "w") as f:
        # Header
        f.write("checkpoint,job,lr,step,staleness,order,curriculum,")
        f.write("gsm8k_4shot_flex,gsm8k_4shot_strict,gsm8k_8shot_flex,gsm8k_8shot_strict,")
        f.write("hendrycks_math,math_qwen,ifeval_prompt,ifeval_inst,mbpp,run_date\n")

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

            # Add metrics
            for key in ["gsm8k_4shot_flex", "gsm8k_4shot_strict", "gsm8k_8shot_flex", "gsm8k_8shot_strict",
                        "hendrycks_math", "math_qwen", "ifeval_prompt", "ifeval_inst", "mbpp"]:
                val = data.get(key)
                if val is not None:
                    row.append(f"{val:.4f}")
                else:
                    row.append("")

            row.append(datetime.now().strftime("%Y-%m-%d"))
            f.write(",".join(row) + "\n")

    print(f"CSV written to: {output_path}")


def write_markdown(checkpoint_results: dict, output_path: Path):
    """Write results to Markdown file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Group by experiment type
    experiments = defaultdict(list)
    for unique_key, data in checkpoint_results.items():
        job_info = data["job_info"]
        step = extract_step(data["ckpt_name"])
        if step:
            experiments[job_info["prefix"]].append((step, data))

    # Sort each experiment by step
    for prefix in experiments:
        experiments[prefix].sort(key=lambda x: x[0])

    with open(output_path, "w") as f:
        f.write("# Evaluation Results\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")

        for prefix in sorted(experiments.keys()):
            exp_data = experiments[prefix]
            if not exp_data:
                continue

            job_info = exp_data[0][1]["job_info"]
            f.write(f"## {prefix}\n\n")
            f.write(f"- **Job**: `{exp_data[0][1]['job']}`\n")
            f.write(f"- **Learning Rate**: {job_info['lr']}\n")
            f.write(f"- **Order**: {job_info['order']}\n")
            f.write(f"- **Staleness**: {job_info['staleness']}\n\n")

            # Table header
            f.write("| Step | GSM8K | MATH | IFEval | MBPP |\n")
            f.write("|------|-------|------|--------|------|\n")

            for step, data in exp_data:
                gsm4 = data.get("gsm8k_4shot_flex")
                math_qwen = data.get("math_qwen")
                ifeval = data.get("ifeval_prompt")
                mbpp = data.get("mbpp")

                row = [
                    str(step),
                    f"{gsm4*100:.1f}%" if gsm4 else "-",
                    f"{math_qwen*100:.1f}%" if math_qwen else "-",
                    f"{ifeval*100:.1f}%" if ifeval else "-",
                    f"{mbpp*100:.1f}%" if mbpp else "-",
                ]
                f.write("| " + " | ".join(row) + " |\n")

            f.write("\n")

    print(f"Markdown written to: {output_path}")


def print_table(checkpoint_results: dict, filter_prefix: str = None):
    """Print a quick results table to stdout."""
    # Group by prefix and sort by step
    experiments = defaultdict(list)
    for unique_key, data in checkpoint_results.items():
        job_info = data.get("job_info", {})
        prefix = job_info.get("prefix", "Unknown")
        if filter_prefix and filter_prefix.lower() not in prefix.lower():
            continue
        step = extract_step(data.get("ckpt_name", ""))
        if step:
            experiments[prefix].append((step, data))

    # Sort each experiment by step
    for prefix in experiments:
        experiments[prefix].sort(key=lambda x: x[0])

    for prefix in sorted(experiments.keys()):
        exp_data = experiments[prefix]
        if not exp_data:
            continue

        print(f"\n{'=' * 80}")
        print(f" {prefix}")
        print(f"{'=' * 80}")

        # Determine which columns to show based on available data
        has_gsm8k = any(d.get("gsm8k_8shot_strict") or d.get("gsm8k_4shot_strict") for _, d in exp_data)
        has_gsm8k_cot = any(d.get("gsm8k_cot_8shot_strict") or d.get("gsm8k_cot_4shot_strict") for _, d in exp_data)
        has_hendrycks = any(d.get("hendrycks_math") for _, d in exp_data)
        has_math_qwen = any(d.get("math_qwen") for _, d in exp_data)
        has_ifeval = any(d.get("ifeval_prompt") for _, d in exp_data)
        has_mbpp = any(d.get("mbpp") for _, d in exp_data)

        # Build header
        header = f"{'Step':<8}"
        if has_gsm8k:
            header += f"{'gsm8k':<10}"
        if has_gsm8k_cot:
            header += f"{'gsm8k_cot':<12}"
        if has_hendrycks:
            header += f"{'hendrycks':<12}"
        if has_math_qwen:
            header += f"{'math_qwen':<12}"
        if has_ifeval:
            header += f"{'ifeval':<10}"
        if has_mbpp:
            header += f"{'mbpp':<10}"

        print(header)
        print("-" * len(header))

        for step, data in exp_data:
            row = f"{step:<8}"

            if has_gsm8k:
                val = data.get("gsm8k_8shot_strict") or data.get("gsm8k_4shot_strict")
                row += f"{f'{val*100:.1f}%' if val else '-':<10}"

            if has_gsm8k_cot:
                val = data.get("gsm8k_cot_8shot_strict") or data.get("gsm8k_cot_4shot_strict")
                row += f"{f'{val*100:.1f}%' if val else '-':<12}"

            if has_hendrycks:
                val = data.get("hendrycks_math")
                row += f"{f'{val*100:.1f}%' if val else '-':<12}"

            if has_math_qwen:
                val = data.get("math_qwen")
                row += f"{f'{val*100:.1f}%' if val else '-':<12}"

            if has_ifeval:
                val = data.get("ifeval_prompt")
                row += f"{f'{val*100:.1f}%' if val else '-':<10}"

            if has_mbpp:
                val = data.get("mbpp")
                row += f"{f'{val*100:.1f}%' if val else '-':<10}"

            print(row)

        print()


def main():
    parser = argparse.ArgumentParser(description="Extract lm_eval results to CSV/Markdown")
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR,
                        help="Directory containing eval results")
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV_OUTPUT,
                        help="Output CSV file path")
    parser.add_argument("--md", type=Path, default=DEFAULT_MD_OUTPUT,
                        help="Output Markdown file path")
    parser.add_argument("--no-csv", action="store_true", help="Skip CSV output")
    parser.add_argument("--no-md", action="store_true", help="Skip Markdown output")
    parser.add_argument("--print-table", "-p", action="store_true",
                        help="Print results table to stdout (skips file output)")
    parser.add_argument("--filter", "-f", type=str,
                        help="Filter results by prefix (case-insensitive substring match)")
    args = parser.parse_args()

    print(f"Reading results from: {args.results_dir}")
    checkpoint_results = collect_results(args.results_dir)
    print(f"Found {len(checkpoint_results)} checkpoints with results")

    if args.print_table:
        print_table(checkpoint_results, args.filter)
    else:
        if not args.no_csv:
            write_csv(checkpoint_results, args.csv)

        if not args.no_md:
            write_markdown(checkpoint_results, args.md)


if __name__ == "__main__":
    main()
