#!/usr/bin/env python3
"""
Verify IFEval/IFBench completions.

- IFEval uses IFMultiConstraintsVerifier (Google IFEval format)
- IFBench uses IFBenchVerifier (AllenAI IFBench format)

Usage:
    python scripts/verify_ifeval_completions.py <completions_file> [--output OUTPUT]

Example:
    python scripts/verify_ifeval_completions.py \
        results/qwen3-1.7B-ifeval-lr5e6-beta1e3_20260125-083856/evals/step100/ifeval_completions.jsonl
"""

import argparse
import json
from pathlib import Path

from tqdm import tqdm

from rlvr_experiments.verifiers.if_multi_constraints import (
    INSTRUCTION_FUNCTIONS,
    _remove_thinking_section,
)
from rlvr_experiments.verifiers.ifbench import IFBenchVerifier, INSTRUCTION_DICT


def is_ifbench_format(instruction_id_list: list) -> bool:
    """Check if instructions are in IFBench format (e.g., 'count:word_count_range')."""
    if not instruction_id_list:
        return False
    # IFBench uses format like 'category:subcategory'
    # IFEval uses format like 'detectable_format:number_placeholders'
    first_id = instruction_id_list[0]
    return first_id in INSTRUCTION_DICT


def verify_single_ifeval(response: str, instruction_id_list: list, kwargs_list: list) -> dict:
    """Verify using IFEval (Google) verifier."""
    if not instruction_id_list:
        return {
            "all_pass": False,
            "pass_count": 0,
            "total_count": 0,
            "score": 0.0,
            "per_instruction": [],
        }

    answer = _remove_thinking_section(response)
    if not answer:
        return {
            "all_pass": False,
            "pass_count": 0,
            "total_count": len(instruction_id_list),
            "score": 0.0,
            "per_instruction": [(iid, False) for iid in instruction_id_list],
        }

    results = []
    pass_count = 0

    for i, instruction_id in enumerate(instruction_id_list):
        kwargs = kwargs_list[i] if i < len(kwargs_list) else {}
        if kwargs is None:
            kwargs = {}

        func = INSTRUCTION_FUNCTIONS.get(instruction_id)
        if func is None:
            results.append((instruction_id, False))
            continue

        kwargs = {k: v for k, v in kwargs.items() if v is not None}
        try:
            ok = func(answer, **kwargs) if kwargs else func(answer)
        except Exception:
            ok = False

        results.append((instruction_id, ok))
        if ok:
            pass_count += 1

    total = len(instruction_id_list)
    return {
        "all_pass": pass_count == total,
        "pass_count": pass_count,
        "total_count": total,
        "score": pass_count / total if total > 0 else 0.0,
        "per_instruction": results,
    }


def verify_single_ifbench(response: str, instruction_id_list: list, kwargs_list: list) -> dict:
    """Verify using IFBench (AllenAI) verifier."""
    verifier = IFBenchVerifier()
    result = verifier.verify(response, instruction_id_list, kwargs_list)
    return {
        "all_pass": result["all_pass"],
        "pass_count": result["pass_count"],
        "total_count": result["total_count"],
        "score": result["pass_count"] / result["total_count"] if result["total_count"] > 0 else 0.0,
        "per_instruction": result["per_instruction"],
    }


def verify_single(response: str, instruction_id_list: list, kwargs_list: list) -> dict:
    """Verify a single completion - auto-detects IFEval vs IFBench format."""
    if is_ifbench_format(instruction_id_list):
        return verify_single_ifbench(response, instruction_id_list, kwargs_list)
    else:
        return verify_single_ifeval(response, instruction_id_list, kwargs_list)


def main():
    parser = argparse.ArgumentParser(description="Verify IFEval/IFBench completions")
    parser.add_argument("completions_file", type=str, help="Path to completions JSONL file")
    parser.add_argument("--output", type=str, help="Output file for verified results (default: same dir, _verified.jsonl)")
    args = parser.parse_args()

    completions_file = Path(args.completions_file)
    if args.output:
        output_file = Path(args.output)
    else:
        output_file = completions_file.parent / completions_file.name.replace(".jsonl", "_verified.jsonl")

    # Load completions
    completions = []
    with open(completions_file) as f:
        for line in f:
            completions.append(json.loads(line))
    print(f"[verify] Loaded {len(completions)} completions from {completions_file}")

    # Verify each completion
    verified_results = []
    prompt_pass = 0
    inst_pass = 0
    inst_total = 0

    for item in tqdm(completions, desc="Verifying"):
        response = item.get("completion", "")
        instruction_id_list = item.get("instruction_id_list", [])
        kwargs_list = item.get("kwargs", [])

        result = verify_single(response, instruction_id_list, kwargs_list)

        # Track metrics
        if result["all_pass"]:
            prompt_pass += 1
        inst_pass += result["pass_count"]
        inst_total += result["total_count"]

        verified_results.append({
            **item,
            "verification": result,
        })

    # Compute summary metrics
    n_prompts = len(completions)
    prompt_level_acc = prompt_pass / n_prompts if n_prompts > 0 else 0.0
    inst_level_acc = inst_pass / inst_total if inst_total > 0 else 0.0

    print(f"\n=== Results for {completions_file.name} ===")
    print(f"Prompt-level strict accuracy: {prompt_pass}/{n_prompts} = {prompt_level_acc:.2%}")
    print(f"Instruction-level accuracy:   {inst_pass}/{inst_total} = {inst_level_acc:.2%}")

    # Save verified results
    with open(output_file, "w") as f:
        for r in verified_results:
            f.write(json.dumps(r) + "\n")
    print(f"[verify] Saved verified results to {output_file}")

    # Save summary
    summary_file = output_file.parent / output_file.name.replace("_verified.jsonl", "_verified_summary.json")
    summary = {
        "completions_file": str(completions_file),
        "n_prompts": n_prompts,
        "prompt_pass": prompt_pass,
        "prompt_level_strict_acc": prompt_level_acc,
        "inst_pass": inst_pass,
        "inst_total": inst_total,
        "inst_level_acc": inst_level_acc,
    }
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[verify] Saved summary to {summary_file}")

    return summary


if __name__ == "__main__":
    main()
