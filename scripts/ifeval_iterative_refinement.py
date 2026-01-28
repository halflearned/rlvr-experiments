#!/usr/bin/env python3
"""
Experiment: Iterative refinement for IFEval constraints.

Give the model its own output + feedback about what's wrong, let it try again.

Usage:
    python scripts/ifeval_iterative_refinement.py --gpu 6
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from datasets import load_dataset
from vllm import LLM, SamplingParams

from rlvr_experiments.verifiers.if_multi_constraints import IFMultiConstraintsVerifier


def describe_constraint(instruction_id: str, kwargs: dict) -> str:
    """Generate a human-readable description of a constraint."""
    descriptions = {
        "detectable_format:sentence_hyphens": lambda k: "All sentences must be connected using hyphens with no spaces between them (e.g., 'First sentence.-Second sentence.-Third sentence.')",
        "detectable_format:number_highlighted_sections": lambda k: f"Include at least {k.get('num_highlights', 'N')} highlighted sections using *asterisks* (e.g., *like this*)",
        "keywords:existence": lambda k: f"Include these keywords: {k.get('keywords', [])}",
        "keywords:frequency": lambda k: f"Use the word '{k.get('keyword', '')}' {k.get('relation', '')} {k.get('frequency', '')} times",
        "keywords:forbidden_words": lambda k: f"Do NOT use these words: {k.get('forbidden_words', [])}",
        "length_constraints:number_sentences": lambda k: f"Use {k.get('relation', '')} {k.get('num_sentences', '')} sentences",
        "length_constraints:number_words": lambda k: f"Use {k.get('relation', '')} {k.get('num_words', '')} words",
        "detectable_format:number_bullet_lists": lambda k: f"Use exactly {k.get('num_bullets', '')} bullet points",
        "detectable_format:json_format": lambda k: "Format the entire response as valid JSON",
        "punctuation:no_comma": lambda k: "Do not use any commas in your response",
        "change_case:english_capital": lambda k: "Write the ENTIRE response in CAPITAL LETTERS",
        "change_case:english_lowercase": lambda k: "Write the entire response in lowercase letters",
        "startend:end_checker": lambda k: f"End your response with the exact phrase: '{k.get('end_phrase', '')}'",
        "startend:quotation": lambda k: "Wrap your entire response in double quotes",
        "copy:copying_multiple": lambda k: f"Repeat the following exactly {k.get('N', 'N')} times, separated by ******",
        "count:count_unique": lambda k: "Every word in your response must be unique (no repeated words)",
    }

    desc_fn = descriptions.get(instruction_id)
    if desc_fn:
        try:
            return desc_fn(kwargs or {})
        except Exception:
            pass
    return instruction_id.replace(":", ": ").replace("_", " ")


def run_iterative_refinement(
    model_path: str,
    gpu_id: int,
    prompt_id: str = "ifeval_8",
    max_iterations: int = 3,
    num_completions: int = 4,
):
    """Run iterative refinement on a specific prompt."""

    # Load dataset
    print("Loading dataset...")
    ds = load_dataset("allenai/IF_multi_constraints_upto5", split="train")

    # Find the prompt
    idx = int(prompt_id.split("_")[1])
    row = ds[idx]

    user_content = ""
    for msg in row["messages"]:
        if msg["role"] == "user":
            user_content = msg["content"]
            break

    ground_truth = row["ground_truth"]

    print(f"\n{'='*70}")
    print(f"PROMPT: {prompt_id}")
    print(f"{'='*70}")
    print(f"\nOriginal prompt:\n{user_content[:500]}...")

    # Parse constraints
    verifier = IFMultiConstraintsVerifier()
    _, inst_ids, kwargs_list = verifier.verify_detailed("test", ground_truth)

    print(f"\nConstraints ({len(inst_ids)}):")
    for i, (inst_id, kw) in enumerate(zip(inst_ids, kwargs_list)):
        print(f"  {i+1}. {describe_constraint(inst_id, kw)}")

    # Initialize model
    print(f"\nLoading model on GPU {gpu_id}...")
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    llm = LLM(
        model=model_path,
        tensor_parallel_size=1,
        max_model_len=4096,
        gpu_memory_utilization=0.4,
    )

    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.95,
        max_tokens=2048,
        n=num_completions,
    )

    # Iteration 0: Initial generation
    current_prompt = user_content
    all_results = []

    # Track which constraints have been satisfied (must maintain all previous + add new)
    best_constraints_satisfied = [False] * len(inst_ids)
    best_completion_so_far = None

    for iteration in range(max_iterations + 1):
        print(f"\n{'='*70}")
        print(f"ITERATION {iteration}")
        print(f"{'='*70}")
        print(f"  Required baseline: {best_constraints_satisfied}")

        if iteration > 0:
            print(f"\nPrompt (truncated):\n{current_prompt[-800:]}")

        outputs = llm.generate([current_prompt], sampling_params)
        completions = [o.text for o in outputs[0].outputs]

        # Verify each
        iteration_results = []
        new_best_found = False

        for i, comp in enumerate(completions):
            per_constraint, _, _ = verifier.verify_detailed(comp, ground_truth)
            score = sum(per_constraint) / len(per_constraint) if per_constraint else 0
            all_passed = all(per_constraint) if per_constraint else False

            # Check if this completion maintains all previous constraints AND adds new ones
            maintains_previous = all(
                per_constraint[j] or not best_constraints_satisfied[j]
                for j in range(len(per_constraint))
            )
            adds_new = any(
                per_constraint[j] and not best_constraints_satisfied[j]
                for j in range(len(per_constraint))
            )
            is_improvement = maintains_previous and adds_new

            iteration_results.append({
                "completion": comp,
                "per_constraint": per_constraint,
                "score": score,
                "all_passed": all_passed,
                "is_improvement": is_improvement,
            })

            # Update best if this is a true improvement
            if is_improvement and not new_best_found:
                new_best_found = True
                best_completion_so_far = comp
                best_per_constraint = per_constraint
                # Update the baseline for next iteration
                best_constraints_satisfied = [
                    best_constraints_satisfied[j] or per_constraint[j]
                    for j in range(len(per_constraint))
                ]

            status = "✓ FULL" if all_passed else f"{sum(per_constraint)}/{len(per_constraint)}"
            imp_marker = " ★IMPROVEMENT" if is_improvement else ""
            print(f"\n  Completion {i+1}: [{status}] score={score:.0%}{imp_marker}")
            print(f"    Per-constraint: {per_constraint}")
            print(f"    Text preview: {comp[:200].replace(chr(10), ' ')}...")

        best_score = sum(best_constraints_satisfied) / len(best_constraints_satisfied)
        all_results.append({
            "iteration": iteration,
            "prompt": current_prompt,
            "results": iteration_results,
            "best_score": best_score,
            "best_constraints_satisfied": list(best_constraints_satisfied),
        })

        # Check if we succeeded
        if any(r["all_passed"] for r in iteration_results):
            print(f"\n{'='*70}")
            print(f"SUCCESS at iteration {iteration}!")
            print(f"{'='*70}")
            break

        if iteration == max_iterations:
            print(f"\n{'='*70}")
            print(f"Max iterations reached without full success")
            print(f"{'='*70}")
            break

        # If no improvement found this iteration, keep using the previous best
        if not new_best_found:
            print(f"\n  No improvement found this iteration, keeping previous best")
            # Use any completion that at least maintains constraints
            for r in iteration_results:
                maintains = all(
                    r["per_constraint"][j] or not best_constraints_satisfied[j]
                    for j in range(len(r["per_constraint"]))
                )
                if maintains:
                    best_completion_so_far = r["completion"]
                    best_per_constraint = r["per_constraint"]
                    break
            else:
                # No completion even maintains - just use first one
                best_completion_so_far = iteration_results[0]["completion"]
                best_per_constraint = iteration_results[0]["per_constraint"]

        # Build feedback for next iteration - show what's STILL missing
        failed_indices = [i for i, satisfied in enumerate(best_constraints_satisfied) if not satisfied]
        failed_constraints = []
        for fi in failed_indices:
            desc = describe_constraint(inst_ids[fi], kwargs_list[fi])
            failed_constraints.append(desc)

        print(f"\n  Updated baseline: {best_constraints_satisfied}")
        print(f"  Still need to satisfy: {[inst_ids[i] for i in failed_indices]}")

        feedback = f"""

Your previous response was:
---
{best_completion_so_far}
---

This response did NOT satisfy the following constraints:
{chr(10).join(f'- {c}' for c in failed_constraints)}

Please try again, making sure to satisfy ALL the constraints this time. Here is the original request:

{user_content}"""

        current_prompt = feedback

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"\nScores by iteration:")
    for r in all_results:
        scores = [res["score"] for res in r["results"]]
        any_full = any(res["all_passed"] for res in r["results"])
        print(f"  Iteration {r['iteration']}: best={r['best_score']:.0%}, avg={sum(scores)/len(scores):.0%}, any_full={any_full}")

    # Save
    output_path = Path(f"/tmp/ifeval_iterative_{prompt_id}.json")
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B-Base")
    parser.add_argument("--gpu", type=int, default=6)
    parser.add_argument("--prompt-id", default="ifeval_8")
    parser.add_argument("--max-iterations", type=int, default=3)
    parser.add_argument("--num-completions", type=int, default=4)

    args = parser.parse_args()

    run_iterative_refinement(
        model_path=args.model,
        gpu_id=args.gpu,
        prompt_id=args.prompt_id,
        max_iterations=args.max_iterations,
        num_completions=args.num_completions,
    )
