#!/usr/bin/env python3
"""
Experiment: Does telling the model which constraints it failed help it succeed?

This script:
1. Loads IFEval prompts from the hard portion of the curriculum
2. Generates k completions for each
3. Identifies prompts where the model often satisfies some but not all constraints
4. Re-runs with feedback about which constraints were violated
5. Compares performance

Usage:
    python scripts/ifeval_constraint_feedback_experiment.py --gpu 6
"""

import argparse
import json
import random
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from datasets import load_dataset
from vllm import LLM, SamplingParams

from rlvr_experiments.verifiers.if_multi_constraints import IFMultiConstraintsVerifier


def load_ifeval_dataset():
    """Load the IF multi-constraints dataset."""
    print("Loading IF_multi_constraints_upto5 dataset...")
    ds = load_dataset("allenai/IF_multi_constraints_upto5", split="train")

    # Build index mapping prompt_id -> row
    id_to_row = {}
    for i, row in enumerate(ds):
        prompt_id = f"ifeval_{i}"
        # Extract user content
        user_content = ""
        for msg in row["messages"]:
            if msg["role"] == "user":
                user_content = msg["content"]
                break
        id_to_row[prompt_id] = {
            "prompt_id": prompt_id,
            "prompt": user_content,
            "ground_truth": row["ground_truth"],
            "constraint_type": row["constraint_type"],
            "constraint": row["constraint"],
        }
    return id_to_row


def load_curriculum_bottom_half(curriculum_path: str) -> list[str]:
    """Load prompt IDs from the bottom (harder) half of curriculum."""
    with open(curriculum_path) as f:
        lines = [l.strip() for l in f.readlines() if l.strip()]
    # Bottom half = harder problems
    return lines[len(lines)//2:]


def describe_constraint(instruction_id: str, kwargs: dict) -> str:
    """Generate a human-readable description of a constraint."""
    # Map instruction IDs to readable descriptions
    descriptions = {
        "keywords:existence": lambda k: f"include the keywords: {k.get('keywords', [])}",
        "keywords:frequency": lambda k: f"use the word '{k.get('keyword', '')}' {k.get('relation', '')} {k.get('frequency', '')} times",
        "keywords:forbidden_words": lambda k: f"avoid these words: {k.get('forbidden_words', [])}",
        "keywords:letter_frequency": lambda k: f"use the letter '{k.get('letter', '')}' {k.get('let_relation', '')} {k.get('let_frequency', '')} times",
        "language:response_language": lambda k: f"respond in {k.get('language', 'the specified language')}",
        "length_constraints:number_sentences": lambda k: f"use {k.get('relation', '')} {k.get('num_sentences', '')} sentences",
        "length_constraints:number_paragraphs": lambda k: f"use exactly {k.get('num_paragraphs', '')} paragraphs separated by ***",
        "length_constraints:number_words": lambda k: f"use {k.get('relation', '')} {k.get('num_words', '')} words",
        "length_constraints:nth_paragraph_first_word": lambda k: f"start paragraph {k.get('nth_paragraph', '')} with the word '{k.get('first_word', '')}'",
        "detectable_content:number_placeholders": lambda k: f"include at least {k.get('num_placeholders', '')} placeholders in [brackets]",
        "detectable_content:postscript": lambda k: f"include a postscript starting with {k.get('postscript_marker', 'P.S.')}",
        "detectable_format:number_bullet_lists": lambda k: f"use exactly {k.get('num_bullets', '')} bullet points",
        "detectable_format:constrained_response": lambda k: "end with 'My answer is yes/no/maybe.'",
        "detectable_format:number_highlighted_sections": lambda k: f"include at least {k.get('num_highlights', '')} highlighted sections with *asterisks*",
        "detectable_format:multiple_sections": lambda k: f"organize into {k.get('num_sections', '')} sections using '{k.get('section_spliter', '')}'",
        "detectable_format:json_format": lambda k: "format the response as valid JSON",
        "detectable_format:title": lambda k: "include a title wrapped in <<double angle brackets>>",
        "combination:two_responses": lambda k: "provide two different responses separated by ******",
        "combination:repeat_prompt": lambda k: f"start by repeating: '{k.get('prompt_to_repeat', '')[:50]}...'",
        "startend:end_checker": lambda k: f"end with the phrase '{k.get('end_phrase', '')}'",
        "change_case:capital_word_frequency": lambda k: f"use {k.get('capital_relation', '')} {k.get('capital_frequency', '')} fully capitalized words",
        "change_case:english_capital": lambda k: "write the entire response in CAPITAL LETTERS",
        "change_case:english_lowercase": lambda k: "write the entire response in lowercase",
        "punctuation:no_comma": lambda k: "do not use any commas",
        "startend:quotation": lambda k: "wrap the entire response in double quotes",
        "first_word:first_word_sent": lambda k: f"start every sentence with the word '{k.get('first_word', '')}'",
        "first_word:first_word_answer": lambda k: f"start your response with the word '{k.get('first_word', '')}'",
        "last_word:last_word_sent": lambda k: f"end every sentence with the word '{k.get('last_word', '')}'",
        "last_word:last_word_answer": lambda k: f"end your response with the word '{k.get('last_word', '')}'",
    }

    desc_fn = descriptions.get(instruction_id)
    if desc_fn:
        try:
            return desc_fn(kwargs)
        except Exception:
            pass

    # Fallback: return the instruction ID in readable form
    return instruction_id.replace(":", " - ").replace("_", " ")


def format_failed_constraints_feedback(
    failed_instruction_ids: list[str],
    failed_kwargs: list[dict]
) -> str:
    """Format feedback about which constraints were failed."""
    if not failed_instruction_ids:
        return ""

    failed_descriptions = []
    for inst_id, kwargs in zip(failed_instruction_ids, failed_kwargs):
        desc = describe_constraint(inst_id, kwargs)
        failed_descriptions.append(f"- {desc}")

    feedback = (
        "\n\n[IMPORTANT: In a previous attempt, you failed to satisfy these constraints. "
        "Please make sure to follow them this time:]\n"
        + "\n".join(failed_descriptions)
    )
    return feedback


def run_experiment(
    model_path: str,
    gpu_id: int,
    num_prompts: int = 10,
    num_completions: int = 8,
    curriculum_path: str = "/efs/rlvr-experiments/curricula/pass-at-64/ifeval_curriculum.txt",
    seed: int = 42,
):
    """Run the constraint feedback experiment."""
    random.seed(seed)

    # Load dataset and curriculum
    id_to_row = load_ifeval_dataset()
    hard_prompt_ids = load_curriculum_bottom_half(curriculum_path)

    # Sample prompts
    sample_ids = random.sample(hard_prompt_ids, min(num_prompts, len(hard_prompt_ids)))
    print(f"\nSelected {len(sample_ids)} prompts from hard portion of curriculum")

    # Get the actual prompts
    prompts_data = []
    for pid in sample_ids:
        if pid in id_to_row:
            prompts_data.append(id_to_row[pid])
        else:
            print(f"Warning: {pid} not found in dataset")

    if not prompts_data:
        print("No valid prompts found!")
        return

    print(f"Loaded {len(prompts_data)} prompts")

    # Initialize model
    print(f"\nLoading model on GPU {gpu_id}...")
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    llm = LLM(
        model=model_path,
        tensor_parallel_size=1,
        max_model_len=4096,
        gpu_memory_utilization=0.4,  # Lower to leave headroom
    )

    # Sampling params for diverse completions
    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.95,
        max_tokens=2048,
        n=num_completions,
    )

    verifier = IFMultiConstraintsVerifier()

    # Phase 1: Generate initial completions
    print(f"\n{'='*60}")
    print("PHASE 1: Initial Generation")
    print(f"{'='*60}")

    prompts = [p["prompt"] for p in prompts_data]
    outputs = llm.generate(prompts, sampling_params)

    results = []
    for prompt_data, output in zip(prompts_data, outputs):
        completions = [o.text for o in output.outputs]

        # Verify each completion with detailed results
        completion_results = []
        for comp in completions:
            per_constraint, inst_ids, kwargs_list = verifier.verify_detailed(
                comp, prompt_data["ground_truth"]
            )
            score = sum(per_constraint) / len(per_constraint) if per_constraint else 0
            completion_results.append({
                "completion": comp[:500] + "..." if len(comp) > 500 else comp,
                "per_constraint": per_constraint,
                "instruction_ids": inst_ids,
                "kwargs_list": kwargs_list,
                "score": score,
                "all_satisfied": all(per_constraint) if per_constraint else False,
            })

        # Compute summary statistics
        scores = [r["score"] for r in completion_results]
        any_all_satisfied = any(r["all_satisfied"] for r in completion_results)
        avg_score = sum(scores) / len(scores)

        results.append({
            "prompt_id": prompt_data["prompt_id"],
            "prompt": prompt_data["prompt"][:300] + "..." if len(prompt_data["prompt"]) > 300 else prompt_data["prompt"],
            "num_constraints": len(completion_results[0]["instruction_ids"]) if completion_results else 0,
            "completions": completion_results,
            "avg_score": avg_score,
            "any_all_satisfied": any_all_satisfied,
            "max_score": max(scores),
            "min_score": min(scores),
        })

        print(f"\n{prompt_data['prompt_id']}:")
        print(f"  Constraints: {results[-1]['num_constraints']}")
        print(f"  Avg score: {avg_score:.2%}, Max: {max(scores):.2%}, Min: {min(scores):.2%}")
        print(f"  Any fully satisfied: {any_all_satisfied}")

    # Phase 2: Find prompts with partial satisfaction (interesting cases)
    print(f"\n{'='*60}")
    print("PHASE 2: Identifying Candidates for Feedback")
    print(f"{'='*60}")

    # Interesting = has partial satisfaction but rarely/never full satisfaction
    # Score between 0.2 and 0.9 average, and not always succeeding
    candidates = [
        r for r in results
        if 0.1 < r["avg_score"] < 0.95 and not r["any_all_satisfied"]
    ]

    print(f"Found {len(candidates)} candidates with partial satisfaction")

    if not candidates:
        print("No suitable candidates found. Trying with relaxed criteria...")
        candidates = [
            r for r in results
            if r["avg_score"] < 1.0  # Any that don't always succeed
        ][:5]

    # Phase 3: Re-run with feedback
    print(f"\n{'='*60}")
    print("PHASE 3: Re-generation with Constraint Feedback")
    print(f"{'='*60}")

    feedback_results = []
    for candidate in candidates:
        # Find the most common failure pattern
        all_failures = []
        for comp_result in candidate["completions"]:
            failed_indices = [
                i for i, passed in enumerate(comp_result["per_constraint"]) if not passed
            ]
            if failed_indices:
                all_failures.append(tuple(failed_indices))

        if not all_failures:
            continue

        # Get the constraints that were failed most often
        from collections import Counter
        failure_counts = Counter(all_failures)
        most_common_failure = failure_counts.most_common(1)[0][0]

        # Get constraint details for feedback
        sample_result = candidate["completions"][0]
        failed_inst_ids = [sample_result["instruction_ids"][i] for i in most_common_failure]
        failed_kwargs = [sample_result["kwargs_list"][i] for i in most_common_failure]

        # Create augmented prompt with feedback
        original_prompt = id_to_row[candidate["prompt_id"]]["prompt"]
        feedback = format_failed_constraints_feedback(failed_inst_ids, failed_kwargs)
        augmented_prompt = original_prompt + feedback

        print(f"\n{candidate['prompt_id']}:")
        print(f"  Original avg score: {candidate['avg_score']:.2%}")
        print(f"  Failed constraints: {failed_inst_ids}")
        print(f"  Feedback added: {feedback[:200]}...")

        # Generate with feedback
        outputs_with_feedback = llm.generate([augmented_prompt], sampling_params)

        # Verify
        ground_truth = id_to_row[candidate["prompt_id"]]["ground_truth"]
        new_completions = [o.text for o in outputs_with_feedback[0].outputs]

        new_scores = []
        new_details = []
        for comp in new_completions:
            per_constraint, inst_ids, kwargs_list = verifier.verify_detailed(comp, ground_truth)
            score = sum(per_constraint) / len(per_constraint) if per_constraint else 0
            new_scores.append(score)
            new_details.append({
                "per_constraint": per_constraint,
                "score": score,
                "all_satisfied": all(per_constraint) if per_constraint else False,
            })

        new_avg = sum(new_scores) / len(new_scores)
        new_any_all = any(d["all_satisfied"] for d in new_details)

        print(f"  NEW avg score: {new_avg:.2%} (delta: {new_avg - candidate['avg_score']:+.2%})")
        print(f"  NEW any fully satisfied: {new_any_all}")

        feedback_results.append({
            "prompt_id": candidate["prompt_id"],
            "original_avg_score": candidate["avg_score"],
            "original_any_all_satisfied": candidate["any_all_satisfied"],
            "failed_constraints": failed_inst_ids,
            "feedback": feedback,
            "new_avg_score": new_avg,
            "new_any_all_satisfied": new_any_all,
            "improvement": new_avg - candidate["avg_score"],
            "new_details": new_details,
        })

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")

    if feedback_results:
        improvements = [r["improvement"] for r in feedback_results]
        avg_improvement = sum(improvements) / len(improvements)
        num_improved = sum(1 for i in improvements if i > 0)

        print(f"Tested {len(feedback_results)} prompts with feedback")
        print(f"Average improvement: {avg_improvement:+.2%}")
        print(f"Prompts that improved: {num_improved}/{len(feedback_results)}")

        # Count new full satisfactions
        new_full = sum(1 for r in feedback_results if r["new_any_all_satisfied"] and not r["original_any_all_satisfied"])
        print(f"Prompts that achieved full satisfaction with feedback: {new_full}")

        print("\nPer-prompt results:")
        for r in sorted(feedback_results, key=lambda x: x["improvement"], reverse=True):
            status = "✓" if r["improvement"] > 0 else "✗"
            full_status = " (FULL!)" if r["new_any_all_satisfied"] and not r["original_any_all_satisfied"] else ""
            print(f"  {status} {r['prompt_id']}: {r['original_avg_score']:.1%} -> {r['new_avg_score']:.1%} ({r['improvement']:+.1%}){full_status}")
    else:
        print("No feedback results to summarize")

    # Save results
    output_path = Path("/tmp/ifeval_feedback_experiment_results.json")
    with open(output_path, "w") as f:
        json.dump({
            "initial_results": results,
            "feedback_results": feedback_results,
        }, f, indent=2, default=str)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="IFEval constraint feedback experiment")
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B-Base", help="Model path")
    parser.add_argument("--gpu", type=int, default=6, help="GPU ID to use")
    parser.add_argument("--num-prompts", type=int, default=10, help="Number of prompts to test")
    parser.add_argument("--num-completions", type=int, default=8, help="Completions per prompt")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    run_experiment(
        model_path=args.model,
        gpu_id=args.gpu,
        num_prompts=args.num_prompts,
        num_completions=args.num_completions,
        seed=args.seed,
    )
