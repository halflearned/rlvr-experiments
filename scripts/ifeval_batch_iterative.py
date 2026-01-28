#!/usr/bin/env python3
"""
Batch run iterative refinement on multiple IFEval prompts.
Skip any that succeed on first try.
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
    kwargs = kwargs or {}
    descriptions = {
        # Format constraints
        "detectable_format:sentence_hyphens": lambda k: "All sentences must be connected using hyphens with no spaces between them (e.g., 'First sentence.-Second sentence.-Third sentence.')",
        "detectable_format:number_highlighted_sections": lambda k: f"Include at least {k.get('num_highlights', 'N')} highlighted sections using *asterisks* (e.g., *like this*)",
        "detectable_format:number_bullet_lists": lambda k: f"Use exactly {k.get('num_bullets', '')} bullet points (using * or - at the start of lines)",
        "detectable_format:json_format": lambda k: "Format the entire response as valid JSON",
        "detectable_format:title": lambda k: "Include a title wrapped in double angle brackets like <<My Title>>",
        "detectable_format:constrained_response": lambda k: "End with exactly one of: 'My answer is yes.' or 'My answer is no.' or 'My answer is maybe.'",
        "detectable_format:multiple_sections": lambda k: f"Organize into at least {k.get('num_sections', 'N')} sections using '{k.get('section_spliter', 'Section')}' followed by numbers",
        "detectable_format:square_brackets": lambda k: "Wrap EVERY word in square brackets like [this] [is] [an] [example]",
        "detectable_format:bigram_wrapping": lambda k: "Wrap every pair of words in <<double angle brackets>>",

        # Keywords constraints
        "keywords:existence": lambda k: f"Include these keywords: {k.get('keywords', [])}",
        "keywords:frequency": lambda k: f"Use the word '{k.get('keyword', '')}' {k.get('relation', '')} {k.get('frequency', '')} times",
        "keywords:forbidden_words": lambda k: f"Do NOT use these words: {k.get('forbidden_words', [])}",
        "keywords:word_once": lambda k: f"Use the word '{k.get('keyword', '')}' exactly once",
        "keywords:letter_frequency": lambda k: f"Use the letter '{k.get('letter', '')}' {k.get('let_relation', '')} {k.get('let_frequency', '')} times",
        "keywords:keyword_specific_position": lambda k: f"The word '{k.get('keyword', '')}' must appear as word #{k.get('m', '')} in sentence #{k.get('n', '')}",
        "keywords:word_count_different_numbers": lambda k: f"Use the word '{k.get('keyword', '')}' {k.get('relation', '')} {k.get('frequency', '')} times",
        "keywords:exclude_word_harder": lambda k: f"Do not include the word '{k.get('keyword', '')}' surrounded by spaces",
        "keywords:no_adjacent_consecutive": lambda k: "No two adjacent words should start with consecutive letters of the alphabet",
        "keywords:palindrome": lambda k: "Include at least one palindrome word (reads same forwards and backwards)",
        "keywords:start_end": lambda k: "The first and last words of your response must be the same",

        # Length constraints
        "length_constraints:number_sentences": lambda k: f"Use {k.get('relation', '')} {k.get('num_sentences', '')} sentences",
        "length_constraints:number_words": lambda k: f"Use {k.get('relation', '')} {k.get('num_words', '')} words",
        "length_constraints:number_paragraphs": lambda k: f"Use exactly {k.get('num_paragraphs', '')} paragraphs separated by ***",
        "length_constraints:nth_paragraph_first_word": lambda k: f"In a response with {k.get('num_paragraphs', '')} paragraphs, paragraph #{k.get('nth_paragraph', '')} must start with '{k.get('first_word', '')}'",

        # Letter/character counting
        "letters:letter_counting": lambda k: f"Use {k.get('relation', '')} {k.get('N', '')} letters total in your response",
        "letters:letter_counting2": lambda k: f"Use the letter '{k.get('letter', '')}' {k.get('let_relation', '')} {k.get('let_frequency', '')} times",

        # Case constraints
        "change_case:english_capital": lambda k: "Write the ENTIRE response in CAPITAL LETTERS",
        "change_case:english_lowercase": lambda k: "Write the entire response in lowercase letters only",
        "change_case:capital_word_frequency": lambda k: f"Use {k.get('capital_relation', '')} {k.get('capital_frequency', '')} fully CAPITALIZED words",

        # Punctuation constraints
        "punctuation:no_comma": lambda k: "Do not use any commas (,) in your response",
        "punctuation:punctuation_dot": lambda k: "Do not use any periods (.) in your response",
        "punctuation:punctuation_exclamation": lambda k: "Do not use any exclamation marks (!) in your response",

        # Start/end constraints
        "startend:end_checker": lambda k: f"End your response with the exact phrase: '{k.get('end_phrase', '')}'",
        "startend:quotation": lambda k: 'Wrap your entire response in double quotes (start with " and end with ")',
        "first_word:first_word_sent": lambda k: f"Start EVERY sentence with the word '{k.get('first_word', '')}'",
        "first_word:first_word_answer": lambda k: f"Start your response with the word '{k.get('first_word', '')}'",
        "last_word:last_word_sent": lambda k: f"End EVERY sentence with the word '{k.get('last_word', '')}'",
        "last_word:last_word_answer": lambda k: f"End your response with the word '{k.get('last_word', '')}'",

        # Detectable content
        "detectable_content:number_placeholders": lambda k: f"Include at least {k.get('num_placeholders', '')} placeholders in [square brackets] like [NAME] or [DATE]",
        "detectable_content:postscript": lambda k: f"Include a postscript starting with {k.get('postscript_marker', 'P.S.')}",

        # Combination/structure constraints
        "combination:two_responses": lambda k: "Provide exactly two different responses separated by ******",
        "combination:repeat_prompt": lambda k: f"Start by repeating this text: '{str(k.get('prompt_to_repeat', ''))[:100]}...'",

        # Copy constraints
        "copy:repeat_phrase": lambda k: f"Repeat the phrase '{k.get('phrase', '')}' exactly {k.get('small_n', '')} times",
        "copy:copy": lambda k: f"Your entire response should be exactly: '{str(k.get('prompt_to_repeat', ''))[:100]}'",
        "copy:copying_simple": lambda k: f"Your entire response should be exactly: '{str(k.get('prompt_to_repeat', ''))[:100]}'",
        "copy:copying_multiple": lambda k: f"Repeat the text exactly {k.get('N', '')} times, separated by ******",
        "new:copy_span_idx": lambda k: f"Your response should be exactly characters {k.get('n_start', '')}-{k.get('n_end', '')} from the prompt",

        # Paragraph constraints
        "paragraphs:paragraphs": lambda k: "Write exactly 2 paragraphs separated by ***",
        "paragraphs:paragraphs2": lambda k: "Write exactly 2 paragraphs separated by a blank line",

        # Count constraints
        "count:lowercase_counting": lambda k: f"Use at most {k.get('N', '')} lowercase words",
        "count:count_unique": lambda k: "Every word in your response must be unique (no repeated words)",
        "count:counting_composition": lambda k: f"Write exactly 3 paragraphs (separated by ***), each with {k.get('n_sent', '')} sentences, each sentence with {k.get('n_words', '')} words",
        "count:count_increment_word": lambda k: f"Use '{k.get('keyword1', '')}' exactly once and '{k.get('keyword2', '')}' exactly twice",

        # Language
        "language:response_language": lambda k: f"Write your entire response in {k.get('language', 'the specified language')}",
    }
    desc_fn = descriptions.get(instruction_id)
    if desc_fn:
        try:
            return desc_fn(kwargs)
        except Exception:
            pass
    return instruction_id.replace(":", ": ").replace("_", " ")


def run_single_prompt(llm, verifier, sampling_params, prompt_id, ds, user_content, ground_truth, max_iterations=3):
    """Run iterative refinement on a single prompt. Returns results dict."""

    # Parse constraints
    _, inst_ids, kwargs_list = verifier.verify_detailed("test", ground_truth)
    num_constraints = len(inst_ids)

    results = {
        "prompt_id": prompt_id,
        "num_constraints": num_constraints,
        "constraint_types": inst_ids,
        "iterations": [],
        "final_best_satisfied": None,
        "success": False,
    }

    # Track which constraints have been satisfied
    best_constraints_satisfied = [False] * num_constraints
    best_completion_so_far = None
    current_prompt = user_content

    for iteration in range(max_iterations + 1):
        outputs = llm.generate([current_prompt], sampling_params)
        completions = [o.text for o in outputs[0].outputs]

        iteration_data = {
            "iteration": iteration,
            "completions": [],
            "any_full_success": False,
            "best_this_iter": None,
        }

        new_best_found = False

        for comp in completions:
            per_constraint, _, _ = verifier.verify_detailed(comp, ground_truth)
            score = sum(per_constraint) / len(per_constraint) if per_constraint else 0
            all_passed = all(per_constraint) if per_constraint else False

            # Check improvement
            maintains_previous = all(
                per_constraint[j] or not best_constraints_satisfied[j]
                for j in range(len(per_constraint))
            )
            adds_new = any(
                per_constraint[j] and not best_constraints_satisfied[j]
                for j in range(len(per_constraint))
            )
            is_improvement = maintains_previous and adds_new

            iteration_data["completions"].append({
                "per_constraint": per_constraint,
                "score": score,
                "all_passed": all_passed,
                "is_improvement": is_improvement,
            })

            if all_passed:
                iteration_data["any_full_success"] = True
                results["success"] = True

            if is_improvement and not new_best_found:
                new_best_found = True
                best_completion_so_far = comp
                best_constraints_satisfied = [
                    best_constraints_satisfied[j] or per_constraint[j]
                    for j in range(len(per_constraint))
                ]

        iteration_data["best_this_iter"] = list(best_constraints_satisfied)
        results["iterations"].append(iteration_data)

        # Check termination
        if iteration_data["any_full_success"]:
            break
        if iteration == max_iterations:
            break

        # If no improvement, pick any that maintains
        if not new_best_found:
            for comp_data, comp in zip(iteration_data["completions"], completions):
                maintains = all(
                    comp_data["per_constraint"][j] or not best_constraints_satisfied[j]
                    for j in range(len(comp_data["per_constraint"]))
                )
                if maintains:
                    best_completion_so_far = comp
                    break
            else:
                best_completion_so_far = completions[0]

        # Build feedback
        failed_indices = [i for i, sat in enumerate(best_constraints_satisfied) if not sat]
        failed_constraints = [describe_constraint(inst_ids[i], kwargs_list[i]) for i in failed_indices]

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

    results["final_best_satisfied"] = list(best_constraints_satisfied)
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B-Base")
    parser.add_argument("--gpu", type=int, default=6)
    parser.add_argument("--max-iterations", type=int, default=3)
    parser.add_argument("--num-completions", type=int, default=4)
    parser.add_argument("--prompt-ids", nargs="+", default=[
        "ifeval_4144", "ifeval_2553", "ifeval_13748", "ifeval_9725", "ifeval_9740",
        "ifeval_8727", "ifeval_72", "ifeval_7364", "ifeval_10723", "ifeval_13665",
    ])
    args = parser.parse_args()

    # Load dataset
    print("Loading dataset...")
    ds = load_dataset("allenai/IF_multi_constraints_upto5", split="train")

    # Build index
    id_to_data = {}
    for i, row in enumerate(ds):
        pid = f"ifeval_{i}"
        user_content = ""
        for msg in row["messages"]:
            if msg["role"] == "user":
                user_content = msg["content"]
                break
        id_to_data[pid] = {
            "user_content": user_content,
            "ground_truth": row["ground_truth"],
        }

    # Initialize model
    print(f"\nLoading model on GPU {args.gpu}...")
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    llm = LLM(
        model=args.model,
        tensor_parallel_size=1,
        max_model_len=4096,
        gpu_memory_utilization=0.4,
    )

    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.95,
        max_tokens=2048,
        n=args.num_completions,
    )

    verifier = IFMultiConstraintsVerifier()

    all_results = []

    for prompt_id in args.prompt_ids:
        if prompt_id not in id_to_data:
            print(f"\nSkipping {prompt_id} - not found")
            continue

        data = id_to_data[prompt_id]
        print(f"\n{'='*60}")
        print(f"Processing {prompt_id}")
        print(f"{'='*60}")

        result = run_single_prompt(
            llm, verifier, sampling_params,
            prompt_id, ds,
            data["user_content"],
            data["ground_truth"],
            max_iterations=args.max_iterations,
        )

        # Check if first iteration had full success - skip these
        if result["iterations"][0]["any_full_success"]:
            print(f"  -> Skipped: succeeded on first try")
            continue

        all_results.append(result)

        # Print summary
        init_best = sum(result["iterations"][0]["best_this_iter"]) / result["num_constraints"]
        final_best = sum(result["final_best_satisfied"]) / result["num_constraints"]
        print(f"  Constraints: {result['num_constraints']} ({result['constraint_types']})")
        print(f"  Initial best: {init_best:.0%}")
        print(f"  Final best: {final_best:.0%}")
        print(f"  Success: {result['success']}")

        # Show progression
        for it in result["iterations"]:
            sat = sum(it["best_this_iter"]) if it["best_this_iter"] else 0
            print(f"    Iter {it['iteration']}: {sat}/{result['num_constraints']} satisfied")

    # Summary
    print(f"\n{'='*60}")
    print("OVERALL SUMMARY")
    print(f"{'='*60}")

    if not all_results:
        print("No prompts to analyze (all succeeded on first try)")
        return

    successes = sum(1 for r in all_results if r["success"])
    print(f"Prompts tested: {len(all_results)}")
    print(f"Full successes: {successes}/{len(all_results)}")

    # Calculate improvement
    improvements = []
    for r in all_results:
        init = sum(r["iterations"][0]["best_this_iter"]) / r["num_constraints"]
        final = sum(r["final_best_satisfied"]) / r["num_constraints"]
        improvements.append(final - init)

    avg_improvement = sum(improvements) / len(improvements) if improvements else 0
    improved_count = sum(1 for i in improvements if i > 0)

    print(f"Average improvement: {avg_improvement:+.1%}")
    print(f"Prompts that improved: {improved_count}/{len(all_results)}")

    print("\nPer-prompt results:")
    for r, imp in zip(all_results, improvements):
        init = sum(r["iterations"][0]["best_this_iter"]) / r["num_constraints"]
        final = sum(r["final_best_satisfied"]) / r["num_constraints"]
        status = "✓" if r["success"] else ("↑" if imp > 0 else "→")
        print(f"  {status} {r['prompt_id']}: {init:.0%} -> {final:.0%} ({imp:+.0%}) [{r['constraint_types']}]")

    # Save
    output_path = Path("/tmp/ifeval_batch_results.json")
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
