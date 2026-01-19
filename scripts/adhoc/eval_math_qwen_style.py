#!/usr/bin/env python3
"""
Evaluate MATH benchmark using Qwen's methodology.

Key differences from lm_eval's hendrycks_math:
1. Prompt: "Question: {problem}\nAnswer: " (not "Problem: X\nAnswer:")
2. Few-shot examples include full CoT with "Let's think step by step" + \boxed{}
3. Answer extraction looks for \boxed{} content
4. Uses sympy for symbolic equivalence checking
"""

import json
import re
import argparse
from pathlib import Path
from datasets import load_dataset
from vllm import LLM, SamplingParams
from tqdm import tqdm

# Qwen's 4-shot examples for MATH (from their examples.py)
MATH_FEWSHOT_EXAMPLES = [
    (
        "Kevin Kangaroo begins hopping on a number line at 0. He wants to get to 1, but he can hop only $\\frac{1}{3}$ of the distance. Each hop tires him out so that he continues to hop $\\frac{1}{3}$ of the remaining distance. How far has he hopped after five hops? Express your answer as a common fraction.",
        "Let's think step by step\nKevin hops $1/3$ of the remaining distance with every hop.\nHis first hop takes $1/3$ closer.\nFor his second hop, he has $2/3$ left to travel, so he hops forward $(2/3)(1/3)$.\nFor his third hop, he has $(2/3)^2$ left to travel, so he hops forward $(2/3)^2(1/3)$.\nIn general, Kevin hops forward $(2/3)^{k-1}(1/3)$ on his $k$th hop.\nWe want to find how far he has hopped after five hops.\nThis is a finite geometric series with first term $1/3$, common ratio $2/3$, and five terms.\nThus, Kevin has hopped $\\frac{\\frac{1}{3}\\left(1-\\left(\\frac{2}{3}\\right)^5\\right)}{1-\\frac{2}{3}} = \\boxed{\\frac{211}{243}}$.\nThe answer is \\frac{211}{243}",
    ),
    (
        "What is the area of the region defined by the equation $x^2+y^2 - 7 = 4y-14x+3$?",
        "Let's think step by step\nWe rewrite the equation as $x^2 + 14x + y^2 - 4y = 10$ and then complete the square,\nresulting in  $(x+7)^2-49 + (y-2)^2-4=10$,\nor $(x+7)^2+(y-2)^2=63$.\nThis is the equation of a circle with center $(-7, 2)$ and radius $\\sqrt{63},$\nso the area of this region is $\\pi r^2 = \\boxed{63\\pi}$.\nThe answer is 63\\pi",
    ),
    (
        "If $x^2+y^2=1$, what is the largest possible value of $|x|+|y|$?",
        "Let's think step by step\nIf $(x,y)$ lies on the circle,\nso does $(x,-y),$ $(-x,-y),$ and $(-x,-y),$ (which all give the same value of $|x| + |y|$),\nso we can assume that $x \\ge 0$ and $y \\ge 0.$\nThen $|x| + |y| = x + y.$  Squaring, we get\n\\[(x + y)^2 = x^2 + 2xy + y^2 = 1 + 2xy.\\]\nNote that $(x - y)^2 \\ge 0.$\nExpanding, we get $x^2 - 2xy + y^2 \\ge 0,$ so $2xy \\le x^2 + y^2 = 1.$\nHence,\\[1 + 2xy \\le 2,\\]which means $x + y \\le \\sqrt{2}.$\nEquality occurs when $x = y = \\frac{1}{\\sqrt{2}},$\nso the maximum value of $|x| + |y|$ is $\\boxed{\\sqrt{2}}.$\nThe answer is \\sqrt{2}",
    ),
    (
        "If $f(x)=\\frac{ax+b}{cx+d}, abcd\\not=0$ and $f(f(x))=x$ for all $x$ in the domain of $f$, what is the value of $a+d$?",
        "Let's think step by step\nThe condition $f(f(x))$ means that $f$ is the inverse of itself,\nso its graph is symmetrical about the line $y = x$.\nWith a rational function of this form, we will have two asymptotes:\na vertical one at $x=-d/c$ if $cx+d$ does not divide $ax+b$,\nand a horizontal one at $y=a/c$,\nif we take the limit of $f(x)$ as $x$ goes to $\\pm\\infty$.\nIn order for $f$ to be its own inverse, the intersection of the asymptotes must lie on the line $y=x$\nso that it and its asymptotes reflect onto themselves.\nThis means that $-d/c=a/c$,\nand therefore $-d=a$ and $a+d=\\boxed{0}$.\nThe answer is 0",
    ),
]


def build_prompt(problem: str, num_shots: int = 4) -> str:
    """Build Qwen-style CoT prompt for MATH."""
    prompt_parts = []

    # Add few-shot examples
    for q, a in MATH_FEWSHOT_EXAMPLES[:num_shots]:
        prompt_parts.append(f"Question: {q}\nAnswer: {a}")

    # Add the test question
    prompt_parts.append(f"Question: {problem}\nAnswer:")

    # Join with triple newlines (Qwen's splitter for "cot" template)
    return "\n\n\n".join(prompt_parts)


def extract_boxed(text: str) -> str | None:
    """Extract content from \boxed{...}, handling nested braces."""
    idx = text.rfind("\\boxed")
    if idx >= 0:
        i = idx
        while i < len(text) and text[i] != '{':
            i += 1
        if i < len(text):
            # Find matching closing brace
            brace_count = 0
            start = i
            while i < len(text):
                if text[i] == '{':
                    brace_count += 1
                elif text[i] == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        return text[start+1:i]
                i += 1

    # Fallback: try "The answer is X" pattern (Qwen's examples use this)
    import re
    match = re.search(r"[Tt]he answer is[:\s]*\$?([^\$\n]+)\$?", text)
    if match:
        answer = match.group(1).strip()
        # Clean up common suffixes
        answer = re.sub(r"\.?\s*$", "", answer)
        return answer

    return None


def normalize_answer(answer: str) -> str:
    """Normalize answer for comparison."""
    if answer is None:
        return ""
    # Basic normalization
    answer = answer.strip()
    answer = answer.replace(" ", "")
    answer = answer.replace("\\left", "")
    answer = answer.replace("\\right", "")
    answer = answer.replace("\\!", "")
    answer = answer.replace("\\,", "")
    answer = answer.replace("\\;", "")
    answer = answer.replace("\\quad", "")
    answer = answer.replace("\\qquad", "")
    answer = answer.replace("dfrac", "frac")
    answer = answer.replace("tfrac", "frac")
    return answer


def is_equiv(pred: str, gold: str) -> bool:
    """Check if two answers are equivalent."""
    pred_norm = normalize_answer(pred)
    gold_norm = normalize_answer(gold)

    # Direct string match
    if pred_norm == gold_norm:
        return True

    # Try sympy comparison using latex2sympy2_extended (same lib as math_verify)
    try:
        from latex2sympy2_extended import latex2sympy
        from sympy import simplify

        pred_expr = latex2sympy(pred_norm)
        gold_expr = latex2sympy(gold_norm)

        diff = simplify(pred_expr - gold_expr)
        if diff == 0:
            return True
    except:
        pass

    return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="/efs/rlvr-experiments/assets/hf/Qwen3-1.7B-Base")
    parser.add_argument("--num-shots", type=int, default=4)
    parser.add_argument("--max-problems", type=int, default=0, help="0 for all")
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--tensor-parallel-size", type=int, default=4)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Set all random seeds for reproducibility
    import random
    import numpy as np
    import torch
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    print(f"Loading model: {args.model}")
    print(f"Random seed: {args.seed}")
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        dtype="bfloat16",
        gpu_memory_utilization=0.8,
        max_model_len=4096,
        seed=args.seed,  # vLLM seed
    )

    sampling_params = SamplingParams(
        temperature=0,
        max_tokens=args.max_tokens,
        stop=["Question:", "\n\nQuestion"],
    )

    # Load MATH test set (all subjects)
    print("Loading MATH dataset...")
    subjects = ["algebra", "counting_and_probability", "geometry",
                "intermediate_algebra", "number_theory", "prealgebra", "precalculus"]
    all_data = []
    for subject in subjects:
        ds = load_dataset("EleutherAI/hendrycks_math", subject, split="test")
        all_data.extend(list(ds))

    # Convert to a simple list format
    dataset = all_data
    print(f"Loaded {len(dataset)} problems across {len(subjects)} subjects")

    if args.max_problems > 0:
        dataset = dataset[:args.max_problems]

    print(f"Evaluating {len(dataset)} problems with {args.num_shots}-shot CoT...")

    # Build prompts
    prompts = [build_prompt(ex["problem"], args.num_shots) for ex in dataset]

    # Generate
    print("Generating responses...")
    outputs = llm.generate(prompts, sampling_params)

    # Evaluate
    correct = 0
    results = []

    for i, (ex, output) in enumerate(tqdm(zip(dataset, outputs), total=len(dataset), desc="Evaluating")):
        response = output.outputs[0].text

        # Extract gold answer from solution
        gold = extract_boxed(ex["solution"])
        pred = extract_boxed(response)

        match = is_equiv(pred, gold) if pred and gold else False
        if match:
            correct += 1

        results.append({
            "idx": i,
            "problem": ex["problem"],
            "level": ex["level"],
            "type": ex["type"],
            "gold": gold,
            "pred": pred,
            "response": response[:500],  # Truncate for storage
            "correct": match,
        })

    accuracy = correct / len(dataset) * 100
    print(f"\n{'='*50}")
    print(f"MATH Accuracy ({args.num_shots}-shot CoT): {accuracy:.2f}%")
    print(f"Correct: {correct}/{len(dataset)}")
    print(f"{'='*50}")

    # Breakdown by level
    level_stats = {}
    for r in results:
        level = r["level"]
        if level not in level_stats:
            level_stats[level] = {"correct": 0, "total": 0}
        level_stats[level]["total"] += 1
        if r["correct"]:
            level_stats[level]["correct"] += 1

    print("\nBreakdown by level:")
    for level in sorted(level_stats.keys()):
        stats = level_stats[level]
        pct = stats["correct"] / stats["total"] * 100
        print(f"  {level}: {stats['correct']}/{stats['total']} ({pct:.1f}%)")

    # Save results
    if args.output:
        output_path = Path(args.output)
    else:
        model_name = Path(args.model).name
        output_path = Path(f"/tmp/math_qwen_style_{model_name}.json")

    with open(output_path, "w") as f:
        json.dump({
            "model": args.model,
            "num_shots": args.num_shots,
            "seed": args.seed,
            "accuracy": accuracy,
            "correct": correct,
            "total": len(dataset),
            "level_stats": level_stats,
            "results": results,
        }, f, indent=2)

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
