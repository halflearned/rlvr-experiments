"""
Ablation: Which GSM8k teacher prompt format helps pass@1 for a base model?

Tests multiple template variants against a plain baseline.

Usage:
  CUDA_VISIBLE_DEVICES=5 python scripts/opsd_teacher_ablation.py --n 500 --gpu 0
"""
import argparse
import re
import random
from datasets import load_dataset
from vllm import LLM, SamplingParams


def extract_gsm8k_answer(text: str) -> str | None:
    """Extract the final number from a GSM8k completion."""
    m = re.search(r"####\s*(-?[\d,]+(?:\.\d+)?)", text)
    if m:
        return m.group(1).replace(",", "")
    numbers = re.findall(r"-?[\d,]+(?:\.\d+)?", text)
    if numbers:
        return numbers[-1].replace(",", "")
    return None


def normalize_answer(ans: str) -> str:
    ans = ans.strip().replace(",", "")
    try:
        return str(int(float(ans)))
    except (ValueError, OverflowError):
        return ans


# ---------------------------------------------------------------------------
# Template variants
# ---------------------------------------------------------------------------

def tpl_plain(question: str, answer: str) -> str:
    """Baseline: plain Q/A format."""
    return f"Q: {question}\nA:"


def tpl_v0_instruct(question: str, answer: str) -> str:
    """V0: Original (instruction-style, known to hurt)."""
    return (
        f"Q: {question}\n"
        f"The answer to this question is {answer}. "
        f"With that in mind, please provide your reasoning.\nA:"
    )


def tpl_v1_repeat(question: str, answer: str) -> str:
    """V1: Show solved example then ask again (few-shot self-repeat)."""
    return (
        f"Q: {question}\n"
        f"A: The answer is {answer}.\n\n"
        f"Q: {question}\nA:"
    )


def tpl_v2_hint_inline(question: str, answer: str) -> str:
    """V2: Inline hint before A: — natural continuation style."""
    return (
        f"Q: {question}\n"
        f"(Answer: {answer})\nA:"
    )


def tpl_v3_worked(question: str, answer: str) -> str:
    """V3: 'Let's verify' style — base models see this in pretraining."""
    return (
        f"Q: {question}\n"
        f"A: Let's work through this step by step. The final answer is {answer}.\n"
        f"Step 1:"
    )


def tpl_v4_answer_first(question: str, answer: str) -> str:
    """V4: Answer first, then reasoning — reverse chain."""
    return (
        f"Q: {question}\n"
        f"A: {answer}\n\n"
        f"Solution:\nQ: {question}\nA:"
    )


TEMPLATES = {
    "plain": tpl_plain,
    "v0_instruct": tpl_v0_instruct,
    "v1_repeat": tpl_v1_repeat,
    "v2_hint_inline": tpl_v2_hint_inline,
    "v3_worked": tpl_v3_worked,
    "v4_answer_first": tpl_v4_answer_first,
}


def evaluate_template(name, tpl_fn, samples, gt_answers, llm, sampling_params):
    """Generate completions and compute pass@1 for a template."""
    prompts = [tpl_fn(s["question"], a) for s, a in zip(samples, gt_answers)]

    # Show first example
    print(f"\n{'='*60}")
    print(f"Template: {name}")
    print(f"{'='*60}")
    print(f"Example:\n{prompts[0]}\n")

    outputs = llm.generate(prompts, sampling_params)

    correct = 0
    results = []
    for i, (output, gt) in enumerate(zip(outputs, gt_answers)):
        completion = output.outputs[0].text
        predicted = extract_gsm8k_answer(completion)
        is_correct = predicted is not None and normalize_answer(predicted) == normalize_answer(gt)
        correct += int(is_correct)
        results.append({
            "idx": i, "gt": gt, "predicted": predicted,
            "correct": is_correct, "completion": completion[:300],
        })

    n = len(samples)
    acc = correct / n
    print(f"Result: {correct}/{n} = {acc:.4f}")
    return acc, results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=500)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--model", type=str, default="/efs/rlvr-experiments/assets/hf/Qwen3-1.7B-Base")
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    # Load GSM8k
    ds = load_dataset("openai/gsm8k", "main", split="test")
    indices = list(range(len(ds)))
    random.shuffle(indices)
    indices = indices[:args.n]
    samples = [ds[i] for i in indices]

    gt_answers = []
    for s in samples:
        m = re.search(r"####\s*(-?[\d,]+(?:\.\d+)?)", s["answer"])
        gt_answers.append(m.group(1).replace(",", "") if m else "")

    # Load model
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    llm = LLM(
        model=args.model,
        max_model_len=2048,  # need more room for repeated-question templates
        gpu_memory_utilization=0.50,
        dtype="bfloat16",
    )

    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_tokens=args.max_tokens,
        stop=["[Question]", "Question:", "Q:", "\n\n\n", "\n\n"],
    )

    # Run all templates
    results_summary = {}
    for name, tpl_fn in TEMPLATES.items():
        acc, results = evaluate_template(name, tpl_fn, samples, gt_answers, llm, sampling_params)
        results_summary[name] = acc

    # Final summary
    print(f"\n{'='*60}")
    print(f"SUMMARY (n={args.n}, temp={args.temperature}, top_k={args.top_k})")
    print(f"{'='*60}")
    plain_acc = results_summary["plain"]
    for name, acc in sorted(results_summary.items(), key=lambda x: -x[1]):
        delta = acc - plain_acc
        marker = " <-- baseline" if name == "plain" else ""
        print(f"  {name:20s}  {acc:.4f}  ({delta:+.4f}){marker}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
