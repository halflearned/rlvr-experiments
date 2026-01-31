#!/usr/bin/env python3
"""
Pass@k evaluation on test sets (Google IFEval, IFBench_test, GSM8k test, MATH test).

Uses the same dataset loaders as eval_checkpoint.py but generates n completions
per prompt with temperature sampling for pass@k computation.

Usage:
    python scripts/eval_pass_at_k_test.py \
        --benchmark ifeval \
        --model-path results/run/checkpoints/step200 \
        --output-dir results/run/evals/ifeval/pass-at-k \
        --n 128 --gpu 0
"""

import argparse
import os
import subprocess
import sys

def find_free_gpu():
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True
        )
        gpus = []
        for line in result.stdout.strip().split("\n"):
            idx, mem = line.split(",")
            gpus.append((int(idx.strip()), int(mem.strip())))
        gpus.sort(key=lambda x: x[1])
        if gpus and gpus[0][1] < 1000:
            return gpus[0][0]
        return None
    except Exception:
        return None

# Parse GPU early
parser_early = argparse.ArgumentParser(add_help=False)
parser_early.add_argument("--gpu", type=int, default=None)
args_early, _ = parser_early.parse_known_args()

if args_early.gpu is None:
    free_gpu = find_free_gpu()
    if free_gpu is not None:
        args_early.gpu = free_gpu
    else:
        args_early.gpu = 0

os.environ["CUDA_VISIBLE_DEVICES"] = str(args_early.gpu)
print(f"[eval] Using GPU {args_early.gpu}")

import json
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Import dataset loaders from eval_checkpoint.py
# We redefine them here to avoid import issues

def load_gsm8k_test():
    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k", "main", split="test")
    rows = []
    for i, item in enumerate(ds):
        prompt = f"Q: {item['question'].strip()}\nA:"
        answer = item["answer"].split("####")[-1].strip()
        rows.append({"id": f"gsm8k_{i}", "prompt": prompt, "gold_answer": answer})
    return rows

def load_math_test():
    from datasets import load_dataset
    subjects = ["algebra", "counting_and_probability", "geometry",
                "intermediate_algebra", "number_theory", "prealgebra", "precalculus"]
    rows = []
    for subject in subjects:
        ds = load_dataset("EleutherAI/hendrycks_math", subject, split="test")
        for i, item in enumerate(ds):
            prompt = f"Problem:\n{item['problem'].strip()}\n\nSolution:"
            rows.append({
                "id": f"math_{subject}_{i}", "prompt": prompt,
                "gold_answer": item["solution"], "level": item.get("level", ""),
                "type": subject,
            })
    return rows

def load_ifeval():
    from datasets import load_dataset
    ds = load_dataset("google/IFEval", split="train")
    rows = []
    for i, item in enumerate(ds):
        rows.append({
            "id": f"ifeval_{i}", "prompt": item["prompt"],
            "instruction_id_list": item.get("instruction_id_list", []),
            "kwargs": item.get("kwargs", []),
        })
    return rows

def load_ifbench():
    from datasets import load_dataset
    ds = load_dataset("allenai/IFBench_test", split="train")
    rows = []
    for i, item in enumerate(ds):
        rows.append({
            "id": f"ifbench_{i}", "prompt": item["prompt"],
            "instruction_id_list": item.get("instruction_id_list", []),
            "kwargs": item.get("kwargs", []),
        })
    return rows


BENCHMARK_CONFIGS = {
    "gsm8k": {"max_tokens": 1024, "max_model_len": 2048,
              "stop_sequences": ["[Question]", "Question:", "Q:", "\n\n\n"]},
    "math": {"max_tokens": 1024, "max_model_len": 2048,
             "stop_sequences": ["Problem:", "\n\n\n"]},
    "ifeval": {"max_tokens": 2048, "max_model_len": 4096, "stop_sequences": None},
    "ifbench": {"max_tokens": 2048, "max_model_len": 4096, "stop_sequences": None},
}

LOADERS = {
    "gsm8k": load_gsm8k_test,
    "math": load_math_test,
    "ifeval": load_ifeval,
    "ifbench": load_ifbench,
}


def verify_gsm8k_multi(all_completions, rows):
    from rlvr_experiments.verifiers.math import MathVerifier
    verifier = MathVerifier(timeout=5.0, max_workers=8)
    results = []
    for completions, row in zip(all_completions, rows):
        scores = [1.0 if verifier.verify(c, row["gold_answer"]) > 0 else 0.0 for c in completions]
        results.append({"scores": scores, "num_correct": sum(1 for s in scores if s > 0),
                        "num_completions": len(scores), "pass_rate": sum(scores) / len(scores)})
    return results


def verify_math_multi(all_completions, rows):
    from rlvr_experiments.verifiers.math import MathVerifier
    verifier = MathVerifier(timeout=5.0, max_workers=8)
    results = []
    for completions, row in zip(all_completions, rows):
        scores = [1.0 if verifier.verify(c, row["gold_answer"]) > 0 else 0.0 for c in completions]
        results.append({"scores": scores, "num_correct": sum(1 for s in scores if s > 0),
                        "num_completions": len(scores), "pass_rate": sum(scores) / len(scores)})
    return results


def verify_ifeval_multi(all_completions, rows):
    from rlvr_experiments.verifiers.if_multi_constraints import (
        INSTRUCTION_FUNCTIONS, _remove_thinking_section,
    )
    results = []
    for completions, row in zip(all_completions, rows):
        instruction_id_list = row.get("instruction_id_list", [])
        kwargs_list = row.get("kwargs", [])
        scores = []
        for completion in completions:
            answer = _remove_thinking_section(completion)
            if not answer or not instruction_id_list:
                scores.append(0.0)
                continue
            pass_count = 0
            for j, instruction_id in enumerate(instruction_id_list):
                kwargs = kwargs_list[j] if j < len(kwargs_list) else {}
                if kwargs is None:
                    kwargs = {}
                func = INSTRUCTION_FUNCTIONS.get(instruction_id)
                if func is None:
                    continue
                kwargs = {k: v for k, v in kwargs.items() if v is not None}
                try:
                    ok = func(answer, **kwargs) if kwargs else func(answer)
                except Exception:
                    ok = False
                if ok:
                    pass_count += 1
            total = len(instruction_id_list)
            scores.append(1.0 if pass_count == total else 0.0)
        results.append({"scores": scores, "num_correct": sum(1 for s in scores if s > 0),
                        "num_completions": len(scores), "pass_rate": sum(scores) / len(scores)})
    return results


def verify_ifbench_multi(all_completions, rows):
    from rlvr_experiments.verifiers.ifbench import IFBenchVerifier
    verifier = IFBenchVerifier()
    results = []
    for completions, row in zip(all_completions, rows):
        instruction_id_list = row.get("instruction_id_list", [])
        kwargs_list = row.get("kwargs", [])
        scores = []
        for completion in completions:
            result = verifier.verify(completion, instruction_id_list, kwargs_list)
            scores.append(1.0 if result["all_pass"] else 0.0)
        results.append({"scores": scores, "num_correct": sum(1 for s in scores if s > 0),
                        "num_completions": len(scores), "pass_rate": sum(scores) / len(scores)})
    return results


MULTI_VERIFIERS = {
    "gsm8k": verify_gsm8k_multi,
    "math": verify_math_multi,
    "ifeval": verify_ifeval_multi,
    "ifbench": verify_ifbench_multi,
}


def main():
    parser = argparse.ArgumentParser(description="Pass@k evaluation on test sets")
    parser.add_argument("--benchmark", type=str, required=True,
                        choices=["gsm8k", "math", "ifeval", "ifbench"])
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--n", type=int, default=128, help="Number of completions per prompt")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Number of prompts to process at once")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    args = parser.parse_args()

    config = BENCHMARK_CONFIGS[args.benchmark]
    max_tokens = config["max_tokens"]
    max_model_len = config["max_model_len"]
    stop_sequences = config["stop_sequences"]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset
    print(f"[eval] Loading {args.benchmark} test set...")
    loader = LOADERS[args.benchmark]
    rows = loader()
    print(f"[eval] Loaded {len(rows)} prompts")

    # Initialize vLLM
    from vllm import LLM, SamplingParams

    print(f"[eval] Loading model from {args.model_path}...")
    llm = LLM(
        model=args.model_path,
        tensor_parallel_size=1,
        dtype="bfloat16",
        gpu_memory_utilization=0.9,
        max_model_len=max_model_len,
        trust_remote_code=True,
    )

    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=max_tokens,
        stop=stop_sequences,
        n=args.n,
    )

    # Process in batches
    all_completions = []
    all_results = []
    t0 = time.time()

    for batch_start in range(0, len(rows), args.batch_size):
        batch_end = min(batch_start + args.batch_size, len(rows))
        batch_rows = rows[batch_start:batch_end]
        batch_prompts = [row["prompt"] for row in batch_rows]

        print(f"\n[eval] Batch {batch_start // args.batch_size}: prompts {batch_start+1}-{batch_end}/{len(rows)}")
        outputs = llm.generate(batch_prompts, sampling_params)

        # Extract n completions per prompt
        for output in outputs:
            completions = [o.text for o in output.outputs]
            all_completions.append(completions)

        # Verify batch
        batch_completions = all_completions[batch_start:batch_end]
        verifier = MULTI_VERIFIERS[args.benchmark]
        batch_results = verifier(batch_completions, batch_rows)
        all_results.extend(batch_results)

        # Progress
        n_done = len(all_results)
        avg_pass = sum(r["pass_rate"] for r in all_results) / n_done if n_done > 0 else 0
        print(f"  Avg pass rate so far: {avg_pass:.4f} ({n_done}/{len(rows)} prompts)")

    gen_time = time.time() - t0
    print(f"\n[eval] Total time: {gen_time:.1f}s")

    # Save results
    results_path = output_dir / "verification_results.jsonl"
    with open(results_path, "w") as f:
        for row, result in zip(rows, all_results):
            f.write(json.dumps({**row, **result, "prompt": row["prompt"][:200]}) + "\n")

    # Compute pass@k
    n = args.n
    pass_at_k = {}
    for k in [1, 2, 4, 8, 16, 32, 64, 128]:
        if k > n:
            break
        passed = sum(1 for r in all_results if any(s > 0 for s in r["scores"][:k]))
        pass_at_k[f"pass@{k}"] = passed / len(all_results) if all_results else 0

    # Save summary
    summary = {
        "benchmark": args.benchmark,
        "model": args.model_path,
        "num_prompts": len(all_results),
        "num_completions_per_prompt": n,
        "temperature": args.temperature,
        "generation_time_s": gen_time,
        "pass_at_k": pass_at_k,
        "avg_pass_rate": sum(r["pass_rate"] for r in all_results) / len(all_results) if all_results else 0,
    }

    summary_path = output_dir / "merged_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*50}")
    print(f"RESULTS: {args.benchmark} pass@k")
    print(f"{'='*50}")
    for k, rate in pass_at_k.items():
        print(f"  {k}: {rate*100:.2f}%")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
