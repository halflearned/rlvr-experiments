#!/usr/bin/env python3
"""Quick APPS verification - 100 problems, 64 completions each."""
import json
import asyncio
import sys
import glob

sys.set_int_max_str_digits(0)

from datasets import load_dataset
from rlvr_experiments.verifiers.code import APPSStdinVerifier, APPSFunctionVerifier
from rlvr_experiments.verifiers.code_executor import CodeExecutor, ExecutorConfig

# Load completions
print("Loading completions...")
completions_by_id = {}
for f in glob.glob("experiments/apps-introductory-pass-rate/*_replica_*.jsonl"):
    with open(f) as fp:
        for line in fp:
            if line.strip():
                d = json.loads(line)
                completions_by_id[d["prompt_id"]] = [c["text"] for c in d["completions"]]

print(f"Loaded {len(completions_by_id)} prompts")

# Load dataset
print("Loading APPS dataset...")
hf = load_dataset("codeparrot/apps", split="train", revision="refs/convert/parquet")

# Get first 100 introductory problems with completions
problems = []
for row in hf:
    if row["difficulty"] != "introductory":
        continue
    pid = f"apps_{row['problem_id']}"
    if pid not in completions_by_id:
        continue
    try:
        io = json.loads(row["input_output"]) if row["input_output"] else {}
    except:
        continue
    if not io.get("inputs") or not io.get("outputs"):
        continue
    inputs = io.get("inputs", [])
    fmt = "stdin" if isinstance(inputs[0], str) else "function_call"
    if fmt == "function_call" and not io.get("fn_name"):
        continue
    problems.append({"pid": pid, "io": io, "fmt": fmt})
    if len(problems) >= 2000:
        break

print(f"Testing {len(problems)} problems")

# Resume from where we left off
START_FROM = 576
correct_total = 5531  # From previous run
verified_total = 36864  # From previous run

print(f"Resuming from prompt {START_FROM}, running total: {correct_total}/{verified_total}")

for i, p in enumerate(problems):
    if i < START_FROM:
        continue
    executor = CodeExecutor(ExecutorConfig(timeout=10.0), max_concurrent=128)
    if p["fmt"] == "stdin":
        verifier = APPSStdinVerifier(executor=executor)
        vp = {"inputs": p["io"]["inputs"], "outputs": p["io"]["outputs"]}
    else:
        verifier = APPSFunctionVerifier(executor=executor)
        vp = {"inputs": p["io"]["inputs"], "outputs": p["io"]["outputs"], "fn_name": p["io"]["fn_name"]}

    completions = completions_by_id[p["pid"]][:64]  # Just 64 for speed

    async def verify():
        tasks = [verifier.verify(vp, c) for c in completions]
        return await asyncio.gather(*tasks)

    results = asyncio.run(verify())
    passed = sum(1 for r in results if r.all_passed)
    correct_total += passed
    verified_total += len(completions)

    pct = passed / 64 * 100
    running_pct = correct_total / verified_total * 100
    print(f"[{i+1}/{len(problems)}] {p['pid']}: {passed}/64 ({pct:.1f}%) | Running: {correct_total}/{verified_total} = {running_pct:.1f}%", flush=True)

print(f"\nFinal: {correct_total}/{verified_total} = {correct_total/verified_total*100:.1f}%")
