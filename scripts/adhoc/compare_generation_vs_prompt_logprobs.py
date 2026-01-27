#!/usr/bin/env python3
"""
Compare logprobs from generation mode vs prompt_logprobs mode in vLLM.

Hypothesis: The rollout uses generation mode (logprobs during sampling),
while reference uses prompt_logprobs mode (logprobs after the fact).
Maybe these give different results for the same tokens?
"""

import os
import torch
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt

os.environ["CUDA_VISIBLE_DEVICES"] = "7"

model_path = "/efs/rlvr-experiments/assets/hf/Qwen3-1.7B-Base"
tokenizer = AutoTokenizer.from_pretrained(model_path)

print("=== Loading vLLM ===")
llm = LLM(
    model=model_path,
    tensor_parallel_size=1,
    dtype="float16",
    gpu_memory_utilization=0.5,
    max_model_len=2561,
)

# Create a prompt
prompt = "Problem:\nWhat is 2 + 2?\n\nSolution:"
prompt_tokens = tokenizer.encode(prompt, add_special_tokens=False)
prompt_len = len(prompt_tokens)

print(f"Prompt: {repr(prompt)}")
print(f"Prompt length: {prompt_len} tokens")

# Method 1: Generate with logprobs (like rollout)
print("\n=== Method 1: Generate with logprobs ===")
sp_gen = SamplingParams(
    max_tokens=50,
    temperature=1.0,
    top_p=0.95,
    top_k=20,
    logprobs=0,  # Return logprobs for sampled tokens
)
outputs = llm.generate([prompt], sampling_params=sp_gen)
gen_output = outputs[0].outputs[0]

print(f"Generated {len(gen_output.token_ids)} tokens")
gen_text = tokenizer.decode(gen_output.token_ids)
print(f"Generated text: {repr(gen_text[:100])}...")

# Extract logprobs from generation
gen_logprobs = []
for i, lp_dict in enumerate(gen_output.logprobs):
    tok_id = gen_output.token_ids[i]
    if tok_id in lp_dict:
        gen_logprobs.append(lp_dict[tok_id].logprob)
    else:
        gen_logprobs.append(None)

print(f"First 10 generation logprobs: {gen_logprobs[:10]}")

# Method 2: Compute prompt_logprobs for the same tokens (like reference)
print("\n=== Method 2: Prompt logprobs for same tokens ===")
full_seq = list(prompt_tokens) + list(gen_output.token_ids)

sp_prompt = SamplingParams(
    max_tokens=1,
    prompt_logprobs=1,
    temperature=1.0,
)
outputs = llm.generate([TokensPrompt(prompt_token_ids=full_seq)], sampling_params=sp_prompt)
prompt_result = outputs[0]

prompt_logprobs = []
if prompt_result.prompt_logprobs:
    for i in range(prompt_len, len(full_seq)):
        if i < len(prompt_result.prompt_logprobs) and prompt_result.prompt_logprobs[i] is not None:
            tok_id = full_seq[i]
            lp_dict = prompt_result.prompt_logprobs[i]
            if tok_id in lp_dict:
                prompt_logprobs.append(lp_dict[tok_id].logprob)
            else:
                prompt_logprobs.append(-100.0)
        else:
            prompt_logprobs.append(-100.0)

print(f"First 10 prompt logprobs: {prompt_logprobs[:10]}")

# Compare
print("\n=== Comparison ===")
print(f"Position | Gen LogP | Prompt LogP | Diff | Token")
print("-" * 70)
max_diff = 0
max_diff_pos = -1
for i in range(min(len(gen_logprobs), len(prompt_logprobs))):
    if gen_logprobs[i] is not None and prompt_logprobs[i] != -100.0:
        diff = abs(gen_logprobs[i] - prompt_logprobs[i])
        tok_id = gen_output.token_ids[i]
        tok_text = tokenizer.decode([tok_id])
        marker = " ***" if diff > 0.1 else ""
        print(f"{i:8d} | {gen_logprobs[i]:8.4f} | {prompt_logprobs[i]:11.4f} | {diff:4.4f} | {repr(tok_text):15s}{marker}")
        if diff > max_diff:
            max_diff = diff
            max_diff_pos = i

print(f"\nMax diff: {max_diff:.4f} at position {max_diff_pos}")

if max_diff > 0.5:
    print("\n*** WARNING: Generation and prompt_logprobs give different results! ***")
else:
    print("\n*** Good: Generation and prompt_logprobs match closely ***")

# Extra test: Same sequence, different temperature
print("\n\n=== Extra test: Temperature effect ===")
for temp in [0.5, 1.0, 2.0]:
    sp = SamplingParams(max_tokens=1, prompt_logprobs=1, temperature=temp)
    outputs = llm.generate([TokensPrompt(prompt_token_ids=full_seq)], sampling_params=sp)
    result = outputs[0]

    # Get logprob at first completion position
    pos = prompt_len
    if result.prompt_logprobs and pos < len(result.prompt_logprobs) and result.prompt_logprobs[pos] is not None:
        tok_id = full_seq[pos]
        lp_dict = result.prompt_logprobs[pos]
        if tok_id in lp_dict:
            lp = lp_dict[tok_id].logprob
            print(f"Temperature {temp}: logprob at pos {pos} = {lp:.4f}")

print("\nDone!")
