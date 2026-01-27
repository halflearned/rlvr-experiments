#!/usr/bin/env python3
"""
Test if the MATH prompt context makes the colon unlikely.

The completion mentions "radius of two smallest circles" and "1.5 meters",
so we'll construct a plausible prompt and test logprobs.
"""

import os
import torch
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt

os.environ["CUDA_VISIBLE_DEVICES"] = "7"

model_path = "/efs/rlvr-experiments/assets/hf/Qwen3-1.7B-Base"
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Construct a plausible MATH prompt
# The completion mentions: radius, two smallest circles, one fourth, large circle, 1.5 meters
prompt = """Problem:
A large circle has a radius of 6 meters. Two smaller circles are inscribed inside the large circle. What is the radius of the two smallest circles?

Solution:"""

# The completion from the dump
completion = """ First, we need to find the radius of the two smallest circles. Notice that the radius of the two smallest circles is one fourth of the radius of the large circle. Thus, the radius of the two smallest circles is $3/2=\\boxed{1.5}$ meters. The answer is: 1.5"""

print(f"Prompt:\n{prompt}")
print(f"\nCompletion:\n{completion}")

# Tokenize
prompt_tokens = tokenizer.encode(prompt, add_special_tokens=False)
completion_tokens = tokenizer.encode(completion, add_special_tokens=False)
full_tokens = prompt_tokens + completion_tokens

print(f"\nPrompt tokens: {len(prompt_tokens)}")
print(f"Completion tokens: {len(completion_tokens)}")
print(f"Full sequence tokens: {len(full_tokens)}")

# Find position 60 in completion
target_pos = 60
if target_pos < len(completion_tokens):
    tok_id = completion_tokens[target_pos]
    tok_text = tokenizer.decode([tok_id])
    print(f"\nCompletion token at position {target_pos}: {tok_id} = {repr(tok_text)}")
    # Also show the context
    context_start = max(0, target_pos - 5)
    context_tokens = completion_tokens[context_start:target_pos + 3]
    context_text = tokenizer.decode(context_tokens)
    print(f"Context: {repr(context_text)}")

# Load vLLM
print(f"\n=== Loading vLLM ===")
llm = LLM(
    model=model_path,
    tensor_parallel_size=1,
    dtype="float16",
    gpu_memory_utilization=0.5,
    max_model_len=2561,
)

# Test 1: Completion only (like our standalone test)
print(f"\n=== Test 1: Completion only (no prompt) ===")
sp = SamplingParams(max_tokens=1, prompt_logprobs=1, temperature=1.0)
prompt_obj = TokensPrompt(prompt_token_ids=completion_tokens)
outputs = llm.generate([prompt_obj], sampling_params=sp)
result = outputs[0]

if result.prompt_logprobs and target_pos < len(result.prompt_logprobs):
    lp_dict = result.prompt_logprobs[target_pos]
    tok_id = completion_tokens[target_pos]
    if tok_id in lp_dict:
        print(f"Position {target_pos}: logprob = {lp_dict[tok_id].logprob:.4f}")
    else:
        print(f"Position {target_pos}: token not in top-k!")

# Test 2: Full sequence (prompt + completion)
print(f"\n=== Test 2: Full sequence (prompt + completion) ===")
prompt_obj = TokensPrompt(prompt_token_ids=full_tokens)
outputs = llm.generate([prompt_obj], sampling_params=sp)
result = outputs[0]

# Position in full sequence = len(prompt_tokens) + target_pos
full_pos = len(prompt_tokens) + target_pos
if result.prompt_logprobs and full_pos < len(result.prompt_logprobs):
    lp_dict = result.prompt_logprobs[full_pos]
    tok_id = full_tokens[full_pos]
    tok_text = tokenizer.decode([tok_id])
    print(f"Full position {full_pos} (completion pos {target_pos}): token={repr(tok_text)}")
    if tok_id in lp_dict:
        print(f"  logprob = {lp_dict[tok_id].logprob:.4f}")
    else:
        print(f"  token not in top-k! Top tokens:")
        for t, info in list(lp_dict.items())[:5]:
            print(f"    {t} ({repr(tokenizer.decode([t]))}): {info.logprob:.4f}")

# Test 3: Show logprobs for positions around target in full sequence
print(f"\n=== Test 3: Positions around target in full sequence ===")
for pos_offset in [-3, -2, -1, 0, 1, 2, 3]:
    pos = len(prompt_tokens) + target_pos + pos_offset
    if pos >= 0 and result.prompt_logprobs and pos < len(result.prompt_logprobs):
        if result.prompt_logprobs[pos] is None:
            continue
        lp_dict = result.prompt_logprobs[pos]
        tok_id = full_tokens[pos]
        tok_text = tokenizer.decode([tok_id])
        if tok_id in lp_dict:
            lp = lp_dict[tok_id].logprob
            marker = " <-- TARGET" if pos_offset == 0 else ""
            print(f"  pos {pos} (offset {pos_offset:+d}): {repr(tok_text):15s} logprob={lp:.4f}{marker}")
        else:
            print(f"  pos {pos} (offset {pos_offset:+d}): {repr(tok_text):15s} NOT IN TOP-K")
