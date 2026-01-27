#!/usr/bin/env python3
"""
Test if including future tokens (padding) affects logprobs at earlier positions.

Hypothesis: vLLM's prompt_logprobs might have a bug where future tokens
affect logprobs at earlier positions (breaking causal masking).
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

# Create a completion that ends with a colon (like the spike case)
completion = " First, we need to find the radius of the two smallest circles. Notice that the radius of the two smallest circles is one fourth of the radius of the large circle. Thus, the radius of the two smallest circles is $3/2=\\boxed{1.5}$ meters. The answer is: 1.5"
tokens = tokenizer.encode(completion, add_special_tokens=False)
print(f"Completion: {len(tokens)} tokens")
print(f"Full text: {completion[:100]}...")

# Find the position of ':' before '1.5'
colon_pos = None
for i, t in enumerate(tokens):
    if tokenizer.decode([t]) == ':':
        # Check if next token is ' ' (space before 1.5)
        if i + 1 < len(tokens) and tokenizer.decode([tokens[i+1]]).strip() in ['', '1']:
            colon_pos = i
print(f"Colon position: {colon_pos}")
if colon_pos:
    print(f"Context around colon: {tokenizer.decode(tokens[max(0,colon_pos-5):colon_pos+3])}")

# Test 1: Compute logprobs with EXACT completion length (no padding)
print("\n=== Test 1: Exact completion length ===")
sp = SamplingParams(max_tokens=1, prompt_logprobs=1, temperature=1.0)
outputs = llm.generate([TokensPrompt(prompt_token_ids=tokens)], sampling_params=sp)
result = outputs[0]

if colon_pos and result.prompt_logprobs and colon_pos < len(result.prompt_logprobs):
    lp_dict = result.prompt_logprobs[colon_pos]
    if tokens[colon_pos] in lp_dict:
        exact_lp = lp_dict[tokens[colon_pos]].logprob
        print(f"Colon logprob (exact length): {exact_lp:.4f}")
    else:
        print(f"Colon not in top-k (exact length)")

# Test 2: Compute logprobs with 100 EOS tokens appended
print("\n=== Test 2: With 100 EOS padding tokens ===")
eos_id = tokenizer.eos_token_id
padded_100 = tokens + [eos_id] * 100
outputs = llm.generate([TokensPrompt(prompt_token_ids=padded_100)], sampling_params=sp)
result = outputs[0]

if colon_pos and result.prompt_logprobs and colon_pos < len(result.prompt_logprobs):
    lp_dict = result.prompt_logprobs[colon_pos]
    if tokens[colon_pos] in lp_dict:
        padded_100_lp = lp_dict[tokens[colon_pos]].logprob
        print(f"Colon logprob (100 EOS padding): {padded_100_lp:.4f}")
    else:
        print(f"Colon not in top-k (100 EOS padding)")

# Test 3: Compute logprobs with 500 EOS tokens appended
print("\n=== Test 3: With 500 EOS padding tokens ===")
padded_500 = tokens + [eos_id] * 500
outputs = llm.generate([TokensPrompt(prompt_token_ids=padded_500)], sampling_params=sp)
result = outputs[0]

if colon_pos and result.prompt_logprobs and colon_pos < len(result.prompt_logprobs):
    lp_dict = result.prompt_logprobs[colon_pos]
    if tokens[colon_pos] in lp_dict:
        padded_500_lp = lp_dict[tokens[colon_pos]].logprob
        print(f"Colon logprob (500 EOS padding): {padded_500_lp:.4f}")
    else:
        print(f"Colon not in top-k (500 EOS padding)")

# Test 4: Compute logprobs with 1000 EOS tokens appended (matching dump length)
print("\n=== Test 4: With 1000 EOS padding tokens (matching dump length) ===")
padded_1000 = tokens + [eos_id] * (1024 - len(tokens))
outputs = llm.generate([TokensPrompt(prompt_token_ids=padded_1000)], sampling_params=sp)
result = outputs[0]

if colon_pos and result.prompt_logprobs and colon_pos < len(result.prompt_logprobs):
    lp_dict = result.prompt_logprobs[colon_pos]
    if tokens[colon_pos] in lp_dict:
        padded_1000_lp = lp_dict[tokens[colon_pos]].logprob
        print(f"Colon logprob (1024 total): {padded_1000_lp:.4f}")
    else:
        print(f"Colon not in top-k (1024 total)")

# Compare all results
print("\n=== Summary ===")
print(f"Exact length ({len(tokens)} tokens): {exact_lp:.4f}")
print(f"With 100 EOS: {padded_100_lp:.4f}")
print(f"With 500 EOS: {padded_500_lp:.4f}")
print(f"With 1024 total: {padded_1000_lp:.4f}")
print(f"Stored ref in dump: -10.3441")
print(f"Stored rollout in dump: -2.0307")

# If causal masking is correct, all these should be nearly identical
if abs(exact_lp - padded_1000_lp) > 0.1:
    print("\n*** WARNING: Future tokens affected logprobs! Causal masking may be broken! ***")
else:
    print("\n*** Good: Future tokens don't affect logprobs (causal masking works) ***")

# Additional check: what logprobs are at the EOS positions?
print("\n=== EOS token logprobs at various positions ===")
for eos_pos in [len(tokens), len(tokens)+1, 134, 200, 500]:
    if eos_pos < len(result.prompt_logprobs) and result.prompt_logprobs[eos_pos] is not None:
        lp_dict = result.prompt_logprobs[eos_pos]
        if eos_id in lp_dict:
            print(f"Position {eos_pos}: EOS logprob = {lp_dict[eos_id].logprob:.4f}")
        else:
            print(f"Position {eos_pos}: EOS not in top-k")

print("\nDone!")
