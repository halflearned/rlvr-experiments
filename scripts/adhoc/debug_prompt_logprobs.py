#!/usr/bin/env python3
"""
Debug vLLM prompt_logprobs behavior - verify what gets returned.
"""

import os
import torch
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt

# Load dump to get the problematic sequence
dump_path = "/tmp/kl_spike_dumps/spike_1769489205270.pt"
dump = torch.load(dump_path, weights_only=False)

response_seq = dump['response_seq'].tolist()
mask_seq = dump['mask_seq'].tolist()
valid_count = sum(1 for m in mask_seq if m > 0.5)
valid_tokens = response_seq[:valid_count]

max_idx = dump['max_token_idx']
print(f"Problematic position: {max_idx}")
print(f"Token at position {max_idx}: {valid_tokens[max_idx]}")

# Set GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "7"

# Load model
model_path = "/efs/rlvr-experiments/assets/hf/Qwen3-1.7B-Base"
tokenizer = AutoTokenizer.from_pretrained(model_path)
print(f"\nToken {valid_tokens[max_idx]} = {repr(tokenizer.decode([valid_tokens[max_idx]]))}")

llm = LLM(
    model=model_path,
    tensor_parallel_size=1,
    dtype="float16",
    gpu_memory_utilization=0.5,
    max_model_len=2561,
)

# Test with different prompt_logprobs settings
print("\n=== Testing prompt_logprobs behavior ===")

for plp in [0, 1, 5, 10]:
    print(f"\n--- prompt_logprobs={plp} ---")
    sp = SamplingParams(
        max_tokens=1,
        prompt_logprobs=plp if plp > 0 else None,
        temperature=1.0,
    )

    prompt = TokensPrompt(prompt_token_ids=valid_tokens)
    outputs = llm.generate([prompt], sampling_params=sp)
    result = outputs[0]

    if result.prompt_logprobs is None:
        print(f"  prompt_logprobs is None")
        continue

    # Check position max_idx
    if max_idx < len(result.prompt_logprobs) and result.prompt_logprobs[max_idx] is not None:
        lp_dict = result.prompt_logprobs[max_idx]
        actual_token = valid_tokens[max_idx]

        print(f"  Position {max_idx}: lp_dict has {len(lp_dict)} entries")
        print(f"  Actual token {actual_token} in dict: {actual_token in lp_dict}")

        if actual_token in lp_dict:
            print(f"  Logprob of actual token: {lp_dict[actual_token].logprob:.4f}")

        # Show what's in the dict
        for token_id, lp_info in sorted(lp_dict.items(), key=lambda x: x[1].logprob, reverse=True):
            tok_text = tokenizer.decode([token_id])
            print(f"    token={token_id} ({repr(tok_text)}): logprob={lp_info.logprob:.4f}")
    else:
        print(f"  Position {max_idx}: not available")

# Also check a few positions around it
print(f"\n=== Positions around {max_idx} (with prompt_logprobs=1) ===")
sp = SamplingParams(max_tokens=1, prompt_logprobs=1, temperature=1.0)
prompt = TokensPrompt(prompt_token_ids=valid_tokens)
outputs = llm.generate([prompt], sampling_params=sp)
result = outputs[0]

for pos in [max_idx - 2, max_idx - 1, max_idx, max_idx + 1, max_idx + 2]:
    if pos < 0 or pos >= len(result.prompt_logprobs):
        continue
    if result.prompt_logprobs[pos] is None:
        print(f"Position {pos}: None")
        continue

    actual_token = valid_tokens[pos]
    lp_dict = result.prompt_logprobs[pos]
    tok_text = tokenizer.decode([actual_token])

    if actual_token in lp_dict:
        lp = lp_dict[actual_token].logprob
        print(f"Position {pos}: token={repr(tok_text)} logprob={lp:.4f} (in dict)")
    else:
        # Show what IS in the dict
        top_tok = list(lp_dict.keys())[0]
        top_lp = lp_dict[top_tok].logprob
        top_text = tokenizer.decode([top_tok])
        print(f"Position {pos}: token={repr(tok_text)} NOT IN DICT! Top: {repr(top_text)} at {top_lp:.4f}")
