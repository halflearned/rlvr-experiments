#!/usr/bin/env python3
"""
Test if batching multiple completions with different lengths causes ref_logprobs corruption.

Hypothesis: When multiple completions with different actual lengths are batched and
passed to vLLM, the logprobs might get mixed up or corrupted at specific positions.
"""

import os
import torch
from pathlib import Path
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

# Create a test prompt
prompt = "The answer is"
prompt_tokens = tokenizer.encode(prompt, add_special_tokens=False)
prompt_len = len(prompt_tokens)

# Create two completions of different lengths
completion1 = ": 1.5 meters."  # Short
completion2 = ": The answer to this question is 1.5 meters, calculated as follows: we divide 6 by 4 to get 1.5."  # Long

tokens1 = tokenizer.encode(completion1, add_special_tokens=False)
tokens2 = tokenizer.encode(completion2, add_special_tokens=False)

print(f"Prompt: {repr(prompt)} ({prompt_len} tokens)")
print(f"Completion 1: {repr(completion1)} ({len(tokens1)} tokens)")
print(f"Completion 2: {repr(completion2)} ({len(tokens2)} tokens)")

# Pad to same length (simulating batching)
max_comp_len = max(len(tokens1), len(tokens2))
padded_len = ((max_comp_len + 7) // 8) * 8  # Round up to 8

tokens1_padded = tokens1 + [tokenizer.eos_token_id] * (padded_len - len(tokens1))
tokens2_padded = tokens2 + [tokenizer.eos_token_id] * (padded_len - len(tokens2))

seq1 = prompt_tokens + tokens1_padded
seq2 = prompt_tokens + tokens2_padded

print(f"\nPadded completion length: {padded_len}")
print(f"Sequence 1 length: {len(seq1)}")
print(f"Sequence 2 length: {len(seq2)}")

# Compute logprobs for each sequence INDIVIDUALLY
print("\n=== Individual computation ===")
sp = SamplingParams(max_tokens=1, prompt_logprobs=1, temperature=1.0)

def get_completion_logprobs(seq, prompt_len, actual_comp_len):
    """Get logprobs for completion portion of sequence."""
    prompt_obj = TokensPrompt(prompt_token_ids=seq)
    outputs = llm.generate([prompt_obj], sampling_params=sp)
    result = outputs[0]

    logprobs = []
    for i in range(prompt_len, prompt_len + actual_comp_len):
        if i < len(result.prompt_logprobs) and result.prompt_logprobs[i] is not None:
            tok_id = seq[i]
            lp_dict = result.prompt_logprobs[i]
            if tok_id in lp_dict:
                logprobs.append(lp_dict[tok_id].logprob)
            else:
                logprobs.append(-100.0)
        else:
            logprobs.append(-100.0)
    return logprobs

lps1_individual = get_completion_logprobs(seq1, prompt_len, len(tokens1))
lps2_individual = get_completion_logprobs(seq2, prompt_len, len(tokens2))

print(f"Seq 1 logprobs (first 10): {lps1_individual[:10]}")
print(f"Seq 2 logprobs (first 10): {lps2_individual[:10]}")

# Now compute in a batch (like VLLMHandle.compute_logprobs does)
print("\n=== Batched computation ===")
import asyncio

async def compute_batched(seqs, prompt_lens, padded_comp_len):
    """Simulate VLLMHandle.compute_logprobs behavior."""
    # This is what compute_logprobs does:
    token_ids_list = []
    for seq, plen in zip(seqs, prompt_lens):
        actual_seq_len = plen + padded_comp_len
        token_ids_list.append(seq[:actual_seq_len])

    # Compute logprobs for all sequences
    results = []
    for token_ids, plen in zip(token_ids_list, prompt_lens):
        prompt_obj = TokensPrompt(prompt_token_ids=token_ids)
        outputs = llm.generate([prompt_obj], sampling_params=sp)
        result = outputs[0]

        completion_logprobs = []
        for i in range(plen, len(token_ids)):
            if i < len(result.prompt_logprobs) and result.prompt_logprobs[i] is not None:
                tok_id = token_ids[i]
                lp_dict = result.prompt_logprobs[i]
                if tok_id in lp_dict:
                    completion_logprobs.append(lp_dict[tok_id].logprob)
                else:
                    completion_logprobs.append(-100.0)
            else:
                completion_logprobs.append(-100.0)
        results.append(completion_logprobs)

    return results

# Run batched computation
seqs = [seq1, seq2]
prompt_lens = [prompt_len, prompt_len]
lps_batched = asyncio.run(compute_batched(seqs, prompt_lens, padded_len))

print(f"Batched seq 1 logprobs (first 10): {lps_batched[0][:10]}")
print(f"Batched seq 2 logprobs (first 10): {lps_batched[1][:10]}")

# Compare
print("\n=== Comparison: Individual vs Batched ===")
print(f"Seq 1: Comparing first {len(tokens1)} tokens (actual completion length)")
max_diff1 = 0
max_diff1_pos = -1
for i in range(len(tokens1)):
    diff = abs(lps1_individual[i] - lps_batched[0][i])
    if diff > max_diff1:
        max_diff1 = diff
        max_diff1_pos = i
print(f"Max diff for seq 1: {max_diff1:.6f} at position {max_diff1_pos}")

print(f"\nSeq 2: Comparing first {len(tokens2)} tokens (actual completion length)")
max_diff2 = 0
max_diff2_pos = -1
for i in range(len(tokens2)):
    diff = abs(lps2_individual[i] - lps_batched[1][i])
    if diff > max_diff2:
        max_diff2 = diff
        max_diff2_pos = i
print(f"Max diff for seq 2: {max_diff2:.6f} at position {max_diff2_pos}")

# Show the padded region for seq1 (where EOS tokens are)
print("\n=== Padding region for seq 1 (positions beyond actual completion) ===")
print(f"Actual completion length: {len(tokens1)}")
print(f"Padded length: {padded_len}")
for i in range(len(tokens1), min(padded_len, len(tokens1) + 5)):
    tok_id = tokens1_padded[i]
    batched_lp = lps_batched[0][i] if i < len(lps_batched[0]) else "N/A"
    print(f"Position {i}: token={tok_id} ({repr(tokenizer.decode([tok_id]))}) logprob={batched_lp}")

print("\nDone!")
