#!/usr/bin/env python3
"""
Minimal script to verify: does the base model really give -9.83 for the colon token?

This loads the EXACT sequence from the spike dump and computes logprobs using
a fresh vLLM instance with base model weights.
"""

import os
import torch
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Load the spike dump
print("=" * 60)
print("STEP 1: Load spike dump")
print("=" * 60)
spike = torch.load("/tmp/kl_spike_dumps/spike_1769498881409.pt")

print(f"Keys in spike dump: {list(spike.keys())}")
print(f"prompt_len: {spike['prompt_len']}")
print(f"max_token_idx (spike position): {spike['max_token_idx']}")
print(f"max_token_id: {spike['max_token_id']}")
print(f"rollout_logprob_at_max: {spike['rollout_logprob_at_max']}")
print(f"ref_logprob_at_max: {spike['ref_logprob_at_max']}")

# Load the 2_ref_computed dump to get the FULL sequence (prompt + completion)
print("\n" + "=" * 60)
print("STEP 2: Load the ref_computed dump to get full sequence")
print("=" * 60)
ref_dump = torch.load("/tmp/lifecycle_dumps/ref_compute_1769498876833.pt")

# The spike is in batch index 1 of this dump
batch_idx = 1
input_ids = ref_dump["input_ids"][batch_idx]
prompt_len = ref_dump["prompt_lens"][batch_idx].item()
completion_len = ref_dump["completion_len"]

print(f"input_ids shape: {input_ids.shape}")
print(f"prompt_len: {prompt_len}")
print(f"completion_len: {completion_len}")

# Get the full sequence
full_seq = input_ids[:prompt_len + completion_len].tolist()
print(f"full_seq length: {len(full_seq)}")

# The spike is at completion position 27, which is full position prompt_len + 27
spike_completion_pos = 27
spike_full_pos = prompt_len + spike_completion_pos
spike_token_id = full_seq[spike_full_pos]

tokenizer = AutoTokenizer.from_pretrained("/efs/rlvr-experiments/assets/hf/Qwen3-1.7B-Base")
print(f"\nSpike position: completion_pos={spike_completion_pos}, full_pos={spike_full_pos}")
print(f"Token at spike: id={spike_token_id}, text={repr(tokenizer.decode([spike_token_id]))}")

# Show context around the spike
print("\nContext around spike:")
for i in range(spike_full_pos - 5, spike_full_pos + 5):
    tok = tokenizer.decode([full_seq[i]])
    marker = " <-- SPIKE" if i == spike_full_pos else ""
    print(f"  pos {i}: {repr(tok)}{marker}")

# Load vLLM with base model
print("\n" + "=" * 60)
print("STEP 3: Load fresh vLLM with BASE model")
print("=" * 60)
model_path = "/efs/rlvr-experiments/assets/hf/Qwen3-1.7B-Base"
llm = LLM(
    model=model_path,
    tensor_parallel_size=1,
    dtype="float16",
    gpu_memory_utilization=0.4,
    max_model_len=2561,
)

# Compute prompt_logprobs
print("\n" + "=" * 60)
print("STEP 4: Compute prompt_logprobs")
print("=" * 60)
sp = SamplingParams(
    max_tokens=1,
    prompt_logprobs=1,
    temperature=1.0,
)

outputs = llm.generate([TokensPrompt(prompt_token_ids=full_seq)], sampling_params=sp)
result = outputs[0]

print(f"prompt_logprobs length: {len(result.prompt_logprobs)}")

# Get logprob at spike position
print("\n" + "=" * 60)
print("STEP 5: Extract logprob at spike position")
print("=" * 60)

lp_dict = result.prompt_logprobs[spike_full_pos]
if lp_dict and spike_token_id in lp_dict:
    base_logprob = lp_dict[spike_token_id].logprob
    print(f"BASE MODEL logprob at spike position: {base_logprob}")
else:
    print(f"Token {spike_token_id} not found in logprobs dict!")
    base_logprob = None

print(f"\nComparison:")
print(f"  Rollout (from spike dump): {spike['rollout_logprob_at_max']}")
print(f"  Ref (from spike dump):     {spike['ref_logprob_at_max']}")
print(f"  Base (just computed):      {base_logprob}")

print("\n" + "=" * 60)
print("STEP 6: Show logprobs around the spike")
print("=" * 60)
print(f"{'pos':>4} | {'token':>15} | {'base_logprob':>12}")
print("-" * 40)
for i in range(spike_full_pos - 5, min(spike_full_pos + 5, len(result.prompt_logprobs))):
    tok_id = full_seq[i]
    tok = tokenizer.decode([tok_id])
    lp_dict = result.prompt_logprobs[i]
    if lp_dict and tok_id in lp_dict:
        lp = lp_dict[tok_id].logprob
    else:
        lp = None
    marker = " <-- SPIKE" if i == spike_full_pos else ""
    print(f"{i:4d} | {repr(tok):>15} | {lp:>12.4f}{marker}" if lp else f"{i:4d} | {repr(tok):>15} | {'N/A':>12}{marker}")

print("\n" + "=" * 60)
print("CONCLUSION")
print("=" * 60)
if base_logprob is not None:
    diff = spike['rollout_logprob_at_max'] - base_logprob
    print(f"Rollout - Base = {spike['rollout_logprob_at_max']:.4f} - ({base_logprob:.4f}) = {diff:.4f}")
    if abs(base_logprob - spike['ref_logprob_at_max']) < 0.01:
        print("Base model matches ref_logprob from training. Ref computation is CORRECT.")
    else:
        print(f"Base model DIFFERS from ref_logprob! Base={base_logprob:.4f}, Ref={spike['ref_logprob_at_max']:.4f}")
