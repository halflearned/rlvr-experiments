#!/usr/bin/env python3
"""
Decode the full completion from a dump to understand the context.
"""

import torch
from transformers import AutoTokenizer

dump_path = "/tmp/kl_spike_dumps/spike_1769489205270.pt"
model_path = "/efs/rlvr-experiments/assets/hf/Qwen3-1.7B-Base"

dump = torch.load(dump_path, weights_only=False)
tokenizer = AutoTokenizer.from_pretrained(model_path)

response_seq = dump['response_seq'].tolist()
mask_seq = dump['mask_seq'].tolist()
valid_count = sum(1 for m in mask_seq if m > 0.5)

print(f"Valid tokens: {valid_count}")
print(f"Total length: {len(response_seq)}")

# Decode valid portion
valid_tokens = response_seq[:valid_count]
completion_text = tokenizer.decode(valid_tokens)

print(f"\n=== Full completion (valid portion) ===")
print(completion_text)

print(f"\n=== Token-by-token around position 60 ===")
for i in range(max(0, 55), min(valid_count, 70)):
    tok_id = response_seq[i]
    tok_text = tokenizer.decode([tok_id])
    ref_lp = dump['ref_logprobs_seq'][i].item()
    trainer_lp = dump['trainer_logprobs_seq'][i].item()
    rollout_lp = dump['rollout_logprobs_seq'][i].item()
    marker = " <-- SPIKE" if i == dump['max_token_idx'] else ""
    print(f"  {i:3d}: {tok_id:6d} {repr(tok_text):15s} ref={ref_lp:8.4f} trainer={trainer_lp:8.4f} rollout={rollout_lp:8.4f}{marker}")

# Check if prompt_len is in dump
if 'prompt_len' in dump:
    print(f"\nPrompt length: {dump['prompt_len']}")
else:
    print(f"\nPrompt length: NOT IN DUMP")

# Check for patterns in ref_logprobs - are there other extreme values?
print(f"\n=== Most extreme ref vs trainer differences ===")
ref_lps = dump['ref_logprobs_seq'][:valid_count]
trainer_lps = dump['trainer_logprobs_seq'][:valid_count]
diffs = [(i, abs(ref_lps[i].item() - trainer_lps[i].item())) for i in range(valid_count)]
diffs.sort(key=lambda x: x[1], reverse=True)
for i, diff in diffs[:10]:
    tok_id = response_seq[i]
    tok_text = tokenizer.decode([tok_id])
    print(f"  Position {i:3d}: diff={diff:8.4f} token={repr(tok_text):15s} ref={ref_lps[i].item():8.4f} trainer={trainer_lps[i].item():8.4f}")
