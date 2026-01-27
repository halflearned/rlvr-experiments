#!/usr/bin/env python3
"""
Trace ref_logprobs computation step by step to find the bug.

The issue: ref_logprobs at certain positions are ~-10 when they should be ~-2.
This happens near the end of completions (within 10 tokens of the valid length).
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

# Load a dump with a spike
dump_dir = Path("/tmp/kl_spike_dumps")
dumps = sorted(dump_dir.glob("spike_*.pt"))

if not dumps:
    print("No dumps found!")
    exit(1)

# Find a dump where the spike is NOT at the very last position
# (so we can see what's happening with the surrounding tokens)
selected_dump = None
for dump_path in dumps:
    dump = torch.load(dump_path, weights_only=False)
    mask_seq = dump['mask_seq'].tolist()
    valid_count = sum(1 for m in mask_seq if m > 0.5)
    spike_pos = dump['max_token_idx']
    from_end = valid_count - spike_pos
    if from_end > 3:  # Want some tokens AFTER the spike
        selected_dump = dump_path
        break

if selected_dump is None:
    selected_dump = dumps[0]  # Fall back to first dump

print(f"Analyzing dump: {selected_dump.name}")
dump = torch.load(selected_dump, weights_only=False)

# Extract info
response_seq = dump['response_seq'].tolist()
mask_seq = dump['mask_seq'].tolist()
ref_logprobs_stored = dump['ref_logprobs_seq']
rollout_logprobs_stored = dump['rollout_logprobs_seq']
trainer_logprobs_stored = dump['trainer_logprobs_seq']

valid_count = sum(1 for m in mask_seq if m > 0.5)
spike_pos = dump['max_token_idx']
spike_token = response_seq[spike_pos]

print(f"\n=== Dump Info ===")
print(f"Valid tokens: {valid_count}")
print(f"Spike position: {spike_pos} (from_end = {valid_count - spike_pos})")
print(f"Spike token: {spike_token} = {repr(tokenizer.decode([spike_token]))}")
print(f"Prompt length (from dump): {dump.get('prompt_len', 'NOT IN DUMP')}")

# Check what the valid completion looks like
valid_tokens = response_seq[:valid_count]
print(f"\n=== Valid Completion ===")
print(tokenizer.decode(valid_tokens))

# Show tokens around spike
print(f"\n=== Tokens around spike ===")
for i in range(max(0, spike_pos - 5), min(valid_count, spike_pos + 5)):
    tok_id = response_seq[i]
    tok_text = tokenizer.decode([tok_id])
    ref_lp = ref_logprobs_stored[i].item()
    rollout_lp = rollout_logprobs_stored[i].item()
    trainer_lp = trainer_logprobs_stored[i].item()
    marker = " <-- SPIKE" if i == spike_pos else ""
    print(f"  {i:3d}: {tok_id:6d} {repr(tok_text):15s} ref={ref_lp:8.4f} rollout={rollout_lp:8.4f} trainer={trainer_lp:8.4f}{marker}")

# Now recompute ref_logprobs with vLLM to see what we should get
print(f"\n=== Loading vLLM ===")
llm = LLM(
    model=model_path,
    tensor_parallel_size=1,
    dtype="float16",
    gpu_memory_utilization=0.5,
    max_model_len=2561,
)

# Test 1: Compute logprobs for ONLY the valid portion
print(f"\n=== Test 1: Valid tokens only (length {valid_count}) ===")
sp = SamplingParams(max_tokens=1, prompt_logprobs=1, temperature=1.0)
prompt_obj = TokensPrompt(prompt_token_ids=valid_tokens)
outputs = llm.generate([prompt_obj], sampling_params=sp)
result = outputs[0]

if result.prompt_logprobs:
    print(f"Recomputed logprobs around spike:")
    for i in range(max(0, spike_pos - 5), min(valid_count, spike_pos + 5)):
        if i < len(result.prompt_logprobs) and result.prompt_logprobs[i] is not None:
            tok_id = valid_tokens[i]
            lp_dict = result.prompt_logprobs[i]
            if tok_id in lp_dict:
                new_lp = lp_dict[tok_id].logprob
                old_lp = ref_logprobs_stored[i].item()
                diff = abs(new_lp - old_lp)
                marker = " <-- SPIKE" if i == spike_pos else ""
                flag = " ***DIFF***" if diff > 1.0 else ""
                print(f"  {i:3d}: recomputed={new_lp:8.4f}  stored_ref={old_lp:8.4f}  diff={diff:.4f}{marker}{flag}")

# Test 2: Compute logprobs with padding (simulating what training does)
# Pad to a bucket length (like make_batch does)
padded_len = 128  # Common bucket size
if valid_count < padded_len:
    padded_tokens = valid_tokens + [tokenizer.eos_token_id] * (padded_len - valid_count)

    print(f"\n=== Test 2: With EOS padding (length {len(padded_tokens)}) ===")
    prompt_obj = TokensPrompt(prompt_token_ids=padded_tokens)
    outputs = llm.generate([prompt_obj], sampling_params=sp)
    result = outputs[0]

    if result.prompt_logprobs:
        print(f"Recomputed logprobs around spike (with padding):")
        for i in range(max(0, spike_pos - 5), min(valid_count + 5, spike_pos + 10)):
            if i < len(result.prompt_logprobs) and result.prompt_logprobs[i] is not None:
                tok_id = padded_tokens[i]
                lp_dict = result.prompt_logprobs[i]
                if tok_id in lp_dict:
                    new_lp = lp_dict[tok_id].logprob
                    if i < valid_count:
                        old_lp = ref_logprobs_stored[i].item()
                        diff = abs(new_lp - old_lp)
                        marker = " <-- SPIKE" if i == spike_pos else ""
                        flag = " ***DIFF***" if diff > 1.0 else ""
                        in_valid = ""
                    else:
                        old_lp = 0.0
                        diff = 0.0
                        marker = " (padding)"
                        flag = ""
                        in_valid = ""
                    print(f"  {i:3d}: tok={repr(tokenizer.decode([tok_id])):15s} recomputed={new_lp:8.4f}  stored_ref={old_lp:8.4f}{marker}{flag}")

# Test 3: What if prompt_len is wrong?
# Let's check if there's a prompt_len in the dump and test with different values
if 'prompt_len' in dump:
    prompt_len = dump['prompt_len']
    print(f"\n=== Test 3: Simulating compute_logprobs with prompt_len={prompt_len} ===")

    # The training code does: seq[:plen + completion_len] where completion_len is padded
    # Let's see if we can reproduce the bug
    seq_with_prompt = list(range(prompt_len))  # Fake prompt tokens
    # We don't have the prompt tokens in the dump! The dump only has completion tokens.
    print(f"NOTE: Dump doesn't contain prompt tokens, only completion tokens")
    print(f"The response_seq in dump IS the completion_ids, not input_ids")
else:
    print(f"\n=== No prompt_len in dump ===")

# Test 4: Check if the completion tokens in dump match what rollout generated
# Compare rollout_logprobs (should be from the generation) vs ref_logprobs
print(f"\n=== Test 4: Rollout vs Ref logprobs comparison ===")
print(f"Comparing stored rollout (from generation) vs stored ref (computed later):")
max_diff_pos = -1
max_diff_val = 0
for i in range(valid_count):
    rollout_lp = rollout_logprobs_stored[i].item()
    ref_lp = ref_logprobs_stored[i].item()
    diff = abs(rollout_lp - ref_lp)
    if diff > max_diff_val:
        max_diff_val = diff
        max_diff_pos = i

print(f"Max difference: position {max_diff_pos}, diff = {max_diff_val:.4f}")
print(f"Token at max diff: {response_seq[max_diff_pos]} = {repr(tokenizer.decode([response_seq[max_diff_pos]]))}")
print(f"Rollout logprob: {rollout_logprobs_stored[max_diff_pos].item():.4f}")
print(f"Ref logprob: {ref_logprobs_stored[max_diff_pos].item():.4f}")
print(f"Trainer logprob: {trainer_logprobs_stored[max_diff_pos].item():.4f}")

# Test 5: Look for patterns - are there multiple large differences?
print(f"\n=== Test 5: All positions with |rollout - ref| > 1.0 ===")
large_diffs = []
for i in range(valid_count):
    rollout_lp = rollout_logprobs_stored[i].item()
    ref_lp = ref_logprobs_stored[i].item()
    diff = abs(rollout_lp - ref_lp)
    if diff > 1.0:
        large_diffs.append((i, response_seq[i], rollout_lp, ref_lp, diff))

if large_diffs:
    for i, tok_id, rollout_lp, ref_lp, diff in large_diffs:
        tok_text = tokenizer.decode([tok_id])
        from_end = valid_count - i
        print(f"  pos={i:3d} (from_end={from_end:2d}): {repr(tok_text):15s} rollout={rollout_lp:8.4f} ref={ref_lp:8.4f} diff={diff:.4f}")
else:
    print("No positions with diff > 1.0")

print("\nDone!")
