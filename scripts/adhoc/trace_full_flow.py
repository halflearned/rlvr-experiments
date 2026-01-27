#!/usr/bin/env python3
"""
Trace the FULL ref_logprobs flow from RolloutSample through TrainSample to Batch.

This script simulates the exact same flow as train_grpo.py to identify where corruption occurs.
"""

import os
import torch
import torch.nn.functional as F
from pathlib import Path
from transformers import AutoTokenizer

os.environ["CUDA_VISIBLE_DEVICES"] = "7"

# Load a dump and simulate the flow
dump_dir = Path("/tmp/kl_spike_dumps")
dumps = sorted(dump_dir.glob("spike_*.pt"))

if not dumps:
    print("No dumps found!")
    exit(1)

# Use the same dump we analyzed before
dump_path = dump_dir / "spike_1769489205270.pt"
if not dump_path.exists():
    dump_path = dumps[0]

print(f"Analyzing dump: {dump_path.name}")
dump = torch.load(dump_path, weights_only=False)

model_path = "/efs/rlvr-experiments/assets/hf/Qwen3-1.7B-Base"
tokenizer = AutoTokenizer.from_pretrained(model_path)

# The dump contains the FINAL tensors after make_batch
# These are already in batch format [B, T] = [1, T] for this sequence
response_seq = dump['response_seq']  # [T]
ref_logprobs_seq = dump['ref_logprobs_seq']  # [T]
rollout_logprobs_seq = dump['rollout_logprobs_seq']  # [T]
mask_seq = dump['mask_seq']  # [T]

print(f"\n=== Dump tensor shapes ===")
print(f"response_seq: {response_seq.shape}")
print(f"ref_logprobs_seq: {ref_logprobs_seq.shape}")
print(f"rollout_logprobs_seq: {rollout_logprobs_seq.shape}")
print(f"mask_seq: {mask_seq.shape}")

valid_count = sum(1 for m in mask_seq.tolist() if m > 0.5)
spike_pos = dump['max_token_idx']

print(f"\n=== Analysis ===")
print(f"Valid tokens (mask > 0.5): {valid_count}")
print(f"Spike position: {spike_pos}")
print(f"Spike token: {response_seq[spike_pos].item()} = {repr(tokenizer.decode([response_seq[spike_pos].item()]))}")

# Check: Is the mask correct? Do completion_ids == pad_token_id where mask == 0?
pad_token_id = tokenizer.eos_token_id
print(f"\n=== Mask vs Token analysis ===")
print(f"PAD token ID: {pad_token_id}")

mismatch_count = 0
for i in range(len(mask_seq)):
    is_pad = (response_seq[i].item() == pad_token_id)
    mask_says_invalid = (mask_seq[i].item() < 0.5)
    if is_pad != mask_says_invalid:
        mismatch_count += 1
        if mismatch_count <= 5:
            print(f"  Mismatch at {i}: token={response_seq[i].item()}, mask={mask_seq[i].item():.1f}, is_pad={is_pad}")
print(f"Total mask mismatches: {mismatch_count}")

# Check the pattern of values around the boundary
print(f"\n=== Values around valid/invalid boundary ===")
for i in range(max(0, valid_count - 5), min(len(mask_seq), valid_count + 5)):
    tok_id = response_seq[i].item()
    tok_text = tokenizer.decode([tok_id])
    mask_val = mask_seq[i].item()
    ref_lp = ref_logprobs_seq[i].item()
    rollout_lp = rollout_logprobs_seq[i].item()
    in_valid = i < valid_count
    print(f"  {i:3d}: tok={repr(tok_text):15s} mask={mask_val:.1f} ref={ref_lp:8.4f} rollout={rollout_lp:8.4f} {'VALID' if in_valid else 'PADDING'}")

# Now let's trace backwards - simulate make_batch and see if truncation/padding could explain the corruption
print("\n\n=== Simulating pad_cat ===")

# The BUCKET sizes used in training
SEQ_LEN_BUCKETS = [256, 384, 512, 640, 768, 896, 1024]
COMPLETION_LEN_BUCKETS = [128, 256, 384, 512]

# Simulate what pad_cat would do
def pad_cat(tensors, pad_value=0, fixed_len=None):
    max_len = fixed_len if fixed_len is not None else max(t.shape[1] for t in tensors)
    return torch.cat([F.pad(t, (0, max_len - t.shape[1]), value=pad_value) for t in tensors])

# If we had a ref_logprobs tensor of shape [1, valid_count], what would happen with different bucket sizes?
print(f"Original valid_count (completion length): {valid_count}")
print(f"Completion len buckets: {COMPLETION_LEN_BUCKETS}")

for bucket in COMPLETION_LEN_BUCKETS:
    if valid_count <= bucket:
        print(f"  Would use bucket: {bucket} (padding needed: {bucket - valid_count})")
        break
else:
    print(f"  Would use max bucket: {COMPLETION_LEN_BUCKETS[-1]} (TRUNCATION: {valid_count - COMPLETION_LEN_BUCKETS[-1]})")

# Check if current tensor length matches a bucket
current_len = len(ref_logprobs_seq)
print(f"\nCurrent tensor length: {current_len}")
for bucket in COMPLETION_LEN_BUCKETS:
    if current_len == bucket:
        print(f"  Matches bucket: {bucket}")
        break
else:
    print(f"  Does NOT match any bucket!")

# The key question: if original ref_logprobs was computed with length X, but then bucketed to length Y,
# what happens at the spike position?
print("\n\n=== Key hypothesis: Position shift due to truncation ===")
if current_len < valid_count:
    print(f"TRUNCATION detected! Tensor length {current_len} < valid count {valid_count}")
    # If truncation happened, positions beyond current_len would be lost
    # But spike is at position 60, which is < valid_count 65, so truncation wouldn't explain it
    print(f"But spike at {spike_pos} < current_len {current_len}, so truncation doesn't explain it directly")
else:
    print(f"No truncation: tensor length {current_len} >= valid count {valid_count}")

# Let's look at if there could be a different kind of corruption
# What if the ref_logprobs were computed with WRONG positions?
print("\n\n=== Hypothesis: Wrong position indexing ===")
print("Comparing ref_logprobs[i] to ref_logprobs[i+offset] for small offsets")
for offset in [-3, -2, -1, 1, 2, 3]:
    target = spike_pos + offset
    if 0 <= target < valid_count:
        ref_at_spike = ref_logprobs_seq[spike_pos].item()
        ref_at_target = ref_logprobs_seq[target].item()
        rollout_at_spike = rollout_logprobs_seq[spike_pos].item()
        # Check if ref_at_target matches rollout_at_spike (would indicate shift)
        diff = abs(ref_at_target - rollout_at_spike)
        print(f"  offset {offset:+d}: ref[{target}]={ref_at_target:.4f}, rollout[{spike_pos}]={rollout_at_spike:.4f}, diff={diff:.4f}")

# Also check: does ref at any position match the bad value -10.34?
print("\n\n=== Searching for -10.34 value pattern ===")
target_value = ref_logprobs_seq[spike_pos].item()
print(f"Looking for values close to {target_value:.4f}")
matches = []
for i in range(len(ref_logprobs_seq)):
    if abs(ref_logprobs_seq[i].item() - target_value) < 0.5:
        matches.append(i)
print(f"Found {len(matches)} positions with similar value: {matches[:20]}")

# What are the logprobs at those positions in rollout?
if matches:
    print("\nRollout values at those positions:")
    for i in matches[:10]:
        tok = tokenizer.decode([response_seq[i].item()])
        print(f"  pos {i}: ref={ref_logprobs_seq[i].item():.4f} rollout={rollout_logprobs_seq[i].item():.4f} token={repr(tok)}")

print("\nDone!")
