#!/usr/bin/env python3
"""
Analyze all dumps to find patterns in ref_logprobs corruption.

Key questions:
1. Is corruption always at the same position relative to completion end?
2. Is the corrupt value always similar (~-10)?
3. Are there multiple corrupted positions per dump?
"""

import torch
from pathlib import Path
from transformers import AutoTokenizer
from collections import Counter

model_path = "/efs/rlvr-experiments/assets/hf/Qwen3-1.7B-Base"
tokenizer = AutoTokenizer.from_pretrained(model_path)

dump_dir = Path("/tmp/kl_spike_dumps")
dumps = sorted(dump_dir.glob("spike_*.pt"))

print(f"Analyzing {len(dumps)} dumps...\n")

# Collect statistics
corruption_stats = []

for dump_path in dumps[:50]:
    try:
        dump = torch.load(dump_path, weights_only=False)

        mask_seq = dump['mask_seq'].tolist()
        ref_lps = dump['ref_logprobs_seq']
        rollout_lps = dump['rollout_logprobs_seq']
        response_seq = dump['response_seq'].tolist()

        valid_count = sum(1 for m in mask_seq if m > 0.5)
        spike_pos = dump['max_token_idx']

        # Find ALL positions where |ref - rollout| > 1.0
        large_diffs = []
        for i in range(valid_count):
            ref_lp = ref_lps[i].item()
            rollout_lp = rollout_lps[i].item()
            diff = abs(ref_lp - rollout_lp)
            if diff > 1.0:
                large_diffs.append({
                    'pos': i,
                    'from_end': valid_count - i,
                    'ref': ref_lp,
                    'rollout': rollout_lp,
                    'diff': diff,
                    'token_id': response_seq[i],
                })

        corruption_stats.append({
            'dump': dump_path.name,
            'valid_count': valid_count,
            'spike_pos': spike_pos,
            'spike_from_end': valid_count - spike_pos,
            'num_corrupt': len(large_diffs),
            'corrupt_positions': large_diffs,
        })

    except Exception as e:
        print(f"Error loading {dump_path.name}: {e}")

# Summarize
print("=== Summary ===")
print(f"Total dumps analyzed: {len(corruption_stats)}")

# How many corrupted positions per dump?
corrupt_counts = Counter(s['num_corrupt'] for s in corruption_stats)
print(f"\nCorrupted positions per dump:")
for count, freq in sorted(corrupt_counts.items()):
    print(f"  {count} corrupted: {freq} dumps")

# Are corruptions always at the same relative position?
print(f"\nCorruption position relative to end (from_end):")
from_end_counter = Counter()
for s in corruption_stats:
    for c in s['corrupt_positions']:
        from_end_counter[c['from_end']] += 1
for from_end, count in sorted(from_end_counter.items())[:20]:
    print(f"  from_end={from_end}: {count}")

# What tokens are corrupted?
print(f"\nCorrupted token IDs:")
token_counter = Counter()
for s in corruption_stats:
    for c in s['corrupt_positions']:
        tok_text = tokenizer.decode([c['token_id']])
        token_counter[f"{c['token_id']}:{repr(tok_text)}"] += 1
for token_info, count in token_counter.most_common(20):
    print(f"  {token_info}: {count}")

# What are the ref logprob values at corrupted positions?
print(f"\nRef logprob values at corrupted positions:")
ref_values = []
for s in corruption_stats:
    for c in s['corrupt_positions']:
        ref_values.append(c['ref'])

if ref_values:
    print(f"  Min: {min(ref_values):.4f}")
    print(f"  Max: {max(ref_values):.4f}")
    print(f"  Mean: {sum(ref_values)/len(ref_values):.4f}")

    # Distribution
    bins = {
        '< -50': 0,
        '-50 to -20': 0,
        '-20 to -10': 0,
        '-10 to -5': 0,
        '-5 to 0': 0,
        '> 0': 0,
    }
    for v in ref_values:
        if v < -50:
            bins['< -50'] += 1
        elif v < -20:
            bins['-50 to -20'] += 1
        elif v < -10:
            bins['-20 to -10'] += 1
        elif v < -5:
            bins['-10 to -5'] += 1
        elif v < 0:
            bins['-5 to 0'] += 1
        else:
            bins['> 0'] += 1
    print(f"\n  Ref value distribution:")
    for bin_name, count in bins.items():
        print(f"    {bin_name}: {count}")

# What are the rollout logprob values (should be correct)?
print(f"\nRollout logprob values at corrupted positions:")
rollout_values = []
for s in corruption_stats:
    for c in s['corrupt_positions']:
        rollout_values.append(c['rollout'])

if rollout_values:
    print(f"  Min: {min(rollout_values):.4f}")
    print(f"  Max: {max(rollout_values):.4f}")
    print(f"  Mean: {sum(rollout_values)/len(rollout_values):.4f}")

# Check: are multiple corruptions in the same dump at consecutive positions?
print(f"\n=== Checking for consecutive corruptions ===")
for s in corruption_stats:
    if len(s['corrupt_positions']) > 1:
        positions = sorted([c['pos'] for c in s['corrupt_positions']])
        gaps = [positions[i+1] - positions[i] for i in range(len(positions)-1)]
        consecutive = sum(1 for g in gaps if g == 1)
        print(f"{s['dump']}: {len(positions)} corruptions, positions={positions}, consecutive={consecutive}")

# Show a few examples of multi-corruption dumps
print(f"\n=== Examples of multi-corruption dumps ===")
multi_corrupt = [s for s in corruption_stats if len(s['corrupt_positions']) > 1][:3]
for s in multi_corrupt:
    print(f"\n{s['dump']}:")
    print(f"  valid_count={s['valid_count']}")
    for c in s['corrupt_positions']:
        tok_text = tokenizer.decode([c['token_id']])
        print(f"  pos={c['pos']} (from_end={c['from_end']}): {repr(tok_text)} ref={c['ref']:.4f} rollout={c['rollout']:.4f}")

print("\nDone!")
