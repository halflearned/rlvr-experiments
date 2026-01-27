#!/usr/bin/env python3
"""
Analyze KL spike positions across multiple dumps to find patterns.
Are spikes always near the end of valid tokens?
"""

import torch
from pathlib import Path

dump_dir = Path("/tmp/kl_spike_dumps")
dumps = sorted(dump_dir.glob("spike_*.pt"))[:30]  # First 30 dumps

print(f"Analyzing {len(dumps)} dumps...\n")

print(f"{'Dump':<30} {'valid_cnt':>10} {'spike_pos':>10} {'from_end':>10} {'spike_ref':>12} {'spike_train':>12} {'diff':>10}")
print("-" * 110)

patterns = {
    "near_end": 0,  # within 10 tokens of end
    "middle": 0,    # more than 10 tokens from end
}

for dump_path in dumps:
    try:
        dump = torch.load(dump_path, weights_only=False)

        mask_seq = dump['mask_seq'].tolist()
        valid_count = sum(1 for m in mask_seq if m > 0.5)
        spike_pos = dump['max_token_idx']
        from_end = valid_count - spike_pos

        ref_lp = dump['ref_logprob_at_max']
        trainer_lp = dump['trainer_logprob_at_max']
        diff = abs(ref_lp - trainer_lp)

        # Classify
        if from_end <= 10:
            patterns["near_end"] += 1
        else:
            patterns["middle"] += 1

        print(f"{dump_path.name:<30} {valid_count:>10} {spike_pos:>10} {from_end:>10} {ref_lp:>12.4f} {trainer_lp:>12.4f} {diff:>10.4f}")

    except Exception as e:
        print(f"{dump_path.name:<30} ERROR: {e}")

print()
print(f"=== Position Patterns ===")
print(f"Near end (within 10 tokens): {patterns['near_end']}")
print(f"Middle (>10 tokens from end): {patterns['middle']}")

# Also check the actual ref_logprobs pattern for outliers
print(f"\n=== Checking if ref values are suspiciously uniform ===")
for dump_path in dumps[:5]:
    try:
        dump = torch.load(dump_path, weights_only=False)
        ref_lps = dump['ref_logprobs_seq']
        mask_seq = dump['mask_seq']
        valid_count = int(mask_seq.sum().item())

        # Get valid ref_logprobs
        valid_ref = ref_lps[:valid_count].tolist()

        # Find how many are exactly -100.0 (fallback value)
        num_fallback = sum(1 for v in valid_ref if abs(v + 100.0) < 0.01)

        # Find how many are extremely negative (< -5)
        num_extreme = sum(1 for v in valid_ref if v < -5 and abs(v + 100.0) > 0.01)

        spike_pos = dump['max_token_idx']
        print(f"\n{dump_path.name}:")
        print(f"  valid_count={valid_count}, spike_pos={spike_pos}")
        print(f"  fallback (-100) values: {num_fallback}")
        print(f"  extreme (<-5) values: {num_extreme}")
        if num_extreme > 0:
            extreme_positions = [i for i, v in enumerate(valid_ref) if v < -5 and abs(v + 100.0) > 0.01]
            print(f"  extreme positions: {extreme_positions}")

    except Exception as e:
        print(f"{dump_path.name}: ERROR: {e}")
