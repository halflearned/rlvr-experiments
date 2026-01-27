#!/usr/bin/env python3
"""
Check what tokens appear at spike positions - are they special (EOS, newlines, etc.)?
"""

import torch
from pathlib import Path
from transformers import AutoTokenizer
from collections import Counter

model_path = "/efs/rlvr-experiments/assets/hf/Qwen3-1.7B-Base"
tokenizer = AutoTokenizer.from_pretrained(model_path)

dump_dir = Path("/tmp/kl_spike_dumps")
dumps = sorted(dump_dir.glob("spike_*.pt"))[:50]

print(f"Analyzing {len(dumps)} dumps...")
print(f"EOS token ID: {tokenizer.eos_token_id}")
print(f"PAD token ID: {tokenizer.pad_token_id}")
print()

token_counter = Counter()
context_samples = []

for dump_path in dumps:
    try:
        dump = torch.load(dump_path, weights_only=False)

        response_seq = dump['response_seq'].tolist()
        spike_pos = dump['max_token_idx']
        spike_token_id = dump['max_token_id']

        token_text = tokenizer.decode([spike_token_id])
        token_counter[f"{spike_token_id}:{repr(token_text)}"] += 1

        # Get context around spike
        context_start = max(0, spike_pos - 3)
        context_end = min(len(response_seq), spike_pos + 3)
        context_tokens = response_seq[context_start:context_end]
        context_text = tokenizer.decode(context_tokens)

        context_samples.append({
            "dump": dump_path.name,
            "token_id": spike_token_id,
            "token_text": repr(token_text),
            "context": repr(context_text),
        })

    except Exception as e:
        print(f"Error loading {dump_path.name}: {e}")

print("=== Most common spike tokens ===")
for token_info, count in token_counter.most_common(20):
    print(f"  {token_info}: {count}")

print("\n=== Context samples (first 10) ===")
for sample in context_samples[:10]:
    print(f"  {sample['dump']}: token={sample['token_text']} context={sample['context']}")

# Check if any are EOS
eos_count = sum(1 for d in dumps if torch.load(d, weights_only=False)['max_token_id'] == tokenizer.eos_token_id)
print(f"\n=== Special tokens ===")
print(f"Spikes at EOS token: {eos_count}")
