#!/usr/bin/env python3
"""
Simulate the exact training flow to reproduce the ref_logprobs corruption bug.

Flow:
1. Generate completions with vLLM (like rollout)
2. Create RolloutSample from outputs
3. Compute ref_logprobs using VLLMHandle-like interface
4. Compare rollout.logprobs vs ref_logprobs
"""

import os
import sys
import torch
import asyncio
from dataclasses import dataclass
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt

os.environ["CUDA_VISIBLE_DEVICES"] = "7"

# Add project to path
sys.path.insert(0, "/efs/rlvr-experiments/src")
from rlvr_experiments.algorithms.grpo import RolloutSample

model_path = "/efs/rlvr-experiments/assets/hf/Qwen3-1.7B-Base"
tokenizer = AutoTokenizer.from_pretrained(model_path)
pad_token_id = tokenizer.eos_token_id

print("=== Loading vLLM ===")
llm = LLM(
    model=model_path,
    tensor_parallel_size=1,
    dtype="float16",
    gpu_memory_utilization=0.5,
    max_model_len=2561,
)

# Create a math-like prompt
prompt = """Problem:
A large circle has a radius of 6 meters. Two smaller circles are inscribed inside. What is the radius of the two smallest circles if they are one fourth of the large circle?

Solution:"""
prompt_tokens = tokenizer.encode(prompt, add_special_tokens=False)
prompt_len = len(prompt_tokens)

print(f"Prompt length: {prompt_len} tokens")

# Step 1: Generate completions (like rollout)
print("\n=== Step 1: Generate completions ===")
sp_gen = SamplingParams(
    max_tokens=100,
    temperature=1.0,
    top_p=0.95,
    top_k=20,
    logprobs=0,  # Get logprobs for sampled tokens
    n=4,  # Multiple completions like GRPO
    stop=["Problem:", "\n\n\n"],
)
outputs = llm.generate([prompt], sampling_params=sp_gen, use_tqdm=False)
response = outputs[0]

print(f"Number of completions: {len(response.outputs)}")
for i, o in enumerate(response.outputs):
    print(f"Completion {i}: {len(o.token_ids)} tokens, finish_reason={o.finish_reason}")
    print(f"  Text: {tokenizer.decode(o.token_ids)[:80]}...")

# Step 2: Create RolloutSample (exactly as training does)
print("\n=== Step 2: Create RolloutSample ===")
rollout_sample = RolloutSample.from_vllm(response, pad_token_id)

print(f"input_ids shape: {rollout_sample.input_ids.shape}")
print(f"completion_ids shape: {rollout_sample.completion_ids.shape}")
print(f"logprobs shape: {rollout_sample.logprobs.shape}")
print(f"prompt_len: {rollout_sample.prompt_len}")
print(f"completion_lens: {rollout_sample.completion_lens}")

# Step 3: Compute ref_logprobs (simulating VLLMHandle.compute_logprobs)
print("\n=== Step 3: Compute ref_logprobs ===")

def compute_ref_logprobs(llm, input_ids, completion_ids, prompt_lens, temperature=1.0):
    """Simulate VLLMHandle.compute_logprobs"""
    batch_size = input_ids.shape[0]
    completion_len = completion_ids.shape[1]

    # Build token_ids_list (exactly as VLLMHandle does)
    token_ids_list = []
    prompt_lens_list = prompt_lens.tolist() if isinstance(prompt_lens, torch.Tensor) else prompt_lens

    for i in range(batch_size):
        seq = input_ids[i].tolist()
        plen = prompt_lens_list[i]
        actual_seq_len = plen + completion_len
        token_ids_list.append(seq[:actual_seq_len])
        print(f"  Seq {i}: len(input_ids[{i}])={len(seq)}, prompt_len={plen}, actual_seq_len={actual_seq_len}, final_len={len(seq[:actual_seq_len])}")

    # Compute logprobs with prompt_logprobs feature
    sp = SamplingParams(max_tokens=1, prompt_logprobs=1, temperature=temperature)

    result = torch.zeros(batch_size, completion_len, dtype=torch.float32)

    for i, (token_ids, plen) in enumerate(zip(token_ids_list, prompt_lens_list)):
        prompt_obj = TokensPrompt(prompt_token_ids=token_ids)
        outputs = llm.generate([prompt_obj], sampling_params=sp, use_tqdm=False)
        out = outputs[0]

        if out.prompt_logprobs is None:
            print(f"  Seq {i}: No prompt_logprobs!")
            continue

        lps = []
        for j in range(plen, len(token_ids)):
            if j < len(out.prompt_logprobs) and out.prompt_logprobs[j] is not None:
                tok_id = token_ids[j]
                lp_dict = out.prompt_logprobs[j]
                if tok_id in lp_dict:
                    lps.append(lp_dict[tok_id].logprob)
                else:
                    lps.append(-100.0)
            else:
                lps.append(-100.0)

        actual_len = min(len(lps), completion_len)
        result[i, :actual_len] = torch.tensor(lps[:actual_len], dtype=torch.float32)
        print(f"  Seq {i}: computed {len(lps)} logprobs, stored {actual_len}")

    return result

prompt_lens = torch.tensor([rollout_sample.prompt_len] * rollout_sample.input_ids.shape[0])
ref_logprobs = compute_ref_logprobs(llm, rollout_sample.input_ids, rollout_sample.completion_ids, prompt_lens)

print(f"ref_logprobs shape: {ref_logprobs.shape}")

# Step 4: Compare rollout.logprobs vs ref_logprobs
print("\n=== Step 4: Comparison ===")

for seq_idx in range(rollout_sample.logprobs.shape[0]):
    actual_len = rollout_sample.completion_lens[seq_idx]
    rollout_lps = rollout_sample.logprobs[seq_idx, :actual_len]
    ref_lps = ref_logprobs[seq_idx, :actual_len]

    diffs = torch.abs(rollout_lps - ref_lps)
    max_diff = diffs.max().item()
    max_diff_pos = diffs.argmax().item()

    print(f"\nSequence {seq_idx} (length {actual_len}):")
    print(f"  Max diff: {max_diff:.4f} at position {max_diff_pos}")

    if max_diff > 0.5:
        print(f"  *** LARGE DIFFERENCE DETECTED ***")
        # Show details around max diff position
        for pos in range(max(0, max_diff_pos-2), min(actual_len, max_diff_pos+3)):
            tok_id = rollout_sample.completion_ids[seq_idx, pos].item()
            tok_text = tokenizer.decode([tok_id])
            r_lp = rollout_lps[pos].item()
            ref_lp = ref_lps[pos].item()
            diff = abs(r_lp - ref_lp)
            marker = " <--" if pos == max_diff_pos else ""
            print(f"    pos {pos}: {repr(tok_text):15s} rollout={r_lp:8.4f} ref={ref_lp:8.4f} diff={diff:.4f}{marker}")
    else:
        # Show first few logprobs
        print(f"  First 5 rollout logprobs: {rollout_lps[:5].tolist()}")
        print(f"  First 5 ref logprobs: {ref_lps[:5].tolist()}")

# Overall statistics
print("\n=== Overall Statistics ===")
for seq_idx in range(rollout_sample.logprobs.shape[0]):
    actual_len = rollout_sample.completion_lens[seq_idx]
    rollout_lps = rollout_sample.logprobs[seq_idx, :actual_len]
    ref_lps = ref_logprobs[seq_idx, :actual_len]
    diffs = torch.abs(rollout_lps - ref_lps)

    large_diffs = (diffs > 0.5).sum().item()
    print(f"Seq {seq_idx}: {large_diffs} positions with diff > 0.5 out of {actual_len}")

print("\nDone!")
