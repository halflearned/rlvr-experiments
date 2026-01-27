#!/usr/bin/env python3
"""
Test how stop tokens affect logprobs in vLLM generation vs prompt_logprobs mode.

Hypothesis: When generation hits a stop token, the token_ids and logprobs
might be handled differently between generation and prompt_logprobs modes.
"""

import os
import torch
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

# Create a prompt that will likely hit a stop token
prompt = "Problem:\nWhat is 2 + 2?\n\nSolution: The answer is 4.\n\nProblem:"
prompt_tokens = tokenizer.encode(prompt, add_special_tokens=False)

# Check what "Problem:" looks like in tokens
problem_tokens = tokenizer.encode("Problem:", add_special_tokens=False)
print(f"'Problem:' tokens: {problem_tokens}")

# Generate with stop tokens (like training does)
print("\n=== Generating with stop tokens ===")
sp_gen = SamplingParams(
    max_tokens=50,
    temperature=1.0,
    top_p=0.95,
    top_k=20,
    logprobs=0,
    stop=["Problem:", "\n\n\n"],
)

# Actually let me use a prompt that won't immediately hit stop
simpler_prompt = "The answer to 2+2 is"
simpler_prompt_tokens = tokenizer.encode(simpler_prompt, add_special_tokens=False)
prompt_len = len(simpler_prompt_tokens)

print(f"Prompt: {repr(simpler_prompt)}")
print(f"Prompt tokens: {simpler_prompt_tokens}")

# Generate without stop (baseline)
print("\n=== Generate WITHOUT stop tokens ===")
sp_no_stop = SamplingParams(
    max_tokens=20,
    temperature=1.0,
    top_p=0.95,
    top_k=20,
    logprobs=0,
)
outputs = llm.generate([simpler_prompt], sampling_params=sp_no_stop, use_tqdm=False)
gen_no_stop = outputs[0].outputs[0]

print(f"Generated tokens: {list(gen_no_stop.token_ids)}")
print(f"Generated text: {repr(tokenizer.decode(gen_no_stop.token_ids))}")
print(f"Number of logprobs dicts: {len(gen_no_stop.logprobs)}")
print(f"Finish reason: {gen_no_stop.finish_reason}")

# Extract logprobs
gen_lps_no_stop = []
for j, lp_dict in enumerate(gen_no_stop.logprobs):
    tok_id = gen_no_stop.token_ids[j]
    if tok_id in lp_dict:
        gen_lps_no_stop.append(lp_dict[tok_id].logprob)
    else:
        gen_lps_no_stop.append(None)
print(f"Logprobs: {gen_lps_no_stop}")

# Generate WITH stop tokens
print("\n=== Generate WITH stop tokens ===")
sp_with_stop = SamplingParams(
    max_tokens=20,
    temperature=1.0,
    top_p=0.95,
    top_k=20,
    logprobs=0,
    stop=["\n"],  # Stop at newline
)
outputs = llm.generate([simpler_prompt], sampling_params=sp_with_stop, use_tqdm=False)
gen_with_stop = outputs[0].outputs[0]

print(f"Generated tokens: {list(gen_with_stop.token_ids)}")
print(f"Generated text: {repr(tokenizer.decode(gen_with_stop.token_ids))}")
print(f"Number of logprobs dicts: {len(gen_with_stop.logprobs)}")
print(f"Finish reason: {gen_with_stop.finish_reason}")

# Check if stop token is in token_ids
newline_token = tokenizer.encode("\n", add_special_tokens=False)[0]
print(f"Newline token ID: {newline_token}")
if newline_token in gen_with_stop.token_ids:
    idx = list(gen_with_stop.token_ids).index(newline_token)
    print(f"Newline found at index {idx}")
else:
    print("Newline NOT in token_ids")

# Extract logprobs
gen_lps_with_stop = []
for j, lp_dict in enumerate(gen_with_stop.logprobs):
    tok_id = gen_with_stop.token_ids[j]
    if tok_id in lp_dict:
        gen_lps_with_stop.append(lp_dict[tok_id].logprob)
    else:
        gen_lps_with_stop.append(None)
print(f"Logprobs: {gen_lps_with_stop}")

# Now compute prompt_logprobs for the same tokens (simulating reference model)
print("\n=== Prompt logprobs for same tokens ===")
full_seq_with_stop = list(simpler_prompt_tokens) + list(gen_with_stop.token_ids)

sp_prompt = SamplingParams(
    max_tokens=1,
    prompt_logprobs=1,
    temperature=1.0,
)
outputs = llm.generate([TokensPrompt(prompt_token_ids=full_seq_with_stop)], sampling_params=sp_prompt, use_tqdm=False)
prompt_result = outputs[0]

prompt_lps = []
for i in range(prompt_len, len(full_seq_with_stop)):
    if i < len(prompt_result.prompt_logprobs) and prompt_result.prompt_logprobs[i] is not None:
        tok_id = full_seq_with_stop[i]
        lp_dict = prompt_result.prompt_logprobs[i]
        if tok_id in lp_dict:
            prompt_lps.append(lp_dict[tok_id].logprob)
        else:
            prompt_lps.append(-100.0)
    else:
        prompt_lps.append(None)
print(f"Prompt logprobs: {prompt_lps}")

# Compare
print("\n=== Comparison (generation vs prompt_logprobs) ===")
print(f"{'Pos':>4} | {'Gen':>10} | {'Prompt':>10} | {'Diff':>8} | Token")
print("-" * 60)
for j in range(len(gen_with_stop.token_ids)):
    gen_lp = gen_lps_with_stop[j] if j < len(gen_lps_with_stop) else None
    prompt_lp = prompt_lps[j] if j < len(prompt_lps) else None
    tok_id = gen_with_stop.token_ids[j]
    tok_text = tokenizer.decode([tok_id])

    if gen_lp is not None and prompt_lp is not None and prompt_lp != -100.0:
        diff = abs(gen_lp - prompt_lp)
        print(f"{j:4d} | {gen_lp:10.4f} | {prompt_lp:10.4f} | {diff:8.4f} | {repr(tok_text)}")
    else:
        print(f"{j:4d} | {'N/A':>10} | {'N/A':>10} | {'N/A':>8} | {repr(tok_text)}")

# Additional test: check EOS token handling
print("\n=== EOS token test ===")
eos_id = tokenizer.eos_token_id
print(f"EOS token ID: {eos_id}")

# Generate until EOS
sp_eos = SamplingParams(
    max_tokens=100,
    temperature=0.5,  # Lower temp for faster EOS
    logprobs=0,
)
outputs = llm.generate([simpler_prompt], sampling_params=sp_eos, use_tqdm=False)
gen_eos = outputs[0].outputs[0]

print(f"Finish reason: {gen_eos.finish_reason}")
print(f"Number of tokens: {len(gen_eos.token_ids)}")
if eos_id in gen_eos.token_ids:
    idx = list(gen_eos.token_ids).index(eos_id)
    print(f"EOS found at index {idx}")
else:
    print("EOS NOT in token_ids")

print("\nDone!")
