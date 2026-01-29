"""Compare logprobs between TorchTitan and vLLM for the same model and inputs.

This is the RIGHT comparison - Titan vs vLLM, not HF vs vLLM.
"""

import torch
import os
import sys
sys.path.insert(0, '/efs/rlvr-experiments/src')

from transformers import AutoTokenizer, AutoModelForCausalLM
from rlvr_experiments.ops import compute_logprobs


def get_titan_style_logprobs(model, tokenizer, prompt, completion_ids, temperature=1.0):
    """Get logprobs using the same compute_logprobs function that Titan uses."""
    # Encode prompt
    prompt_ids = tokenizer.encode(prompt, return_tensors='pt').to(model.device)
    prompt_len = prompt_ids.shape[1]

    # Full sequence = prompt + completion
    full_ids = torch.cat([prompt_ids, completion_ids.unsqueeze(0).to(model.device)], dim=1)

    with torch.no_grad():
        outputs = model(full_ids)
        logits = outputs.logits  # [1, seq_len, vocab_size]

    # Use the exact same compute_logprobs function that Titan uses
    # completion_ids is shape [comp_len], need to make it [1, comp_len]
    comp_ids_2d = completion_ids.unsqueeze(0).to(model.device)
    prompt_lens = torch.tensor([prompt_len], device=model.device)

    logprobs, _ = compute_logprobs(
        logits,
        comp_ids_2d,
        temperature=temperature,
        align=True,
        prompt_lens=prompt_lens,
    )

    return logprobs.squeeze(0).cpu()


def get_vllm_logprobs(model_path, prompt, completion_text, tp_size=1, temperature=1.0):
    """Get logprobs using vLLM with prompt_logprobs mode."""
    from vllm import LLM, SamplingParams

    # Initialize vLLM with specified TP
    llm = LLM(
        model=model_path,
        tensor_parallel_size=tp_size,
        dtype='float16',
        gpu_memory_utilization=0.5,
        max_model_len=512,
    )

    # Get tokenizer
    tokenizer = llm.get_tokenizer()

    # Encode the full text
    full_text = prompt + completion_text
    full_ids = tokenizer.encode(full_text)
    prompt_ids = tokenizer.encode(prompt)
    completion_ids = full_ids[len(prompt_ids):]

    # Use prompt_logprobs to get logprobs for all tokens
    # NOTE: vLLM returns logprobs BEFORE temperature scaling (raw logits -> log_softmax)
    sampling_params = SamplingParams(
        prompt_logprobs=1,  # Get logprobs for prompt tokens
        max_tokens=1,  # We don't need generation
        temperature=temperature,
    )

    outputs = llm.generate([full_text], sampling_params)

    # Extract logprobs for completion tokens from prompt_logprobs
    prompt_logprobs = outputs[0].prompt_logprobs

    # prompt_logprobs[i] contains logprob for token i given tokens 0..i-1
    # We want logprobs for completion tokens, which start at len(prompt_ids)
    comp_logprobs = []
    for i in range(len(prompt_ids), len(full_ids)):
        token_id = full_ids[i]
        if prompt_logprobs[i] is not None and token_id in prompt_logprobs[i]:
            comp_logprobs.append(prompt_logprobs[i][token_id].logprob)
        else:
            comp_logprobs.append(float('nan'))

    del llm
    torch.cuda.empty_cache()

    return torch.tensor(comp_logprobs), torch.tensor(completion_ids)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, default="0")
    parser.add_argument('--tp', type=int, default=1)
    parser.add_argument('--temperature', type=float, default=1.0)
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    model_path = "/efs/rlvr-experiments/assets/hf/Qwen3-1.7B-Base"

    # Test prompt and completion - use a real MATH-style completion with boxed
    prompt = """Problem:
Find all real solutions to $x^4+(2-x)^4=34$.

Solution:"""
    completion = """ Let me solve this step by step.

Let $y = x - 1$. Then $x = y + 1$ and $2 - x = 1 - y$.

So we need $(y+1)^4 + (1-y)^4 = 34$.

Expanding:
$(y+1)^4 = y^4 + 4y^3 + 6y^2 + 4y + 1$
$(1-y)^4 = y^4 - 4y^3 + 6y^2 - 4y + 1$

Adding: $2y^4 + 12y^2 + 2 = 34$
$y^4 + 6y^2 - 16 = 0$
$(y^2 + 8)(y^2 - 2) = 0$

So $y^2 = 2$ giving $y = \\pm\\sqrt{2}$.

Therefore $x = 1 \\pm \\sqrt{2}$.

The answer is $\\boxed{1-\\sqrt{2}, 1+\\sqrt{2}}$."""

    print(f"Testing with TP={args.tp} on GPU(s) {os.environ['CUDA_VISIBLE_DEVICES']}")
    print(f"Temperature: {args.temperature}")
    print(f"Prompt: {repr(prompt)}")
    print(f"Completion: {repr(completion)}")
    print()

    # Get vLLM logprobs FIRST (before loading HF model to avoid memory issues)
    print("Getting vLLM logprobs...")
    vllm_logprobs, completion_ids = get_vllm_logprobs(
        model_path, prompt, completion, tp_size=args.tp, temperature=args.temperature
    )
    print(f"vLLM logprobs: {vllm_logprobs}")

    # Get Titan-style logprobs
    print("\nGetting Titan-style logprobs...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map='cuda:0'
    )
    titan_logprobs = get_titan_style_logprobs(
        model, tokenizer, prompt, completion_ids, temperature=args.temperature
    )
    print(f"Titan logprobs: {titan_logprobs}")

    # Compare
    diff = vllm_logprobs - titan_logprobs
    ratio = torch.exp(diff)

    print(f"\nDifference (vLLM - Titan): {diff}")
    print(f"Ratio (exp(diff)):         {ratio}")
    print(f"\nStats:")
    print(f"  Max abs diff: {diff.abs().max().item():.6f}")
    print(f"  Mean abs diff: {diff.abs().mean().item():.6f}")
    print(f"  Max ratio: {ratio.max().item():.6f}")
    print(f"  Min ratio: {ratio.min().item():.6f}")

    # Also check if the issue is dtype-related
    print("\n\nNow testing with bfloat16 (Titan uses bfloat16 sometimes)...")
    model_bf16 = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map='cuda:0'
    )
    titan_logprobs_bf16 = get_titan_style_logprobs(
        model_bf16, tokenizer, prompt, completion_ids, temperature=args.temperature
    )
    print(f"Titan logprobs (bf16): {titan_logprobs_bf16}")

    diff_bf16 = vllm_logprobs - titan_logprobs_bf16
    print(f"Diff (vLLM - Titan bf16): {diff_bf16}")
    print(f"Max abs diff (bf16): {diff_bf16.abs().max().item():.6f}")

if __name__ == '__main__':
    main()
