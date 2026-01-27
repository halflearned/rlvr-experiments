#!/usr/bin/env python3
"""
Verify reference logprobs by loading the base model independently and
computing logprobs for sequences from KL spike dumps.

This helps diagnose whether the reference logprobs computed during training
match what we'd get from a fresh vLLM instance with the same base model.
"""

import argparse
import os
import torch
from pathlib import Path
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


def load_dump(dump_path: str) -> dict:
    """Load a KL spike dump file."""
    return torch.load(dump_path, weights_only=False)


def compute_logprobs_standalone(
    llm: LLM,
    token_ids: list[int],
    prompt_len: int,
    temperature: float = 1.0,
) -> list[float]:
    """Compute logprobs using vLLM's prompt_logprobs feature.

    Matches the settings used in vllm_engine_actor.py get_logprobs().
    """
    from vllm.inputs import TokensPrompt

    # Match exact params from vllm_engine_actor.py get_logprobs()
    sp = SamplingParams(
        max_tokens=1,  # vLLM requires at least 1
        prompt_logprobs=1,  # Get logprob of each prompt token
        temperature=temperature,
    )

    prompt = TokensPrompt(prompt_token_ids=token_ids)
    outputs = llm.generate([prompt], sampling_params=sp)

    if not outputs or not outputs[0].prompt_logprobs:
        return [0.0] * (len(token_ids) - prompt_len)

    result = outputs[0]
    completion_logprobs = []

    # Extract logprobs for completion tokens (after prompt_len)
    # prompt_logprobs[i] contains logprob info for token at position i
    # The logprob at position i is P(token[i] | token[0:i])
    for i in range(prompt_len, len(token_ids)):
        if i < len(result.prompt_logprobs) and result.prompt_logprobs[i] is not None:
            token_id = token_ids[i]
            lp_dict = result.prompt_logprobs[i]
            if token_id in lp_dict:
                completion_logprobs.append(lp_dict[token_id].logprob)
            else:
                # Token not in top-k, use a small default
                completion_logprobs.append(-100.0)
        else:
            completion_logprobs.append(-100.0)

    return completion_logprobs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dump_path", type=str, help="Path to KL spike dump file")
    parser.add_argument("--model", type=str,
                       default="/efs/rlvr-experiments/assets/hf/Qwen3-1.7B-Base",
                       help="Path to base model")
    parser.add_argument("--gpu", type=int, default=0, help="GPU to use")
    args = parser.parse_args()

    # Set GPU before importing vLLM internals
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    # Load dump
    print(f"Loading dump: {args.dump_path}")
    dump = load_dump(args.dump_path)

    print(f"\n=== Dump Info ===")
    print(f"kl_max: {dump['kl_max']:.4f}")
    print(f"max_batch_idx: {dump['max_batch_idx']}")
    print(f"max_token_idx: {dump['max_token_idx']}")
    print(f"max_token_id: {dump['max_token_id']}")

    # Get the sequence
    response_seq = dump['response_seq'].tolist()
    ref_logprobs_seq = dump['ref_logprobs_seq'].tolist()
    trainer_logprobs_seq = dump['trainer_logprobs_seq'].tolist()
    rollout_logprobs_seq = dump['rollout_logprobs_seq'].tolist()
    mask_seq = dump['mask_seq'].tolist()

    # Find valid token count
    valid_count = sum(1 for m in mask_seq if m > 0.5)
    print(f"Valid tokens (mask=1): {valid_count}")
    print(f"Response seq length: {len(response_seq)}")

    # Load tokenizer for decoding
    print(f"\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    # Decode the problematic token
    max_idx = dump['max_token_idx']
    token_id = dump['max_token_id']
    token_text = tokenizer.decode([token_id])
    print(f"\nProblematic token at position {max_idx}:")
    print(f"  token_id: {token_id}")
    print(f"  token_text: {repr(token_text)}")
    print(f"  trainer_lp (from dump): {dump['trainer_logprob_at_max']:.4f}")
    print(f"  ref_lp (from dump): {dump['ref_logprob_at_max']:.4f}")
    print(f"  rollout_lp (from dump): {dump['rollout_logprob_at_max']:.4f}")

    # Show context around the problematic token
    context_start = max(0, max_idx - 5)
    context_end = min(len(response_seq), max_idx + 5)
    context_tokens = response_seq[context_start:context_end]
    context_text = tokenizer.decode(context_tokens)
    print(f"\nContext around problematic token:")
    print(f"  {repr(context_text)}")

    # We need to reconstruct the full input_ids (prompt + completion)
    # The dump only has completion_ids (response_seq), so we need to get prompt somehow
    # For now, let's just compute logprobs on the response_seq assuming prompt_len=0
    # This won't match exactly but will show if there's a systematic issue

    print(f"\n=== Loading vLLM with base model ===")
    print(f"Settings: dtype=float16, max_model_len=2561 (matching reference config)")

    # Match reference vLLM settings from config
    llm = LLM(
        model=args.model,
        tensor_parallel_size=1,
        dtype="float16",  # Match reference config
        gpu_memory_utilization=0.90,  # Match reference config
        max_model_len=2561,  # Match reference config
        enable_prefix_caching=False,  # Match reference config
        enable_chunked_prefill=True,  # Match reference config
    )

    # Compute logprobs treating the entire response as a "prompt"
    # This gives us P(token[i] | token[0:i]) for each position
    print(f"\n=== Computing logprobs independently ===")

    # Only use valid tokens (non-padding)
    valid_response = response_seq[:valid_count]

    standalone_logprobs = compute_logprobs_standalone(
        llm,
        valid_response,
        prompt_len=0,  # Treat all as "completion" to get logprobs for all positions
    )

    print(f"\n=== Comparison at problematic position {max_idx} ===")
    if max_idx < len(standalone_logprobs):
        standalone_lp = standalone_logprobs[max_idx]
        ref_lp = ref_logprobs_seq[max_idx]
        trainer_lp = trainer_logprobs_seq[max_idx]
        rollout_lp = rollout_logprobs_seq[max_idx]

        print(f"Position {max_idx} (token {repr(token_text)}):")
        print(f"  standalone (fresh vLLM): {standalone_lp:.4f}")
        print(f"  ref (from dump):         {ref_lp:.4f}")
        print(f"  trainer (from dump):     {trainer_lp:.4f}")
        print(f"  rollout (from dump):     {rollout_lp:.4f}")
        print(f"\nDifferences:")
        print(f"  |standalone - ref|:     {abs(standalone_lp - ref_lp):.4f}")
        print(f"  |standalone - trainer|: {abs(standalone_lp - trainer_lp):.4f}")
        print(f"  |ref - trainer|:        {abs(ref_lp - trainer_lp):.4f}")

    # Also show a few other positions for comparison
    print(f"\n=== Comparison at other positions ===")
    positions_to_check = [0, 10, 20, max_idx-1, max_idx, max_idx+1, valid_count-1]
    positions_to_check = [p for p in positions_to_check if 0 <= p < min(len(standalone_logprobs), valid_count)]
    positions_to_check = sorted(set(positions_to_check))

    print(f"{'Pos':<6} {'Token':<15} {'Standalone':<12} {'Ref':<12} {'Trainer':<12} {'Rollout':<12} {'|Ref-Stnd|':<12}")
    print("-" * 90)
    for pos in positions_to_check:
        tok_id = response_seq[pos]
        tok_text = tokenizer.decode([tok_id])[:12]
        standalone = standalone_logprobs[pos] if pos < len(standalone_logprobs) else float('nan')
        ref = ref_logprobs_seq[pos]
        trainer = trainer_logprobs_seq[pos]
        rollout = rollout_logprobs_seq[pos]
        diff = abs(standalone - ref) if pos < len(standalone_logprobs) else float('nan')

        marker = " <-- SPIKE" if pos == max_idx else ""
        print(f"{pos:<6} {repr(tok_text):<15} {standalone:<12.4f} {ref:<12.4f} {trainer:<12.4f} {rollout:<12.4f} {diff:<12.4f}{marker}")


if __name__ == "__main__":
    main()
