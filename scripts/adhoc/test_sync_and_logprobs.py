#!/usr/bin/env python3
"""Investigate ratio_max discrepancy: generation-time vs compute logprobs.

This script tests the hypothesis that generation-time logprobs differ from compute_logprobs,
which would explain the ratio_max discrepancy in GRPO training.

Run on secondary node with 8 GPUs:
    ssh ubuntu@172.31.17.116 "source /efs/rlvr-experiments/.venv/bin/activate && \
        CUDA_VISIBLE_DEVICES=0,1,2 python /efs/rlvr-experiments/scripts/adhoc/test_sync_and_logprobs.py"
"""

import os
import sys
import asyncio

# Set GPUs before any imports
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0,1,2")

sys.path.insert(0, "/efs/rlvr-experiments/src")

import torch
import yaml
import tempfile
import ray


def create_config(assets_path, tp_trainer=2, tp_vllm=1):
    """Create config with Titan trainer and vLLM rollout."""
    return {
        "run": {"name": "sync_logprobs_test"},
        "model": {"path": assets_path},
        "tokenizer": {
            "pretrained_model_name_or_path": assets_path,
            "use_fast": False,
        },
        "training": {
            "num_epochs": 1,
            "prompts_per_rollout_sync": 1,
            "prompts_per_reference_sync": 1,
            "prompts_per_optim_step": 1,
            "prompts_per_forward_backward": 1,
            "seq_len_buckets": [512],
            "completion_len_buckets": [256],
            "completions_per_micro_batch": 8,
        },
        "verifier": {"num_workers": 1},
        "loss": {"name": "grpo", "beta": 0.001, "eps": 0.2},
        "data": {"dataset": "gsm8k", "split": "train"},
        "data_iter": {"system_prompt": "", "assistant_prefix": "", "skip_chat_template": True},
        "sampling": {"temperature": 1.0, "max_tokens": 128, "n": 4, "logprobs": 0},
        "buffer": {"max_reads": 1},
        "roles": [
            {
                "name": "trainer",
                "kind": "titan",
                "config": {
                    "trainable": True,
                    "profiling": {"enable_profiling": False},
                    "metrics": {"log_freq": 1, "enable_tensorboard": False},
                    "model": {
                        "name": "qwen3",
                        "flavor": "1.7B",
                        "hf_assets_path": assets_path,
                    },
                    "optimizer": {"name": "AdamW", "lr": 1e-3, "eps": 1e-7},
                    "lr_scheduler": {"warmup_steps": 0},
                    "training": {
                        "seq_len": 512,
                        "dtype": "float16",
                        "mixed_precision_param": "float16",
                        "mixed_precision_reduce": "float32",
                    },
                    "parallelism": {
                        "data_parallel_replicate_degree": 1,
                        "data_parallel_shard_degree": 1,
                        "fsdp_reshard_after_forward": "default",
                        "tensor_parallel_degree": tp_trainer,
                        "context_parallel_degree": 1,
                        "disable_loss_parallel": False,
                    },
                    "checkpoint": {"enable": True, "initial_load_in_hf": True},
                    "activation_checkpoint": {"mode": "selective", "selective_ac_option": "op"},
                    "compile": {"enable": False},
                },
            },
            {
                "name": "rollout",
                "kind": "vllm",
                "config": {
                    "model": assets_path,
                    "max_concurrent_per_replica": 4,
                    "max_num_seqs": 16,
                    "tensor_parallel_size": tp_vllm,
                    "data_parallel_size": 1,
                    "max_model_len": 512,
                    "gpu_memory_utilization": 0.5,
                    "dtype": "float16",
                    "logprobs_mode": "raw_logprobs",
                    "enable_prefix_caching": False,
                    "enable_chunked_prefill": True,
                },
            },
        ],
        "sync": {
            "chunk_mb": 100,
            "wiring": [{"src": "trainer", "dst": "rollout"}],
        },
    }


async def test_generation_vs_compute_logprobs():
    """Compare generation-time logprobs vs compute_logprobs - THE KEY TEST."""
    print("\n" + "=" * 80)
    print("TEST: Generation-time vs Compute Logprobs")
    print("=" * 80)

    from rlvr_experiments.runtime import Runtime
    from rlvr_experiments.syncing import sync_titan_to_vllm
    from transformers import AutoTokenizer

    assets_path = "/efs/rlvr-experiments/assets/hf/Qwen3-1.7B-Base"

    config = create_config(assets_path, tp_trainer=2, tp_vllm=1)
    config_path = tempfile.mktemp(suffix=".yaml")
    with open(config_path, "w") as f:
        yaml.dump(config, f)

    runtime = await Runtime.from_plan(config_path)
    await runtime.start()

    trainer = runtime.roles["trainer"]
    rollout = runtime.roles["rollout"]
    tokenizer = AutoTokenizer.from_pretrained(assets_path, use_fast=False)

    # First sync so both have same weights
    print("\n--- Initial sync (Titan -> vLLM) ---")
    await sync_titan_to_vllm(trainer, rollout, wire_dtype="float16", trainer_version=trainer.version)

    # Test multiple prompts to get a range of results
    prompts = [
        "Question: What is 2+2?\nAnswer:",
        "Question: Calculate 15 times 7.\nAnswer:",
        "Question: What is the square root of 144?\nAnswer:",
    ]

    all_gen_vs_titan_ratios = []
    all_gen_vs_vllm_ratios = []
    all_vllm_vs_titan_ratios = []

    for prompt_idx, prompt in enumerate(prompts):
        prompt_ids = tokenizer.encode(prompt, return_tensors="pt")
        prompt_len = prompt_ids.shape[1]

        print(f"\n--- Prompt {prompt_idx}: {repr(prompt[:50])}... ---")

        # Generate with logprobs=1 to get per-token logprobs
        response = await rollout.generate_single(prompt, temperature=1.0, max_tokens=32, n=1, logprobs=1)

        completion_text = response.outputs[0].text
        completion_ids = tokenizer.encode(completion_text, add_special_tokens=False)
        full_ids = prompt_ids[0].tolist() + completion_ids

        print(f"Completion: {repr(completion_text[:50])}...")
        print(f"Completion length: {len(completion_ids)} tokens")

        # Extract generation-time logprobs
        gen_logprobs_list = []
        vllm_output = response.outputs[0]
        if hasattr(vllm_output, 'logprobs') and vllm_output.logprobs:
            for i, lp_dict in enumerate(vllm_output.logprobs):
                if lp_dict is not None:
                    token_id = vllm_output.token_ids[i]
                    if token_id in lp_dict:
                        lp_info = lp_dict[token_id]
                        lp_value = lp_info.logprob if hasattr(lp_info, 'logprob') else lp_info
                        gen_logprobs_list.append(lp_value)
                    else:
                        gen_logprobs_list.append(float('nan'))
                else:
                    gen_logprobs_list.append(float('nan'))

        gen_logprobs = torch.tensor(gen_logprobs_list)

        # Compute logprobs from both vLLM and Titan
        full_ids_tensor = torch.tensor([full_ids])
        completion_ids_tensor = torch.tensor([completion_ids])
        prompt_lens_tensor = torch.tensor([prompt_len])

        vllm_compute_logprobs = await rollout.compute_logprobs(
            full_ids_tensor,
            completion_ids_tensor,
            prompt_lens_tensor,
            temperature=1.0,
        )
        vllm_compute_logprobs = vllm_compute_logprobs.squeeze(0)

        titan_compute_logprobs = await trainer.compute_logprobs(
            full_ids_tensor.cuda(),
            completion_ids_tensor.cuda(),
            prompt_lens_tensor,
            temperature=1.0,
        )
        titan_compute_logprobs = titan_compute_logprobs.squeeze(0).cpu()

        # Comparison
        min_len = min(len(gen_logprobs), len(vllm_compute_logprobs), len(titan_compute_logprobs))

        # Gen vs vLLM compute
        gen_vs_vllm = gen_logprobs[:min_len] - vllm_compute_logprobs[:min_len]
        gen_vs_vllm_valid = gen_vs_vllm[~torch.isnan(gen_vs_vllm)]
        if len(gen_vs_vllm_valid) > 0:
            ratio_gen_vllm = torch.exp(gen_vs_vllm_valid)
            all_gen_vs_vllm_ratios.append(ratio_gen_vllm)
            print(f"  Gen vs vLLM compute: max_ratio={ratio_gen_vllm.max().item():.4f}, max_diff={gen_vs_vllm_valid.abs().max().item():.4f}")

        # Gen vs Titan compute
        gen_vs_titan = gen_logprobs[:min_len] - titan_compute_logprobs[:min_len]
        gen_vs_titan_valid = gen_vs_titan[~torch.isnan(gen_vs_titan)]
        if len(gen_vs_titan_valid) > 0:
            ratio_gen_titan = torch.exp(gen_vs_titan_valid)
            all_gen_vs_titan_ratios.append(ratio_gen_titan)
            print(f"  Gen vs Titan compute: max_ratio={ratio_gen_titan.max().item():.4f}, max_diff={gen_vs_titan_valid.abs().max().item():.4f}")

        # vLLM vs Titan compute
        vllm_vs_titan = vllm_compute_logprobs[:min_len] - titan_compute_logprobs[:min_len]
        ratio_vllm_titan = torch.exp(vllm_vs_titan)
        all_vllm_vs_titan_ratios.append(ratio_vllm_titan)
        print(f"  vLLM vs Titan compute: max_ratio={ratio_vllm_titan.max().item():.4f}, max_diff={vllm_vs_titan.abs().max().item():.4f}")

        # Per-token breakdown for first prompt
        if prompt_idx == 0:
            print("\n  Per-token breakdown:")
            print("  Pos | Token         | Gen LP  | vLLM LP | Titan LP | Gen-Titan")
            print("  " + "-" * 70)
            for i in range(min(10, min_len)):
                token_id = completion_ids[i]
                decoded = repr(tokenizer.decode([token_id]))[:12].ljust(12)
                gen_lp = gen_logprobs[i].item() if i < len(gen_logprobs) else float('nan')
                vllm_lp = vllm_compute_logprobs[i].item()
                titan_lp = titan_compute_logprobs[i].item()
                diff = gen_lp - titan_lp if not torch.isnan(torch.tensor(gen_lp)) else float('nan')
                print(f"  {i:3d} | {decoded} | {gen_lp:7.4f} | {vllm_lp:7.4f} | {titan_lp:8.4f} | {diff:9.4f}")

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    if all_gen_vs_vllm_ratios:
        combined = torch.cat(all_gen_vs_vllm_ratios)
        print(f"Generation vs vLLM compute_logprobs:")
        print(f"  Ratio max: {combined.max().item():.4f}")
        print(f"  Ratio min: {combined.min().item():.4f}")
        print(f"  Ratio mean: {combined.mean().item():.4f}")

    if all_gen_vs_titan_ratios:
        combined = torch.cat(all_gen_vs_titan_ratios)
        print(f"\nGeneration vs Titan compute_logprobs:")
        print(f"  Ratio max: {combined.max().item():.4f}")
        print(f"  Ratio min: {combined.min().item():.4f}")
        print(f"  Ratio mean: {combined.mean().item():.4f}")

    if all_vllm_vs_titan_ratios:
        combined = torch.cat(all_vllm_vs_titan_ratios)
        print(f"\nvLLM compute vs Titan compute:")
        print(f"  Ratio max: {combined.max().item():.4f}")
        print(f"  Ratio min: {combined.min().item():.4f}")
        print(f"  Ratio mean: {combined.mean().item():.4f}")

    os.remove(config_path)

    # Return key metrics
    if all_gen_vs_titan_ratios:
        return torch.cat(all_gen_vs_titan_ratios).max().item()
    return 1.0


async def test_titan_vllm_after_sync():
    """Test that Titan and vLLM match after sync using compute_logprobs."""
    print("\n" + "=" * 80)
    print("TEST: Titan vs vLLM compute_logprobs After Sync")
    print("=" * 80)

    from rlvr_experiments.runtime import Runtime
    from rlvr_experiments.syncing import sync_titan_to_vllm
    from transformers import AutoTokenizer

    assets_path = "/efs/rlvr-experiments/assets/hf/Qwen3-1.7B-Base"

    config = create_config(assets_path, tp_trainer=2, tp_vllm=1)
    config_path = tempfile.mktemp(suffix=".yaml")
    with open(config_path, "w") as f:
        yaml.dump(config, f)

    runtime = await Runtime.from_plan(config_path)
    await runtime.start()

    trainer = runtime.roles["trainer"]
    rollout = runtime.roles["rollout"]
    tokenizer = AutoTokenizer.from_pretrained(assets_path, use_fast=False)

    # Test input
    prompt = "Question: What is 2+2?\nAnswer: The answer is"
    prompt_ids = tokenizer.encode(prompt, return_tensors="pt")
    prompt_len = prompt_ids.shape[1]

    # Fixed completion for comparison
    completion = " 4."
    completion_ids = tokenizer.encode(completion, add_special_tokens=False)
    full_ids = prompt_ids[0].tolist() + completion_ids

    full_ids_tensor = torch.tensor([full_ids])
    completion_ids_tensor = torch.tensor([completion_ids])
    prompt_lens_tensor = torch.tensor([prompt_len])

    print(f"\nPrompt: {prompt}")
    print(f"Completion: {completion}")

    print("\n--- BEFORE sync ---")
    vllm_lp_before = await rollout.compute_logprobs(
        full_ids_tensor, completion_ids_tensor, prompt_lens_tensor, temperature=1.0
    )
    titan_lp_before = await trainer.compute_logprobs(
        full_ids_tensor.cuda(), completion_ids_tensor.cuda(), prompt_lens_tensor, temperature=1.0
    )

    diff_before = titan_lp_before.cpu() - vllm_lp_before
    ratio_before = torch.exp(diff_before)
    print(f"Titan vs vLLM:")
    print(f"  Max diff: {diff_before.abs().max().item():.6f}")
    print(f"  Max ratio: {ratio_before.max().item():.6f}")

    print("\n--- Syncing Titan -> vLLM ---")
    await sync_titan_to_vllm(trainer, rollout, wire_dtype="float16", trainer_version=trainer.version)

    print("\n--- AFTER sync ---")
    vllm_lp_after = await rollout.compute_logprobs(
        full_ids_tensor, completion_ids_tensor, prompt_lens_tensor, temperature=1.0
    )

    diff_after = titan_lp_before.cpu() - vllm_lp_after
    ratio_after = torch.exp(diff_after)
    print(f"Titan vs vLLM:")
    print(f"  Max diff: {diff_after.abs().max().item():.6f}")
    print(f"  Max ratio: {ratio_after.max().item():.6f}")

    # Check if vLLM changed
    vllm_change = vllm_lp_after - vllm_lp_before
    print(f"\nvLLM change from sync:")
    print(f"  Max change: {vllm_change.abs().max().item():.6f}")

    os.remove(config_path)
    return ratio_after.max().item()


async def main():
    if ray.is_initialized():
        ray.shutdown()

    num_gpus = len(os.environ.get("CUDA_VISIBLE_DEVICES", "0,1,2").split(","))
    print(f"Initializing Ray with {num_gpus} GPUs")
    ray.init(num_gpus=num_gpus)

    try:
        # First test: Titan vs vLLM after sync
        ratio_sync = await test_titan_vllm_after_sync()

        # Re-initialize Ray
        ray.shutdown()
        ray.init(num_gpus=num_gpus)

        # Second test: Generation vs compute logprobs
        ratio_gen = await test_generation_vs_compute_logprobs()

        print("\n" + "=" * 80)
        print("FINAL SUMMARY")
        print("=" * 80)
        print(f"Titan vs vLLM compute_logprobs after sync: ratio_max = {ratio_sync:.4f}")
        print(f"Generation-time vs Titan compute_logprobs: ratio_max = {ratio_gen:.4f}")

        if ratio_sync > 1.05:
            print("\n⚠️  Titan and vLLM have significant logprob differences even after sync!")
            print("   This suggests weights aren't being transferred correctly.")
        if ratio_gen > 1.05:
            print("\n⚠️  Generation-time logprobs differ from compute_logprobs!")
            print("   This is likely the cause of ratio_max discrepancy in training.")

    finally:
        ray.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
