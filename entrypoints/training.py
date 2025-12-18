import argparse
import asyncio
import time
import torch

from rlvr_experiments.runtime import Runtime
from rlvr_experiments.syncing import sync_titan_to_vllm, sync_titan_to_titan
from rlvr_experiments.losses import GRPOLoss
from rlvr_experiments.verifiers import MathVerifier
from rlvr_experiments.vllm_utils import VLLMOutput

from transformers import AutoTokenizer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="RLVR Experiments Entrypoint")
    p.add_argument(
        "config",
        type=str,
        help="Path to the YAML configuration file.",
    )
    return p.parse_args()


def apply_template(prompt: str, tokenizer, tokenize=False) -> str:
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=tokenize,
        enable_thinking=False,
        add_generation_prompt=True,
    )

def scramble(sentence):
    import random
    words = sentence.split()
    random.shuffle(words)
    return " ".join(words)


async def main() -> None:
    args = parse_args()

    runtime = await Runtime.from_plan(args.config)
    await runtime.start()

    # Get roles
    trainer = runtime.roles["trainer"]
    reference = runtime.roles["reference"]
    rollout = runtime.roles["rollout"]

    # Ensure the reference starts identical to the trainer checkpoint.
    # This is especially important when trainer TP != reference TP, since any
    # mismatch at step 0 can make the KL term explode.
    await sync_titan_to_titan(trainer, reference)
    print("Synchronized trainer weights to reference model (startup).")

    loss_fn = GRPOLoss(beta=0.1, eps=0.2)

    verifier = MathVerifier()

    avg_rewards = []

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B", use_fast=False)
    system_prompt = r"Answer this math question with a number within \\boxed{}. "
    problem_prompts = [
        "Simplify the expression  ((7/12) + (5/18)) / (31/36)"
    ]   
    problem_answers = [
        "1",
    ]

    templates = [apply_template(system_prompt + p + "?", tokenizer) for p in problem_prompts]

    with open("reward_log.txt", "w") as f:
        f.write("")

    num_iterations = 200
    for i in range(num_iterations):
        iter_start = time.perf_counter()
        print("\n\n" + "*" * 20 + f" ITERATION {i+1}/{num_iterations} " + "*" * 20)

        if i % 10 == 0 and i > 0:
            t_sync_ref_start = time.perf_counter()
            await sync_titan_to_titan(trainer, reference)
            t_sync_ref_end = time.perf_counter()
            print(f"[TIMING] sync_titan_to_titan: {t_sync_ref_end - t_sync_ref_start:.3f}s")
            print("Synchronized trainer weights to reference model.")

        if i % 5 == 0:
            # Sync trainer weights to rollout vLLM
            t_sync_vllm_start = time.perf_counter()
            await sync_titan_to_vllm(trainer, rollout)
            t_sync_vllm_end = time.perf_counter()
            print(f"[TIMING] sync_titan_to_vllm: {t_sync_vllm_end - t_sync_vllm_start:.3f}s")
            print("Synchronized trainer weights to rollout model.")

        # Generate responses from rollout vLLM
        # https://qwen.readthedocs.io/en/latest/deployment/vllm.html#thinking-non-thinking-modes
        t_gen_start = time.perf_counter()
        sampling_params = dict(temperature=0.6, top_p=0.95, top_k=20, max_tokens=512, logprobs=0, n=20)
        responses = await rollout.generate(templates, **sampling_params)
        t_gen_end = time.perf_counter()
        print(f"[TIMING] vLLM generation: {t_gen_end - t_gen_start:.3f}s")
        response = responses[0]
        print("generated:", response.outputs[0].text)

        # For now, only support single prompt per batch
        t_vllm_parse_start = time.perf_counter()
        vllm_output = VLLMOutput(response)

        # Prepare inputs for trainer and reference models
        full_input_ids, completion_ids, completion_mask, completion_logprobs = vllm_output.get_tensors(tokenizer)
        t_vllm_parse_end = time.perf_counter()
        print(f"[TIMING] VLLMOutput parsing: {t_vllm_parse_end - t_vllm_parse_start:.3f}s")

        # Get logprobs from reference model (computed inside actor to avoid serializing huge logits)
        input_dict = {
            "input": full_input_ids,
        }
        torch.cuda.synchronize()
        t_ref_start = time.perf_counter()
        reference_logprobs = await reference.compute_logprobs_step(input_dict, completion_ids)
        torch.cuda.synchronize()
        t_ref_end = time.perf_counter()
        print(f"[TIMING] reference logprobs: {t_ref_end - t_ref_start:.3f}s")
        print(f"got reference logprobs: shape={reference_logprobs.shape}, min={reference_logprobs.min():.4f}, max={reference_logprobs.max():.4f}, num_zeros={(reference_logprobs == 0).sum()}")

        if i == 0:
            # Use compute_logprobs_step to avoid serializing huge logits tensor through Ray
            trainer_logprobs_probe = await trainer.compute_logprobs_step(input_dict, completion_ids)
            max_abs_diff = (
                trainer_logprobs_probe.detach().cpu() - reference_logprobs.detach().cpu()
            ).abs().max().item()
            print(f"startup trainer/reference logprob max_abs_diff={max_abs_diff:.6f}")

        # Compute rewards
        t_reward_start = time.perf_counter()
        rewards = verifier.verify_batch(
            responses=vllm_output.completion_texts(),
            targets=problem_answers * len(response.outputs),
            return_dtype=torch.float32,
        )
        t_reward_end = time.perf_counter()
        print(f"[TIMING] reward computation: {t_reward_end - t_reward_start:.3f}s")
        print(f"rewards: {rewards}")

        avg_rewards.append(rewards.mean().item())
        print(f"avg rewards so far: {avg_rewards}")
        with open("reward_log.txt", "a") as f:
            f.write(f"{rewards.mean().item()}\n")

        if torch.allclose(rewards, rewards[0]):
            print("All rewards are identical, skipping this batch.")
            continue


        # GRPO loss - passes logits to loss function which computes logprobs internally
        # Args after input_dict: response (completion_ids), ref_logprobs, rollout_logprobs, rewards, padding_mask
        torch.cuda.synchronize()
        t_loss_start = time.perf_counter()
        loss = await trainer.compute_loss_and_backward_step(
            loss_fn, input_dict, completion_ids, reference_logprobs, completion_logprobs,
            rewards, padding_mask=completion_mask
        )
        torch.cuda.synchronize()
        t_loss_end = time.perf_counter()
        print(f"[TIMING] compute_loss_and_backward: {t_loss_end - t_loss_start:.3f}s")
        print(f"computed loss and backward pass. loss: {loss}")

        # Update trainer model
        torch.cuda.synchronize()
        t_opt_start = time.perf_counter()
        grad_norm = await trainer.optimizer_step()
        torch.cuda.synchronize()
        t_opt_end = time.perf_counter()
        print(f"[TIMING] optimizer step: {t_opt_end - t_opt_start:.3f}s")
        print(f"optimizer step done. grad_norm={grad_norm:.6f}")

        # Model update check - commented out for now since we've verified gradients flow
        # and the varying sequence lengths cause shape mismatches across iterations
        # torch.cuda.synchronize()
        # t_check_start = time.perf_counter()
        # trainer_logprobs_check = await trainer.compute_logprobs_step(input_dict, completion_ids)
        # torch.cuda.synchronize()
        # t_check_end = time.perf_counter()
        # print(f"[TIMING] model update check forward: {t_check_end - t_check_start:.3f}s")
        # if prev_trainer_logprobs is not None and prev_trainer_logprobs.shape == trainer_logprobs_check.shape:
        #     logprob_diff = (trainer_logprobs_check.detach().cpu() - prev_trainer_logprobs).abs().mean().item()
        #     logprob_max_diff = (trainer_logprobs_check.detach().cpu() - prev_trainer_logprobs).abs().max().item()
        #     print(f"[MODEL UPDATE CHECK] logprob change from prev iter: mean={logprob_diff:.6f}, max={logprob_max_diff:.6f}")
        #     if logprob_diff < 1e-6:
        #         print("[WARNING] Model logprobs not changing - gradients may not be flowing!")
        # prev_trainer_logprobs = trainer_logprobs_check.detach().cpu().clone()

        torch.cuda.synchronize()
        iter_end = time.perf_counter()
        print(f"[TIMING] total iteration: {iter_end - iter_start:.3f}s")
        print("✓ Success.")


    


if __name__ == "__main__":
    asyncio.run(main())
