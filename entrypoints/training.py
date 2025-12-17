import argparse
import asyncio
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
        print("\n\n" + "*" * 20 + f" ITERATION {i+1}/{num_iterations} " + "*" * 20)

        if i % 10 == 0 and i > 0:
            await sync_titan_to_titan(trainer, reference)
            print("Synchronized trainer weights to reference model.")

        if i % 5 == 0:
            # Sync trainer weights to rollout vLLM
            await sync_titan_to_vllm(trainer, rollout)
            print("Synchronized trainer weights to rollout model.")

        # Generate responses from rollout vLLM
        # https://qwen.readthedocs.io/en/latest/deployment/vllm.html#thinking-non-thinking-modes
        sampling_params = dict(temperature=0.6, top_p=0.95, top_k=20, max_tokens=512, logprobs=0, n=20)
        responses = await rollout.generate(templates, **sampling_params)
        response = responses[0]
        print("generated:", response.outputs[0].text)

        # For now, only support single prompt per batch
        vllm_output = VLLMOutput(response)

        # Prepare inputs for trainer and reference models
        full_input_ids, completion_ids, completion_mask, completion_logprobs = vllm_output.get_tensors(tokenizer)

        # Get logprobs from reference model
        input_dict = {
            "input": full_input_ids,
            "completion_ids": completion_ids,
        }
        reference_logprobs = await reference.forward_step(input_dict)
        print("got reference logprobs.")

        # Compute rewards
        rewards = verifier.verify_batch(
            responses=vllm_output.completion_texts(),
            targets=problem_answers * len(response.outputs),
            return_dtype=torch.float32,
        )
        print(f"rewards: {rewards}")

        avg_rewards.append(rewards.mean().item())
        print(f"avg rewards so far: {avg_rewards}")
        with open("reward_log.txt", "a") as f:
            f.write(f"{rewards.mean().item()}\n")

        if torch.allclose(rewards, rewards[0]):
            print("All rewards are identical, skipping this batch.")
            continue


        # GRPO loss
        loss = await trainer.compute_loss_and_backward_step(
            loss_fn, input_dict, reference_logprobs, completion_logprobs,
            rewards, padding_mask=completion_mask
        )
        print(f"computed loss and backward pass. loss: {loss}")

        # Update trainer model
        grad_norm = await trainer.optimizer_step()
        print(f"optimizer step done. grad_norm={grad_norm:.6f}")

        print("✓ Success.")


    


if __name__ == "__main__":
    asyncio.run(main())
