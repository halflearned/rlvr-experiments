import argparse
import asyncio
import torch

from rlvr_experiments.runtime import Runtime
from rlvr_experiments.syncing import sync_titan_to_vllm
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


    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B", use_fast=False)
    system_prompt = r"This math question has been scrambled. Unscrambled and answer with a number within \\boxed{}. "
    problem_prompts = [
        "what is the integral of x^2 over the unit-interval",
    ]
    problem_answers = [
        "0.333333",
    ]
    templates = [apply_template(system_prompt + scramble(p) + "?", tokenizer) for p in problem_prompts]

    num_iterations = 2
    for i in range(num_iterations):
        print("\n\n" + "*" * 20 + f" ITERATION {i+1}/10 " + "*" * 20)

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

        # Get logprobs from trainer and reference models
        input_dict = {
            "input": full_input_ids,
            "completion_ids": completion_ids,
        }
        trainer_logprobs = await trainer.forward_step(input_dict)
        print("trainer logprobs", trainer_logprobs[0, :10])
        print("got trainer logprobs.")

        reference_logprobs = await reference.forward_step(input_dict)
        print("reference logprobs", reference_logprobs[0, :10])
        print("got reference logprobs.")

        # Compute rewards
        rewards = verifier.verify_batch(
            responses=vllm_output.completion_texts(),
            targets=problem_answers * len(response.outputs),
            return_dtype=torch.float32,
        )
        print(f"rewards: {rewards}")

        # GRPO loss
        loss = await trainer.compute_loss_and_backward_step(
            loss_fn, trainer_logprobs, reference_logprobs, completion_logprobs,
            rewards, padding_mask=completion_mask
        )
        print(f"computed loss and backward pass. loss: {loss}")

        # Update trainer model
        await trainer.optimizer_step()
        print("optimizer step done.")

    # Sync trainer weights to rollout vLLM
    await sync_titan_to_vllm(trainer, rollout)
    print("Synchronized trainer weights to rollout vLLM.")

    print("✓ Success.")


    


if __name__ == "__main__":
    asyncio.run(main())
