import argparse
import asyncio
import torch

from rlvr_experiments.runtime import Runtime
from rlvr_experiments.syncing import sync_titan_to_vllm
from rlvr_experiments.losses import GRPOLoss
from rlvr_experiments.verifiers import MathVerifier

from torch.nn.utils.rnn import pad_sequence
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
        enable_thinking=True,
        add_generation_prompt=True,
    )


def parse_vllm_responses(responses):
    rollout_logprobs = []
    rollout_token_ids = []
    rollout_text = []
    for resp in responses:
        print(f"Response has {len(resp.outputs)} outputs.")
        for i, completion in enumerate(resp.outputs):
            print(f"Iterating over completion {i}, which has length {len(completion.logprobs)}")
            rollout_token_ids.append(completion.token_ids)
            token_logprobs = [
                next(iter(token_dict.values())).logprob
                for token_dict in completion.logprobs
            ]
            rollout_logprobs.append(token_logprobs)
            rollout_text.append(completion.text)
    
    print("Padding rollout logprobs...")
    print("lengths", [len(lp) for lp in rollout_logprobs])
    rollout_logprobs = pad_sequence(
        [torch.tensor(lp) for lp in rollout_logprobs],
        batch_first=True,
        padding_value=0.0,  # zero?
    )
    return rollout_token_ids, rollout_logprobs, rollout_text


def scramble(sentence: str) -> str:
    import random
    words = sentence.split()
    random.shuffle(words)
    return ' '.join(words)


async def main() -> None:
    args = parse_args()

    runtime = await Runtime.from_plan(args.config)
    await runtime.start()

    # Get roles
    trainer = runtime.roles["trainer"]
    reference = runtime.roles["reference"]
    rollout = runtime.roles["rollout"]

    loss_fn = GRPOLoss(beta=0.1, eps=0.2)

    verifier = MathVerifier(tolerance=1e-6, partial_credit=0.1)

    # Recommended sampling params for thinking mode
    # https://qwen.readthedocs.io/en/latest/deployment/vllm.html#thinking-non-thinking-modes
    sampling_params = dict(temperature=0.6, top_p=0.95, top_k=20, max_tokens=2048, logprobs=0, n=10)
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B", use_fast=False)
    system_prompt = "This math question has been scrambled. There is straightforward and not a trick question. Answer with a number within <answer></answer> tags: "
    problem_prompts = [
        "what is the integral of x^2 from 0 to 1",
        #"what is the derivative of sin(x) at x=0",
        #"what is 12 times 12 plus 5",
    ]
    problem_answers = [
        "0.333333",
        #"1.0",
        #"149",
    ]
    templates = [apply_template(system_prompt + p + "?", tokenizer) for p in problem_prompts]


    for i in range(2):
        print("\n\n" + "*" * 20 + f" ITERATION {i+1}/10 " + "*" * 20)
        responses = await rollout.generate(templates, **sampling_params)

        limits = await rollout.debug_effective_limits()
        print("Rollout effective limits:", limits)

        for i, resp in enumerate(responses):
            print(f"prompt {i}: {len(resp.outputs)} outputs")
            for j, out in enumerate(resp.outputs):
                print("  ", j, out.finish_reason)


        print("Got rollout responses:", len(responses))
        rollout_token_ids, rollout_logprobs, rollout_text = parse_vllm_responses(responses)
        print("Got rollout logprobs.")
        print("Rollout texts[0]:", rollout_text[0])
        print("Total texts:", len(rollout_text))



        encoded = tokenizer.pad({"input_ids": rollout_token_ids}, padding=True, return_tensors="pt")
        padding_mask = encoded["attention_mask"]  # for later

        print("Print shape for debugging:", encoded["input_ids"].shape)
        input_dict = {"input": encoded["input_ids"]}

        trainer_logprobs = await trainer.forward_step(input_dict)
        print("Got trainer logprobs.")

        reference_logprobs = await reference.forward_step(input_dict)
        print("Got reference logprobs.")

        # Rewards
        aligned_answers = [
            answer
            for resp, answer in zip(responses, problem_answers)
            for _ in resp.outputs
        ]
        rewards = torch.tensor([
            verifier(None, r, a) for r, a in zip(rollout_text, aligned_answers)
        ])
        print(f"Got rewards: {rewards}")

        # GRPO loss
        loss = await trainer.compute_loss_and_backward_step(
            loss_fn, trainer_logprobs, reference_logprobs, rollout_logprobs, rewards, padding_mask
        )
        print("Computed loss and backward pass")
        print(f"Loss: {loss}")

        await trainer.optimizer_step()
        print("Performed optimizer step.")

    # Sync trainer weights to rollout vLLM
    await sync_titan_to_vllm(trainer, rollout)
    print("Synchronized trainer weights to rollout vLLM.")

    print("✓ Success.")


    


if __name__ == "__main__":
    asyncio.run(main())
