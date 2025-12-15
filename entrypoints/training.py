import argparse
import asyncio
import torch

from rlvr_experiments.runtime import Runtime
from rlvr_experiments.syncing import sync_titan_to_vllm
from rlvr_experiments.losses import GRPOLoss

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
    for resp in responses:
        completion = resp.outputs[0]  # TODO: when n > 1 fix this
        rollout_token_ids.append(completion.token_ids)
        token_logprobs = [
            token_dict[token_id].logprob
            for token_id, token_dict in zip(completion.token_ids, completion.logprobs)
        ]
        rollout_logprobs.append(token_logprobs)
    
    rollout_logprobs = pad_sequence(
        [torch.tensor(lp) for lp in rollout_logprobs],
        batch_first=True,
        padding_value=0.0,  # zero?
    )
    return rollout_token_ids, rollout_logprobs


async def main() -> None:
    args = parse_args()

    runtime = await Runtime.from_plan(args.config)
    await runtime.start()

    # Get roles
    trainer = runtime.roles["trainer"]
    reference = runtime.roles["reference"]
    rollout = runtime.roles["rollout"]

    loss_fn = GRPOLoss(beta=0.1, eps=0.2)


    # Recommended sampling params for thinking mode
    # https://qwen.readthedocs.io/en/latest/deployment/vllm.html#thinking-non-thinking-modes
    sampling_params = dict(temperature=0.6, top_p=0.95, top_k=20, max_tokens=1024, logprobs=True)
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B", use_fast=False)
    system_prompt = "Provide a final answer within <answer></answer> tags. "
    problem_prompts = ["The capital of France is", "2+2=", "The square root of 1000 is"]
    templates = [apply_template(system_prompt + p, tokenizer) for p in problem_prompts]

    responses = await rollout.generate(templates, **sampling_params)
    rollout_token_ids, rollout_logprobs = parse_vllm_responses(responses)
    print("Got rollout logprobs.")

    encoded = tokenizer.pad({"input_ids": rollout_token_ids}, padding=True, return_tensors="pt")
    padding_mask = encoded["attention_mask"]  # for later
    input_dict = {"input": encoded["input_ids"]}

    trainer_logprobs = await trainer.forward_step(input_dict)
    print("Got trainer logprobs.")

    reference_logprobs = await reference.forward_step(input_dict)
    print("Got reference logprobs.")

    # Rewards (dummy for now)
    rewards = torch.tensor([1.0, 0.0, -1.0])
    print("Prepared rewards.")

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
