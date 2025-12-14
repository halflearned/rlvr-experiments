import argparse
import asyncio

from rlvr_experiments.runtime import Runtime
from rlvr_experiments.syncing import sync_titan_to_vllm

from transformers import AutoTokenizer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="RLVR Experiments Entrypoint")
    p.add_argument(
        "config",
        type=str,
        help="Path to the YAML configuration file.",
    )
    return p.parse_args()


async def main() -> None:
    args = parse_args()

    runtime = await Runtime.from_plan(args.config)
    await runtime.start()

    # Get roles
    trainer = runtime.roles["trainer"]
    reference = runtime.roles["reference"]
    rollout = runtime.roles["rollout"]

    sampling_params = {"max_tokens": 16, "temperature": 0.0, "top_p": 1.0}

    # Recommended sampling params for thinking mode
    # https://qwen.readthedocs.io/en/latest/deployment/vllm.html#thinking-non-thinking-modes
    sampling_params = dict(temperature=0.6, top_p=0.95, top_k=20, max_tokens=1024)
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B", use_fast=False)
    prompts = ["The capital of France is", "2+2=", "The square root of 1000 is"]
    templates = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            enable_thinking=True,
            add_generation_prompt=True,
        )
        for prompt in prompts
    ]

    print("Rollout generation test 1...")
    output = await rollout.generate(templates, **sampling_params)
    for out in output:
        print("Generation:", out.outputs[0].text)
    print("✓ Success.")

    # Sync trainer weights to rollout vLLM
    await sync_titan_to_vllm(trainer, rollout)

    print("Rollout generation test 2...")
    output = await rollout.generate(templates, **sampling_params)
    for out in output:
        print("Generation:", out.outputs[0].text)
    print("✓ Success.")


    


if __name__ == "__main__":
    asyncio.run(main())
