import argparse
import asyncio
import torch

from rlvr_experiments.runtime import Runtime
from rlvr_experiments.syncing import sync_titan_to_vllm, sync_titan_to_titan
from rlvr_experiments.tracer import (
    dump_traces,
    get_tracer,
    init_global_tracer,
    set_current_task_name,
    traced,
)
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
    p.add_argument(
        "--trace",
        type=str,
        default=None,
        help="Override RLVR_TRACE_PATH with this output file.",
    )
    return p.parse_args()


def apply_template(prompt: str, tokenizer, tokenize=False) -> str:
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=tokenize,
        enable_thinking=False,
        add_generation_prompt=True,
    )


async def continuous_rollout_producer(
    rollout,
    buffer,
    tokenizer,
    templates: list[str],
    problem_answers: list[str],
    verifier,
    sampling_params: dict,
    version: int,
) -> None:
    """
    Continuously generates rollouts and pushes them to the buffer.
    Automatically stops when sync_titan_to_vllm is called.
    """
    set_current_task_name("rollout")

    while True:
        if rollout.is_stopped():
            print("[ROLLOUT PRODUCER] Stop signal received, exiting.")
            break

        responses = await rollout.generate(templates, **sampling_params)
        response = responses[0]
        print(f"[ROLLOUT PRODUCER] generated: {response.outputs[0].text[:100]}...")

        # Parse vLLM output
        vllm_output = VLLMOutput(response)
        full_input_ids, completion_ids, completion_mask, completion_logprobs = vllm_output.get_tensors(tokenizer)

        # Compute rewards
        rewards = verifier.verify_batch(
            responses=vllm_output.completion_texts(),
            targets=problem_answers * len(response.outputs),
            return_dtype=torch.float32,
        )
        print(f"[ROLLOUT PRODUCER] rewards: mean={rewards.mean().item():.3f}")

        # Store as a plain dict
        entry = {
            "full_input_ids": full_input_ids,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "completion_logprobs": completion_logprobs,
            "rewards": rewards,
        }
        await buffer.put(entry, version)
        print(f"[ROLLOUT PRODUCER] pushed entry to buffer (size={buffer.size()})")





@traced()
async def train_on_rollout(
    entry: dict,
    trainer,
    reference,
    loss_fn,
) -> tuple[float, float] | None:
    """
    Train on a single rollout entry (dict). Returns (loss, grad_norm) or None if skipped.
    """
    rewards = entry["rewards"]

    # Skip if all rewards are identical (no gradient signal)
    if torch.allclose(rewards, rewards[0]):
        print("[TRAINER] All rewards identical, skipping batch.")
        return None

    input_dict = {"input": entry["full_input_ids"]}
    completion_ids = entry["completion_ids"]
    completion_logprobs = entry["completion_logprobs"]
    completion_mask = entry["completion_mask"]

    # Get logprobs from reference model
    reference_logprobs = await reference.compute_logprobs_step(
        input_dict,
        completion_ids,
    )

    # Compute loss and backward
    loss = await trainer.compute_loss_and_backward_step(
        loss_fn,
        input_dict,
        completion_ids,
        reference_logprobs,
        completion_logprobs,
        rewards,
        padding_mask=completion_mask,
    )
    print(f"[TRAINER] loss={loss:.4f}")

    # Optimizer step
    grad_norm = await trainer.optimizer_step()
    print(f"[TRAINER] grad_norm={grad_norm:.6f}")

    return loss, grad_norm


async def main() -> None:
    args = parse_args()

    if args.trace:
        init_global_tracer(args.trace)
    set_current_task_name("main")

    runtime = await Runtime.from_plan(args.config)
    await runtime.start()

    # Get roles
    trainer = runtime.roles["trainer"]
    reference = runtime.roles["reference"]
    rollout = runtime.roles["rollout"]
    buffer = runtime.buffer
    
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

    sampling_params = dict(temperature=0.6, top_p=0.95, top_k=20, max_tokens=512, logprobs=0, n=20)

    with open("reward_log.txt", "w") as f:
        f.write("")

    # Training configuration
    num_epochs = 3  # Number of weight sync cycles
    iterations_per_epoch = 3  # Training iterations before syncing weights
    sync_reference_every = 1  # Sync reference model every N epochs

    global_iteration = 0

    for epoch in range(num_epochs):
        print("\n\n" + "=" * 60)
        print(f" EPOCH {epoch + 1}/{num_epochs} ")
        print("=" * 60)

        current_version = epoch  # Version tracks weight version

        rollout.start_producer(
            continuous_rollout_producer(
                rollout=rollout,
                buffer=buffer,
                tokenizer=tokenizer,
                templates=templates,
                problem_answers=problem_answers,
                verifier=verifier,
                sampling_params=sampling_params,
                version=current_version,
            )
        )
        print(f"[MAIN] Started continuous rollout producer (version={current_version})")

        # Train for n iterations
        trained_iterations = 0
        while trained_iterations < iterations_per_epoch:
            global_iteration += 1
            print(f"\n{'*' * 20} ITERATION {global_iteration} (epoch {epoch+1}, iter {trained_iterations+1}/{iterations_per_epoch}) {'*' * 20}")

            # Pop from buffer (blocks until available)
            entry = await buffer.pop(min_version=current_version)
            rewards_mean = entry["rewards"].mean().item()
            print(f"[TRAINER] Got rollout from buffer (rewards_mean={rewards_mean:.3f})")

            avg_rewards.append(rewards_mean)
            with open("reward_log.txt", "a") as f:
                f.write(f"{rewards_mean}\n")

            # Train on this entry
            result = await train_on_rollout(entry, trainer, reference, loss_fn)

            if result is not None:
                trained_iterations += 1

        # Sync weights: trainer -> rollout (vLLM)
        # This automatically stops the rollout producer, syncs, then resumes
        await sync_titan_to_vllm(trainer, rollout)
        print("[MAIN] Synced trainer -> rollout vLLM")

        # Periodically sync trainer -> reference
        if (epoch + 1) % sync_reference_every == 0:
            await sync_titan_to_titan(trainer, reference)
            print("[MAIN] Synced trainer -> reference")

        print(f"[MAIN] avg rewards so far: {avg_rewards[-10:]}")  # Last 10

    print("\n" + "=" * 60)
    print(" TRAINING COMPLETE ")
    print("=" * 60)
    print(f"Total iterations: {global_iteration}")
    print(f"Final avg reward (last 10): {sum(avg_rewards[-10:])/min(10, len(avg_rewards)):.3f}")

    tracer = get_tracer()
    if tracer is not None:
        dump_traces()
        print(f"[TRACE] wrote {tracer.path}")

    


if __name__ == "__main__":
    asyncio.run(main())
