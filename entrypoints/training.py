import argparse
import asyncio
import torch

from rlvr_experiments.data import DataIterator, load_gsm8k
from rlvr_experiments.runtime import Runtime
from rlvr_experiments.syncing import sync_titan_to_vllm, sync_titan_to_titan
from rlvr_experiments.tracer import (
    close_tracer,
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
    p = argparse.ArgumentParser(description="RLVR Training")
    p.add_argument("config", type=str, help="Path to the YAML config file.")
    return p.parse_args()


async def continuous_rollout_producer(
    rollout,
    buffer,
    data_iter: DataIterator,
    tokenizer,
    verifier,
    sampling_params: dict,
    version: int,
) -> None:
    """
    Continuously generates rollouts and pushes them to the buffer.
    Stops when sync_titan_to_vllm is called or data is exhausted.
    """
    set_current_task_name("rollout")

    while True:
        if rollout.is_stopped():
            break

        batch = await data_iter.next_batch()
        if batch is None:
            break

        templates = batch["templates"]
        answers = batch["answers"]

        responses = await rollout.generate(templates, **sampling_params)

        for i, response in enumerate(responses):
            vllm_output = VLLMOutput(response)
            full_input_ids, completion_ids, completion_mask, completion_logprobs = (
                vllm_output.get_tensors(tokenizer)
            )

            rewards = verifier.verify_batch(
                responses=vllm_output.completion_texts(),
                targets=[answers[i]] * len(response.outputs),
                return_dtype=torch.float32,
            )

            entry = {
                "full_input_ids": full_input_ids,
                "completion_ids": completion_ids,
                "completion_mask": completion_mask,
                "completion_logprobs": completion_logprobs,
                "rewards": rewards,
            }
            await buffer.put(entry, version)


@traced()
async def train_step(
    entry: dict,
    trainer,
    reference,
    loss_fn,
) -> tuple[float, float] | None:
    """
    Train on a single rollout entry. Returns (loss, grad_norm) or None if skipped.
    """
    rewards = entry["rewards"]

    # Skip if all rewards are identical (no gradient signal)
    # TODO: do something better / more elegant here, or at least move it out?
    if torch.allclose(rewards, rewards[0]):
        return None

    input_dict = {"input": entry["full_input_ids"]}
    completion_ids = entry["completion_ids"]
    completion_logprobs = entry["completion_logprobs"]
    completion_mask = entry["completion_mask"]

    reference_logprobs = await reference.compute_logprobs_step(
        input_dict,
        completion_ids,
    )

    loss = await trainer.compute_loss_and_backward_step(
        loss_fn,
        input_dict,
        completion_ids,
        reference_logprobs,
        completion_logprobs,
        rewards,
        padding_mask=completion_mask,
    )

    grad_norm = await trainer.optimizer_step()

    return loss, grad_norm


async def main() -> None:
    args = parse_args()
    runtime = await Runtime.from_plan(args.config)
    plan = runtime.plan

    if plan.trace_path:
        init_global_tracer(plan.trace_path)
    set_current_task_name("main")

    await runtime.start()

    # Roles
    trainer = runtime.roles["trainer"]
    reference = runtime.roles["reference"]
    rollout = runtime.roles["rollout"]
    buffer = runtime.buffer

    # Config
    tokenizer = AutoTokenizer.from_pretrained(**plan.tokenizer)
    loss_fn = GRPOLoss(**plan.loss)
    verifier = MathVerifier()

    ds = load_gsm8k(**plan.data)
    data_iter = DataIterator(ds, tokenizer=tokenizer, **plan.data_iter)

    sampling_params = {**plan.sampling, "logprobs": 0}

    num_epochs = plan.training["num_epochs"]
    iterations_per_epoch = plan.training.get("iterations_per_epoch")  # None = full epoch
    sync_reference_every = plan.training["sync_reference_every"]

    # Training loop
    for epoch in range(num_epochs):
        data_iter.new_epoch(seed=epoch)

        producer_task = rollout.start_producer(
            continuous_rollout_producer(
                rollout=rollout,
                buffer=buffer,
                data_iter=data_iter,
                tokenizer=tokenizer,
                verifier=verifier,
                sampling_params=sampling_params,
                version=epoch,
            )
        )

        trained = 0
        while iterations_per_epoch is None or trained < iterations_per_epoch:
            # Check if producer finished (data exhausted)
            if producer_task.done():
                # Drain remaining buffer entries
                if buffer.size() == 0:
                    break

            entry = await buffer.pop(min_version=epoch)
            result = await train_step(entry, trainer, reference, loss_fn)
            if result is not None:
                trained += 1

        await sync_titan_to_vllm(trainer, rollout)

        if (epoch + 1) % sync_reference_every == 0:
            await sync_titan_to_titan(trainer, reference)

        # Export trained model in HuggingFace format
        if True:  # TODO: save or not
            await trainer.export_to_hf(f"outputs/{plan.run['name']}")

    if get_tracer():
        close_tracer()


if __name__ == "__main__":
    asyncio.run(main())
