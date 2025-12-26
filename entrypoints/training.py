import argparse
import asyncio
import logging
import torch

from rlvr_experiments.data import DataIterator, load_gsm8k, load_dummy

logger = logging.getLogger(__name__)
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
    batch_num = 0
    total_entries_produced = 0

    logger.info(f"producer: Starting, version={version}")

    while True:
        if rollout.is_stopped():
            logger.info("producer: Rollout stopped, breaking")
            break

        logger.debug(f"producer: Fetching next data batch (batch_num={batch_num})")
        batch = await data_iter.next_batch()
        if batch is None:
            logger.info("producer: Data exhausted, breaking")
            break

        batch_num += 1
        templates = batch["templates"]
        answers = batch["answers"]
        logger.info(f"producer: Got data batch {batch_num}, {len(templates)} templates, generating...")

        responses = await rollout.generate(templates, **sampling_params)
        logger.debug(f"producer: Generate complete, got {len(responses)} responses")

        for i, response in enumerate(responses):
            vllm_output = VLLMOutput(response)
            full_input_ids, completion_ids, completion_mask, completion_logprobs = (
                vllm_output.get_tensors(tokenizer)
            )

            logger.info(f"Example rollout:\n{vllm_output.completion_texts()[0]}")

            rewards = verifier.verify_batch(
                responses=vllm_output.completion_texts(),
                targets=[answers[i]] * len(response.outputs),
                return_dtype=torch.float32,
            )
            logger.info(f"producer: Rewards for response {i}: sum={rewards.sum().item():.2f}, mean={rewards.mean().item():.2f}")

            if tracer := get_tracer():
                gen_lengths = [len(out.token_ids) for out in response.outputs]
                tracer.counter("rollout", {
                    "avg_reward": rewards.mean().item(),
                    "avg_gen_length": sum(gen_lengths) / len(gen_lengths),
                })

            entry = {
                "full_input_ids": full_input_ids,
                "completion_ids": completion_ids,
                "completion_mask": completion_mask,
                "completion_logprobs": completion_logprobs,
                "rewards": rewards,
            }
            logger.debug(f"producer: Putting entry {total_entries_produced} to buffer...")
            await buffer.put(entry, version)
            total_entries_produced += 1
            logger.debug(f"producer: Entry {total_entries_produced} put complete, buffer_size={buffer.size()}")

    logger.info(f"producer: Finished, total_entries_produced={total_entries_produced}")


@traced()
async def train_step(
    batch: dict,
    trainer,
    reference,
    loss_fn,
) -> tuple[float, float]:
    """
    Train on a batched rollout entry. Returns (loss, grad_norm).

    The batch contains pre-computed per-group normalized advantages.
    """
    input_dict = {"input": batch["full_input_ids"]}
    completion_ids = batch["completion_ids"]
    rollout_logprobs = batch["completion_logprobs"]
    completion_mask = batch["completion_mask"]
    advantages = batch["advantages"]

    # TODO: it'd be nice to make this concurrent with the model forward.
    # Maybe create a task and then await it later.
    reference_logprobs = await reference.compute_logprobs_step(
        input_dict,
        completion_ids,
    )

    loss = await trainer.compute_loss_and_backward_step(
        loss_fn,
        input_dict,
        completion_ids,
        reference_logprobs,
        rollout_logprobs,
        advantages,
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
    # ds = load_dummy(**plan.data)
    data_iter = DataIterator(ds, tokenizer=tokenizer, **plan.data_iter)

    sampling_params = {**plan.sampling, "logprobs": 0}

    num_epochs = plan.training["num_epochs"]
    iterations_per_epoch = plan.training.get("iterations_per_epoch")
    sync_reference_every = plan.training["sync_reference_every"]
    train_batch_size = plan.training.get("train_batch_size", 1)

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
        null_batch_count = 0
        while iterations_per_epoch is None or trained < iterations_per_epoch:
            # Check if producer finished (data exhausted)
            producer_done = producer_task.done()
            buffer_size = buffer.size()
            logger.debug(f"train_loop: trained={trained}, producer_done={producer_done}, buffer_size={buffer_size}")

            if producer_done:
                # Drain remaining buffer entries
                if buffer_size == 0:
                    logger.info("train_loop: Producer done and buffer empty, breaking")
                    break
                logger.debug(f"train_loop: Producer done but buffer has {buffer_size} items, draining...")

            logger.debug(f"train_loop: Calling buffer.pop(batch_size={train_batch_size}, min_version={epoch})")
            batch = await buffer.pop(batch_size=train_batch_size, min_version=epoch)
            if batch is None:
                null_batch_count += 1
                logger.debug(f"train_loop: Got None batch (#{null_batch_count}), skipping")
                # All samples had zero reward variance - skip this batch
                continue

            null_batch_count = 0  # Reset on success
            logger.debug("train_loop: Got valid batch, calling train_step")

            loss, grad_norm = await train_step(batch, trainer, reference, loss_fn)
            trained += 1
            logger.debug(f"train_loop: train_step complete, loss={loss:.4f}, grad_norm={grad_norm:.4f}")

            if trained % 10 == 0:
                stats = buffer.get_stats()
                logger.info(f"step={trained} loss={loss:.4f} grad_norm={grad_norm:.4f} buffer={stats}")

        stats = buffer.get_stats()
        logger.info(f"Epoch {epoch}: Completed {trained} train steps. Buffer stats: {stats}")

        logger.info(f"Epoch {epoch}: Syncing trainer to vLLM...")
        await sync_titan_to_vllm(trainer, rollout)
        logger.debug(f"Epoch {epoch}: sync_titan_to_vllm done")

        if (epoch + 1) % sync_reference_every == 0:
            logger.info(f"Epoch {epoch}: Syncing trainer to reference...")
            await sync_titan_to_titan(trainer, reference)
            logger.debug(f"Epoch {epoch}: sync_titan_to_titan done")

        # Export trained model in HuggingFace format
        if False: # TODO: save or not
            logger.info(f"Epoch {epoch}: Exporting to HuggingFace...")
            await trainer.export_to_hf(f"outputs/{plan.run['name']}")
            logger.debug(f"Epoch {epoch}: export_to_hf done")

        logger.info(f"Epoch {epoch}: complete")

    logger.info("Training complete, closing tracer...")
    if get_tracer():
        close_tracer()
    logger.info("Done")


if __name__ == "__main__":
    asyncio.run(main())
