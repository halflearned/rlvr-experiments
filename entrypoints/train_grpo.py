"""GRPO training for code generation.

The algorithm is visible, the async machinery is hidden.
"""

import asyncio
import argparse

from transformers import AutoTokenizer

from rlvr_experiments.data import DataIterator, load_mbpp, load_humaneval
from rlvr_experiments.losses import GRPOLoss
from rlvr_experiments.rollout import run_epoch
from rlvr_experiments.runtime import Runtime
from rlvr_experiments.verifiers import VerifierPool, MBPPVerifier, HumanEvalVerifier
from rlvr_experiments.syncing import sync_titan_to_vllm, sync_titan_to_titan
from rlvr_experiments.tracer import trace_span

DATASETS = {
    "humaneval": (load_humaneval, HumanEvalVerifier),
    "mbpp": (load_mbpp, MBPPVerifier),
}


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str)
    args = parser.parse_args()

    # --- Setup ---
    runtime = await Runtime.from_plan(args.config)
    plan = runtime.plan
    await runtime.start()

    trainer = runtime.roles["trainer"]
    reference = runtime.roles["reference"]
    rollout = runtime.roles["rollout"]
    buffer = runtime.buffer

    tokenizer = AutoTokenizer.from_pretrained(**plan.tokenizer)
    pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    data_cfg = dict(plan.data)
    dataset_name = data_cfg.pop("dataset")
    load_fn, verifier_cls = DATASETS[dataset_name]

    verifier = VerifierPool(verifier_cls, **plan.verifier)

    data_iter = DataIterator(load_fn(**data_cfg), tokenizer=tokenizer, **plan.data_iter)
    loss_fn = GRPOLoss(**plan.loss)

    num_epochs = plan.training["num_epochs"]
    max_steps = plan.training.get("iterations_per_epoch")
    batch_size = plan.training.get("train_batch_size") or 1
    sync_ref_every = plan.training["sync_reference_every"]

    # --- Training loop ---
    for epoch in range(num_epochs):
        data_iter.new_epoch(seed=epoch)

        async for step, batch in run_epoch(
            rollout, data_iter, buffer,
            reward=verifier.verify_completions,
            pad_token_id=pad_token_id,
            batch_size=batch_size,
            sampling_params=plan.sampling,
            epoch=epoch,
        ):
            # The GRPO algorithm
            with trace_span("train_step"):
                with trace_span("ref_logprobs"):
                    ref_logprobs = await reference.compute_logprobs(batch.input_ids, batch.completion_ids)

                with trace_span("forward_backward"):
                    loss = await trainer.forward_backward(
                        loss_fn, batch.input_ids,
                        loss_args=(batch.completion_ids, ref_logprobs, batch.logprobs, batch.rewards),
                        loss_kwargs={"padding_mask": batch.mask},
                    )

                with trace_span("optim_step"):
                    grad_norm = await trainer.optim_step()

            avg_reward = batch.rewards.mean().item()
            runtime.tracer.counter("metrics", {"loss": loss, "grad_norm": grad_norm, "avg_reward": avg_reward})

            print(f"[epoch {epoch}] step={step} loss={loss:.4f} grad_norm={grad_norm:.4f} reward={avg_reward:.2f}")

            if max_steps and step >= max_steps:
                break

        print(f"Epoch {epoch} complete: {step} steps")

        # Sync weights
        await sync_titan_to_vllm(trainer, rollout)
        if (epoch + 1) % sync_ref_every == 0:
            await sync_titan_to_titan(trainer, reference)


if __name__ == "__main__":
    asyncio.run(main())
