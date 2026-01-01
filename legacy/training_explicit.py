import argparse
import asyncio
from dataclasses import dataclass
import logging
import math
import os
import time
import torch
import torch.nn.functional as F

from transformers import AutoTokenizer

from rlvr_experiments.data import DataIterator, load_humaneval, load_mbpp
from rlvr_experiments.losses import GRPOLoss
from rlvr_experiments.runtime import Runtime
from rlvr_experiments.sample_logger import init_sample_logger, log_sample, close_sample_logger
from rlvr_experiments.syncing import sync_titan_to_vllm, sync_titan_to_titan
from rlvr_experiments.tracer import init_global_tracer, trace_span, set_current_task_name
from rlvr_experiments.verifiers import VerifierPool, HumanEvalVerifier, MBPPVerifier

DATASET_REGISTRY = {
    "humaneval": (load_humaneval, HumanEvalVerifier),
    "mbpp": (load_mbpp, MBPPVerifier),
}

logger = logging.getLogger(__name__)


@dataclass
class RolloutSample:
    """One prompt with N completions."""

    input_ids: torch.Tensor          # [N, seq_len]
    completion_ids: torch.Tensor     # [N, completion_len]
    logprobs: torch.Tensor           # [N, completion_len]
    rewards: list[float]             # [N]
    problem_id: str = ""

    @classmethod
    def from_vllm(cls, response, pad_token_id, rewards, problem_id: str = ""):
        prompt = response.prompt_token_ids
        outputs = response.outputs
        n = len(outputs)

        seqs = [prompt + list(o.token_ids) for o in outputs]
        max_seq_len = max(len(s) for s in seqs)
        input_ids = torch.full((n, max_seq_len), pad_token_id, dtype=torch.long)
        for i, seq in enumerate(seqs):
            input_ids[i, :len(seq)] = torch.tensor(seq)

        max_completion_len = max(len(o.token_ids) for o in outputs)
        completion_ids = torch.full((n, max_completion_len), pad_token_id, dtype=torch.long)
        logprobs = torch.zeros((n, max_completion_len), dtype=torch.float32)

        for i, o in enumerate(outputs):
            L = len(o.token_ids)
            completion_ids[i, -L:] = torch.tensor(o.token_ids)
            logprobs[i, -L:] = torch.tensor([o.logprobs[j][o.token_ids[j]].logprob for j in range(L)])

        return cls(input_ids, completion_ids, logprobs, rewards, problem_id)


def make_batch(samples, pad_token_id):
    """Batch samples, compute advantages. Returns None if all zero-variance."""
    valid = [s for s in samples if torch.tensor(s.rewards, dtype=torch.float32).std() > 1e-6]
    if not valid:
        return None, samples  # Return original samples for logging

    def pad_cat(tensors, pad_value=0):
        max_len = max(t.shape[1] for t in tensors)
        return torch.cat([F.pad(t, (0, max_len - t.shape[1]), value=pad_value) for t in tensors])

    def advantages(s):
        rewards = torch.tensor(s.rewards, dtype=torch.float32)
        a = (rewards - rewards.mean()) / rewards.std().clamp(min=1e-6)
        return a[:, None].expand(-1, s.logprobs.shape[1])

    return {
        "input_ids": pad_cat([s.input_ids for s in valid], pad_value=pad_token_id),
        "completion_ids": pad_cat([s.completion_ids for s in valid], pad_value=pad_token_id),
        "completion_mask": pad_cat([(s.completion_ids != pad_token_id).long() for s in valid]),
        "logprobs": pad_cat([s.logprobs for s in valid]),
        "advantages": pad_cat([advantages(s) for s in valid]),
    }, valid


async def produce_rollouts(rollout, buffer, *, data_iter, pad_token_id, verifier, sampling_params, epoch):
    """Generate rollouts with pipelined verification.

    Verification runs in background while GPU generates next batch.
    """
    set_current_task_name("rollout")

    async def verify_and_buffer(response, problem):
        """Verify completions and push to buffer."""
        completions = [out.text for out in response.outputs]
        rewards, _ = await verifier.verify_completions(problem, completions)
        problem_id = str(problem.get('task_id', 'unknown'))
        await buffer.put(RolloutSample.from_vllm(response, pad_token_id, rewards, problem_id), epoch)

    async with asyncio.TaskGroup() as tasks:
        while not rollout.is_stopped():
            batch = await data_iter.next_batch()
            if batch is None:
                break

            responses = await rollout.generate(batch["templates"], **sampling_params)

            for response, problem in zip(responses, batch["problems"]):
                tasks.create_task(verify_and_buffer(response, problem))


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str)
    args = parser.parse_args()

    runtime = await Runtime.from_plan(args.config)
    plan = runtime.plan

    tracer = init_global_tracer(plan.trace_path)

    # Sample logger - goes next to trace file
    if plan.trace_path:
        sample_path = os.path.join(os.path.dirname(plan.trace_path) or ".", "samples.jsonl")
        init_sample_logger(sample_path)
        logger.info(f"Sample logging to {sample_path}")

    await runtime.start()

    trainer = runtime.roles["trainer"]
    reference = runtime.roles["reference"]
    rollout = runtime.roles["rollout"]
    buffer = runtime.buffer

    # Config
    tok_cfg = plan.tokenizer
    data_cfg = dict(plan.data)  # Copy so we can pop
    train_cfg = plan.training
    verifier_cfg = getattr(plan, "verifier", {})

    tokenizer = AutoTokenizer.from_pretrained(**tok_cfg)
    pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    loss_fn = GRPOLoss(**plan.loss)

    # Dataset selection
    dataset_name = data_cfg.pop("dataset", None)
    if not dataset_name or dataset_name not in DATASET_REGISTRY:
        raise ValueError(f"data.dataset required, one of: {list(DATASET_REGISTRY.keys())}")

    load_fn, verifier_cls = DATASET_REGISTRY[dataset_name]
    verifier = VerifierPool(
        verifier_cls,
        num_workers=verifier_cfg.get("num_workers", 4),
        timeout=verifier_cfg.get("timeout", 10.0),
        max_concurrent=verifier_cfg.get("max_concurrent", 4),
    )

    data_iter = DataIterator(load_fn(**data_cfg), tokenizer=tokenizer, **plan.data_iter)
    sampling_params = {**plan.sampling, "logprobs": 0}

    num_epochs = train_cfg["num_epochs"]
    iterations_per_epoch = train_cfg.get("iterations_per_epoch") or math.inf
    sync_reference_every = train_cfg["sync_reference_every"]
    train_batch_size = train_cfg.get("train_batch_size") or 1

    set_current_task_name("trainer")

    for epoch in range(num_epochs):
        data_iter.new_epoch(seed=epoch)

        trained = 0
        async for samples in rollout.run_epoch(
            produce_rollouts, buffer,
            batch_size=train_batch_size,
            min_version=epoch,
            data_iter=data_iter, pad_token_id=pad_token_id,
            verifier=verifier, sampling_params=sampling_params, epoch=epoch,
        ):
            if trained >= iterations_per_epoch:
                break

            print(f"\r[epoch {epoch}] step={trained} buffer={buffer.size()} training...          ", end="", flush=True)

            # Log reward stats
            all_rewards = [r for s in samples for r in s.rewards]
            tracer.counter("rewards", {
                "mean": sum(all_rewards) / len(all_rewards) if all_rewards else 0,
                "num_positive": sum(1 for r in all_rewards if r > 0),
            })

            with trace_span(None, "make_batch"):
                batch, valid_samples = make_batch(samples, pad_token_id)

            # Log training events
            for s in samples:
                r = torch.tensor(s.rewards, dtype=torch.float32)
                log_sample("training",
                    problem_id=s.problem_id, step=trained, epoch=epoch,
                    rewards=s.rewards, included=r.std() > 1e-6)

            if batch is None:
                tracer.counter("skipped", {"zero_variance_batches": 1})
                continue

            avg_reward = sum(sum(s.rewards) / len(s.rewards) for s in valid_samples) / len(valid_samples)

            with trace_span(None, "train_step"):
                with trace_span(None, "ref_logprobs"):
                    ref_logprobs = await reference.compute_logprobs(batch["input_ids"], batch["completion_ids"])
                with trace_span(None, "forward_backward"):
                    loss = await trainer.forward_backward(
                        loss_fn, batch["input_ids"],
                        loss_args=(batch["completion_ids"], ref_logprobs, batch["logprobs"], batch["advantages"]),
                        loss_kwargs={"padding_mask": batch["completion_mask"]},
                    )
                with trace_span(None, "optim_step"):
                    grad_norm = await trainer.optim_step()

            trained += 1
            tracer.counter("metrics", {"avg_reward": avg_reward, "loss": loss, "grad_norm": grad_norm})

            if trained % 10 == 0:
                logger.info(f"step={trained} loss={loss:.4f} grad_norm={grad_norm:.4f} avg_reward={avg_reward:.4f}")

        logger.info(f"Epoch {epoch}: {trained} steps")

        await sync_titan_to_vllm(trainer, rollout)
        if (epoch + 1) % sync_reference_every == 0:
            await sync_titan_to_titan(trainer, reference)


if __name__ == "__main__":
    asyncio.run(main())
