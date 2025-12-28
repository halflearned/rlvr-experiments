import argparse
import asyncio
import logging
import math
import torch
import torch.nn.functional as F

from transformers import AutoTokenizer

from rlvr_experiments.data import DataIterator, load_dummy
from rlvr_experiments.losses import GRPOLoss
from rlvr_experiments.runtime import Runtime
from rlvr_experiments.syncing import sync_titan_to_vllm, sync_titan_to_titan
from rlvr_experiments.tracer import init_global_tracer, trace_span
from rlvr_experiments.verifiers import MathVerifier

logger = logging.getLogger(__name__)


class RolloutSample:
    """One prompt with N completions."""

    def __init__(self, input_ids, completion_ids, logprobs, rewards):
        self.input_ids = input_ids          # [N, seq_len]
        self.completion_ids = completion_ids  # [N, completion_len]
        self.logprobs = logprobs            # [N, completion_len]
        self.rewards = rewards              # [N]

    @classmethod
    def from_vllm(cls, response, pad_token_id, rewards):
        prompt = response.prompt_token_ids
        outputs = response.outputs
        n = len(outputs)

        # Build input_ids (prompt + completion), right-padded
        seqs = [prompt + list(o.token_ids) for o in outputs]
        max_seq_len = max(len(s) for s in seqs)
        input_ids = torch.full((n, max_seq_len), pad_token_id, dtype=torch.long)
        for i, seq in enumerate(seqs):
            input_ids[i, :len(seq)] = torch.tensor(seq)

        # Build completion_ids and logprobs, left-padded
        max_completion_len = max(len(o.token_ids) for o in outputs)
        completion_ids = torch.full((n, max_completion_len), pad_token_id, dtype=torch.long)
        logprobs = torch.zeros((n, max_completion_len), dtype=torch.float32)

        for i, o in enumerate(outputs):
            L = len(o.token_ids)
            completion_ids[i, -L:] = torch.tensor(o.token_ids)
            logprobs[i, -L:] = torch.tensor([o.logprobs[j][o.token_ids[j]].logprob for j in range(L)])

        return cls(input_ids, completion_ids, logprobs, rewards)

    @staticmethod
    def batch(samples):
        """Batch samples, compute advantages. Returns None if all zero-variance."""
        valid = [s for s in samples if s.rewards.std() > 1e-6]
        if not valid:
            return None

        def pad_cat(tensors):
            max_len = max(t.shape[1] for t in tensors)
            return torch.cat([F.pad(t, (0, max_len - t.shape[1])) for t in tensors])

        def advantages(s):
            a = (s.rewards - s.rewards.mean()) / s.rewards.std().clamp(min=1e-6)
            return a[:, None].expand(-1, s.logprobs.shape[1])  # repeated along completion length

        # B = total completions across all valid samples, L = max completion length
        return {
            "input_ids": pad_cat([s.input_ids for s in valid]),          # [B, prompt_len + L]
            "completion_ids": pad_cat([s.completion_ids for s in valid]),  # [B, L]
            "completion_mask": pad_cat([(s.completion_ids != 0).long() for s in valid]),  # [B, L]
            "logprobs": pad_cat([s.logprobs for s in valid]),            # [B, L]
            "advantages": pad_cat([advantages(s) for s in valid]),       # [B, L]
        }


async def produce_rollouts(rollout, buffer, data_iter, pad_token_id, verifier, sampling_params, epoch):
    """Generate rollouts and push to buffer until stopped or data exhausted."""
    while not rollout.is_stopped():
        batch = await data_iter.next_batch()
        if batch is None:
            break

        responses = await rollout.generate(batch["templates"], **sampling_params)

        for i, response in enumerate(responses):
            completions = [out.text for out in response.outputs]
            target = batch["answers"][i]
            rewards = verifier.verify_batch(
                responses=completions,
                targets=[target] * len(completions),
                return_dtype=torch.float32,
            )
            await buffer.put(RolloutSample.from_vllm(response, pad_token_id, rewards), epoch)


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str)
    args = parser.parse_args()

    runtime = await Runtime.from_plan(args.config)
    plan = runtime.plan

    init_global_tracer(plan.trace_path)
    await runtime.start()

    # Unpack runtime components
    trainer = runtime.roles["trainer"]
    reference = runtime.roles["reference"]
    rollout = runtime.roles["rollout"]
    buffer = runtime.buffer

    # Setup
    tokenizer = AutoTokenizer.from_pretrained(**plan.tokenizer)
    pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    loss_fn = GRPOLoss(**plan.loss)
    verifier = MathVerifier()
    data_iter = DataIterator(load_dummy(**plan.data), tokenizer=tokenizer, **plan.data_iter)
    sampling_params = {**plan.sampling, "logprobs": 0}

    num_epochs = plan.training["num_epochs"]
    iterations_per_epoch = plan.training.get("iterations_per_epoch") or math.inf
    sync_reference_every = plan.training["sync_reference_every"]
    train_batch_size = plan.training.get("train_batch_size") or 1

    for epoch in range(num_epochs):
        data_iter.new_epoch(seed=epoch)
        rollout.start_producers(produce_rollouts, buffer, data_iter, pad_token_id, verifier, sampling_params, epoch)

        trained = 0
        while trained < iterations_per_epoch:
            if rollout.producers_done() and buffer.size() == 0:
                break

            samples = await buffer.pop_batch(train_batch_size, min_version=epoch)
            batch = RolloutSample.batch(samples)
            if batch is None:
                continue

            with trace_span(None, "train_step"):
                ref_logprobs = await reference.compute_logprobs(batch["input_ids"], batch["completion_ids"])
                loss = await trainer.forward_backward(
                    loss_fn,
                    batch["input_ids"],
                    loss_args=(batch["completion_ids"], ref_logprobs, batch["logprobs"], batch["advantages"]),
                    loss_kwargs={"padding_mask": batch["completion_mask"]},
                )
                grad_norm = await trainer.optim_step()
            trained += 1

            if trained % 10 == 0:
                logger.info(f"step={trained} loss={loss:.4f} grad_norm={grad_norm:.4f}")

        await rollout.stop_producers()
        logger.info(f"Epoch {epoch}: {trained} steps")

        await sync_titan_to_vllm(trainer, rollout)
        if (epoch + 1) % sync_reference_every == 0:
            await sync_titan_to_titan(trainer, reference)


if __name__ == "__main__":
    asyncio.run(main())
