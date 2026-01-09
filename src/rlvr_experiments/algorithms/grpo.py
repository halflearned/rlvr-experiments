"""Rollout types and epoch runner."""

import asyncio
from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn.functional as F

from ..rollout_logger import log_rollout
from ..tracer import get_tracer, trace_span


# Sentinel to signal producer completion
class _Done:
    pass

DONE = _Done()


@dataclass
class Batch:
    """A training batch - generic container for rollout data."""
    input_ids: torch.Tensor       # [B, seq_len]
    completion_ids: torch.Tensor  # [B, completion_len]
    logprobs: torch.Tensor        # [B, completion_len]
    rewards: torch.Tensor         # [B]
    mask: torch.Tensor            # [B, completion_len]
    prompt_lens: torch.Tensor     # [B] - length of prompt for each sample
    ref_logprobs: torch.Tensor | None = None  # [B, completion_len] - from reference model


@dataclass
class RolloutSample:
    """Raw rollout data from vLLM - one prompt with N completions."""
    input_ids: torch.Tensor       # [N, seq_len] - prompt + completion
    completion_ids: torch.Tensor  # [N, completion_len]
    logprobs: torch.Tensor        # [N, completion_len]
    prompt_len: int               # Length of the prompt (same for all completions)
    finish_reasons: list[str]     # e.g. ["stop", "length", "stop", ...]
    completion_lens: list[int]    # Actual completion lengths before padding

    @classmethod
    def from_vllm(cls, response, pad_token_id: int):
        prompt = response.prompt_token_ids
        outputs = response.outputs
        n = len(outputs)
        prompt_len = len(prompt)

        seqs = [prompt + list(o.token_ids) for o in outputs]
        max_seq_len = max(len(s) for s in seqs)
        input_ids = torch.full((n, max_seq_len), pad_token_id, dtype=torch.long)
        for i, seq in enumerate(seqs):
            input_ids[i, :len(seq)] = torch.tensor(seq)

        max_completion_len = max(len(o.token_ids) for o in outputs)
        completion_ids = torch.full((n, max_completion_len), pad_token_id, dtype=torch.long)
        logprobs = torch.zeros((n, max_completion_len), dtype=torch.float32)

        finish_reasons = []
        completion_lens = []

        for i, o in enumerate(outputs):
            L = len(o.token_ids)
            completion_ids[i, :L] = torch.tensor(o.token_ids)
            logprobs[i, :L] = torch.tensor([o.logprobs[j][o.token_ids[j]].logprob for j in range(L)])
            finish_reasons.append(getattr(o, 'finish_reason', None) or 'unknown')
            completion_lens.append(L)

        return cls(input_ids, completion_ids, logprobs, prompt_len, finish_reasons, completion_lens)


@dataclass
class TrainSample:
    """Training-ready sample: rollout + rewards + ref_logprobs."""
    rollout: RolloutSample
    rewards: list[float]
    ref_logprobs: torch.Tensor


@dataclass
class BatchStats:
    """Statistics about a training batch for monitoring padding/truncation waste."""
    seq_lens: list[int]  # Per-completion actual sequence lengths
    completion_lens: list[int]  # Per-completion actual completion lengths
    padded_seq_len: int  # Final padded length
    padded_completion_len: int  # Final padded length
    finish_reasons: dict[str, int]  # e.g. {"stop": 10, "length": 6}
    rewards: list[float]

    @classmethod
    def from_samples(
        cls,
        samples: list["TrainSample"],
        padded_seq_len: int,
        padded_completion_len: int,
    ) -> "BatchStats":
        """Compute stats from training samples."""
        seq_lens = []
        completion_lens = []
        finish_reasons: dict[str, int] = {}
        rewards = []

        for s in samples:
            r = s.rollout
            n = r.input_ids.size(0)
            for i in range(n):
                comp_len = r.completion_lens[i]
                seq_lens.append(r.prompt_len + comp_len)
                completion_lens.append(comp_len)
                reason = r.finish_reasons[i]
                finish_reasons[reason] = finish_reasons.get(reason, 0) + 1
            rewards.extend(s.rewards)

        return cls(
            seq_lens=seq_lens,
            completion_lens=completion_lens,
            padded_seq_len=padded_seq_len,
            padded_completion_len=padded_completion_len,
            finish_reasons=finish_reasons,
            rewards=rewards,
        )

    def trace(self, tracer) -> None:
        """Emit all batch statistics to tracer."""
        n = len(self.seq_lens)
        seq_padding_tokens = [self.padded_seq_len - l for l in self.seq_lens]
        completion_padding_tokens = [self.padded_completion_len - l for l in self.completion_lens]
        total_seq_padding = sum(seq_padding_tokens)
        total_comp_padding = sum(completion_padding_tokens)

        tracer.counter("batch.padding", {
            "seq_padding_pct": 100 * total_seq_padding / (n * self.padded_seq_len) if n else 0,
            "completion_padding_pct": 100 * total_comp_padding / (n * self.padded_completion_len) if n else 0,
            "padded_seq_len": self.padded_seq_len,
            "padded_completion_len": self.padded_completion_len,
            "seq_lens": self.seq_lens,
            "completion_lens": self.completion_lens,
            "seq_padding_tokens": seq_padding_tokens,
            "completion_padding_tokens": completion_padding_tokens,
        })

        tracer.counter("batch.finish_reasons", self.finish_reasons)

        tracer.counter("batch.reward_vs_len", {
            "rewards": self.rewards,
            "completion_lens": self.completion_lens,
        })

        tracer.counter("rewards", {
            "mean": sum(self.rewards) / len(self.rewards) if self.rewards else 0,
            "max": max(self.rewards) if self.rewards else 0,
            "num_positive": sum(1 for r in self.rewards if r > 0),
        })


def make_batch(
    samples: list[TrainSample],
    pad_token_id: int,
    max_seq_len: int | None = None,
    max_completion_len: int | None = None,
) -> tuple[Batch, BatchStats]:
    """Combine training samples into a batch."""
    def pad_cat(tensors, pad_value=0, fixed_len=None):
        max_len = fixed_len if fixed_len is not None else max(t.shape[1] for t in tensors)
        return torch.cat([F.pad(t, (0, max_len - t.shape[1]), value=pad_value) for t in tensors])

    rollouts = [s.rollout for s in samples]

    # Determine padded lengths
    natural_max_seq = max(r.input_ids.shape[1] for r in rollouts)
    natural_max_comp = max(r.completion_ids.shape[1] for r in rollouts)
    padded_seq_len = max_seq_len or natural_max_seq
    padded_completion_len = max_completion_len or natural_max_comp

    # Build prompt_lens
    prompt_lens_list = [r.prompt_len for r in rollouts for _ in range(r.input_ids.size(0))]

    # Build ref_logprobs if present
    has_ref = samples[0].ref_logprobs is not None
    ref_logprobs = pad_cat([s.ref_logprobs for s in samples], fixed_len=padded_completion_len) if has_ref else None

    batch = Batch(
        input_ids=pad_cat([r.input_ids for r in rollouts], pad_value=pad_token_id, fixed_len=max_seq_len),
        completion_ids=pad_cat([r.completion_ids for r in rollouts], pad_value=pad_token_id, fixed_len=max_completion_len),
        logprobs=pad_cat([r.logprobs for r in rollouts], fixed_len=max_completion_len),
        rewards=torch.cat([torch.tensor(s.rewards, dtype=torch.float32) for s in samples]),
        mask=pad_cat([(r.completion_ids != pad_token_id).float() for r in rollouts], fixed_len=max_completion_len),
        prompt_lens=torch.tensor(prompt_lens_list, dtype=torch.long),
        ref_logprobs=ref_logprobs,
    )

    stats = BatchStats.from_samples(samples, padded_seq_len, padded_completion_len)
    return batch, stats


async def grpo_samples(
    rollout,
    data_iter,
    buffer,
    *,
    verifier_fn: Callable,
    pad_token_id: int,
    prompts_per_batch: int,
    sampling_params: dict | None = None,
    num_epochs: int | None = None,
    max_steps: int | None = None,
    max_seq_len: int | None = None,
    max_completion_len: int | None = None,
    max_staleness: int = 0,
):
    """Run generation/verification and yield training batches. """
    if num_epochs is None and max_steps is None:
        raise ValueError("Must specify either num_epochs or max_steps")
    if num_epochs is not None and max_steps is not None:
        raise ValueError("Cannot specify both num_epochs and max_steps")

    tracer = get_tracer()
    sp = {**(sampling_params or {}), "logprobs": 0}
    num_producers = rollout.num_replicas

    global_step = 0
    epoch = 0

    while True:
        # Check termination conditions
        if num_epochs is not None and epoch >= num_epochs:
            break
        if max_steps is not None and global_step >= max_steps:
            break

        # Start new epoch
        data_iter.new_epoch(seed=epoch)
        tracer.counter("epoch", {"epoch": epoch})

        done_count = 0

        async def produce():
            """Generate completions, compute rewards, push to buffer."""
            async def verify_and_push(response, problem, prompt, version):
                completions = [out.text for out in response.outputs]
                rewards = await verifier_fn(problem, completions, version=version)
                sample = RolloutSample.from_vllm(response, pad_token_id, rewards, version)
                await buffer.put(sample, version)
                log_rollout(
                    prompt_id=problem.get("prompt_id", "unknown"),
                    prompt=prompt,
                    completions=completions,
                    rewards=rewards,
                    version=version,
                )

            async with asyncio.TaskGroup() as tg:
                for item in data_iter:
                    version = rollout.model_version
                    response = await rollout.generate_single(item["template"], **sp)
                    tg.create_task(verify_and_push(response, item["problem"], item["prompt"], version))

            await buffer.put(DONE, -1)  # version=-1 means never evict

        producers = [asyncio.create_task(produce()) for _ in range(num_producers)]

        samples = []
        try:
            while done_count < num_producers:
                # Check if any producer crashed
                for p in producers:
                    if p.done() and p.exception():
                        raise p.exception()

                min_version = max(0, rollout.model_version - max_staleness)
                item = await buffer.pop(min_version)

                if isinstance(item, _Done):
                    done_count += 1
                    continue

                # Skip zero-variance samples (all completions have same reward)
                # This ensures batch_size means "valid prompts", not "total prompts"
                if torch.tensor(item.rewards, dtype=torch.float32).std() < 1e-6:
                    tracer.counter("skipped", {"zero_variance_samples": 1})
                    buffer.stats.record_filtered(item.version)
                    continue

                # Skip samples that exceed fixed lengths
                seq_len = item.input_ids.shape[1]
                comp_len = item.completion_ids.shape[1]
                if max_seq_len is not None and seq_len > max_seq_len:
                    print(f"[run] WARNING: dropping sample with seq_len={seq_len} > max_seq_len={max_seq_len}")
                    tracer.counter("skipped", {"seq_too_long": 1})
                    continue
                if max_completion_len is not None and comp_len > max_completion_len:
                    print(f"[run] WARNING: dropping sample with completion_len={comp_len} > max_completion_len={max_completion_len}")
                    tracer.counter("skipped", {"completion_too_long": 1})
                    continue

                samples.append(item)

                if len(samples) >= prompts_per_batch:
                    batch, batch_stats = make_batch(samples, pad_token_id, max_seq_len, max_completion_len)
                    batch_stats.trace(tracer)

                    samples = []
                    global_step += 1
                    yield global_step, epoch, batch

                    if max_steps is not None and global_step >= max_steps:
                        return

        finally:
            for t in producers:
                t.cancel()
            await asyncio.gather(*producers, return_exceptions=True)

        tracer.counter("epoch_complete", {"epoch": epoch, "steps_in_epoch": global_step})
        epoch += 1
