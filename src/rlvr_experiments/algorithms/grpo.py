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


@dataclass
class RolloutSample:
    """One prompt with N completions."""
    input_ids: torch.Tensor
    completion_ids: torch.Tensor
    logprobs: torch.Tensor
    rewards: list[float]
    prompt_len: int  # Length of the prompt (same for all completions in this sample)
    version: int = -1  # Model version that generated this sample
    # Per-completion metadata from vLLM
    finish_reasons: list[str] | None = None  # e.g. ["stop", "length", "stop", ...]
    completion_lens: list[int] | None = None  # Actual completion lengths before padding

    @classmethod
    def from_vllm(cls, response, pad_token_id: int, rewards: list[float], version: int = -1):
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

        # Track per-completion metadata
        finish_reasons = []
        completion_lens = []

        for i, o in enumerate(outputs):
            L = len(o.token_ids)
            # Left-align to match input_ids (prompt + completion, then padding)
            completion_ids[i, :L] = torch.tensor(o.token_ids)
            logprobs[i, :L] = torch.tensor([o.logprobs[j][o.token_ids[j]].logprob for j in range(L)])
            # Track finish reason and actual length
            finish_reasons.append(getattr(o, 'finish_reason', None) or 'unknown')
            completion_lens.append(L)

        return cls(input_ids, completion_ids, logprobs, rewards, prompt_len, version,
                   finish_reasons, completion_lens)


@dataclass
class BatchStats:
    """Statistics about a training batch for monitoring padding/truncation waste."""
    # Sequence length stats
    actual_seq_lens: list[int]  # Per-sample actual sequence lengths
    padded_seq_len: int  # Final padded length
    seq_padding_tokens: int  # Total padding tokens added
    seq_padding_pct: float  # % of batch that is padding
    # Completion length stats
    actual_completion_lens: list[int]  # Per-sample actual completion lengths
    padded_completion_len: int  # Final padded length
    completion_padding_tokens: int  # Total padding tokens added
    completion_padding_pct: float  # % of completions that is padding
    # Stop reason distribution
    finish_reasons: dict[str, int]  # e.g. {"stop": 10, "length": 6}


def make_batch(
    samples: list[RolloutSample],
    pad_token_id: int,
    max_seq_len: int | None = None,
    max_completion_len: int | None = None,
) -> tuple[Batch, BatchStats]:
    """Combine pre-filtered samples into a training batch (pad and concatenate).

    Args:
        samples: List of RolloutSample objects (already filtered for validity)
        pad_token_id: Token ID to use for padding
        max_seq_len: If provided, pad input_ids to this fixed length (avoids recompilation)
        max_completion_len: If provided, pad completion_ids to this fixed length

    Returns:
        (Batch, BatchStats) tuple with the batch and statistics about padding/truncation.

    Note: Filtering (zero-variance, length limits) is done upstream in run().
    """
    assert samples, "make_batch requires non-empty samples list"

    def pad_cat(tensors, pad_value=0, fixed_len=None):
        max_len = fixed_len if fixed_len is not None else max(t.shape[1] for t in tensors)
        return torch.cat([F.pad(t, (0, max_len - t.shape[1]), value=pad_value) for t in tensors])

    # Build prompt_lens: each sample has N completions with the same prompt_len
    prompt_lens_list = []
    for s in samples:
        n_completions = s.input_ids.size(0)
        prompt_lens_list.extend([s.prompt_len] * n_completions)

    # Compute actual sequence lengths per completion
    actual_seq_lens = []
    actual_completion_lens = []
    finish_reasons: dict[str, int] = {}

    for s in samples:
        n_completions = s.input_ids.size(0)
        for i in range(n_completions):
            # Actual seq len = prompt_len + completion_len for this completion
            comp_len = s.completion_lens[i] if s.completion_lens else s.completion_ids.size(1)
            actual_seq_lens.append(s.prompt_len + comp_len)
            actual_completion_lens.append(comp_len)
            # Track finish reasons
            if s.finish_reasons:
                reason = s.finish_reasons[i]
                finish_reasons[reason] = finish_reasons.get(reason, 0) + 1

    # Compute padding stats
    natural_max_seq = max(t.shape[1] for t in [s.input_ids for s in samples])
    natural_max_comp = max(t.shape[1] for t in [s.completion_ids for s in samples])

    padded_seq_len = max_seq_len if max_seq_len else natural_max_seq
    padded_completion_len = max_completion_len if max_completion_len else natural_max_comp

    n_samples = len(actual_seq_lens)
    total_actual_seq_tokens = sum(actual_seq_lens)
    total_padded_seq_tokens = n_samples * padded_seq_len
    seq_padding_tokens = total_padded_seq_tokens - total_actual_seq_tokens

    total_actual_comp_tokens = sum(actual_completion_lens)
    total_padded_comp_tokens = n_samples * padded_completion_len
    completion_padding_tokens = total_padded_comp_tokens - total_actual_comp_tokens

    stats = BatchStats(
        actual_seq_lens=actual_seq_lens,
        padded_seq_len=padded_seq_len,
        seq_padding_tokens=seq_padding_tokens,
        seq_padding_pct=100 * seq_padding_tokens / total_padded_seq_tokens if total_padded_seq_tokens > 0 else 0,
        actual_completion_lens=actual_completion_lens,
        padded_completion_len=padded_completion_len,
        completion_padding_tokens=completion_padding_tokens,
        completion_padding_pct=100 * completion_padding_tokens / total_padded_comp_tokens if total_padded_comp_tokens > 0 else 0,
        finish_reasons=finish_reasons,
    )

    batch = Batch(
        input_ids=pad_cat([s.input_ids for s in samples], pad_value=pad_token_id, fixed_len=max_seq_len),
        completion_ids=pad_cat([s.completion_ids for s in samples], pad_value=pad_token_id, fixed_len=max_completion_len),
        logprobs=pad_cat([s.logprobs for s in samples], fixed_len=max_completion_len),
        rewards=torch.cat([torch.tensor(s.rewards, dtype=torch.float32) for s in samples]),
        mask=pad_cat([(s.completion_ids != pad_token_id).float() for s in samples], fixed_len=max_completion_len),
        prompt_lens=torch.tensor(prompt_lens_list, dtype=torch.long),
    )

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
    """Run generation/verification and yield training batches.

    Args:
        rollout: VLLMHandle for generation
        data_iter: DataIterator yielding {"templates": [...], "problems": [...]}
        buffer: DataBuffer for producer/consumer coordination
        verifier_fn: async (problem, completions) -> list[float]
        pad_token_id: For padding tensors
        prompts_per_batch: Number of prompts (valid, non-zero-variance) per training batch.
                          Total samples = prompts_per_batch * n (completions per prompt).
        sampling_params: vLLM sampling parameters (must include 'n' for completions per prompt)
        num_epochs: Number of epochs to run (mutually exclusive with max_steps)
        max_steps: Maximum steps to run (mutually exclusive with num_epochs)
        max_seq_len: If provided, pad input_ids to fixed length (avoids recompilation)
        max_completion_len: If provided, pad completions to fixed length
        max_staleness: Allow samples up to this many versions old (0 = only current version)

    Yields:
        (step, epoch, batch) tuples. Step is global (doesn't reset per epoch).
        Zero-variance samples are filtered upstream.

    Mid-epoch sync: When sync_titan_to_vllm is called, it increments rollout.model_version.
    Samples older than (current_version - max_staleness) are discarded from the buffer.
    """
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
        producer_batch_counts = {}  # Track batches fetched per producer

        async def produce():
            """Generate completions, compute rewards, push to buffer."""
            async def process_one(response, problem, prompt, version):
                completions = [out.text for out in response.outputs]
                rewards = await verifier_fn(problem, completions)
                sample = RolloutSample.from_vllm(response, pad_token_id, rewards, version)
                await buffer.put(sample, version)

                # Log rollout for debugging/analysis
                prompt_id = problem.get("prompt_id", "unknown")
                log_rollout(
                    prompt_id=prompt_id,
                    prompt=prompt,
                    completions=completions,
                    rewards=rewards,
                    version=version,
                )

            async with asyncio.TaskGroup() as tg:
                while True:
                    data_batch = await data_iter.next_batch()
                    if data_batch is None:
                        break
                    version = rollout.model_version
                    responses = await rollout.generate(data_batch["templates"], **sp)
                    for response, problem, prompt in zip(responses, data_batch["problems"], data_batch["prompts"]):
                        tg.create_task(process_one(response, problem, prompt, version))

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
                    all_rewards = [r for s in samples for r in s.rewards]
                    tracer.counter("rewards", {
                        "mean": sum(all_rewards) / len(all_rewards) if all_rewards else 0,
                        "max": max(all_rewards) if all_rewards else 0,
                        "num_positive": sum(1 for r in all_rewards if r > 0),
                    })

                    batch, batch_stats = make_batch(samples, pad_token_id, max_seq_len, max_completion_len)

                    # Emit batch stats to tracer (include raw lengths for cumulative quantile tracking)
                    # Compute per-sample padding token counts (padded_len - actual_len)
                    seq_padding_tokens = [batch_stats.padded_seq_len - l for l in batch_stats.actual_seq_lens]
                    completion_padding_tokens = [batch_stats.padded_completion_len - l for l in batch_stats.actual_completion_lens]
                    tracer.counter("batch.padding", {
                        "seq_padding_pct": batch_stats.seq_padding_pct,
                        "completion_padding_pct": batch_stats.completion_padding_pct,
                        "padded_seq_len": batch_stats.padded_seq_len,
                        "padded_completion_len": batch_stats.padded_completion_len,
                        # Raw lengths for histogram/quantile computation in viz
                        "seq_lens": batch_stats.actual_seq_lens,
                        "completion_lens": batch_stats.actual_completion_lens,
                        # Per-sample padding token counts for absolute waste visualization
                        "seq_padding_tokens": seq_padding_tokens,
                        "completion_padding_tokens": completion_padding_tokens,
                    })

                    # Emit finish reason distribution
                    if batch_stats.finish_reasons:
                        tracer.counter("batch.finish_reasons", batch_stats.finish_reasons)

                    # Emit reward vs completion length pairs for scatter plot
                    tracer.counter("batch.reward_vs_len", {
                        "rewards": all_rewards,
                        "completion_lens": batch_stats.actual_completion_lens,
                    })

                    samples = []
                    global_step += 1
                    yield global_step, epoch, batch

                    if max_steps is not None and global_step >= max_steps:
                        return

            # Yield any remaining samples as final batch
            # NOTE: Commented out to avoid CUDA graph recompilation from partial batches
            # if samples:
            #     batch = make_batch(samples, pad_token_id, max_seq_len, max_completion_len)
            #     if batch is not None:
            #         global_step += 1
            #         yield global_step, epoch, batch
            
            #         if max_steps is not None and global_step >= max_steps:
            #             return

        finally:
            for t in producers:
                t.cancel()
            await asyncio.gather(*producers, return_exceptions=True)

        tracer.counter("epoch_complete", {"epoch": epoch, "steps_in_epoch": global_step})
        epoch += 1
