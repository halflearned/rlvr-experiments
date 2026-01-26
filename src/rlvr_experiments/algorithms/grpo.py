"""Rollout types and batch utilities."""

from dataclasses import dataclass

import torch
import torch.nn.functional as F


class RewardStats:
    """Accumulates reward statistics across all samples (including filtered ones)."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.total_prompts = 0
        self.total_completions = 0
        self.total_reward_sum = 0.0
        self.all_correct = 0  # prompts where all completions correct
        self.all_wrong = 0    # prompts where all completions wrong
        self.used_prompts = 0  # prompts with variance (used for training)
        self.used_reward_sum = 0.0
        self.used_completions = 0

    def record(self, rewards: list[float], used: bool):
        """Record stats for one prompt's completions."""
        self.total_prompts += 1
        self.total_completions += len(rewards)
        self.total_reward_sum += sum(rewards)

        all_same = all(r == rewards[0] for r in rewards)
        if all_same:
            if rewards[0] > 0.5:
                self.all_correct += 1
            else:
                self.all_wrong += 1

        if used:
            self.used_prompts += 1
            self.used_reward_sum += sum(rewards)
            self.used_completions += len(rewards)

    def get_metrics(self) -> dict:
        """Return metrics dict, then reset."""
        if self.total_prompts == 0:
            return {}
        metrics = {
            "reward_used": self.used_reward_sum / self.used_completions if self.used_completions > 0 else 0.0,
            "reward_overall": self.total_reward_sum / self.total_completions if self.total_completions > 0 else 0.0,
            "frac_all_correct": self.all_correct / self.total_prompts,
            "frac_all_wrong": self.all_wrong / self.total_prompts,
        }
        self.reset()
        return metrics


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

    @classmethod
    def merge(cls, samples: list["RolloutSample"], pad_token_id: int) -> "RolloutSample":
        """Merge multiple RolloutSample chunks for the same prompt."""
        if not samples:
            raise ValueError("RolloutSample.merge requires at least one sample")
        prompt_len = samples[0].prompt_len
        if any(s.prompt_len != prompt_len for s in samples):
            raise ValueError("All RolloutSample chunks must share the same prompt_len")

        input_rows = [row for s in samples for row in s.input_ids]
        comp_rows = [row for s in samples for row in s.completion_ids]
        logprob_rows = [row for s in samples for row in s.logprobs]

        max_seq_len = max(r.numel() for r in input_rows)
        max_comp_len = max(r.numel() for r in comp_rows)

        input_ids = torch.full((len(input_rows), max_seq_len), pad_token_id, dtype=torch.long)
        completion_ids = torch.full((len(comp_rows), max_comp_len), pad_token_id, dtype=torch.long)
        logprobs = torch.zeros((len(logprob_rows), max_comp_len), dtype=torch.float32)

        for i, row in enumerate(input_rows):
            input_ids[i, : row.numel()] = row
        for i, row in enumerate(comp_rows):
            completion_ids[i, : row.numel()] = row
        for i, row in enumerate(logprob_rows):
            logprobs[i, : row.numel()] = row

        finish_reasons = [r for s in samples for r in s.finish_reasons]
        completion_lens = [l for s in samples for l in s.completion_lens]

        return cls(input_ids, completion_ids, logprobs, prompt_len, finish_reasons, completion_lens)


@dataclass
class TrainSample:
    """Training-ready sample: rollout + rewards + ref_logprobs."""
    rollout: RolloutSample
    rewards: list[float]
    ref_logprobs: torch.Tensor
    item_id: str  # prompt_id for status tracking
    trainer_version: int  # trainer version when weights were synced to vLLM
    dataset: str = "unknown"  # dataset/verifier_type for mixed training


@dataclass
class BatchStats:
    """Statistics about a training batch for monitoring padding/truncation waste."""
    seq_lens: list[int]  # Per-completion actual sequence lengths
    completion_lens: list[int]  # Per-completion actual completion lengths
    padded_seq_len: int  # Final padded length
    padded_completion_len: int  # Final padded length
    finish_reasons: dict[str, int]  # e.g. {"stop": 10, "length": 6}
    rewards: list[float]
    datasets: list[str]  # Per-completion dataset name

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
        datasets = []

        for s in samples:
            r = s.rollout
            n = r.input_ids.size(0)
            for i in range(n):
                comp_len = r.completion_lens[i]
                seq_lens.append(r.prompt_len + comp_len)
                completion_lens.append(comp_len)
                reason = r.finish_reasons[i]
                finish_reasons[reason] = finish_reasons.get(reason, 0) + 1
                datasets.append(s.dataset)
            rewards.extend(s.rewards)

        return cls(
            seq_lens=seq_lens,
            completion_lens=completion_lens,
            padded_seq_len=padded_seq_len,
            padded_completion_len=padded_completion_len,
            finish_reasons=finish_reasons,
            rewards=rewards,
            datasets=datasets,
        )

    def trace(self, tracer, step: int | None = None) -> None:
        """Emit all batch statistics to tracer.

        Args:
            tracer: The tracer instance to emit events to.
            step: Optional training step number to include in events.
        """
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
            **({"step": step} if step is not None else {}),
        })

        tracer.counter("batch.finish_reasons", self.finish_reasons)

        tracer.counter("batch.reward_vs_len", {
            "rewards": self.rewards,
            "completion_lens": self.completion_lens,
            "datasets": self.datasets,
            **({"step": step} if step is not None else {}),
        })

        tracer.counter("rewards", {
            "mean": sum(self.rewards) / len(self.rewards) if self.rewards else 0,
            "max": max(self.rewards) if self.rewards else 0,
            "num_positive": sum(1 for r in self.rewards if r > 0),
        })


def _bucket_size(size: int, buckets: list[int]) -> int:
    """Round up size to nearest bucket for torch.compile cache efficiency."""
    for b in buckets:
        if size <= b:
            return b
    return buckets[-1]  # Clamp to largest bucket


# Default buckets for sequence and completion lengths
SEQ_LEN_BUCKETS = [256, 384, 512, 640, 768, 896, 1024]
COMPLETION_LEN_BUCKETS = [128, 256, 384, 512]


def make_batch(
    samples: list[TrainSample],
    pad_token_id: int,
    seq_len_buckets: list[int] | None = None,
    completion_len_buckets: list[int] | None = None,
) -> tuple[Batch, BatchStats]:
    """Combine training samples into a batch."""
    def pad_cat(tensors, pad_value=0, fixed_len=None):
        max_len = fixed_len if fixed_len is not None else max(t.shape[1] for t in tensors)
        return torch.cat([F.pad(t, (0, max_len - t.shape[1]), value=pad_value) for t in tensors])

    rollouts = [s.rollout for s in samples]

    # Compute natural lengths from data
    natural_max_seq = max(r.input_ids.shape[1] for r in rollouts)
    natural_max_comp = max(r.completion_ids.shape[1] for r in rollouts)
    max_prompt_len = max(r.prompt_len for r in rollouts)

    # Bucket completion length for torch.compile cache efficiency
    comp_buckets = completion_len_buckets or COMPLETION_LEN_BUCKETS
    padded_completion_len = natural_max_comp
    for bucket in comp_buckets:
        if natural_max_comp <= bucket:
            padded_completion_len = bucket
            break
    else:
        padded_completion_len = comp_buckets[-1]

    # Sequence length must be >= max_prompt_len + padded_completion_len for gather to work
    # (compute_logprobs gathers positions [prompt_len-1, prompt_len+completion_len-2])
    min_seq_len_needed = max_prompt_len + padded_completion_len

    # Bucket sequence length
    seq_buckets = seq_len_buckets or SEQ_LEN_BUCKETS
    padded_seq_len = min_seq_len_needed
    for bucket in seq_buckets:
        if min_seq_len_needed <= bucket:
            padded_seq_len = bucket
            break
    else:
        padded_seq_len = seq_buckets[-1]

    # If seq_len was clamped to max bucket, we must also clamp completion_len
    # to ensure completion fits within sequence: prompt_len + completion_len <= seq_len
    max_completion_for_seq = padded_seq_len - max_prompt_len
    if padded_completion_len > max_completion_for_seq:
        padded_completion_len = max_completion_for_seq

    # Build prompt_lens
    prompt_lens_list = [r.prompt_len for r in rollouts for _ in range(r.input_ids.size(0))]

    # Build ref_logprobs if present
    has_ref = samples[0].ref_logprobs is not None
    ref_logprobs = pad_cat([s.ref_logprobs for s in samples], fixed_len=padded_completion_len) if has_ref else None

    batch = Batch(
        input_ids=pad_cat([r.input_ids for r in rollouts], pad_value=pad_token_id, fixed_len=padded_seq_len),
        completion_ids=pad_cat([r.completion_ids for r in rollouts], pad_value=pad_token_id, fixed_len=padded_completion_len),
        logprobs=pad_cat([r.logprobs for r in rollouts], fixed_len=padded_completion_len),
        rewards=torch.cat([torch.tensor(s.rewards, dtype=torch.float32) for s in samples]),
        mask=pad_cat([(r.completion_ids != pad_token_id).float() for r in rollouts], fixed_len=padded_completion_len),
        prompt_lens=torch.tensor(prompt_lens_list, dtype=torch.long),
        ref_logprobs=ref_logprobs,
    )

    stats = BatchStats.from_samples(samples, padded_seq_len, padded_completion_len)
    print(f"[make_batch] B={batch.input_ids.shape[0]}, seq_len={padded_seq_len}, comp_len={padded_completion_len}")
    return batch, stats

