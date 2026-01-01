"""Batch and sample types for RL training."""

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class Batch:
    """A training batch with everything needed for GRPO."""
    input_ids: torch.Tensor       # [B, seq_len]
    completion_ids: torch.Tensor  # [B, completion_len]
    logprobs: torch.Tensor        # [B, completion_len]
    advantages: torch.Tensor      # [B, completion_len]
    mask: torch.Tensor            # [B, completion_len]
    avg_reward: float


@dataclass
class RolloutSample:
    """One prompt with N completions."""
    input_ids: torch.Tensor
    completion_ids: torch.Tensor
    logprobs: torch.Tensor
    rewards: list[float]

    @classmethod
    def from_vllm(cls, response, pad_token_id: int, rewards: list[float]):
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

        return cls(input_ids, completion_ids, logprobs, rewards)


def make_batch(samples: list[RolloutSample], pad_token_id: int) -> Batch | None:
    """Combine samples into a training batch. Returns None if all zero-variance."""
    valid = [s for s in samples if torch.tensor(s.rewards, dtype=torch.float32).std() > 1e-6]
    if not valid:
        return None

    def pad_cat(tensors, pad_value=0):
        max_len = max(t.shape[1] for t in tensors)
        return torch.cat([F.pad(t, (0, max_len - t.shape[1]), value=pad_value) for t in tensors])

    def advantages(s):
        rewards = torch.tensor(s.rewards, dtype=torch.float32)
        a = (rewards - rewards.mean()) / rewards.std().clamp(min=1e-6)
        return a[:, None].expand(-1, s.logprobs.shape[1])

    avg_reward = sum(sum(s.rewards) / len(s.rewards) for s in valid) / len(valid)

    return Batch(
        input_ids=pad_cat([s.input_ids for s in valid], pad_value=pad_token_id),
        completion_ids=pad_cat([s.completion_ids for s in valid], pad_value=pad_token_id),
        logprobs=pad_cat([s.logprobs for s in valid]),
        advantages=pad_cat([advantages(s) for s in valid]),
        mask=pad_cat([(s.completion_ids != pad_token_id).long() for s in valid]),
        avg_reward=avg_reward,
    )
