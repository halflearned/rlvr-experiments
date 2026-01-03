import torch

from .ops import compute_logprobs


def _rewards_to_advantages(rewards: torch.Tensor) -> torch.Tensor:
    """GRPO: normalize rewards to zero mean, unit variance."""
    return (rewards - rewards.mean()) / rewards.std().clamp(min=1e-6)


class GRPOLoss(torch.nn.Module):
    """
    GRPO Loss following the DeepSeekMath paper.

    Takes logits directly (not logprobs) and computes logprobs internally,
    following the torchforge pattern for proper DTensor gradient handling.
    """
    def __init__(self, beta: float, eps: float = 0.0):
        super().__init__()
        self.beta = beta
        self.eps = eps

    def forward(
        self,
        logits: torch.Tensor,           # [B, seq_len, vocab] – model output logits
        response: torch.Tensor,         # [B, T] – completion token ids
        ref_logprobs: torch.Tensor,     # [B, T] – pre-computed reference log π_ref
        rollout_logprobs: torch.Tensor, # [B, T] – pre-computed rollout log π_{θ_old}
        rewards: torch.Tensor,          # [B] – rewards
        padding_mask: torch.Tensor,     # [B, T], 1 for tokens, 0 for pad
        prompt_lens: torch.Tensor | None = None,  # [B] – prompt lengths for proper slicing
    ):
        # Compute trainer logprobs from logits (keeps gradient flow)
        trainer_logprobs = compute_logprobs(logits, response, prompt_lens=prompt_lens)

        # Move other tensors to same device, ensure float32
        ref_logprobs = ref_logprobs.to(trainer_logprobs.device).float()
        rollout_logprobs = rollout_logprobs.to(trainer_logprobs.device).float()
        padding_mask = padding_mask.to(trainer_logprobs.device).float()

        # Zero out padded positions BEFORE computing ratios/KL to avoid numerical issues
        trainer_logprobs_masked = trainer_logprobs * padding_mask
        ref_logprobs = ref_logprobs * padding_mask
        rollout_logprobs = rollout_logprobs * padding_mask

        # Compute advantages from rewards and broadcast over tokens
        adv = _rewards_to_advantages(rewards).to(trainer_logprobs.device).float()
        while adv.ndim < trainer_logprobs_masked.ndim:
            adv = adv.unsqueeze(-1)

        # importance ratio
        log_ratio_policy = trainer_logprobs_masked - rollout_logprobs
        ratio = torch.exp(log_ratio_policy)  # π_θ / π_{θ_old}

        clipped_ratio = torch.clamp(ratio, 1.0 - self.eps, 1.0 + self.eps)

        unclipped_obj = ratio * adv
        clipped_obj = clipped_ratio * adv
        surrogate = torch.minimum(unclipped_obj, clipped_obj)  # PPO clip

        # kl-div per token, unbiased estimator
        log_ratio_ref = ref_logprobs - trainer_logprobs_masked  # log(π_ref / π_θ)
        kl_t = torch.exp(log_ratio_ref) - log_ratio_ref - 1.0

        # Debug: log differences between logprobs
        with torch.no_grad():
            diff_trainer_ref = (trainer_logprobs_masked - ref_logprobs).abs()
            diff_trainer_rollout = (trainer_logprobs_masked - rollout_logprobs).abs()
            print(
                f"[GRPO DEBUG] "
                f"trainer_lp: [{trainer_logprobs_masked.min().item():.2f}, {trainer_logprobs_masked.max().item():.2f}]  "
                f"ref_lp: [{ref_logprobs.min().item():.2f}, {ref_logprobs.max().item():.2f}]  "
                f"rollout_lp: [{rollout_logprobs.min().item():.2f}, {rollout_logprobs.max().item():.2f}]  "
                f"|trainer-ref|_max: {diff_trainer_ref.max().item():.4f}  "
                f"|trainer-rollout|_max: {diff_trainer_rollout.max().item():.4f}  "
                f"kl_max: {kl_t.max().item():.4f}  "
                f"ratio_max: {ratio.max().item():.4f}",
                flush=True
            )

        # per-token loss with mask
        per_token_loss = -(surrogate - self.beta * kl_t)
        per_token_loss = per_token_loss * padding_mask

        # length normalization
        lengths = padding_mask.sum(dim=1).clamp(min=1.0)

        # final loss
        per_group_loss = per_token_loss.sum(dim=1) / lengths
        loss = per_group_loss.mean()
        return loss


class SimpleGRPOLoss(torch.nn.Module):
    """
    Simplified GRPO Loss following the torchforge pattern.

    Takes logits directly and computes logprobs internally for proper gradient handling.
    Inspired by the Hugging Face TRL implementation:
    https://github.com/huggingface/trl/blob/417915a3e4d3e3bc8d7b196594308b8eabf928be/trl/trainer/grpo_trainer.py#L1624
    """

    def __init__(self, beta: float = 0.1, **kwargs):
        super().__init__()
        self.beta = beta

    def forward(
        self,
        logits: torch.Tensor,       # [B, seq_len, vocab] – model output logits
        response: torch.Tensor,     # [B, T] – completion token ids
        ref_logprobs: torch.Tensor, # [B, T] – pre-computed reference log π_ref
        rewards: torch.Tensor,      # [B] – rewards
        padding_mask: torch.Tensor, # [B, T], 1 for tokens, 0 for pad
    ):
        # Compute trainer logprobs from logits (keeps gradient flow)
        logprobs = compute_logprobs(logits, response)

        # Move other tensors to same device
        ref_logprobs = ref_logprobs.to(logprobs.device).float()
        padding_mask = padding_mask.to(logprobs.device).float()

        # Compute advantages from rewards
        advantages = _rewards_to_advantages(rewards).to(logprobs.device).float()
        if advantages.ndim == 1:
            advantages = advantages.unsqueeze(-1)

        kl = torch.exp(ref_logprobs - logprobs) - (ref_logprobs - logprobs) - 1
        per_token_policy_loss = torch.exp(logprobs - logprobs.detach()) * advantages
        per_token_loss = -(per_token_policy_loss - self.beta * kl)
        loss = (
            ((per_token_loss * padding_mask).sum(dim=1))
            / (padding_mask.sum(dim=1).clamp(min=1.0))
        ).mean()
        return loss
