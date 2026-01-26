import torch

from .ops import compute_logprobs


import torch


def compute_grpo_advantages(
    rewards: torch.Tensor,
    group_size: int | None = None,
    group_sizes: list[int] | None = None,
) -> torch.Tensor:
    """
    Notes:
        - In multi-prompt batches, group_size MUST be the per-prompt sample count G,
          and the batch must be laid out as contiguous groups:
              [prompt0 sample0..G-1, prompt1 sample0..G-1, ...]
        - This function is pure tensor math; it does NOT do cross-rank synchronization.
          If using DDP and rewards/advantages need to be normalized globally per prompt,
          compute rewards per-prompt on a single rank or all-gather before calling this.
    """
    rewards = rewards.float()
    eps = 1e-6

    if group_sizes is not None:
        B = rewards.shape[0]
        if sum(group_sizes) != B:
            raise ValueError(f"sum(group_sizes) must equal B={B}, got {sum(group_sizes)}")
        out = torch.empty_like(rewards, dtype=torch.float32)
        idx = 0
        for g in group_sizes:
            if g <= 0:
                raise ValueError(f"group_sizes must be positive, got {g}")
            group = rewards[idx : idx + g]
            mean = group.mean()
            std = group.std(unbiased=False).clamp_min(eps)
            out[idx : idx + g] = (group - mean) / std
            idx += g
        return out

    if group_size is None:
        mean = rewards.mean()
        std = rewards.std(unbiased=False).clamp_min(eps)
        return (rewards - mean) / std

    B = rewards.shape[0]
    if B % group_size != 0:
        raise ValueError(f"B={B} must be divisible by group_size={group_size}")

    r = rewards.view(-1, group_size)                          # [num_groups, G]
    mean = r.mean(dim=1, keepdim=True)                        # [num_groups, 1]
    std = r.std(dim=1, keepdim=True, unbiased=False).clamp_min(eps)
    a = (r - mean) / std
    return a.view(-1)


def compute_drgrpo_advantages(
    rewards: torch.Tensor,
    group_size: int | None = None,
    group_sizes: list[int] | None = None,
) -> torch.Tensor:
    """
    Dr. GRPO advantages: per-group mean-centering only (no std normalization).

        A_i = r_i - mean(r_group)

    If group_size is None, this treats the entire batch as one group.

    Args:
        rewards: [B] rewards (float tensor).
        group_size: Optional group size G. If provided, B must be divisible by G.

    Returns:
        [B] centered advantages.

    Notes:
        Same batching/layout assumptions as compute_grpo_advantages().
    """
    rewards = rewards.float()

    if group_sizes is not None:
        B = rewards.shape[0]
        if sum(group_sizes) != B:
            raise ValueError(f"sum(group_sizes) must equal B={B}, got {sum(group_sizes)}")
        out = torch.empty_like(rewards, dtype=torch.float32)
        idx = 0
        for g in group_sizes:
            if g <= 0:
                raise ValueError(f"group_sizes must be positive, got {g}")
            group = rewards[idx : idx + g]
            mean = group.mean()
            out[idx : idx + g] = group - mean
            idx += g
        return out

    if group_size is None:
        return rewards - rewards.mean()

    B = rewards.shape[0]
    if B % group_size != 0:
        raise ValueError(f"B={B} must be divisible by group_size={group_size}")

    r = rewards.view(-1, group_size)                          # [num_groups, G]
    mean = r.mean(dim=1, keepdim=True)                        # [num_groups, 1]
    a = r - mean
    return a.view(-1)


class GRPOLoss(torch.nn.Module):
    """
    GRPO loss (DeepSeekMath-style): PPO surrogate with KL penalty and per-response
    length normalization.

    Expects:
      - advantages are precomputed (compute_grpo_advantages)
      - padding_mask is 1 for valid response tokens and 0 for pad
      - logits correspond to the model being optimized (grad flows through logits)
    """

    def __init__(self, beta: float, eps: float = 0.2):
        super().__init__()
        self.beta = float(beta)
        self.eps = float(eps)
        self._last_debug: dict | None = None

    def get_debug_metrics(self) -> dict | None:
        metrics = self._last_debug
        self._last_debug = None
        return metrics

    def forward(
        self,
        logits: torch.Tensor,            # [B, seq_len, vocab]
        response: torch.Tensor,          # [B, T]
        ref_logprobs: torch.Tensor,      # [B, T]
        rollout_logprobs: torch.Tensor,  # [B, T]
        advantages: torch.Tensor,        # [B]
        padding_mask: torch.Tensor,      # [B, T]
        prompt_lens: torch.Tensor | None = None,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        trainer_logprobs = compute_logprobs(
            logits,
            response,
            temperature=temperature,
            prompt_lens=prompt_lens,
        )

        device = trainer_logprobs.device
        ref_lp = ref_logprobs.to(device=device, dtype=torch.float32)
        old_lp = rollout_logprobs.to(device=device, dtype=torch.float32)
        mask = padding_mask.to(device=device, dtype=torch.float32)

        # broadcast advantages to [B, T]
        adv = advantages.to(device=device, dtype=torch.float32).unsqueeze(-1)

        # PPO ratio: πθ / πold
        log_ratio = trainer_logprobs - old_lp
        ratio = torch.exp(log_ratio)
        if self.eps > 0:
            ratio_clip = torch.clamp(ratio, 1.0 - self.eps, 1.0 + self.eps)
            surrogate = torch.minimum(ratio * adv, ratio_clip * adv)
        else:
            surrogate = ratio * adv

        # KL penalty (Schulman-style approximator used in GRPO implementations):
        # log_ratio_ref = log(πθ/πref)
        log_ratio_ref = trainer_logprobs - ref_lp
        kl_t = torch.exp(log_ratio_ref) - log_ratio_ref - 1.0

        per_token_loss = -(surrogate - self.beta * kl_t)
        per_token_loss = per_token_loss * mask

        lengths = mask.sum(dim=1).clamp_min(1.0)
        loss = (per_token_loss.sum(dim=1) / lengths).mean()

        # debug on valid tokens only
        with torch.no_grad():
            valid = mask.bool()
            if valid.any():
                kl_mean = kl_t[valid].mean().item()
                kl_max = kl_t[valid].max().item()
                ratio_max = ratio[valid].max().item()
                # Entropy approximation: mean negative logprob of selected tokens at rollout time
                entropy_mean = -old_lp[valid].mean().item()
                # Clip fraction: what fraction of tokens were clipped
                if self.eps > 0:
                    clipped = (ratio[valid] < 1.0 - self.eps) | (ratio[valid] > 1.0 + self.eps)
                    clip_frac = clipped.float().mean().item()
                else:
                    clip_frac = 0.0
            else:
                kl_mean = kl_max = ratio_max = entropy_mean = clip_frac = 0.0

            self._last_debug = {
                "kl_mean": kl_mean,
                "kl_max": kl_max,
                "ratio_max": ratio_max,
                "entropy_mean": entropy_mean,
                "clip_frac": clip_frac,
            }
            print(f"[GRPO DEBUG] {self._last_debug}")

        return loss


class DrGRPOLoss(torch.nn.Module):
    """
    Dr. GRPO loss: same surrogate + KL penalty, but removes per-response length
    normalization. Uses a fixed constant C for scale.

    Expects:
      - advantages are mean-centered (compute_drgrpo_advantages)
      - padding_mask is 1 for valid response tokens and 0 for pad
    """

    def __init__(self, beta: float, eps: float = 0.0, C: float = 2048.0):
        super().__init__()
        self.beta = float(beta)
        self.eps = float(eps)
        self.C = float(C)
        self._last_debug: dict | None = None

    def get_debug_metrics(self) -> dict | None:
        metrics = self._last_debug
        self._last_debug = None
        return metrics

    def forward(
        self,
        logits: torch.Tensor,            # [B, seq_len, vocab]
        response: torch.Tensor,          # [B, T]
        ref_logprobs: torch.Tensor,      # [B, T]
        rollout_logprobs: torch.Tensor,  # [B, T]
        advantages: torch.Tensor,        # [B]
        padding_mask: torch.Tensor,      # [B, T]
        prompt_lens: torch.Tensor | None = None,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        trainer_logprobs = compute_logprobs(
            logits,
            response,
            temperature=temperature,
            prompt_lens=prompt_lens,
        )

        device = trainer_logprobs.device
        ref_lp = ref_logprobs.to(device=device, dtype=torch.float32)
        old_lp = rollout_logprobs.to(device=device, dtype=torch.float32)
        mask = padding_mask.to(device=device, dtype=torch.float32)

        adv = advantages.to(device=device, dtype=torch.float32).unsqueeze(-1)

        log_ratio = trainer_logprobs - old_lp
        ratio = torch.exp(log_ratio)
        if self.eps > 0:
            ratio_clip = torch.clamp(ratio, 1.0 - self.eps, 1.0 + self.eps)
            surrogate = torch.minimum(ratio * adv, ratio_clip * adv)
        else:
            surrogate = ratio * adv

        log_ratio_ref = trainer_logprobs - ref_lp
        kl_t = torch.exp(log_ratio_ref) - log_ratio_ref - 1.0

        per_token_loss = -(surrogate - self.beta * kl_t)
        per_token_loss = per_token_loss * mask

        loss = (per_token_loss.sum(dim=1) / self.C).mean()

        with torch.no_grad():
            valid = mask.bool()
            if valid.any():
                kl_mean = kl_t[valid].mean().item()
                kl_max = kl_t[valid].max().item()
                ratio_max = ratio[valid].max().item()
                # Entropy approximation: mean negative logprob of selected tokens at rollout time
                entropy_mean = -old_lp[valid].mean().item()
                # Clip fraction: what fraction of tokens were clipped
                if self.eps > 0:
                    clipped = (ratio[valid] < 1.0 - self.eps) | (ratio[valid] > 1.0 + self.eps)
                    clip_frac = clipped.float().mean().item()
                else:
                    clip_frac = 0.0
            else:
                kl_mean = kl_max = ratio_max = entropy_mean = clip_frac = 0.0
            self._last_debug = {
                "kl_mean": kl_mean,
                "kl_max": kl_max,
                "ratio_max": ratio_max,
                "entropy_mean": entropy_mean,
                "clip_frac": clip_frac,
            }
            print(f"[GRPO DEBUG] {self._last_debug}")

        return loss
