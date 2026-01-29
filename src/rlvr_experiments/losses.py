import torch
import json
import os
from pathlib import Path

from .ops import compute_logprobs


_KL_SPIKE_DUMP_DIR = Path("/tmp/kl_spike_dumps")
_KL_SPIKE_THRESHOLD = 1000.0  # dump when kl_max exceeds this


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
        trainer_logprobs, token_entropy = compute_logprobs(
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

        # KL penalty (Schulman k3 estimator, matching torchforge):
        # r = log(π_ref / π_θ), then KL ≈ exp(r) - r - 1 estimates KL(π_θ ∥ π_ref)
        log_ratio_ref = ref_lp - trainer_logprobs  # log(π_ref / π_θ)
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
                entropy_mean = token_entropy.to(device)[valid].mean().item()
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

            # Dump token-level data when kl_max spikes
            if kl_max > _KL_SPIKE_THRESHOLD:
                _KL_SPIKE_DUMP_DIR.mkdir(parents=True, exist_ok=True)
                import time
                dump_file = _KL_SPIKE_DUMP_DIR / f"spike_{int(time.time()*1000)}.pt"

                # Find the token with max KL
                kl_flat = kl_t[valid]
                max_idx = kl_flat.argmax().item()

                # Get indices in original tensor
                valid_indices = valid.nonzero(as_tuple=False)
                max_pos = valid_indices[max_idx]  # [batch_idx, token_idx]
                batch_idx, token_idx = max_pos[0].item(), max_pos[1].item()

                dump_data = {
                    "kl_max": kl_max,
                    "kl_mean": kl_mean,
                    "max_batch_idx": batch_idx,
                    "max_token_idx": token_idx,
                    "max_token_id": response[batch_idx, token_idx].item(),
                    "trainer_logprob_at_max": trainer_logprobs[batch_idx, token_idx].item(),
                    "ref_logprob_at_max": ref_lp[batch_idx, token_idx].item(),
                    "rollout_logprob_at_max": old_lp[batch_idx, token_idx].item(),
                    "log_ratio_ref_at_max": log_ratio_ref[batch_idx, token_idx].item(),
                    # Save full tensors for the problematic sequence
                    "response_seq": response[batch_idx].cpu(),
                    "trainer_logprobs_seq": trainer_logprobs[batch_idx].cpu(),
                    "ref_logprobs_seq": ref_lp[batch_idx].cpu(),
                    "rollout_logprobs_seq": old_lp[batch_idx].cpu(),
                    "kl_seq": kl_t[batch_idx].cpu(),
                    "mask_seq": mask[batch_idx].cpu(),
                    # Also save prompt_lens for debugging
                    "prompt_len": prompt_lens[batch_idx].item() if prompt_lens is not None else None,
                }
                torch.save(dump_data, dump_file)
                print(f"[KL SPIKE] Dumped to {dump_file}: batch={batch_idx} token={token_idx} "
                      f"token_id={dump_data['max_token_id']} "
                      f"trainer_lp={dump_data['trainer_logprob_at_max']:.4f} "
                      f"ref_lp={dump_data['ref_logprob_at_max']:.4f} "
                      f"log_ratio_ref={dump_data['log_ratio_ref_at_max']:.4f}")

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
        trainer_logprobs, token_entropy = compute_logprobs(
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

        # KL penalty (Schulman k3 estimator): r = log(π_ref/π_θ)
        log_ratio_ref = ref_lp - trainer_logprobs  # log(π_ref / π_θ)
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
                entropy_mean = token_entropy.to(device)[valid].mean().item()
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



import torch
from .ops import compute_logprobs


class DAPOLoss(torch.nn.Module):
    """
    DAPO loss:
      - Clip-Higher / decoupled clipping: clip(ratio, 1-eps_low, 1+eps_high)
      - Token-level loss normalization within each prompt-group:
            loss = mean_over_prompts( sum_{i,t in group} loss_{i,t} / (#valid tokens in group) )

    Notes:
      - To match DAPO, set beta=0.0 (no KL penalty term).
      - Requires batches laid out as contiguous groups per prompt:
            [prompt0 sample0..G-1, prompt1 sample0..G-1, ...]
        If variable G per prompt, pass group_sizes instead of group_size.
    """

    def __init__(self, eps_low: float = 0.2, eps_high: float = 0.28, beta: float = 0.0):
        super().__init__()
        self.eps_low = float(eps_low)
        self.eps_high = float(eps_high)
        self.beta = float(beta)
        self._last_debug: dict | None = None

    def get_debug_metrics(self) -> dict | None:
        metrics = self._last_debug
        self._last_debug = None
        return metrics

    @staticmethod
    def _token_level_group_reduce(
        per_token_loss: torch.Tensor,  # [B, T] already masked (pad positions are 0)
        mask: torch.Tensor,            # [B, T] float {0,1}
        group_size: int | None,
        group_sizes: list[int] | None,
    ) -> torch.Tensor:
        # Returns a scalar tensor.
        if group_sizes is not None:
            B = per_token_loss.shape[0]
            if sum(group_sizes) != B:
                raise ValueError(f"sum(group_sizes) must equal B={B}, got {sum(group_sizes)}")
            losses = []
            idx = 0
            for g in group_sizes:
                if g <= 0:
                    raise ValueError(f"group_sizes must be positive, got {g}")
                l_sum = per_token_loss[idx : idx + g].sum()
                tok = mask[idx : idx + g].sum().clamp_min(1.0)
                losses.append(l_sum / tok)
                idx += g
            return torch.stack(losses).mean()

        if group_size is None:
            return per_token_loss.sum() / mask.sum().clamp_min(1.0)

        B = per_token_loss.shape[0]
        if B % group_size != 0:
            raise ValueError(f"B={B} must be divisible by group_size={group_size}")

        # [num_groups, G, T]
        l = per_token_loss.view(-1, group_size, per_token_loss.shape[1])
        m = mask.view(-1, group_size, mask.shape[1])

        group_loss = l.sum(dim=(1, 2)) / m.sum(dim=(1, 2)).clamp_min(1.0)  # [num_groups]
        return group_loss.mean()

    def forward(
        self,
        logits: torch.Tensor,            # [B, seq_len, vocab]
        response: torch.Tensor,          # [B, T]
        rollout_logprobs: torch.Tensor,  # [B, T] logprobs under behavior/old policy
        advantages: torch.Tensor,        # [B] (typically GRPO-style group-normalized rewards)
        padding_mask: torch.Tensor,      # [B, T] 1 for valid response tokens, 0 for pad
        prompt_lens: torch.Tensor | None = None,
        temperature: float = 1.0,
        *,
        group_size: int | None = None,
        group_sizes: list[int] | None = None,
        ref_logprobs: torch.Tensor | None = None,  # only used if beta>0
    ) -> torch.Tensor:
        trainer_logprobs, token_entropy = compute_logprobs(
            logits,
            response,
            temperature=temperature,
            prompt_lens=prompt_lens,
        )  # [B, T]

        device = trainer_logprobs.device
        old_lp = rollout_logprobs.to(device=device, dtype=torch.float32)
        mask = padding_mask.to(device=device, dtype=torch.float32)
        adv = advantages.to(device=device, dtype=torch.float32).unsqueeze(-1)  # [B, 1]

        # PPO ratio: πθ / πold
        log_ratio = trainer_logprobs - old_lp
        ratio = torch.exp(log_ratio)

        # Clip-Higher / decoupled clipping range
        ratio_clip = torch.clamp(ratio, 1.0 - self.eps_low, 1.0 + self.eps_high)

        # PPO surrogate
        surrogate = torch.minimum(ratio * adv, ratio_clip * adv)

        # Optional KL penalty (DAPO sets this to 0 / removes KL)
        if self.beta > 0.0:
            if ref_logprobs is None:
                raise ValueError("ref_logprobs must be provided when beta > 0")
            ref_lp = ref_logprobs.to(device=device, dtype=torch.float32)
            # KL penalty (Schulman k3 estimator): r = log(π_ref/π_θ)
            log_ratio_ref = ref_lp - trainer_logprobs  # log(π_ref / π_θ)
            kl_t = torch.exp(log_ratio_ref) - log_ratio_ref - 1.0
            per_token_loss = -(surrogate - self.beta * kl_t)
        else:
            kl_t = None
            per_token_loss = -surrogate

        # Apply padding mask
        per_token_loss = per_token_loss * mask  # [B, T], pad positions are 0

        # Token-level (per-prompt-group) normalization
        loss = self._token_level_group_reduce(per_token_loss, mask, group_size, group_sizes)

        # Debug (valid tokens only)
        with torch.no_grad():
            valid = mask.bool()
            if valid.any():
                ratio_max = ratio[valid].max().item()
                entropy_mean = token_entropy.to(device)[valid].mean().item()
                up_clip_frac = (ratio[valid] > 1.0 + self.eps_high).float().mean().item()
                low_clip_frac = (ratio[valid] < 1.0 - self.eps_low).float().mean().item()
                clip_frac = ((ratio[valid] != ratio_clip[valid]).float().mean().item())
                if kl_t is not None:
                    kl_mean = kl_t[valid].mean().item()
                    kl_max = kl_t[valid].max().item()
                else:
                    kl_mean = kl_max = 0.0
            else:
                ratio_max = entropy_mean = up_clip_frac = low_clip_frac = clip_frac = 0.0
                kl_mean = kl_max = 0.0

            self._last_debug = {
                "ratio_max": ratio_max,
                "entropy_mean": entropy_mean,
                "clip_frac": clip_frac,
                "up_clip_frac": up_clip_frac,
                "low_clip_frac": low_clip_frac,
                "kl_mean": kl_mean,
                "kl_max": kl_max,
            }

        return loss
