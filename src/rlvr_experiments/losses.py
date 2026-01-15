import torch

from .ops import compute_logprobs


def compute_advantages(rewards: torch.Tensor, group_size: int | None = None) -> torch.Tensor:
    """Compute GRPO advantages from rewards.

    Normalizes rewards to zero mean, unit variance. When group_size is provided,
    normalization is done per-group (for multi-prompt batches where each prompt's
    completions should be normalized separately).

    Args:
        rewards: [B] tensor of rewards
        group_size: If provided, normalize within groups of this size.
                    B must be divisible by group_size.

    Returns:
        [B] tensor of normalized advantages

    This MUST be called before passing to GRPOLoss. The loss function expects
    pre-computed advantages, not raw rewards. This is required for:
    - Chunked loss computation (advantages must be normalized across full batch)
    - DDP training (each rank sees only a shard, but advantages need global stats)
    - Multi-prompt batches (each prompt's completions normalized separately)
    """
    if group_size is None:
        # Single group: normalize across entire batch
        mean = rewards.mean()
        std = rewards.std().clamp(min=1e-6)
        return (rewards - mean) / std
    else:
        # Multi-group: normalize within each group
        B = rewards.shape[0]
        if B % group_size != 0:
            raise ValueError(f"Batch size {B} not divisible by group_size {group_size}")

        grouped = rewards.view(-1, group_size)
        mean = grouped.mean(dim=1, keepdim=True)
        std = grouped.std(dim=1, keepdim=True).clamp(min=1e-6)
        normalized = (grouped - mean) / std
        return normalized.view(-1)


def compute_drgrpo_advantages(rewards: torch.Tensor, group_size: int | None = None) -> torch.Tensor:
    """Compute Dr. GRPO advantages from rewards.

    Subtracts the mean reward as an unbiased baseline, without std normalization.
    When group_size is provided, centering is done per-group (for multi-prompt
    batches where each prompt's completions should be centered separately).

    Args:
        rewards: [B] tensor of rewards
        group_size: If provided, center within groups of this size.
                    B must be divisible by group_size.

    Returns:
        [B] tensor of centered advantages

    This MUST be called before passing to DrGRPOLoss. The loss function expects
    pre-computed advantages, not raw rewards.
    """
    if group_size is None:
        mean = rewards.mean()
        return rewards - mean

    B = rewards.shape[0]
    if B % group_size != 0:
        raise ValueError(f"Batch size {B} not divisible by group_size {group_size}")

    grouped = rewards.view(-1, group_size)
    mean = grouped.mean(dim=1, keepdim=True)
    centered = grouped - mean
    return centered.view(-1)


class GRPOLoss(torch.nn.Module):
    """
    GRPO Loss following the DeepSeekMath paper.

    Takes logits directly (not logprobs) and computes logprobs internally,
    following the torchforge pattern for proper DTensor gradient handling.

    IMPORTANT: This loss expects pre-computed advantages, not raw rewards.
    Call compute_advantages(rewards) before passing to this loss function.
    This is required for correctness with chunked computation and DDP.
    """
    def __init__(self, beta: float, eps: float = 0.0):
        super().__init__()
        self.beta = beta
        self.eps = eps
        self._last_debug: dict | None = None

    def get_debug_metrics(self) -> dict | None:
        """Return debug metrics from the last forward pass, then clear them."""
        metrics = self._last_debug
        self._last_debug = None
        return metrics

    def forward(
        self,
        logits: torch.Tensor,           # [B, seq_len, vocab] – model output logits
        response: torch.Tensor,         # [B, T] – completion token ids
        ref_logprobs: torch.Tensor,     # [B, T] – pre-computed reference log π_ref
        rollout_logprobs: torch.Tensor, # [B, T] – pre-computed rollout log π_{θ_old}
        advantages: torch.Tensor,       # [B] – pre-computed advantages (use compute_advantages())
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

        # Broadcast advantages over tokens
        adv = advantages.to(trainer_logprobs.device).float()
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
            # Compute mean KL over non-padded tokens
            kl_sum = (kl_t * padding_mask).sum().item()
            kl_count = padding_mask.sum().item()
            kl_mean = kl_sum / kl_count if kl_count > 0 else 0.0
            kl_max = kl_t.max().item()
            ratio_max = ratio.max().item()
            diff_ref_max = diff_trainer_ref.max().item()
            diff_rollout_max = diff_trainer_rollout.max().item()
            print(
                f"[GRPO DEBUG] "
                f"trainer_lp: [{trainer_logprobs_masked.min().item():.2f}, {trainer_logprobs_masked.max().item():.2f}]  "
                f"ref_lp: [{ref_logprobs.min().item():.2f}, {ref_logprobs.max().item():.2f}]  "
                f"rollout_lp: [{rollout_logprobs.min().item():.2f}, {rollout_logprobs.max().item():.2f}]  "
                f"|trainer-ref|_max: {diff_ref_max:.4f}  "
                f"|trainer-rollout|_max: {diff_rollout_max:.4f}  "
                f"kl_mean: {kl_mean:.4f}  "
                f"kl_max: {kl_max:.4f}  "
                f"ratio_max: {ratio_max:.4f}",
                flush=True
            )
            # Store debug metrics for caller to emit to tracer
            self._last_debug = {
                "kl_mean": kl_mean,
                "kl_max": kl_max,
                "ratio_max": ratio_max,
                "diff_trainer_ref_max": diff_ref_max,
                "diff_trainer_rollout_max": diff_rollout_max,
            }

        # per-token loss with mask
        per_token_loss = -(surrogate - self.beta * kl_t)
        per_token_loss = per_token_loss * padding_mask

        # length normalization
        lengths = padding_mask.sum(dim=1).clamp(min=1.0)

        # final loss
        per_group_loss = per_token_loss.sum(dim=1) / lengths
        loss = per_group_loss.mean()
        return loss


class DrGRPOLoss(torch.nn.Module):
    """
    Dr. GRPO Loss (GRPO done right).

    Removes response-length normalization and uses advantages without std
    normalization. See compute_drgrpo_advantages for the recommended advantage
    computation.
    """

    def __init__(self, beta: float, eps: float = 0.0):
        super().__init__()
        self.beta = beta
        self.eps = eps
        self._last_debug: dict | None = None

    def get_debug_metrics(self) -> dict | None:
        """Return debug metrics from the last forward pass, then clear them."""
        metrics = self._last_debug
        self._last_debug = None
        return metrics

    def forward(
        self,
        logits: torch.Tensor,           # [B, seq_len, vocab] – model output logits
        response: torch.Tensor,         # [B, T] – completion token ids
        ref_logprobs: torch.Tensor,     # [B, T] – pre-computed reference log π_ref
        rollout_logprobs: torch.Tensor, # [B, T] – pre-computed rollout log π_{θ_old}
        advantages: torch.Tensor,       # [B] – pre-computed advantages (use compute_drgrpo_advantages())
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

        # Broadcast advantages over tokens
        adv = advantages.to(trainer_logprobs.device).float()
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
            # Compute mean KL over non-padded tokens
            kl_sum = (kl_t * padding_mask).sum().item()
            kl_count = padding_mask.sum().item()
            kl_mean = kl_sum / kl_count if kl_count > 0 else 0.0
            kl_max = kl_t.max().item()
            ratio_max = ratio.max().item()
            diff_ref_max = diff_trainer_ref.max().item()
            diff_rollout_max = diff_trainer_rollout.max().item()
            print(
                f"[DrGRPO DEBUG] "
                f"trainer_lp: [{trainer_logprobs_masked.min().item():.2f}, {trainer_logprobs_masked.max().item():.2f}]  "
                f"ref_lp: [{ref_logprobs.min().item():.2f}, {ref_logprobs.max().item():.2f}]  "
                f"rollout_lp: [{rollout_logprobs.min().item():.2f}, {rollout_logprobs.max().item():.2f}]  "
                f"|trainer-ref|_max: {diff_ref_max:.4f}  "
                f"|trainer-rollout|_max: {diff_rollout_max:.4f}  "
                f"kl_mean: {kl_mean:.4f}  "
                f"kl_max: {kl_max:.4f}  "
                f"ratio_max: {ratio_max:.4f}",
                flush=True
            )
            # Store debug metrics for caller to emit to tracer
            self._last_debug = {
                "kl_mean": kl_mean,
                "kl_max": kl_max,
                "ratio_max": ratio_max,
                "diff_trainer_ref_max": diff_ref_max,
                "diff_trainer_rollout_max": diff_rollout_max,
            }

        # per-token loss with mask
        per_token_loss = -(surrogate - self.beta * kl_t)
        per_token_loss = per_token_loss * padding_mask

        # final loss (no length normalization)
        per_group_loss = per_token_loss.sum(dim=1)
        loss = per_group_loss.mean()
        return loss


class SimpleGRPOLoss(torch.nn.Module):
    """
    Simplified GRPO Loss following the torchforge pattern.

    Takes logits directly and computes logprobs internally for proper gradient handling.
    Inspired by the Hugging Face TRL implementation:
    https://github.com/huggingface/trl/blob/417915a3e4d3e3bc8d7b196594308b8eabf928be/trl/trainer/grpo_trainer.py#L1624

    IMPORTANT: This loss expects pre-computed advantages, not raw rewards.
    Call compute_advantages(rewards) before passing to this loss function.
    """

    def __init__(self, beta: float = 0.1, **kwargs):
        super().__init__()
        self.beta = beta

    def forward(
        self,
        logits: torch.Tensor,       # [B, seq_len, vocab] – model output logits
        response: torch.Tensor,     # [B, T] – completion token ids
        ref_logprobs: torch.Tensor, # [B, T] – pre-computed reference log π_ref
        advantages: torch.Tensor,   # [B] – pre-computed advantages (use compute_advantages())
        padding_mask: torch.Tensor, # [B, T], 1 for tokens, 0 for pad
    ):
        # Compute trainer logprobs from logits (keeps gradient flow)
        logprobs = compute_logprobs(logits, response)

        # Move other tensors to same device
        ref_logprobs = ref_logprobs.to(logprobs.device).float()
        padding_mask = padding_mask.to(logprobs.device).float()

        # Broadcast advantages over tokens
        adv = advantages.to(logprobs.device).float()
        if adv.ndim == 1:
            adv = adv.unsqueeze(-1)

        kl = torch.exp(ref_logprobs - logprobs) - (ref_logprobs - logprobs) - 1
        per_token_policy_loss = torch.exp(logprobs - logprobs.detach()) * adv
        per_token_loss = -(per_token_policy_loss - self.beta * kl)
        loss = (
            ((per_token_loss * padding_mask).sum(dim=1))
            / (padding_mask.sum(dim=1).clamp(min=1.0))
        ).mean()
        return loss
