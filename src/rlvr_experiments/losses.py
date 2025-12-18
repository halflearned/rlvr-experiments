import torch


class GRPOLoss(torch.nn.Module):
    """ following the deekseekmath paper """
    def __init__(self, beta: float, eps: float = 0.0):
        super().__init__()
        self.beta = beta
        self.eps = eps

    def forward(
        self,
        trainer_logprobs,    # [B, T] – log π_θ
        reference_logprobs,  # [B, T] – log π_ref
        rollout_logprobs,    # [B, T] – log π_{θ_old}
        rewards,             # [B] or [B, G] or [B, G*...] – scalar per sequence
        padding_mask,        # [B, T], 1 for tokens, 0 for pad
    ):
        # Handle DTensor inputs by extracting local tensors for operations that don't support DTensor
        from torch.distributed.tensor import DTensor

        def to_local(x):
            if isinstance(x, DTensor):
                return x.to_local()
            return x

        trainer_logprobs = to_local(trainer_logprobs).float()
        reference_logprobs = to_local(reference_logprobs).float()
        rollout_logprobs = to_local(rollout_logprobs).float()
        rewards = to_local(rewards).float()
        padding_mask = to_local(padding_mask).float()

        # DEBUG: Check inputs BEFORE any masking
        import os
        rank = int(os.environ.get("RANK", 0))
        print(f"[Rank {rank}] === LOSS INPUTS (before masking) ===")
        print(f"[Rank {rank}] trainer_logprobs: shape={trainer_logprobs.shape}, min={trainer_logprobs.min():.4f}, max={trainer_logprobs.max():.4f}, num_zeros={(trainer_logprobs == 0).sum()}")
        print(f"[Rank {rank}] reference_logprobs: shape={reference_logprobs.shape}, min={reference_logprobs.min():.4f}, max={reference_logprobs.max():.4f}, num_zeros={(reference_logprobs == 0).sum()}")
        print(f"[Rank {rank}] rollout_logprobs: shape={rollout_logprobs.shape}, min={rollout_logprobs.min():.4f}, max={rollout_logprobs.max():.4f}, num_zeros={(rollout_logprobs == 0).sum()}")
        print(f"[Rank {rank}] Sample trainer[0,:10]: {trainer_logprobs[0,:10]}")
        print(f"[Rank {rank}] Sample reference[0,:10]: {reference_logprobs[0,:10]}")
        print(f"[Rank {rank}] Sample rollout[0,:10]: {rollout_logprobs[0,:10]}")
        print(f"[Rank {rank}] padding_mask: shape={padding_mask.shape}, sum={padding_mask.sum()}, sample[0,:10]={padding_mask[0,:10]}")

        # Zero out padded positions BEFORE computing ratios/KL to avoid numerical issues
        # Different models may produce different logprobs for padded positions
        trainer_logprobs = trainer_logprobs * padding_mask
        reference_logprobs = reference_logprobs * padding_mask
        rollout_logprobs = rollout_logprobs * padding_mask

        # advantage normalization (fp32 to avoid bf16 underflow when rewards have low variance)
        reward_std = rewards.std(unbiased=False)
        adv = (rewards - rewards.mean()) / (reward_std.clamp_min(1e-6) + 1e-8)
        # broadcast over tokens if rewards is [B] or [B, G]
        while adv.ndim < trainer_logprobs.ndim:
            adv = adv.unsqueeze(-1)  # now same rank as [B, T...]

        # importance ratio
        log_ratio_policy = trainer_logprobs - rollout_logprobs
        print(f"[Rank {rank}] log_ratio_policy (trainer-rollout): min={log_ratio_policy.min():.4f}, max={log_ratio_policy.max():.4f}, abs_max_idx={log_ratio_policy.abs().argmax()}")
        ratio = torch.exp(log_ratio_policy)  # π_θ / π_{θ_old}
        print(f"[Rank {rank}] ratio: min={ratio.min():.4e}, max={ratio.max():.4e}, has_inf={torch.isinf(ratio).any()}")

        clipped_ratio = torch.clamp(ratio, 1.0 - self.eps, 1.0 + self.eps)

        unclipped_obj = ratio * adv
        clipped_obj = clipped_ratio * adv
        surrogate = torch.minimum(unclipped_obj, clipped_obj)  # PPO clip
        print(f"[Rank {rank}] surrogate: min={surrogate.min():.4e}, max={surrogate.max():.4e}")

        # kl-div per token, unbiased estimator
        log_ratio_ref = reference_logprobs - trainer_logprobs  # log(π_ref / π_θ)
        print(f"[Rank {rank}] log_ratio_ref (ref-trainer): min={log_ratio_ref.min():.4f}, max={log_ratio_ref.max():.4f}")

        # DEBUG: Find the worst position and print the actual values
        worst_idx = log_ratio_ref.abs().argmax()
        worst_row = worst_idx // log_ratio_ref.shape[1]
        worst_col = worst_idx % log_ratio_ref.shape[1]
        print(f"[Rank {rank}] WORST POSITION: [{worst_row}, {worst_col}]")
        print(f"[Rank {rank}]   reference_logprobs[{worst_row},{worst_col}] = {reference_logprobs[worst_row, worst_col]:.4f}")
        print(f"[Rank {rank}]   trainer_logprobs[{worst_row},{worst_col}] = {trainer_logprobs[worst_row, worst_col]:.4f}")
        print(f"[Rank {rank}]   rollout_logprobs[{worst_row},{worst_col}] = {rollout_logprobs[worst_row, worst_col]:.4f}")
        print(f"[Rank {rank}]   padding_mask[{worst_row},{worst_col}] = {padding_mask[worst_row, worst_col]:.0f}")
        print(f"[Rank {rank}]   log_ratio_ref at worst = {log_ratio_ref[worst_row, worst_col]:.4f}")

        kl_t = torch.exp(log_ratio_ref) - log_ratio_ref - 1.0  # KL estimator
        print(f"[Rank {rank}] kl_t: min={kl_t.min():.4e}, max={kl_t.max():.4e}, has_inf={torch.isinf(kl_t).any()}")

        # per-token loss with mask
        per_token_loss = -(surrogate - self.beta * kl_t)
        print(f"[Rank {rank}] per_token_loss (before mask): min={per_token_loss.min():.4e}, max={per_token_loss.max():.4e}")
        per_token_loss = per_token_loss * padding_mask
        print(f"[Rank {rank}] per_token_loss (after mask): min={per_token_loss.min():.4e}, max={per_token_loss.max():.4e}")

        # length normalization
        lengths = padding_mask.sum(dim=1).clamp(min=1.0)

        # final loss
        per_group_loss = per_token_loss.sum(dim=1) / lengths
        loss = per_group_loss.mean()
        return loss



class SimpleGRPOLoss(torch.nn.Module):
    """Simplified GRPO Loss for simplified single step updates
    Copied from torchforge. In turn, they say: inspired by the Hugging Face TRL implementation
        https://github.com/huggingface/trl/blob/417915a3e4d3e3bc8d7b196594308b8eabf928be/trl/trainer/grpo_trainer.py#L1624.
    """

    def __init__(self, beta: float = 0.1):
        super().__init__()
        self.beta = beta

    def forward(self, logprobs, ref_logprobs, advantages, padding_mask):
        kl = torch.exp(ref_logprobs - logprobs) - (ref_logprobs - logprobs) - 1
        per_token_policy_loss = torch.exp(logprobs - logprobs.detach()) * advantages
        per_token_loss = -(per_token_policy_loss - self.beta * kl)
        loss = (
            ((per_token_loss * padding_mask).sum(dim=1))
            / (padding_mask.sum(dim=1).clamp(min=1.0))
        ).mean()
        return loss
