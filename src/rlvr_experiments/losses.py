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
        rollout_logprobs,  # [B, T] – log π_{θ_old}
        rewards,             # [B] or [B, G] or [B, G*...] – scalar per sequence
        padding_mask,        # [B, T], 1 for tokens, 0 for pad
    ):
        # advantage normalization
        adv = (rewards - rewards.mean()) / (rewards.std(unbiased=False) + 1e-8)
        # broadcast over tokens if rewards is [B] or [B, G]
        while adv.ndim < trainer_logprobs.ndim:
            adv = adv.unsqueeze(-1)  # now same rank as [B, T...]

        # importance ratio
        ratio = torch.exp(trainer_logprobs - rollout_logprobs)  # π_θ / π_{θ_old}
        clipped_ratio = torch.clamp(ratio, 1.0 - self.eps, 1.0 + self.eps)
        
        unclipped_obj = ratio * adv
        clipped_obj = clipped_ratio * adv
        surrogate = torch.minimum(unclipped_obj, clipped_obj)  # PPO clip

        # kl-div per token, unbiased estimator
        log_ratio_ref = reference_logprobs - trainer_logprobs  # log(π_ref / π_θ)
        kl_t = torch.exp(log_ratio_ref) - log_ratio_ref - 1.0  # KL estimator

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
