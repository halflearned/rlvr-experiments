import torch
import torch.nn.functional as F
from torch.distributed.tensor import DTensor


def compute_logprobs(
    logits: torch.Tensor | DTensor,
    input_ids: torch.Tensor,
    temperature: float = 1.0,
    align: bool = True,
) -> torch.Tensor:
    """
    Computes the log probabilities of the input tokens given the model logits and temperature.
    Always converts inputs to fp32 for numerical stability.

    When `align=True`, assumes `input_ids` contains only the target tokens (e.g., the
    completion portion) and slices the logits so that each row predicts the next token in
    `input_ids`. When `align=False`, assumes logits and input_ids are already one-to-one.
    """
    import os
    rank = int(os.environ.get("RANK", 0))

    # DEBUG: Check if logits is DTensor and what we receive
    if isinstance(logits, DTensor):
        print(f"[compute_logprobs Rank {rank}] INPUT logits is DTensor! placements={logits.placements}, shape={logits.shape}")
        # Convert DTensor to full tensor for correct cross-entropy
        logits = logits.full_tensor()
        print(f"[compute_logprobs Rank {rank}] After full_tensor: shape={logits.shape}, min={logits.min():.4f}, max={logits.max():.4f}")
    else:
        print(f"[compute_logprobs Rank {rank}] INPUT logits is Tensor shape={logits.shape}, min={logits.min():.4f}, max={logits.max():.4f}, sample[0,0,:3]={logits[0,0,:3]}")

    scaled_logits = logits if temperature == 1.0 else logits / temperature

    if align:
        target_len = input_ids.size(1)
        if target_len == 0:
            return torch.zeros(
                (logits.size(0), 0),
                device=input_ids.device,
                dtype=torch.float32,
            )
        # logits[:, i] predicts token at position i+1, so drop the last position
        sliced_logits = scaled_logits[:, -target_len - 1 : -1, :]
    else:
        sliced_logits = scaled_logits

    sliced_logits = sliced_logits.to(input_ids.device).float()

    batch_size, seq_len, vocab_size = sliced_logits.shape
    logprobs = -F.cross_entropy(
        sliced_logits.reshape(-1, vocab_size),
        input_ids.reshape(-1).long(),
        reduction="none",
    )

    return logprobs.reshape(batch_size, seq_len)
