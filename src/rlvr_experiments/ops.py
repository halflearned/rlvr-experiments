import torch
import torch.nn.functional as F
from torch.distributed.tensor import DTensor

import logging
import time
import os

logger = logging.getLogger(__name__)

# Enable profiling via environment variable
_PROFILE_OPS = os.environ.get("RLVR_PROFILE_OPS", "0") == "1"


def compute_logprobs(
    logits: torch.Tensor | DTensor,
    input_ids: torch.Tensor,
    temperature: float = 1.0,
    align: bool = True,
    prompt_lens: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Computes the log probabilities of the input tokens given the model logits and temperature.
    Always converts inputs to fp32 for numerical stability.

    When `align=True`, assumes `input_ids` contains only the target tokens (e.g., the
    completion portion) and slices the logits so that each row predicts the next token in
    `input_ids`. When `align=False`, assumes logits and input_ids are already one-to-one.

    If `prompt_lens` is provided (shape [B]), uses it to correctly slice logits for each
    sample based on where the completion starts. Otherwise, assumes completion is at the
    end of the sequence (which only works if there's no trailing padding).

    Supports DTensor inputs - when loss_parallel is enabled (disable_loss_parallel=false),
    the cross_entropy will handle vocab-sharded logits natively without all-gathers.
    """
    if _PROFILE_OPS:
        torch.cuda.synchronize()
        t0 = time.perf_counter()

    scaled_logits = logits if temperature == 1.0 else logits / temperature

    if align:
        target_len = input_ids.size(1)
        if target_len == 0:
            return torch.zeros(
                (logits.size(0), 0),
                device=input_ids.device,
                dtype=torch.float32,
            )

        if prompt_lens is not None:
            # Per-sample slicing based on prompt length
            # logits[:, prompt_len-1 : prompt_len-1+target_len] predicts completion tokens
            batch_size = logits.size(0)
            sliced_list = []
            for i in range(batch_size):
                plen = prompt_lens[i].item()
                # logits[i, plen-1] predicts token at position plen (first completion token)
                start = plen - 1
                end = start + target_len
                sliced_list.append(scaled_logits[i, start:end, :])
            sliced_logits = torch.stack(sliced_list, dim=0)
        else:
            # Legacy: assume completion is at the end (no trailing padding)
            sliced_logits = scaled_logits[:, -target_len - 1 : -1, :]
    else:
        sliced_logits = scaled_logits

    if _PROFILE_OPS:
        torch.cuda.synchronize()
        t1 = time.perf_counter()

    # Keep bfloat16 if input is bfloat16 - cross_entropy supports it and saves memory
    sliced_logits = sliced_logits.to(input_ids.device)

    if _PROFILE_OPS:
        torch.cuda.synchronize()
        t2 = time.perf_counter()

    batch_size, seq_len, vocab_size = sliced_logits.shape
    logprobs = -F.cross_entropy(
        sliced_logits.reshape(-1, vocab_size),
        input_ids.reshape(-1).long(),
        reduction="none",
    )

    if _PROFILE_OPS:
        torch.cuda.synchronize()
        t3 = time.perf_counter()
        logger.info(
            f"[PROFILE compute_logprobs] "
            f"slice={1000*(t1-t0):.1f}ms, "
            f"to_float={1000*(t2-t1):.1f}ms, "
            f"cross_entropy={1000*(t3-t2):.1f}ms, "
            f"TOTAL={1000*(t3-t0):.1f}ms | "
            f"shape=[{batch_size}, {seq_len}, {vocab_size}]"
        )

    return logprobs.reshape(batch_size, seq_len)
