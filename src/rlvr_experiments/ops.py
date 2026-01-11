import torch
import torch.nn.functional as F
from torch.distributed.tensor import DTensor, Replicate, DeviceMesh

import logging
import time
import os

logger = logging.getLogger(__name__)

# Enable profiling via environment variable
_PROFILE_OPS = os.environ.get("RLVR_PROFILE_OPS", "0") == "1"


def _get_prompt_groups(prompt_lens: torch.Tensor) -> list[tuple[int, int, int]]:
    """Find contiguous groups with same prompt length.

    Returns list of (start_idx, end_idx, prompt_len) tuples.
    Used for DTensor-compatible slicing when samples from the same prompt
    are contiguous in the batch.
    """
    groups = []
    start = 0
    current_len = prompt_lens[0].item()
    for i in range(1, len(prompt_lens)):
        if prompt_lens[i].item() != current_len:
            groups.append((start, i, current_len))
            start = i
            current_len = prompt_lens[i].item()
    groups.append((start, len(prompt_lens), current_len))
    return groups


def _compute_logprobs_dtensor(
    logits: DTensor,
    input_ids: torch.Tensor,
    prompt_lens: torch.Tensor,
    target_len: int,
) -> torch.Tensor:
    """DTensor-compatible compute_logprobs using group-based slicing.

    Instead of using gather (which doesn't work with vocab-sharded DTensor),
    we process each prompt group separately using simple slicing. This works
    because samples from the same prompt have the same prompt_len and are
    contiguous in the batch.

    Must be called inside a loss_parallel() context.
    """
    vocab_size = logits.shape[-1]
    mesh = logits.device_mesh

    # Get groups of samples with same prompt length
    groups = _get_prompt_groups(prompt_lens)

    # Process each group
    results = []
    for start, end, prompt_len in groups:
        # Simple slice - works with DTensor because we're slicing on batch and seq dims
        # which are replicated, not the vocab dim which is sharded
        group_logits = logits[start:end, prompt_len - 1 : prompt_len - 1 + target_len, :]
        group_ids = input_ids[start:end]

        # Wrap ids as replicated DTensor for cross_entropy compatibility
        group_ids_dt = DTensor.from_local(group_ids, mesh, [Replicate()])

        # Compute logprobs via cross_entropy
        group_logprobs = -F.cross_entropy(
            group_logits.reshape(-1, vocab_size),
            group_ids_dt.reshape(-1).long(),
            reduction="none",
        )
        group_logprobs = group_logprobs.reshape(end - start, target_len)
        results.append(group_logprobs)

    # Concatenate results
    logprobs = torch.cat(results, dim=0)

    # Convert back to regular tensor if it's a DTensor
    if isinstance(logprobs, DTensor):
        logprobs = logprobs.to_local()

    return logprobs


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

    Supports DTensor inputs - when logits is a vocab-sharded DTensor (from TP), this
    function uses group-based slicing to avoid gather operations that don't work with
    sharded tensors. Must be called inside a loss_parallel() context when using DTensor.
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

        # DTensor path: use group-based slicing instead of gather
        if isinstance(scaled_logits, DTensor) and prompt_lens is not None:
            prompt_lens = prompt_lens.to(input_ids.device)
            logprobs = _compute_logprobs_dtensor(scaled_logits, input_ids, prompt_lens, target_len)

            if _PROFILE_OPS:
                torch.cuda.synchronize()
                t1 = time.perf_counter()
                logger.info(
                    f"[PROFILE compute_logprobs DTensor] "
                    f"TOTAL={1000*(t1-t0):.1f}ms | "
                    f"shape=[{logits.size(0)}, {target_len}, {logits.size(2)}]"
                )

            return logprobs

        # Regular tensor path
        if prompt_lens is not None:
            # Per-sample slicing based on prompt length using advanced indexing
            # logits[:, prompt_len-1 : prompt_len-1+target_len] predicts completion tokens
            batch_size = logits.size(0)
            seq_len = logits.size(1)
            vocab_size = logits.size(2)

            # Ensure prompt_lens is on same device as logits
            prompt_lens = prompt_lens.to(logits.device)

            # Build indices: for each sample i, we want positions [prompt_len[i]-1, ..., prompt_len[i]-1+target_len-1]
            # Shape: [batch, target_len]
            offsets = torch.arange(target_len, device=logits.device).unsqueeze(0)  # [1, T]
            starts = (prompt_lens - 1).unsqueeze(1)  # [B, 1]
            indices = starts + offsets  # [B, T]

            # Use gather to extract the slices
            # Expand indices to match logits shape [B, T, V]
            indices_expanded = indices.unsqueeze(-1).expand(-1, -1, vocab_size)  # [B, T, V]
            sliced_logits = scaled_logits.gather(1, indices_expanded)  # [B, T, V]
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
