"""HuggingFace-compatible precision functions for TorchTitan models.

These functions were created to achieve numerical parity between TorchTitan
and HuggingFace implementations. They are NOT actively used - experiments
showed they don't achieve exact parity and add complexity.

Kept here for reference and potential future investigation.

Key differences between HF and TorchTitan:
1. RMSNorm: HF computes variance in float32, PyTorch nn.RMSNorm stays in input dtype
2. RoPE: HF computes cos/sin in float32 and keeps them float32 during application
"""

import torch
import torch.nn as nn

from torchtitan.tools.logging import logger


class HFCompatibleRMSNorm(nn.Module):
    """RMSNorm that matches HuggingFace's implementation.

    HF computes variance in float32 for numerical stability, then casts back
    to the input dtype. PyTorch's nn.RMSNorm stays in the input dtype, causing
    small numerical differences that accumulate across layers.

    This implementation matches HF behavior for inference/export compatibility.
    """

    def __init__(self, normalized_shape, eps=1e-6, elementwise_affine=True, device=None, dtype=None):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine

        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(normalized_shape, device=device, dtype=dtype))
        else:
            self.register_parameter('weight', None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        # Compute in float32 for numerical stability (matches HF)
        x_f32 = x.to(torch.float32)
        variance = x_f32.pow(2).mean(-1, keepdim=True)
        x_normed = x_f32 * torch.rsqrt(variance + self.eps)
        x_normed = x_normed.to(input_dtype)

        if self.weight is not None:
            return self.weight * x_normed
        return x_normed

    def extra_repr(self):
        return f"{self.normalized_shape}, eps={self.eps}, elementwise_affine={self.elementwise_affine}"


def replace_rmsnorm_with_hf_compatible(module: nn.Module) -> None:
    """Recursively replace nn.RMSNorm with HFCompatibleRMSNorm in-place.

    This preserves the weights while changing the forward implementation
    to match HuggingFace's float32 variance computation.
    """
    for name, child in list(module.named_children()):
        if isinstance(child, nn.RMSNorm):
            # Create HF-compatible version with same config
            new_norm = HFCompatibleRMSNorm(
                normalized_shape=child.normalized_shape,
                eps=child.eps,
                elementwise_affine=child.weight is not None,
                device=child.weight.device if child.weight is not None else None,
                dtype=child.weight.dtype if child.weight is not None else None,
            )
            # Copy weights
            if child.weight is not None:
                new_norm.weight.data.copy_(child.weight.data)
            # Replace in parent
            setattr(module, name, new_norm)
        else:
            # Recurse into children
            replace_rmsnorm_with_hf_compatible(child)


def apply_rotary_emb_hf_compatible(
    xq: torch.Tensor, xk: torch.Tensor, rope_cache: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary embeddings matching HuggingFace's precision.

    HuggingFace computes RoPE in float32 (via torch.autocast disabled), while
    TorchTitan precomputes cos/sin and casts to bf16 when storing. This causes
    small numerical differences that accumulate across positions.

    This version recomputes cos/sin in float32 from the cached angles to match HF.
    The rope_cache contains [cos, sin] concatenated, but may be in bf16.
    We extract the underlying angles and recompute cos/sin in float32.
    """
    # Import here to avoid circular dependency
    from torchtitan.models.qwen3.model.model import reshape_for_broadcast, rotate_half

    head_dim = xq.shape[-1]
    input_dtype = xq.dtype

    # reshape for broadcast
    rope_cache = reshape_for_broadcast(rope_cache, xq)

    # Extract cos and sin from cache
    # rope_cache shape: [1, seq_len, 1, head_dim * 2] containing [cos | sin]
    cached_cos = rope_cache[..., :head_dim]
    cached_sin = rope_cache[..., head_dim:]

    # The cached values might be in bf16. To match HF, we need float32 cos/sin.
    # We can't perfectly recover the original angles from bf16 cos/sin,
    # but we can do the computation in float32 for better precision.
    #
    # Option 1: Just cast to float32 (still has bf16 quantization in values)
    # Option 2: Recompute from inv_freq (requires access to model config)
    #
    # For now, we cast to float32 and do computation there. This helps
    # because the multiply-add operations are more precise in float32.
    cos = cached_cos.to(device=xq.device, dtype=torch.float32)
    sin = cached_sin.to(device=xq.device, dtype=torch.float32)

    # Compute in float32, then cast back (matches HF's autocast(enabled=False))
    xq_f32 = xq.to(torch.float32)
    xk_f32 = xk.to(torch.float32)

    xq_out = (xq_f32 * cos) + (rotate_half(xq_f32) * sin)
    xk_out = (xk_f32 * cos) + (rotate_half(xk_f32) * sin)

    return xq_out.to(input_dtype), xk_out.to(input_dtype)


def patch_rope_for_hf_compatibility() -> None:
    """Monkey-patch TorchTitan's apply_rotary_emb to match HuggingFace precision.

    This patches both the module-level function and the Attention.forward method's
    __globals__ to ensure the patched version is used by all Attention layers.
    """
    import torchtitan.models.qwen3.model.model as qwen3_model
    from torchtitan.models.qwen3.model.model import Attention

    old_fn = Attention.forward.__globals__.get("apply_rotary_emb")
    logger.info(f"Before patch: Attention.forward.__globals__[apply_rotary_emb] = {old_fn}")

    # Patch the module namespace
    qwen3_model.apply_rotary_emb = apply_rotary_emb_hf_compatible

    # Patch the Attention.forward method's __globals__ directly
    # This ensures the compiled bytecode's LOAD_GLOBAL picks up our function
    Attention.forward.__globals__["apply_rotary_emb"] = apply_rotary_emb_hf_compatible

    new_fn = Attention.forward.__globals__.get("apply_rotary_emb")
    logger.info(f"After patch: Attention.forward.__globals__[apply_rotary_emb] = {new_fn}")
    logger.info(f"Patch successful: {new_fn is apply_rotary_emb_hf_compatible}")


def convert_rope_cache_to_float32(module: nn.Module) -> None:
    """Convert rope_cache buffer to float32 for HF-compatible precision.

    TorchTitan stores precomputed cos/sin in the model's default dtype (bf16),
    but HuggingFace computes them fresh in float32 each forward pass.
    This converts the buffer to float32 so the patched apply_rotary_emb
    can use full precision values.
    """
    if hasattr(module, 'rope_cache') and module.rope_cache is not None:
        old_dtype = module.rope_cache.dtype
        if old_dtype != torch.float32:
            # Re-register buffer with float32 dtype
            # We need to recompute from scratch since bf16->f32 loses precision
            from torchtitan.models.qwen3.model.model import precompute_rope_cache

            # Get model args from the module
            if hasattr(module, 'model_args'):
                args = module.model_args
                new_cache = precompute_rope_cache(
                    args.head_dim,
                    args.max_seq_len,
                    args.rope_theta,
                )
                module.register_buffer("rope_cache", new_cache.to(module.rope_cache.device), persistent=False)
                logger.info(f"Converted rope_cache to float32 (was {old_dtype})")
            else:
                logger.warning(f"Found rope_cache in {type(module).__name__} but no model_args - cannot recompute")
        else:
            logger.info(f"rope_cache already float32 in {type(module).__name__}")

    # Recurse for nested modules (though rope_cache is typically at top level)
    for child in module.children():
        convert_rope_cache_to_float32(child)
