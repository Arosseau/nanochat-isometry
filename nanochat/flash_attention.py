"""
Unified Flash Attention interface with automatic FA3/FA2/SDPA switching.

Priority: FA3 (Hopper) > FA2 (Ampere/Ada/Hopper) > PyTorch SDPA (fallback).

Only FA3 and FA2 support sliding window attention efficiently.
SDPA fallback ignores window_size and does full attention (works, but slow).

Usage (drop-in replacement for FA3):
    from nanochat.flash_attention import flash_attn

    # Training (no KV cache)
    y = flash_attn.flash_attn_func(q, k, v, causal=True, window_size=window_size)

    # Inference (with KV cache)
    y = flash_attn.flash_attn_with_kvcache(q, k_cache, v_cache, k=k, v=v, ...)
"""
import torch
import torch.nn.functional as F


# =============================================================================
# Detection: FA3 (Hopper only)
# =============================================================================
def _load_flash_attention_3():
    """Try to load Flash Attention 3 (requires Hopper GPU, sm90)."""
    if not torch.cuda.is_available():
        return None
    try:
        major, _ = torch.cuda.get_device_capability()
        if major != 9:
            return None
        import os
        os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
        from kernels import get_kernel
        return get_kernel('varunneal/flash-attention-3').flash_attn_interface
    except Exception:
        return None


# =============================================================================
# Detection: FA2 (Ampere sm80+, Ada sm89, Hopper sm90)
# =============================================================================
def _load_flash_attention_2():
    """Try to load Flash Attention 2 (requires sm80+, install: pip install flash-attn)."""
    if not torch.cuda.is_available():
        return None
    try:
        major, _ = torch.cuda.get_device_capability()
        if major < 8:
            return None
        import flash_attn as _fa2_pkg
        return _fa2_pkg
    except Exception:
        return None


_fa3 = _load_flash_attention_3()
_fa2 = _load_flash_attention_2()
HAS_FA3 = _fa3 is not None
HAS_FA2 = _fa2 is not None

# Override for testing: set to 'fa3', 'fa2', 'sdpa', or None (auto)
_override_impl = None


def _resolve_impl():
    """Decide once which implementation to use: 'fa3', 'fa2', or 'sdpa'."""
    if _override_impl == 'fa3':
        assert HAS_FA3, "Cannot override to FA3: not available on this hardware"
        return 'fa3'
    if _override_impl == 'fa2':
        assert HAS_FA2, "Cannot override to FA2: flash-attn not installed"
        return 'fa2'
    if _override_impl == 'sdpa':
        return 'sdpa'
    if HAS_FA3:
        from nanochat.common import COMPUTE_DTYPE
        if COMPUTE_DTYPE == torch.bfloat16:
            return 'fa3'
        # FA3 only supports bf16/fp8; fall through to FA2/SDPA for fp16/fp32
    if HAS_FA2:
        return 'fa2'
    return 'sdpa'


IMPL = _resolve_impl()
USE_FA3 = (IMPL == 'fa3')
USE_FA2 = (IMPL == 'fa2')


# =============================================================================
# SDPA helpers (fallback only — no sliding window support)
# =============================================================================
def _sdpa_attention(q, k, v, window_size, enable_gqa):
    """
    SDPA attention. q, k, v are (B, H, T, D) format.
    window_size is ignored (SDPA has no sliding window support).
    """
    Tq = q.size(2)
    Tk = k.size(2)

    # Full context, same length
    if Tq == Tk:
        return F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=enable_gqa)

    # Single token generation
    if Tq == 1:
        window = window_size[0]
        if window >= 0 and window < Tk:
            start = max(0, Tk - (window + 1))
            k = k[:, :, start:, :]
            v = v[:, :, start:, :]
        return F.scaled_dot_product_attention(q, k, v, is_causal=False, enable_gqa=enable_gqa)

    # Chunk inference: build explicit causal mask
    device = q.device
    row_idx = (Tk - Tq) + torch.arange(Tq, device=device).unsqueeze(1)
    col_idx = torch.arange(Tk, device=device).unsqueeze(0)
    mask = col_idx <= row_idx
    return F.scaled_dot_product_attention(q, k, v, attn_mask=mask, enable_gqa=enable_gqa)


# =============================================================================
# Public API: Same interface as FA3
# =============================================================================
def flash_attn_func(q, k, v, causal=False, window_size=(-1, -1)):
    """
    Flash Attention for training (no KV cache).

    Args:
        q, k, v: Tensors of shape (B, T, H, D)
        causal: Whether to use causal masking
        window_size: (left, right) sliding window. -1 means unlimited.
                     Respected by FA3 and FA2. Ignored by SDPA fallback.

    Returns:
        Output tensor of shape (B, T, H, D)
    """
    if USE_FA3:
        return _fa3.flash_attn_func(q, k, v, causal=causal, window_size=window_size)

    if USE_FA2:
        return _fa2.flash_attn_func(q, k, v, causal=causal, window_size=window_size)

    # SDPA fallback: transpose (B, T, H, D) -> (B, H, T, D)
    q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
    enable_gqa = q.size(1) != k.size(1)
    y = _sdpa_attention(q, k, v, window_size, enable_gqa)
    return y.transpose(1, 2)


def flash_attn_with_kvcache(q, k_cache, v_cache, k=None, v=None, cache_seqlens=None,
                            causal=False, window_size=(-1, -1)):
    """
    Flash Attention with KV cache for inference.

    Args:
        q: Queries, shape (B, T_new, H, D)
        k_cache, v_cache: Pre-allocated cache tensors, shape (B, T_max, H_kv, D)
        k, v: New keys/values to insert, shape (B, T_new, H_kv, D)
        cache_seqlens: Current position in cache, shape (B,) int32
        causal: Whether to use causal masking
        window_size: (left, right) sliding window. Respected by FA3 and FA2.

    Returns:
        Output tensor of shape (B, T_new, H, D)
    """
    if USE_FA3:
        return _fa3.flash_attn_with_kvcache(
            q, k_cache, v_cache, k=k, v=v, cache_seqlens=cache_seqlens,
            causal=causal, window_size=window_size
        )

    if USE_FA2:
        return _fa2.flash_attn_with_kvcache(
            q, k_cache, v_cache, k=k, v=v, cache_seqlens=cache_seqlens,
            causal=causal, window_size=window_size
        )

    # SDPA fallback: manually manage KV cache
    B, T_new, H, D = q.shape
    pos = cache_seqlens[0].item()

    if k is not None and v is not None:
        k_cache[:, pos:pos+T_new, :, :] = k
        v_cache[:, pos:pos+T_new, :, :] = v

    end_pos = pos + T_new
    k_full = k_cache[:, :end_pos, :, :]
    v_full = v_cache[:, :end_pos, :, :]

    q_sdpa = q.transpose(1, 2)
    k_sdpa = k_full.transpose(1, 2)
    v_sdpa = v_full.transpose(1, 2)
    enable_gqa = q_sdpa.size(1) != k_sdpa.size(1)
    y = _sdpa_attention(q_sdpa, k_sdpa, v_sdpa, window_size, enable_gqa)
    return y.transpose(1, 2)


# =============================================================================
# Export: flash_attn module interface (drop-in replacement for FA3)
# =============================================================================
from types import SimpleNamespace
flash_attn = SimpleNamespace(
    flash_attn_func=flash_attn_func,
    flash_attn_with_kvcache=flash_attn_with_kvcache,
)
