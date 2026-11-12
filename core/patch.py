"""
VANTAGE-V-AI 1-Line Drop-in Acceleration & Memory Optimizer.

Usage:
    import vantage
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-3B-Instruct")
    vantage.enable(model, max_active_tokens=256, sink_tokens=4, quant_mode="fp8")

    # Regular Hugging Face generation now automatically runs with bounded KV memory!
    out = model.generate(input_ids, max_new_tokens=64)
"""

from functools import wraps
from typing import Optional, Union

import torch
from core.adapters.transformers import VantageStreamingCache
from core.kv import QUANT_LOSSLESS, QUANT_FP8, QUANT_TERNARY_158


def _resolve_quant_mode(mode: Union[str, int]) -> int:
    if isinstance(mode, int):
        return mode
    m = str(mode).lower()
    if "fp8" in m:
        return QUANT_FP8
    if "ternary" in m or "158" in m or "bitnet" in m or "2bit" in m:
        return QUANT_TERNARY_158
    return QUANT_LOSSLESS


def enable(
    model: torch.nn.Module,
    max_active_tokens: int = 256,
    block_size: int = 32,
    sink_tokens: int = 4,
    quant_mode: Union[str, int] = "lossless",
) -> torch.nn.Module:
    """
    Patches a Hugging Face CausalLM model to automatically use VantageStreamingCache.
    
    Args:
        model: Hugging Face PreTrainedModel instance (e.g. Qwen, Llama, Mistral, GPT).
        max_active_tokens: Bounded attention window length in VRAM / RAM.
        block_size: Chunk size for amortized block offload.
        sink_tokens: Attention sink tokens retained permanently at start of sequence.
        quant_mode: 'lossless' | 'fp8' | 'ternary_158'.
    
    Returns:
        The patched model ready for inference.
    """
    orig_generate = model.generate
    active_quant = _resolve_quant_mode(quant_mode)

    @wraps(orig_generate)
    def patched_generate(*args, **kwargs):
        if kwargs.get("past_key_values") is None:
            kwargs["past_key_values"] = VantageStreamingCache(
                max_active_tokens=max_active_tokens,
                block_size=block_size,
                sink_tokens=sink_tokens,
                quant_mode=active_quant,
            )
        return orig_generate(*args, **kwargs)

    model.generate = patched_generate
    model._vantage_enabled = True
    model._vantage_config = {
        "max_active_tokens": max_active_tokens,
        "block_size": block_size,
        "sink_tokens": sink_tokens,
        "quant_mode": quant_mode,
    }
    return model


def disable(model: torch.nn.Module) -> torch.nn.Module:
    """Removes Vantage streaming patch and restores standard Hugging Face generate."""
    if hasattr(model, "_vantage_enabled") and hasattr(model.generate, "__wrapped__"):
        model.generate = model.generate.__wrapped__
        delattr(model, "_vantage_enabled")
        if hasattr(model, "_vantage_config"):
            delattr(model, "_vantage_config")
    return model


# Aliases
patch = enable
unpatch = disable
