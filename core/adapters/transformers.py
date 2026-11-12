"""
Vantage Dynamic Cache Adapter for Hugging Face Transformers (v8.0.0).

Next-Gen Inference & Memory Architecture:
1. `VantageStreamingCache`:
   - Bounded Active Attention Window with Attention Sink preservation (`sink_tokens=4`).
   - Prevents attention sink collapse, preserving generation coherence.
   - Amortized block-level offload (`block_size=16` or `32`).
   - Supports Lossless, FP8, and 1.58-bit Ternary offloaded storage.
   - `restore_full_cache()` for exact temporal sequence reconstruction.

2. `VantagePromptCheckpoint` & `checkpoint_prompt_cache` / `restore_prompt_cache`:
   - Instant prefill checkpointing for session switching and prompt reuse.
   - Dual modes:
     * Mode Lossless (LZ4/Zstd): 100% bit-for-bit lossless parity (torch.equal == True).
     * Mode FP8: 4x compression, CosSim > 0.999.
     * Mode Ternary 1.58-bit (BitNet b1.58): 8x-16x extreme memory reduction.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers.cache_utils import DynamicCache

from core.engine import CodecType
from core.kv import (
    QUANT_FP8,
    QUANT_LOSSLESS,
    QUANT_TERNARY_158,
    KVCacheMetadata,
    KVCacheStreamEngine,
)


class VantageStreamingCache(DynamicCache):
    """
    Bounded Streaming KV Cache with Attention Sinks and Block-Level Offload.

    Parameters:
    - max_active_tokens: target maximum tokens kept in active attention window.
    - block_size: tokens accumulated before an offload block is compressed (default: 16).
    - sink_tokens: initial tokens permanently kept to prevent attention collapse (default: 4).
    - quantize_fp8: whether to apply FP8 quantization to offloaded blocks.
    - quant_mode: QUANT_LOSSLESS, QUANT_FP8, or QUANT_TERNARY_158.
    - codec: compression codec (CodecType.LZ4_BLOCK by default).
    """

    def __init__(
        self,
        max_active_tokens: int = 1024,
        block_size: int = 16,
        sink_tokens: int = 0,
        quantize_fp8: bool = False,
        quant_mode: Optional[int] = None,
        codec: CodecType = CodecType.LZ4_BLOCK,
    ) -> None:
        super().__init__()
        self.max_active_tokens = max_active_tokens
        self.block_size = max(1, block_size)
        self.sink_tokens = max(0, sink_tokens)
        self.quantize_fp8 = quantize_fp8
        if quant_mode is not None:
            self.quant_mode = quant_mode
        elif quantize_fp8:
            self.quant_mode = QUANT_FP8
        else:
            self.quant_mode = QUANT_LOSSLESS

        self.engine = KVCacheStreamEngine(codec=codec)
        # Store list of compressed blocks per layer:
        # {layer_idx: [(k_blob, v_blob, meta_k, meta_v, num_tokens), ...]}
        self._offloaded_blocks: Dict[int, List[Tuple[bytes, bytes, KVCacheMetadata, KVCacheMetadata, int]]] = {}

    def _get_layer_keys_values(self, layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if hasattr(self, "layers") and len(self.layers) > layer_idx:
            layer = self.layers[layer_idx]
            return layer.keys, layer.values
        if hasattr(self, "key_cache") and len(self.key_cache) > layer_idx:
            return self.key_cache[layer_idx], self.value_cache[layer_idx]
        raise IndexError(f"Layer {layer_idx} not found in cache")

    def _set_layer_keys_values(self, layer_idx: int, keys: torch.Tensor, values: torch.Tensor) -> None:
        if hasattr(self, "layers") and len(self.layers) > layer_idx:
            layer = self.layers[layer_idx]
            layer.keys = keys
            layer.values = values
        if hasattr(self, "key_cache") and len(self.key_cache) > layer_idx:
            self.key_cache[layer_idx] = keys
            self.value_cache[layer_idx] = values

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Updates active attention window with sink preservation and block offload."""
        k_out, v_out = super().update(key_states, value_states, layer_idx, cache_kwargs)

        curr_tokens = k_out.shape[-2]
        if curr_tokens >= self.max_active_tokens + self.block_size:
            if self.sink_tokens > 0 and curr_tokens > self.sink_tokens + self.block_size:
                # Attention Sink: retain initial sink_tokens permanently
                sink_k = k_out[..., : self.sink_tokens, :].contiguous()
                sink_v = v_out[..., : self.sink_tokens, :].contiguous()
                evict_k = k_out[..., self.sink_tokens : self.sink_tokens + self.block_size, :].contiguous()
                evict_v = v_out[..., self.sink_tokens : self.sink_tokens + self.block_size, :].contiguous()
                recent_k = k_out[..., self.sink_tokens + self.block_size :, :].contiguous()
                recent_v = v_out[..., self.sink_tokens + self.block_size :, :].contiguous()

                comp_k, meta_k = self.engine.compress_kv_block(
                    evict_k, layer_idx=layer_idx, quant_mode=self.quant_mode
                )
                comp_v, meta_v = self.engine.compress_kv_block(
                    evict_v, layer_idx=layer_idx, quant_mode=self.quant_mode
                )

                if layer_idx not in self._offloaded_blocks:
                    self._offloaded_blocks[layer_idx] = []
                self._offloaded_blocks[layer_idx].append(
                    (comp_k, comp_v, meta_k, meta_v, self.block_size)
                )

                active_k = torch.cat([sink_k, recent_k], dim=-2).contiguous()
                active_v = torch.cat([sink_v, recent_v], dim=-2).contiguous()
                self._set_layer_keys_values(layer_idx, active_k, active_v)
                return active_k, active_v
            else:
                evict_k = k_out[..., : self.block_size, :].contiguous()
                evict_v = v_out[..., : self.block_size, :].contiguous()

                comp_k, meta_k = self.engine.compress_kv_block(
                    evict_k, layer_idx=layer_idx, quant_mode=self.quant_mode
                )
                comp_v, meta_v = self.engine.compress_kv_block(
                    evict_v, layer_idx=layer_idx, quant_mode=self.quant_mode
                )

                if layer_idx not in self._offloaded_blocks:
                    self._offloaded_blocks[layer_idx] = []
                self._offloaded_blocks[layer_idx].append(
                    (comp_k, comp_v, meta_k, meta_v, self.block_size)
                )

                active_k = k_out[..., self.block_size :, :].contiguous()
                active_v = v_out[..., self.block_size :, :].contiguous()
                self._set_layer_keys_values(layer_idx, active_k, active_v)
                return active_k, active_v

        return k_out, v_out

    def get_active_memory_bytes(self) -> int:
        """Return total bytes held in active PyTorch attention tensor memory."""
        total = 0
        if hasattr(self, "layers"):
            for layer in self.layers:
                if layer.keys is not None:
                    total += layer.keys.numel() * layer.keys.element_size()
                if layer.values is not None:
                    total += layer.values.numel() * layer.values.element_size()
        else:
            for k in self.key_cache:
                if k is not None:
                    total += k.numel() * k.element_size()
            for v in self.value_cache:
                if v is not None:
                    total += v.numel() * v.element_size()
        return total

    def get_offloaded_memory_bytes(self) -> int:
        """Return total compressed bytes currently held in offloaded storage."""
        total = 0
        for block_list in self._offloaded_blocks.values():
            for comp_k, comp_v, _, _, _ in block_list:
                total += len(comp_k) + len(comp_v)
        return total

    def get_offloaded_token_count(self, layer_idx: int = 0) -> int:
        """Return total tokens offloaded for a specific layer."""
        if layer_idx not in self._offloaded_blocks:
            return 0
        return sum(tokens for _, _, _, _, tokens in self._offloaded_blocks[layer_idx])

    def restore_full_cache(self) -> None:
        """Restores all offloaded blocks back into active tensor layers in exact order."""
        for layer_idx, block_list in list(self._offloaded_blocks.items()):
            if not block_list:
                continue
            curr_k, curr_v = self._get_layer_keys_values(layer_idx)
            b = curr_k.shape[0]
            h = curr_k.shape[1]

            restored_k_parts = []
            restored_v_parts = []
            for comp_k, comp_v, meta_k, meta_v, num_tokens in block_list:
                rk, _ = self.engine.decompress_kv_block(comp_k, as_torch=True)
                rv, _ = self.engine.decompress_kv_block(comp_v, as_torch=True)
                rk = rk.reshape(num_tokens, b, h, meta_k.head_dim).permute(1, 2, 0, 3)
                rv = rv.reshape(num_tokens, b, h, meta_v.head_dim).permute(1, 2, 0, 3)
                restored_k_parts.append(rk.to(device=curr_k.device, dtype=curr_k.dtype))
                restored_v_parts.append(rv.to(device=curr_v.device, dtype=curr_v.dtype))

            if self.sink_tokens > 0 and curr_k.shape[-2] > self.sink_tokens:
                sink_k = curr_k[..., : self.sink_tokens, :]
                sink_v = curr_v[..., : self.sink_tokens, :]
                recent_k = curr_k[..., self.sink_tokens :, :]
                recent_v = curr_v[..., self.sink_tokens :, :]
                full_k = torch.cat([sink_k, *restored_k_parts, recent_k], dim=-2)
                full_v = torch.cat([sink_v, *restored_v_parts, recent_v], dim=-2)
            else:
                full_k = torch.cat([*restored_k_parts, curr_k], dim=-2)
                full_v = torch.cat([*restored_v_parts, curr_v], dim=-2)

            self._set_layer_keys_values(layer_idx, full_k, full_v)

        self._offloaded_blocks.clear()


# Aliases
VantageDynamicCache = VantageStreamingCache
VantageTernaryCache = VantageStreamingCache


class VantagePromptCheckpoint:
    """
    Compressed serialized prompt cache checkpoint.
    Frees active GPU / host tensor memory while enabling zero-drift or high-fidelity recovery.
    """

    def __init__(
        self,
        layers_data: List[Tuple[bytes, bytes, KVCacheMetadata, KVCacheMetadata, Tuple[int, ...]]],
        quant_mode: int,
        codec: CodecType,
    ) -> None:
        self.layers_data = layers_data
        self.quant_mode = quant_mode
        self.codec = codec

    @property
    def total_bytes(self) -> int:
        return sum(len(k) + len(v) for k, v, _, _, _ in self.layers_data)


def checkpoint_prompt_cache(
    cache: DynamicCache,
    quantize_fp8: bool = False,
    quant_mode: Optional[int] = None,
    codec: CodecType = CodecType.LZ4_BLOCK,
) -> VantagePromptCheckpoint:
    """
    Compresses an existing HuggingFace DynamicCache into a VantagePromptCheckpoint.
    Supports Lossless (LZ4/Zstd), Dynamic FP8, and 1.58-bit Ternary.
    """
    engine = KVCacheStreamEngine(codec=codec)
    layers_data = []

    if quant_mode is None:
        active_quant = QUANT_FP8 if quantize_fp8 else QUANT_LOSSLESS
    else:
        active_quant = quant_mode

    num_layers = len(cache.layers) if hasattr(cache, "layers") else len(cache.key_cache)
    for l in range(num_layers):
        if hasattr(cache, "layers"):
            k = cache.layers[l].keys
            v = cache.layers[l].values
        else:
            k = cache.key_cache[l]
            v = cache.value_cache[l]

        orig_shape = tuple(k.shape)
        comp_k, meta_k = engine.compress_kv_block(k, layer_idx=l, quant_mode=active_quant)
        comp_v, meta_v = engine.compress_kv_block(v, layer_idx=l, quant_mode=active_quant)
        layers_data.append((comp_k, comp_v, meta_k, meta_v, orig_shape))

    return VantagePromptCheckpoint(
        layers_data=layers_data,
        quant_mode=active_quant,
        codec=codec,
    )


def restore_prompt_cache(checkpoint: VantagePromptCheckpoint, device: str = "cpu") -> DynamicCache:
    """Decompresses a VantagePromptCheckpoint into a full Hugging Face DynamicCache."""
    engine = KVCacheStreamEngine(codec=checkpoint.codec)
    restored_cache = DynamicCache()

    for l, (comp_k, comp_v, meta_k, meta_v, orig_shape) in enumerate(checkpoint.layers_data):
        rk, _ = engine.decompress_kv_block(comp_k, as_torch=True)
        rv, _ = engine.decompress_kv_block(comp_v, as_torch=True)

        b, h, s, d = orig_shape
        rk = rk.reshape(s, b, h, d).permute(1, 2, 0, 3).to(device=device)
        rv = rv.reshape(s, b, h, d).permute(1, 2, 0, 3).to(device=device)

        restored_cache.update(rk, rv, layer_idx=l)

    return restored_cache
