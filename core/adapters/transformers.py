"""
Vantage Dynamic Cache Adapter for Hugging Face Transformers.

Seamless drop-in replacement for `transformers.DynamicCache`:
Supports both older transformers (`key_cache` / `value_cache` lists) and
modern transformers v5.x (`layers[layer_idx].keys` / `values`).
Automatically offloads older attention KV prefix tokens via Vantage
KVCacheStreamEngine to keep GPU / RAM active tensor memory bounded.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch
from transformers.cache_utils import DynamicCache

from core.engine import CodecType
from core.kv import KVCacheStreamEngine


class VantageDynamicCache(DynamicCache):
    """
    HuggingFace DynamicCache powered by Vantage KV-Cache Streaming Engine.

    Parameters:
    - max_active_tokens: maximum number of recent tokens kept uncompressed in active memory.
    - quantize_fp8: whether to apply Vantage dynamic FP8 quantization during offload (2x reduction).
    - codec: compression codec (CodecType.LZ4_BLOCK by default).
    """

    def __init__(
        self,
        max_active_tokens: int = 1024,
        quantize_fp8: bool = False,
        codec: CodecType = CodecType.LZ4_BLOCK,
    ) -> None:
        super().__init__()
        self.max_active_tokens = max_active_tokens
        self.quantize_fp8 = quantize_fp8
        self.engine = KVCacheStreamEngine(codec=codec)
        # Store compressed blobs per layer: {layer_idx: (compressed_k_bytes, compressed_v_bytes, excess_len)}
        self._offloaded_blobs: Dict[int, Tuple[bytes, bytes, int]] = {}

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
        """
        Updates the cache with new `key_states` and `value_states` for layer `layer_idx`.
        Restores compressed prefix if present, updates, and offloads excess tokens.
        """
        # Restore offloaded prefix if present
        if layer_idx in self._offloaded_blobs:
            comp_k, comp_v, excess = self._offloaded_blobs.pop(layer_idx)
            rest_k, meta_k = self.engine.decompress_kv_block(comp_k, as_torch=True)
            rest_v, meta_v = self.engine.decompress_kv_block(comp_v, as_torch=True)
            # Reshape back to 4D [batch, heads, seq, dim]
            b = key_states.shape[0]
            h = key_states.shape[1]
            rest_k = rest_k.reshape(excess, b, h, meta_k.head_dim).permute(1, 2, 0, 3)
            rest_v = rest_v.reshape(excess, b, h, meta_v.head_dim).permute(1, 2, 0, 3)
            rest_k = rest_k.to(device=key_states.device, dtype=key_states.dtype)
            rest_v = rest_v.to(device=value_states.device, dtype=value_states.dtype)

            # Prepend restored prefix back to current layer states before update
            curr_k, curr_v = self._get_layer_keys_values(layer_idx)
            self._set_layer_keys_values(
                layer_idx,
                torch.cat([rest_k, curr_k], dim=-2),
                torch.cat([rest_v, curr_v], dim=-2),
            )

        # Standard HuggingFace DynamicCache update
        k_out, v_out = super().update(key_states, value_states, layer_idx, cache_kwargs)

        # If sequence length exceeds budget, offload older prefix
        curr_tokens = k_out.shape[-2]
        if curr_tokens > self.max_active_tokens:
            excess = curr_tokens - self.max_active_tokens
            prefix_k = k_out[..., :excess, :].contiguous()
            prefix_v = v_out[..., :excess, :].contiguous()

            comp_k, _ = self.engine.compress_kv_block(prefix_k, layer_idx=layer_idx, quantize_fp8=self.quantize_fp8)
            comp_v, _ = self.engine.compress_kv_block(prefix_v, layer_idx=layer_idx, quantize_fp8=self.quantize_fp8)
            self._offloaded_blobs[layer_idx] = (comp_k, comp_v, excess)

            # Retain active window
            active_k = k_out[..., excess:, :].contiguous()
            active_v = v_out[..., excess:, :].contiguous()
            self._set_layer_keys_values(layer_idx, active_k, active_v)
            return active_k, active_v

        return k_out, v_out

    def get_offloaded_memory_bytes(self) -> int:
        """Return total compressed bytes currently offloaded in storage."""
        total = 0
        for comp_k, comp_v, _ in self._offloaded_blobs.values():
            total += len(comp_k) + len(comp_v)
        return total

    def restore_full_cache(self) -> None:
        """Restores all offloaded prefixes back into active tensor layers."""
        for layer_idx in list(self._offloaded_blobs.keys()):
            comp_k, comp_v, excess = self._offloaded_blobs.pop(layer_idx)
            rest_k, meta_k = self.engine.decompress_kv_block(comp_k, as_torch=True)
            rest_v, meta_v = self.engine.decompress_kv_block(comp_v, as_torch=True)
            curr_k, curr_v = self._get_layer_keys_values(layer_idx)
            b, h = curr_k.shape[0], curr_k.shape[1]
            rest_k = rest_k.reshape(excess, b, h, meta_k.head_dim).permute(1, 2, 0, 3)
            rest_v = rest_v.reshape(excess, b, h, meta_v.head_dim).permute(1, 2, 0, 3)
            rest_k = rest_k.to(device=curr_k.device, dtype=curr_k.dtype)
            rest_v = rest_v.to(device=curr_v.device, dtype=curr_v.dtype)
            self._set_layer_keys_values(
                layer_idx,
                torch.cat([rest_k, curr_k], dim=-2),
                torch.cat([rest_v, curr_v], dim=-2),
            )
