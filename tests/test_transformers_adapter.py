"""Unit tests for Vantage DynamicCache adapter for Hugging Face Transformers."""

import pytest
import torch
from transformers.cache_utils import DynamicCache

from core.adapters.transformers import (
    VantageDynamicCache,
    VantageStreamingCache,
    checkpoint_prompt_cache,
    restore_prompt_cache,
)


def test_vantage_streaming_cache_block_chunked_offload():
    max_active = 32
    block_size = 16
    cache = VantageStreamingCache(
        max_active_tokens=max_active,
        block_size=block_size,
        quantize_fp8=False,
    )

    # Feed 80 tokens step by step
    for step in range(80):
        k = torch.randn(1, 4, 1, 64, dtype=torch.float32)
        v = torch.randn(1, 4, 1, 64, dtype=torch.float32)
        cache.update(k, v, layer_idx=0)

    # Active tokens should be bounded to around max_active (within block_size margin)
    keys, values = cache._get_layer_keys_values(0)
    assert keys.shape[-2] <= max_active + block_size
    assert values.shape[-2] <= max_active + block_size

    # Tokens offloaded must be non-zero and multiple of block_size
    offloaded_tokens = cache.get_offloaded_token_count(0)
    assert offloaded_tokens > 0
    assert offloaded_tokens % block_size == 0
    assert cache.get_offloaded_memory_bytes() > 0


def test_vantage_streaming_cache_full_restore():
    max_active = 32
    block_size = 16
    cache = VantageStreamingCache(
        max_active_tokens=max_active,
        block_size=block_size,
        quantize_fp8=False,
    )

    total_tokens = 64
    all_k = []
    all_v = []
    for _ in range(total_tokens):
        k = torch.randn(1, 4, 1, 64, dtype=torch.float32)
        v = torch.randn(1, 4, 1, 64, dtype=torch.float32)
        all_k.append(k)
        all_v.append(v)
        cache.update(k, v, layer_idx=0)

    expected_k = torch.cat(all_k, dim=-2)
    expected_v = torch.cat(all_v, dim=-2)

    # Before restore, active keys are bounded
    keys, _ = cache._get_layer_keys_values(0)
    assert keys.shape[-2] < total_tokens

    # After restore, active keys must match total_tokens exactly and be bit-for-bit identical!
    cache.restore_full_cache()
    restored_k, restored_v = cache._get_layer_keys_values(0)
    assert restored_k.shape[-2] == total_tokens
    assert torch.equal(restored_k, expected_k)
    assert torch.equal(restored_v, expected_v)


def test_vantage_prompt_checkpoint_lossless_bit_parity():
    # Simulate a prompt prefill of 512 tokens across 4 layers
    prompt_len = 512
    cache = DynamicCache()
    for l in range(4):
        k = torch.randn(1, 8, prompt_len, 64, dtype=torch.float32)
        v = torch.randn(1, 8, prompt_len, 64, dtype=torch.float32)
        cache.update(k, v, layer_idx=l)

    # Checkpoint prefilled prompt cache
    checkpoint = checkpoint_prompt_cache(cache, quantize_fp8=False)
    assert checkpoint.total_bytes > 0

    # Restore from checkpoint
    restored_cache = restore_prompt_cache(checkpoint)

    # Verify 100% bit-for-bit equality across all layers
    for l in range(4):
        orig_k = cache.layers[l].keys if hasattr(cache, "layers") else cache.key_cache[l]
        orig_v = cache.layers[l].values if hasattr(cache, "layers") else cache.value_cache[l]
        rest_k = restored_cache.layers[l].keys if hasattr(restored_cache, "layers") else restored_cache.key_cache[l]
        rest_v = restored_cache.layers[l].values if hasattr(restored_cache, "layers") else restored_cache.value_cache[l]

        assert torch.equal(orig_k, rest_k), f"Layer {l} keys mismatch"
        assert torch.equal(orig_v, rest_v), f"Layer {l} values mismatch"


def test_vantage_prompt_checkpoint_fp8_compression():
    prompt_len = 512
    cache = DynamicCache()
    for l in range(2):
        k = torch.randn(1, 8, prompt_len, 64, dtype=torch.float32)
        v = torch.randn(1, 8, prompt_len, 64, dtype=torch.float32)
        cache.update(k, v, layer_idx=l)

    checkpoint_fp8 = checkpoint_prompt_cache(cache, quantize_fp8=True)
    restored_cache = restore_prompt_cache(checkpoint_fp8)

    for l in range(2):
        orig_k = cache.layers[l].keys if hasattr(cache, "layers") else cache.key_cache[l]
        rest_k = restored_cache.layers[l].keys if hasattr(restored_cache, "layers") else restored_cache.key_cache[l]

        # Cosine similarity should be > 0.999
        cossim = torch.nn.functional.cosine_similarity(orig_k.flatten(), rest_k.flatten(), dim=0)
        assert cossim.item() > 0.999


def test_vantage_streaming_cache_attention_sinks():
    """Verify that initial sink tokens are strictly preserved during streaming offload."""
    sink_tokens = 4
    max_active = 16
    block_size = 8

    cache = VantageStreamingCache(
        max_active_tokens=max_active,
        block_size=block_size,
        sink_tokens=sink_tokens,
        quantize_fp8=False,
    )

    # Track initial tokens
    all_k = []
    all_v = []
    for step in range(32):
        k = torch.full((1, 2, 1, 32), fill_value=float(step), dtype=torch.float32)
        v = torch.full((1, 2, 1, 32), fill_value=float(step), dtype=torch.float32)
        all_k.append(k)
        all_v.append(v)
        cache.update(k, v, layer_idx=0)

    # Check active keys: the first sink_tokens tokens must match steps 0, 1, 2, 3!
    active_k, _ = cache._get_layer_keys_values(0)
    # Total active tokens should be <= max_active + block_size
    assert active_k.shape[-2] <= max_active + block_size

    # Verify first 4 tokens in active cache are the sink tokens (0, 1, 2, 3)
    for i in range(sink_tokens):
        assert torch.all(active_k[..., i, :] == float(i)), f"Sink token {i} was corrupted or evicted!"


def test_vantage_prompt_checkpoint_ternary_158():
    from core.kv import QUANT_TERNARY_158

    prompt_len = 256
    cache = DynamicCache()
    for l in range(2):
        k = torch.randn(1, 4, prompt_len, 64, dtype=torch.float32)
        v = torch.randn(1, 4, prompt_len, 64, dtype=torch.float32)
        cache.update(k, v, layer_idx=l)

    checkpoint = checkpoint_prompt_cache(cache, quant_mode=QUANT_TERNARY_158)
    restored = restore_prompt_cache(checkpoint)

    for l in range(2):
        orig_k = cache.layers[l].keys if hasattr(cache, "layers") else cache.key_cache[l]
        rest_k = restored.layers[l].keys if hasattr(restored, "layers") else restored.key_cache[l]

        assert orig_k.shape == rest_k.shape
        cossim = torch.nn.functional.cosine_similarity(orig_k.flatten(), rest_k.flatten(), dim=0)
        assert cossim.item() > 0.70

