"""Unit tests for Vantage DynamicCache adapter for Hugging Face Transformers."""

import pytest
import torch

from core.adapters.transformers import VantageDynamicCache


def test_vantage_dynamic_cache_basic_update():
    cache = VantageDynamicCache(max_active_tokens=64, quantize_fp8=False)

    # Simulate 2 layers of generation step-by-step
    for step in range(30):
        # Shape: [batch_size=1, num_heads=4, seq_len=1, head_dim=64]
        k = torch.randn(1, 4, 1, 64, dtype=torch.float32)
        v = torch.randn(1, 4, 1, 64, dtype=torch.float32)

        k_out0, v_out0 = cache.update(k, v, layer_idx=0)
        k_out1, v_out1 = cache.update(k, v, layer_idx=1)

        assert k_out0.shape[-2] == step + 1
        assert k_out1.shape[-2] == step + 1


def test_vantage_dynamic_cache_bounded_offload_and_restore():
    max_active = 32
    cache = VantageDynamicCache(max_active_tokens=max_active, quantize_fp8=False)

    # Feed 80 tokens (exceeds max_active = 32)
    for _ in range(80):
        k = torch.randn(1, 4, 1, 64, dtype=torch.float32)
        v = torch.randn(1, 4, 1, 64, dtype=torch.float32)
        k_out, v_out = cache.update(k, v, layer_idx=0)

    # The active in-memory tensor length must be bounded to max_active
    keys, values = cache._get_layer_keys_values(0)
    assert keys.shape[-2] == max_active
    assert values.shape[-2] == max_active
    assert cache.get_offloaded_memory_bytes() > 0


def test_vantage_dynamic_cache_fp8_quantized_offload():
    max_active = 32
    cache = VantageDynamicCache(max_active_tokens=max_active, quantize_fp8=True)

    for _ in range(64):
        k = torch.randn(1, 4, 1, 64, dtype=torch.float32)
        v = torch.randn(1, 4, 1, 64, dtype=torch.float32)
        cache.update(k, v, layer_idx=0)

    keys, values = cache._get_layer_keys_values(0)
    assert keys.shape[-2] == max_active
    assert values.shape[-2] == max_active
    assert cache.get_offloaded_memory_bytes() > 0
