"""Unit tests for Vantage V-AI KV-Cache Streaming Engine."""

import io
import os
import sys
import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.engine import CodecType
from core.kv import KVCacheMetadata, KVCacheStreamEngine, compute_quantization_error

try:
    import torch
except ImportError:
    torch = None


def test_kv_metadata_roundtrip():
    meta = KVCacheMetadata(layer_idx=31, num_heads=32, head_dim=128, num_tokens=2048, dtype_str="float16")
    b = meta.to_bytes()
    restored, size = KVCacheMetadata.from_bytes(b)
    assert restored == meta
    assert size == len(b)


@pytest.mark.parametrize("codec", [CodecType.LZ4_BLOCK, CodecType.LZ4_FRAME])
def test_numpy_kv_block_lossless_roundtrip(codec):
    engine = KVCacheStreamEngine(codec=codec)
    # Simulate Llama-3-8B KV block: [num_tokens=512, num_heads=8, head_dim=128] float16
    raw_arr = np.random.randn(512, 8, 128).astype(np.float16)

    payload, meta = engine.compress_kv_block(raw_arr, layer_idx=15)
    assert meta.layer_idx == 15
    assert meta.num_tokens == 512
    assert meta.num_heads == 8
    assert meta.head_dim == 128

    restored_arr, restored_meta = engine.decompress_kv_block(payload, as_torch=False)
    assert restored_meta == meta
    np.testing.assert_array_equal(restored_arr, raw_arr)


@pytest.mark.skipif(torch is None, reason="PyTorch is not available")
def test_torch_kv_cache_multi_layer_stream_offload():
    engine = KVCacheStreamEngine(codec=CodecType.LZ4_BLOCK)

    # Simulate 4-layer KV cache of 1024 tokens in float32 / float16
    kv_cache = {}
    for layer in range(4):
        k = torch.randn(1024, 8, 128, dtype=torch.float32)
        v = torch.randn(1024, 8, 128, dtype=torch.float32)
        kv_cache[layer] = (k, v)

    stream = io.BytesIO()
    total_bytes = engine.stream_offload_cache(kv_cache, stream)
    assert total_bytes > 0
    assert total_bytes == len(stream.getvalue())

    # Reload
    stream.seek(0)
    reloaded = engine.stream_reload_cache(stream, as_torch=True)

    assert set(reloaded.keys()) == {0, 1, 2, 3}
    for layer in range(4):
        k_orig, v_orig = kv_cache[layer]
        k_rel, v_rel = reloaded[layer]
        assert torch.equal(k_orig, k_rel)
        assert torch.equal(v_orig, v_rel)


@pytest.mark.skipif(torch is None, reason="PyTorch is not available")
def test_torch_kv_cache_fp8_quantized_stream_offload():
    engine = KVCacheStreamEngine(codec=CodecType.LZ4_BLOCK)

    # Simulate FP16 Attention tensors
    k = torch.randn(512, 8, 128, dtype=torch.float16)
    v = torch.randn(512, 8, 128, dtype=torch.float16)
    kv_cache = {0: (k, v)}

    stream = io.BytesIO()
    total_bytes = engine.stream_offload_cache(kv_cache, stream, quantize_fp8=True)
    raw_bytes = k.numel() * k.element_size() + v.numel() * v.element_size()

    # FP8 must achieve ~2x compression on raw FP16
    assert raw_bytes / total_bytes >= 1.9

    stream.seek(0)
    reloaded = engine.stream_reload_cache(stream, as_torch=True)
    k_rel, v_rel = reloaded[0]

    # Verify bounded numerical error from FP8 quantization (cosine similarity > 0.999)
    cos_sim = torch.nn.functional.cosine_similarity(k.flatten().float(), k_rel.flatten().float(), dim=0)
    assert cos_sim.item() > 0.999

    err = compute_quantization_error(k, k_rel)
    assert err.mean_absolute_error < 0.02
    assert err.root_mean_squared_error < 0.03
    assert err.relative_error < 0.03
    assert err.cosine_similarity > 0.999


def test_ternary_158_pack_unpack_roundtrip():
    from core.kv import pack_ternary_2bit, unpack_ternary_2bit

    # Create synthetic ternary array with exact {-1, 0, +1} values
    rng = np.random.default_rng(42)
    original = rng.choice([-1.0, 0.0, 1.0], size=(64, 8, 128)).astype(np.float32)

    packed = pack_ternary_2bit(original)

    # 64 * 8 * 128 = 65,536 elements -> 65536 / 4 = 16,384 bytes
    assert len(packed) == original.size // 4
    # FP32 raw bytes: 65,536 * 4 = 262,144 bytes -> 16x reduction!
    assert (original.size * 4) / len(packed) == 16.0

    restored = unpack_ternary_2bit(packed, target_size=original.size, scale=1.0)
    restored = restored.reshape(original.shape).astype(np.float32)

    np.testing.assert_array_equal(original, restored)


@pytest.mark.skipif(torch is None, reason="PyTorch is not available")
def test_torch_kv_cache_ternary_158_stream_offload():
    from core.kv import QUANT_TERNARY_158

    engine = KVCacheStreamEngine(codec=CodecType.LZ4_BLOCK)

    # Simulate FP16 Attention tensors
    k = torch.randn(512, 8, 128, dtype=torch.float16)
    v = torch.randn(512, 8, 128, dtype=torch.float16)
    kv_cache = {0: (k, v)}

    stream = io.BytesIO()
    total_bytes = engine.stream_offload_cache(kv_cache, stream, quant_mode=QUANT_TERNARY_158)
    raw_bytes = k.numel() * k.element_size() + v.numel() * v.element_size()

    # Ternary packed (2-bit per element) achieves > 7.5x compression on raw FP16
    compression_ratio = raw_bytes / total_bytes
    assert compression_ratio >= 7.0

    stream.seek(0)
    reloaded = engine.stream_reload_cache(stream, as_torch=True)
    k_rel, v_rel = reloaded[0]

    assert k_rel.shape == k.shape
    assert v_rel.shape == v.shape

    # Cosine similarity between ternary quant and continuous weights is positive and aligned
    cos_sim_k = torch.nn.functional.cosine_similarity(k.flatten().float(), k_rel.flatten().float(), dim=0)
    assert cos_sim_k.item() > 0.70  # Standard expected alignment for 1.58-bit ternary quantization

