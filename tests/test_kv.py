"""Unit tests for Vantage V-AI KV-Cache Streaming Engine."""

import io
import os
import sys
import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.engine import CodecType
from core.kv import KVCacheMetadata, KVCacheStreamEngine

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
