"""Unit tests for Ternary Latent Attention (TLA) Engine."""

import pytest
import torch
import torch.nn.functional as F

from core.tla import TernaryLatentAttentionEngine, TLACompressionStats


def test_tla_initialization():
    engine = TernaryLatentAttentionEngine(
        num_heads=8,
        head_dim=128,
        latent_dim=256,
        rope_dim=32,
    )
    assert engine.num_heads == 8
    assert engine.head_dim == 128
    assert engine.latent_dim == 256
    assert engine.rope_dim == 32
    assert engine.total_kv_dim == 2048


def test_tla_project_and_unpack_4d():
    engine = TernaryLatentAttentionEngine(
        num_heads=4,
        head_dim=64,
        latent_dim=128,
        rope_dim=32,
    )
    b, h, s, d = 2, 4, 32, 64
    k = torch.randn(b, h, s, d, dtype=torch.float16)
    v = torch.randn(b, h, s, d, dtype=torch.float16)
    rope_k = torch.randn(s, 32, dtype=torch.float16)

    packed_latent, scale, orig_shape, ret_rope = engine.project_and_pack_torch(k, v, rope_k)

    assert packed_latent is not None
    assert scale > 0.0
    assert orig_shape == k.shape

    # Packed elements: 2 * 32 tokens * 128 latent elements = 8192 elements -> 2048 packed bytes
    assert packed_latent.numel() == (b * s * 128) // 4

    k_rec, v_rec = engine.unpack_and_reconstruct_torch(packed_latent, scale, orig_shape, ret_rope)
    assert k_rec.shape == k.shape
    assert v_rec.shape == v.shape
    assert k_rec.dtype == torch.float16
    assert v_rec.dtype == torch.float16


def test_tla_compression_benchmark():
    engine = TernaryLatentAttentionEngine(
        num_heads=8,
        head_dim=128,
        latent_dim=512,
        rope_dim=64,
    )
    stats = engine.evaluate_benchmark(seq_len=512)
    assert isinstance(stats, TLACompressionStats)
    assert stats.ratio_vs_standard > 1.0
    assert stats.ratio_vs_deepseek_mla > 1.0
    assert stats.percent_savings_vs_standard > 90.0
    assert stats.percent_savings_vs_deepseek_mla > 50.0
