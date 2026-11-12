"""Unit tests for Vantage 1-line enable/patch API and GPU-native ternary tensor packing."""

import pytest
import torch
from transformers import AutoConfig, AutoModelForCausalLM

import vantage
from vantage import enable, disable, pack_ternary_torch, unpack_ternary_torch


def test_pack_ternary_torch_tensor_roundtrip():
    # 4D attention tensor: [batch=1, heads=4, seq=16, dim=64]
    t = torch.randn(1, 4, 16, 64, dtype=torch.float32)
    packed, scale, orig_shape = pack_ternary_torch(t)

    assert orig_shape == t.shape
    # 4 elements packed into 1 uint8 byte -> exactly 4x reduction in elements
    assert packed.numel() == t.numel() // 4
    # Compared to FP16 (2 bytes/elem), uint8 is 2 * 4 = 8x smaller!
    raw_fp16_bytes = t.numel() * 2
    assert raw_fp16_bytes / packed.numel() == 8.0

    restored = unpack_ternary_torch(packed, scale, orig_shape)
    assert restored.shape == t.shape
    assert restored.dtype == torch.float16

    # Verify positive directional alignment
    cos = torch.nn.functional.cosine_similarity(t.flatten().float(), restored.flatten().float(), dim=0)
    assert cos.item() > 0.85


def test_vantage_enable_disable_monkeypatch():
    config = AutoConfig.from_pretrained("gpt2")
    model = AutoModelForCausalLM.from_config(config)

    # Enable Vantage 1-line patch
    vantage.enable(model, max_active_tokens=64, sink_tokens=4, quant_mode="fp8")
    assert getattr(model, "_vantage_enabled", False) is True

    # Run dummy generate step without explicit past_key_values
    input_ids = torch.randint(0, 1000, (1, 32), dtype=torch.long)
    out = model.generate(input_ids, max_new_tokens=4)
    assert out.shape[-1] == 36

    # Disable Vantage patch
    vantage.disable(model)
    assert hasattr(model, "_vantage_enabled") is False
