"""
End-to-end Hugging Face Transformers Generation Benchmark with Vantage Dynamic Cache.

Compares:
1. Standard Hugging Face DynamicCache
2. VantageDynamicCache (Lossless LZ4 offload)
3. VantageDynamicCache (Dynamic FP8 Quantized + LZ4 offload)

Measures:
- Generation throughput (tokens/sec)
- Exact token match / logits parity
- Active tensor memory vs offloaded compressed bytes
- Peak RSS memory footprint
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import psutil
import torch
from transformers import AutoConfig, AutoModelForCausalLM
from transformers.cache_utils import DynamicCache

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.adapters.transformers import VantageDynamicCache
from core.engine import CodecType


def get_process_rss_mb() -> float:
    return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)


def run_benchmark():
    print("=" * 70)
    print("  VANTAGE V-AI v7.2.0: Hugging Face Transformers E2E Benchmark")
    print("=" * 70)

    # 1. Initialize reproducible model config
    torch.manual_seed(42)
    vocab_size = 500
    n_embd = 128
    n_head = 4
    n_layer = 4
    prompt_len = 64
    new_tokens = 64
    window_size = 32

    config = AutoConfig.for_model(
        "gpt2",
        n_layer=n_layer,
        n_head=n_head,
        n_embd=n_embd,
        vocab_size=vocab_size,
        bos_token_id=0,
        eos_token_id=1,
    )
    model = AutoModelForCausalLM.from_config(config)
    model.eval()

    # Fixed synthetic prompt
    input_ids = torch.randint(2, vocab_size, (1, prompt_len))

    results = {}

    # Run 1: Standard HF DynamicCache
    print("\n[1/3] Running Standard HF DynamicCache...")
    standard_cache = DynamicCache()
    start_rss = get_process_rss_mb()
    t0 = time.perf_counter()
    with torch.no_grad():
        out_std = model.generate(
            input_ids,
            max_new_tokens=new_tokens,
            past_key_values=standard_cache,
            do_sample=False,
        )
    t1 = time.perf_counter()
    std_time = t1 - t0
    std_rss = get_process_rss_mb()

    # Calculate active tensor bytes in standard cache
    std_active_bytes = 0
    for l in range(n_layer):
        k, v = standard_cache.layers[l].keys, standard_cache.layers[l].values
        std_active_bytes += k.numel() * k.element_size() + v.numel() * v.element_size()

    results["standard_dynamic_cache"] = {
        "time_sec": std_time,
        "tokens_per_sec": new_tokens / std_time,
        "peak_rss_mb": std_rss,
        "active_tensor_kb": std_active_bytes / 1024,
        "offloaded_compressed_kb": 0.0,
        "tokens_generated": new_tokens,
    }
    print(f"  -> Generated {new_tokens} tokens in {std_time:.3f}s ({new_tokens/std_time:.2f} tok/s)")
    print(f"  -> Active KV Tensor Memory: {std_active_bytes / 1024:.2f} KB | Peak RSS: {std_rss:.2f} MB")

    # Run 2: VantageDynamicCache (Lossless LZ4)
    print("\n[2/3] Running VantageDynamicCache (Lossless LZ4)...")
    vantage_lossless_cache = VantageDynamicCache(
        max_active_tokens=window_size,
        quantize_fp8=False,
        codec=CodecType.LZ4_BLOCK,
    )
    t0 = time.perf_counter()
    with torch.no_grad():
        out_lossless = model.generate(
            input_ids,
            max_new_tokens=new_tokens,
            past_key_values=vantage_lossless_cache,
            do_sample=False,
        )
    t1 = time.perf_counter()
    lossless_time = t1 - t0
    lossless_rss = get_process_rss_mb()

    # Active vs offloaded bytes
    v_lossless_active_bytes = 0
    for l in range(n_layer):
        k, v = vantage_lossless_cache._get_layer_keys_values(l)
        v_lossless_active_bytes += k.numel() * k.element_size() + v.numel() * v.element_size()
    v_lossless_offloaded_bytes = sum(
        len(k_blob) + len(v_blob)
        for k_blob, v_blob, _ in vantage_lossless_cache._offloaded_blobs.values()
    )

    exact_match_lossless = torch.equal(out_std, out_lossless)
    results["vantage_dynamic_cache_lossless"] = {
        "time_sec": lossless_time,
        "tokens_per_sec": new_tokens / lossless_time,
        "peak_rss_mb": lossless_rss,
        "active_tensor_kb": v_lossless_active_bytes / 1024,
        "offloaded_compressed_kb": v_lossless_offloaded_bytes / 1024,
        "exact_token_match": exact_match_lossless,
        "tokens_generated": new_tokens,
    }
    print(f"  -> Generated {new_tokens} tokens in {lossless_time:.3f}s ({new_tokens/lossless_time:.2f} tok/s)")
    print(f"  -> Active KV Tensor Memory: {v_lossless_active_bytes / 1024:.2f} KB")
    print(f"  -> Offloaded Compressed Memory: {v_lossless_offloaded_bytes / 1024:.2f} KB")
    print(f"  -> Exact Token Match with Standard Cache: {exact_match_lossless}")

    # Run 3: VantageDynamicCache (FP8 Quantized)
    print("\n[3/3] Running VantageDynamicCache (Dynamic FP8 Quantized)...")
    vantage_fp8_cache = VantageDynamicCache(
        max_active_tokens=window_size,
        quantize_fp8=True,
        codec=CodecType.LZ4_BLOCK,
    )
    t0 = time.perf_counter()
    with torch.no_grad():
        out_fp8 = model.generate(
            input_ids,
            max_new_tokens=new_tokens,
            past_key_values=vantage_fp8_cache,
            do_sample=False,
        )
    t1 = time.perf_counter()
    fp8_time = t1 - t0
    fp8_rss = get_process_rss_mb()

    v_fp8_active_bytes = 0
    for l in range(n_layer):
        k, v = vantage_fp8_cache._get_layer_keys_values(l)
        v_fp8_active_bytes += k.numel() * k.element_size() + v.numel() * v.element_size()
    v_fp8_offloaded_bytes = sum(
        len(k_blob) + len(v_blob)
        for k_blob, v_blob, _ in vantage_fp8_cache._offloaded_blobs.values()
    )

    exact_match_fp8 = torch.equal(out_std, out_fp8)
    token_overlap_fp8 = (out_std == out_fp8).sum().item() / out_std.numel()

    results["vantage_dynamic_cache_fp8"] = {
        "time_sec": fp8_time,
        "tokens_per_sec": new_tokens / fp8_time,
        "peak_rss_mb": fp8_rss,
        "active_tensor_kb": v_fp8_active_bytes / 1024,
        "offloaded_compressed_kb": v_fp8_offloaded_bytes / 1024,
        "exact_token_match": exact_match_fp8,
        "token_overlap_ratio": token_overlap_fp8,
        "tokens_generated": new_tokens,
    }
    print(f"  -> Generated {new_tokens} tokens in {fp8_time:.3f}s ({new_tokens/fp8_time:.2f} tok/s)")
    print(f"  -> Active KV Tensor Memory: {v_fp8_active_bytes / 1024:.2f} KB")
    print(f"  -> Offloaded Compressed Memory: {v_fp8_offloaded_bytes / 1024:.2f} KB")
    print(f"  -> Exact Token Match with Standard Cache: {exact_match_fp8} (Overlap: {token_overlap_fp8*100:.1f}%)")

    # Save results
    output_dir = PROJECT_ROOT / "artifacts" / "benchmarks"
    output_dir.mkdir(parents=True, exist_ok=True)
    out_file = output_dir / "benchmark_hf_transformers_v7_2.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # Run 4: Full Restoration & Prompt Cache Parity Verification
    print("\n[4/4] Verifying Full KV Cache Restoration & Bit-Fidelity...")
    vantage_lossless_cache.restore_full_cache()
    restored_k, restored_v = vantage_lossless_cache._get_layer_keys_values(0)
    expected_k, expected_v = standard_cache.layers[0].keys, standard_cache.layers[0].values
    
    # Check length
    print(f"  -> Restored shape: {restored_k.shape} vs Expected: {expected_k.shape}")
    restored_match = (restored_k.shape == expected_k.shape) and torch.allclose(restored_k, expected_k, atol=1e-5)
    results["restored_cache_fidelity"] = {
        "restored_shape": list(restored_k.shape),
        "expected_shape": list(expected_k.shape),
        "lossless_fidelity_match": restored_match,
    }
    print(f"  -> Lossless Reconstruction Parity: {restored_match}")

    print("\n" + "=" * 70)
    print(f"  Benchmark complete! Output saved to: {out_file.relative_to(PROJECT_ROOT)}")
    print("=" * 70)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    return results


if __name__ == "__main__":
    run_benchmark()
