"""
Benchmark Vantage KV-Cache Streaming & Offload against Raw PyTorch Serialization.

Simulates Llama-3-8B KV-Cache:
- 32 Layers
- 8 KV Heads (Grouped-Query Attention)
- 128 Head Dim
- 2048 to 8192 Tokens Context Window
- Float16 Tensor Representation
"""

from __future__ import annotations

import io
import json
import statistics
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from core.engine import CodecType
from core.kv import KVCacheStreamEngine, compute_quantization_error


def benchmark_llama3_kv_cache(tokens: int = 4096, num_layers: int = 8) -> dict:
    """Benchmark offloading and reloading an 8-layer slice of Llama-3-8B KV Cache."""
    heads = 8
    head_dim = 128
    print(f"\n[BENCHMARK] Simulating Llama-3 KV-Cache: {num_layers} layers, {tokens} tokens, {heads} heads, {head_dim} dim (float16)")

    # Build KV cache tensors (simulate attention outputs)
    kv_cache = {}
    total_raw_bytes = 0
    for layer in range(num_layers):
        k = torch.randn(tokens, heads, head_dim, dtype=torch.float16)
        v = torch.randn(tokens, heads, head_dim, dtype=torch.float16)
        kv_cache[layer] = (k, v)
        total_raw_bytes += k.numel() * k.element_size() + v.numel() * v.element_size()

    raw_mb = total_raw_bytes / (1024 * 1024)
    print(f"  Raw Tensor Volume: {raw_mb:.2f} MB")

    engine = KVCacheStreamEngine(codec=CodecType.LZ4_BLOCK)

    # 1. Benchmark Vantage FP8 Offload (Quantize + Stream)
    fp8_times = []
    fp8_bytes = 0
    for _ in range(5):
        stream = io.BytesIO()
        t0 = time.perf_counter()
        fp8_bytes = engine.stream_offload_cache(kv_cache, stream, quantize_fp8=True)
        fp8_times.append(time.perf_counter() - t0)

    med_fp8_offload = statistics.median(fp8_times)
    fp8_ratio = total_raw_bytes / fp8_bytes

    # 2. Benchmark Vantage Lossless Offload
    offload_times = []
    compressed_bytes = 0
    for _ in range(5):
        stream = io.BytesIO()
        t0 = time.perf_counter()
        compressed_bytes = engine.stream_offload_cache(kv_cache, stream, quantize_fp8=False)
        offload_times.append(time.perf_counter() - t0)

    med_offload = statistics.median(offload_times)
    offload_gb_s = (total_raw_bytes / (1024**3)) / med_offload

    # 3. Benchmark Vantage Reload (Stream Read + Decompress)
    stream.seek(0)
    reload_times = []
    for _ in range(5):
        stream.seek(0)
        t0 = time.perf_counter()
        reloaded = engine.stream_reload_cache(stream, as_torch=True)
        reload_times.append(time.perf_counter() - t0)

    med_reload = statistics.median(reload_times)
    reload_gb_s = (total_raw_bytes / (1024**3)) / med_reload

    # Correctness check: bit-for-bit identity on lossless
    for layer in range(num_layers):
        assert torch.equal(kv_cache[layer][0], reloaded[layer][0])
        assert torch.equal(kv_cache[layer][1], reloaded[layer][1])

    # 4. Rigorous Quantization Error Profiling (MAE, RMSE, Max Error, Rel Error, Cosine)
    stream_fp8 = io.BytesIO()
    engine.stream_offload_cache(kv_cache, stream_fp8, quantize_fp8=True)
    stream_fp8.seek(0)
    reloaded_fp8 = engine.stream_reload_cache(stream_fp8, as_torch=True)
    layer_errors = [
        compute_quantization_error(kv_cache[l][0], reloaded_fp8[l][0])
        for l in range(num_layers)
    ]
    max_abs_err = max(e.max_absolute_error for e in layer_errors)
    avg_mae = statistics.mean(e.mean_absolute_error for e in layer_errors)
    avg_rmse = statistics.mean(e.root_mean_squared_error for e in layer_errors)
    avg_rel_err = statistics.mean(e.relative_error for e in layer_errors)
    min_cos_sim = min(e.cosine_similarity for e in layer_errors)

    # 5. Baseline: PyTorch torch.save()
    save_stream = io.BytesIO()
    save_times = []
    for _ in range(5):
        save_stream.seek(0)
        t0 = time.perf_counter()
        torch.save(kv_cache, save_stream)
        save_times.append(time.perf_counter() - t0)
    med_torch_save = statistics.median(save_times)

    result = {
        "tokens": tokens,
        "layers": num_layers,
        "raw_mb": round(raw_mb, 2),
        "fp8_compressed_mb": round(fp8_bytes / (1024 * 1024), 2),
        "fp8_compression_ratio": round(fp8_ratio, 2),
        "fp8_offload_ms": round(med_fp8_offload * 1000, 2),
        "error_metrics": {
            "max_absolute_error": round(max_abs_err, 6),
            "mean_absolute_error": round(avg_mae, 6),
            "root_mean_squared_error": round(avg_rmse, 6),
            "mean_relative_error": round(avg_rel_err, 6),
            "min_cosine_similarity": round(min_cos_sim, 6),
        },
        "lossless_offload_ms": round(med_offload * 1000, 2),
        "lossless_reload_ms": round(med_reload * 1000, 2),
        "torch_save_ms": round(med_torch_save * 1000, 2),
        "bit_for_bit_lossless": True,
    }
    print(f"  FP8 Offload: {result['fp8_offload_ms']} ms | Memory: {result['raw_mb']} MB -> {result['fp8_compressed_mb']} MB ({result['fp8_compression_ratio']}x reduction)")
    print(f"  Error Metrics: MAE={result['error_metrics']['mean_absolute_error']} | RMSE={result['error_metrics']['root_mean_squared_error']} | RelErr={result['error_metrics']['mean_relative_error']} | CosSim={result['error_metrics']['min_cosine_similarity']}")
    print(f"  Lossless Offload: {result['lossless_offload_ms']} ms | Reload: {result['lossless_reload_ms']} ms")
    return result


def main():
    res_4k = benchmark_llama3_kv_cache(tokens=4096, num_layers=8)
    res_8k = benchmark_llama3_kv_cache(tokens=8192, num_layers=8)

    out_file = REPO_ROOT / "benchmarks" / "results" / "kv_cache_benchmark.json"
    out_file.write_text(json.dumps({"llama3_4k": res_4k, "llama3_8k": res_8k}, indent=2))
    print(f"\n[OK] Results saved to {out_file}")


if __name__ == "__main__":
    main()
