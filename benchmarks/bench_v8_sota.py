"""VANTAGE-V-AI v8.0.0 SOTA Frontier Benchmark Suite.

Benchmarks:
1. 1.58-bit Ternary vs FP8 vs Lossless KV Compression & Fidelity
2. Pretrained GPT-2 Generation with Attention Sinks vs Standard HF DynamicCache
3. Hyperdimensional Associative Memory Sub-Millisecond Bitwise Retrieval
"""

import gc
import json
import os
import platform
import sys
import threading
import time
from typing import Dict, List, Any

import numpy as np
import psutil
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.adapters.transformers import (
    VantageDynamicCache,
    VantageStreamingCache,
    checkpoint_prompt_cache,
    restore_prompt_cache,
)
from core.kv import (
    QUANT_FP8,
    QUANT_LOSSLESS,
    QUANT_TERNARY_158,
    KVCacheStreamEngine,
    compute_quantization_error,
    pack_ternary_2bit,
    unpack_ternary_2bit,
)
from core.memory import HyperdimensionalMemory


class PeakMemoryTracker:
    def __init__(self, interval_sec: float = 0.005):
        self.interval = interval_sec
        self.running = False
        self.peak_rss = 0
        self.process = psutil.Process()
        self._thread = None

    def _monitor(self):
        while self.running:
            try:
                rss = self.process.memory_info().rss
                if rss > self.peak_rss:
                    self.peak_rss = rss
            except Exception:
                pass
            time.sleep(self.interval)

    def start(self):
        self.peak_rss = self.process.memory_info().rss
        self.running = True
        self._thread = threading.Thread(target=self._monitor, daemon=True)
        self._thread.start()

    def stop(self) -> int:
        self.running = False
        if self._thread is not None:
            self._thread.join(timeout=0.5)
        rss = self.process.memory_info().rss
        if rss > self.peak_rss:
            self.peak_rss = rss
        return self.peak_rss


def benchmark_kv_compression():
    print("--- 1. KV-Cache Multi-Tier Compression Benchmark ---")
    results = {}
    engine = KVCacheStreamEngine()

    shapes = [
        ("ctx_512", (512, 12, 64)),   # GPT-2 small layer
        ("ctx_2048", (2048, 12, 64)),
        ("ctx_4096", (4096, 12, 64)),
    ]

    for label, shape in shapes:
        results[label] = {}
        # Generate synthetic FP16 tensor
        t = torch.randn(*shape, dtype=torch.float16)
        raw_bytes = t.numel() * t.element_size()

        # 1. Lossless LZ4
        t0 = time.perf_counter()
        payload_lz4, meta_lz4 = engine.compress_kv_block(t, quantize_fp8=False, quant_mode=QUANT_LOSSLESS)
        t_lz4_enc = time.perf_counter() - t0

        t0 = time.perf_counter()
        restored_lz4, _ = engine.decompress_kv_block(payload_lz4, as_torch=True)
        t_lz4_dec = time.perf_counter() - t0

        assert torch.equal(t, restored_lz4), "Lossless parity failure"
        lz4_ratio = raw_bytes / len(payload_lz4)

        # 2. FP8 Lossy Quantization
        t0 = time.perf_counter()
        payload_fp8, meta_fp8 = engine.compress_kv_block(t, quantize_fp8=True, quant_mode=QUANT_FP8)
        t_fp8_enc = time.perf_counter() - t0

        t0 = time.perf_counter()
        restored_fp8, _ = engine.decompress_kv_block(payload_fp8, as_torch=True)
        t_fp8_dec = time.perf_counter() - t0

        err_fp8 = compute_quantization_error(t, restored_fp8)
        fp8_ratio = raw_bytes / len(payload_fp8)

        # 3. 1.58-bit Ternary Quantization (BitNet format)
        t0 = time.perf_counter()
        payload_ternary, meta_ternary = engine.compress_kv_block(t, quant_mode=QUANT_TERNARY_158)
        t_ternary_enc = time.perf_counter() - t0

        t0 = time.perf_counter()
        restored_ternary, _ = engine.decompress_kv_block(payload_ternary, as_torch=True)
        t_ternary_dec = time.perf_counter() - t0

        err_ternary = compute_quantization_error(t, restored_ternary)
        ternary_ratio = raw_bytes / len(payload_ternary)

        results[label] = {
            "raw_bytes": raw_bytes,
            "lossless": {
                "compressed_bytes": len(payload_lz4),
                "ratio_vs_fp16": round(lz4_ratio, 2),
                "enc_mb_s": round((raw_bytes / (1024 * 1024)) / max(t_lz4_enc, 1e-6), 1),
                "dec_mb_s": round((raw_bytes / (1024 * 1024)) / max(t_lz4_dec, 1e-6), 1),
                "exact_bit_parity": True,
            },
            "fp8": {
                "compressed_bytes": len(payload_fp8),
                "ratio_vs_fp16": round(fp8_ratio, 2),
                "enc_mb_s": round((raw_bytes / (1024 * 1024)) / max(t_fp8_enc, 1e-6), 1),
                "dec_mb_s": round((raw_bytes / (1024 * 1024)) / max(t_fp8_dec, 1e-6), 1),
                "cosine_similarity": round(err_fp8.cosine_similarity, 5),
                "mae": round(err_fp8.mean_absolute_error, 5),
            },
            "ternary_158": {
                "compressed_bytes": len(payload_ternary),
                "ratio_vs_fp16": round(ternary_ratio, 2),
                "enc_mb_s": round((raw_bytes / (1024 * 1024)) / max(t_ternary_enc, 1e-6), 1),
                "dec_mb_s": round((raw_bytes / (1024 * 1024)) / max(t_ternary_dec, 1e-6), 1),
                "cosine_similarity": round(err_ternary.cosine_similarity, 5),
                "mae": round(err_ternary.mean_absolute_error, 5),
            },
        }
        print(f"[{label}] Raw: {raw_bytes/1024:.1f}KB | LZ4: {lz4_ratio:.2f}x | FP8: {fp8_ratio:.2f}x (cos={err_fp8.cosine_similarity:.4f}) | Ternary: {ternary_ratio:.2f}x (cos={err_ternary.cosine_similarity:.4f})")

    return results


def benchmark_hd_memory():
    print("\n--- 2. Hyperdimensional Associative Memory Benchmark ---")
    hd = HyperdimensionalMemory(input_dim=768, hd_dim=4096, seed=42)

    # Populate with 5,000 vectors
    n_items = 5000
    rng = np.random.default_rng(42)
    vectors = rng.standard_normal((n_items, 768)).astype(np.float32)

    t0 = time.perf_counter()
    for i in range(n_items):
        hd.store(f"doc_{i}", vectors[i], metadata={"idx": i})
    insert_sec = time.perf_counter() - t0
    insert_rate = n_items / insert_sec

    # Measure Query Latency
    n_queries = 200
    query_indices = rng.integers(0, n_items, size=n_queries)
    
    # Warmup
    hd.query(vectors[0], top_k=5)

    latencies_us = []
    recalls_top1 = 0

    for idx in query_indices:
        q_vec = vectors[idx] + 0.05 * rng.standard_normal(768).astype(np.float32)  # noisy query
        t_start = time.perf_counter()
        res = hd.query(q_vec, top_k=1)
        t_query = (time.perf_counter() - t_start) * 1e6  # microseconds
        latencies_us.append(t_query)
        if res and res[0].key == f"doc_{idx}":
            recalls_top1 += 1

    avg_latency_us = float(np.mean(latencies_us))
    p95_latency_us = float(np.percentile(latencies_us, 95))
    recall_rate = recalls_top1 / n_queries

    print(f"Items: {n_items} | Insert: {insert_rate:.1f} items/s")
    print(f"Mean Query Latency: {avg_latency_us:.1f} µs ({avg_latency_us/1000:.3f} ms) | P95: {p95_latency_us:.1f} µs")
    print(f"Noisy Query 1-NN Recall: {recall_rate * 100:.1f}%")

    return {
        "capacity_items": n_items,
        "input_dim": 768,
        "hd_dim_bits": 4096,
        "memory_footprint_bytes": n_items * (4096 // 8),
        "insert_throughput_items_per_sec": round(insert_rate, 1),
        "query_latency_mean_us": round(avg_latency_us, 1),
        "query_latency_p95_us": round(p95_latency_us, 1),
        "top1_noisy_recall_percent": round(recall_rate * 100, 1),
    }


def benchmark_pretrained_gpt2_streaming(n_trials: int = 5):
    print("\n--- 3. Pretrained GPT-2 Generation Benchmark with Attention Sinks ---")
    model_name = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.eval()

    long_prompt_text = (
        "The architecture of modern artificial intelligence systems requires fundamental innovations "
        "in memory hierarchy and streaming computational efficiency. As context lengths scale from thousands "
        "to millions of tokens, the conventional quadratic complexity of self-attention mechanisms imposes "
        "prohibitive memory bandwidth and capacity constraints on hardware accelerators. To address these "
        "challenges, researcher communities have explored hierarchical KV-cache offloading, sliding window attention, "
        "and structured quantization formats including FP8 and 1.58-bit ternary representations. "
        "However, sliding window attention frequently suffers from catastrophic perplexity collapse because initial tokens, "
        "known as attention sinks, absorb disproportionate attention scores regardless of semantic distance. "
        "By permanently retaining initial sink tokens in active memory while offloading intermediate history "
        "into packed binary hyperdimensional stores, next-generation engines achieve bounded constant memory "
        "without sacrificing autoregressive generation coherence."
    )
    tokens = tokenizer.encode(long_prompt_text)
    # Replicate to reach exactly 512 tokens
    while len(tokens) < 512:
        tokens = (tokens + tokens)[:512]
    tokens = tokens[:512]
    input_ids = torch.tensor([tokens], dtype=torch.long)
    new_tokens = 64

    configs = [
        ("standard_hf", "Standard HF DynamicCache (Full Context)", None),
        ("vantage_sliding_nosink", "Vantage Sliding Window (No Sink, W=256)", {"sink_tokens": 0, "max_active": 256, "quant_mode": QUANT_LOSSLESS}),
        ("vantage_sota_sink4", "Vantage SOTA (Sink=4, W=256, Lossless)", {"sink_tokens": 4, "max_active": 256, "quant_mode": QUANT_LOSSLESS}),
        ("vantage_sota_sink4_fp8", "Vantage SOTA (Sink=4, W=256, FP8)", {"sink_tokens": 4, "max_active": 256, "quant_mode": QUANT_FP8}),
    ]

    results = {}

    for cfg_id, cfg_label, cfg_params in configs:
        throughputs = []
        peak_rss_mb = []
        generated_token_ids_list = []
        active_kv_kb_list = []
        offloaded_kv_kb_list = []

        for trial in range(n_trials):
            gc.collect()
            tracker = PeakMemoryTracker(interval_sec=0.005)
            tracker.start()

            # Create fresh cache
            if cfg_params is None:
                past_cache = DynamicCache()
            else:
                past_cache = VantageStreamingCache(
                    max_active_tokens=cfg_params["max_active"],
                    block_size=32,
                    sink_tokens=cfg_params["sink_tokens"],
                    quant_mode=cfg_params["quant_mode"],
                )

            t0 = time.perf_counter()
            with torch.no_grad():
                out = model.generate(
                    input_ids,
                    past_key_values=past_cache,
                    max_new_tokens=new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )
            dur = time.perf_counter() - t0
            peak_rss = tracker.stop()

            tok_s = new_tokens / dur
            throughputs.append(tok_s)
            peak_rss_mb.append(peak_rss / (1024 * 1024))
            generated_token_ids = out[0, input_ids.shape[-1]:].tolist()
            generated_token_ids_list.append(generated_token_ids)

            if isinstance(past_cache, VantageStreamingCache):
                active_kb = past_cache.get_active_memory_bytes() / 1024
                offloaded_kb = past_cache.get_offloaded_memory_bytes() / 1024
            else:
                # Standard HF
                tot_bytes = sum(
                    layer.keys.numel() * layer.keys.element_size() + layer.values.numel() * layer.values.element_size()
                    for layer in past_cache.layers
                ) if hasattr(past_cache, "layers") else sum(
                    past_cache.key_cache[l].numel() * past_cache.key_cache[l].element_size() +
                    past_cache.value_cache[l].numel() * past_cache.value_cache[l].element_size()
                    for l in range(len(past_cache.key_cache))
                )
                active_kb = tot_bytes / 1024
                offloaded_kb = 0.0

            active_kv_kb_list.append(active_kb)
            offloaded_kv_kb_list.append(offloaded_kb)

        sample_text = tokenizer.decode(generated_token_ids_list[0])
        results[cfg_id] = {
            "label": cfg_label,
            "throughput_mean": round(float(np.mean(throughputs)), 2),
            "throughput_std": round(float(np.std(throughputs)), 2),
            "peak_rss_mb_mean": round(float(np.mean(peak_rss_mb)), 2),
            "active_kv_kb": round(float(np.mean(active_kv_kb_list)), 2),
            "offloaded_kv_kb": round(float(np.mean(offloaded_kv_kb_list)), 2),
            "generated_tokens": generated_token_ids_list[0],
            "sample_generation_text": sample_text[:120] + "...",
        }
        print(f"[{cfg_id}] Throughput: {results[cfg_id]['throughput_mean']} ± {results[cfg_id]['throughput_std']} tok/s | Active KV: {results[cfg_id]['active_kv_kb']} KB | Offloaded: {results[cfg_id]['offloaded_kv_kb']} KB | Peak RSS: {results[cfg_id]['peak_rss_mb_mean']} MB")

    # Evaluate exact token overlap against standard_hf baseline strictly on generated tokens
    std_tokens = results["standard_hf"]["generated_tokens"]
    for cfg_id in results:
        cfg_toks = results[cfg_id]["generated_tokens"]
        matches = sum(1 for a, b in zip(std_tokens, cfg_toks) if a == b)
        overlap_pct = (matches / len(std_tokens)) * 100
        results[cfg_id]["generated_token_overlap_percent"] = round(overlap_pct, 2)
        results[cfg_id]["matched_tokens_count"] = f"{matches}/{len(std_tokens)}"

    return results


def main():
    print(f"=== Starting VANTAGE-V-AI v8.0.0 SOTA Benchmark ===")
    print(f"Platform: {platform.platform()} | CPU: {platform.processor()} | PyTorch: {torch.__version__}")
    
    t_start_total = time.perf_counter()
    kv_comp = benchmark_kv_compression()
    hd_mem = benchmark_hd_memory()
    gpt2_stream = benchmark_pretrained_gpt2_streaming(n_trials=5)
    total_time = time.perf_counter() - t_start_total

    final_payload = {
        "metadata": {
            "version": "8.0.0",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
            "platform": platform.platform(),
            "cpu": platform.processor(),
            "pytorch_version": torch.__version__,
            "total_benchmark_time_seconds": round(total_time, 2),
        },
        "kv_cache_compression": kv_comp,
        "hyperdimensional_associative_memory": hd_mem,
        "pretrained_gpt2_e2e_streaming": gpt2_stream,
    }

    out_dir = os.path.join(os.path.dirname(__file__), "..", "artifacts", "benchmarks")
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, "benchmark_v8_sota.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(final_payload, f, indent=2)

    print(f"\n[DONE] Saved comprehensive benchmark artifact to {out_file}")


if __name__ == "__main__":
    main()
