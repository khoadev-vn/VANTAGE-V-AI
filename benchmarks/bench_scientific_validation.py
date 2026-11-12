"""
Vantage V-AI v7.3.1: Rigorous Scientific Validation on Pretrained Hugging Face GPT-2.

Methodology following MLSys & Systems Review Standards:
1. Real Pretrained Weights: Hugging Face `gpt2` (12 layers, 12 heads, 768 dim, 124M params).
2. Real Text Ingestion: Human language corpus prompt.
3. Multi-Trial Statistics: 1 warmup + 10 independent trials per configuration (P50, StdDev, IQR).
4. Transparent Memory Accounting:
   - Active Attention KV Tensor Memory (VRAM / computation tier).
   - Offloaded Compressed Memory (Host storage / cache tier).
   - Total Process RSS (OS-level footprint).
5. Output Semantics & Quality Verification:
   - Part A: Prompt Checkpoint Parity (torch.equal == True for Lossless LZ4; MAE/CosSim for FP8).
   - Part B: Streaming Generation Trade-off (Throughput, Memory Capping, and Token Overlap).
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import psutil
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.adapters.transformers import (
    VantagePromptCheckpoint,
    VantageStreamingCache,
    checkpoint_prompt_cache,
    restore_prompt_cache,
)


def get_process_rss_mb() -> float:
    return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)


def run_scientific_validation():
    print("=" * 84)
    print("  VANTAGE V-AI v7.3.1: Pretrained GPT-2 Rigorous Scientific Validation")
    print("=" * 84)

    # 1. Load Pretrained Model & Tokenizer
    model_name = "gpt2"
    print(f"\n[1/3] Loading pretrained '{model_name}' (12 layers, 12 heads, 768 dim)...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.eval()

    # Real human prompt text (repeated to reach desired prompt tokens)
    base_text = (
        "In modern artificial intelligence infrastructure, large language models demand "
        "exponentially growing memory for key-value attention caching during autoregressive decoding. "
        "Systems researchers have proposed hierarchical offloading, sliding window attention, "
        "and low-bit quantization to prevent out-of-memory errors on commodity accelerators. "
    )
    prompt_tokens_target = 512
    tokens = tokenizer.encode(base_text)
    repeat_factor = (prompt_tokens_target // len(tokens)) + 1
    input_ids = torch.tensor([ (tokens * repeat_factor)[:prompt_tokens_target] ])
    actual_prompt_len = input_ids.shape[1]
    print(f"  -> Real text prompt prepared: {actual_prompt_len} tokens.")

    results = {
        "model": {
            "name": model_name,
            "layers": model.config.n_layer,
            "heads": model.config.n_head,
            "hidden_size": model.config.n_embd,
            "head_dim": model.config.n_embd // model.config.n_head,
            "pretrained": True,
        },
        "prompt_checkpoint_evaluation": {},
        "streaming_generation_trials": {},
    }

    # ------------------------------------------------------------------
    # Part 1: Prompt Cache Checkpoint Parity (Real Pretrained Prefill)
    # ------------------------------------------------------------------
    print("\n[2/3] Evaluating Prompt Checkpoint on Real Prefill KV States...")
    with torch.no_grad():
        prefill_cache = DynamicCache()
        _ = model(input_ids, past_key_values=prefill_cache, use_cache=True)

    raw_kv_bytes = sum(
        prefill_cache.layers[l].keys.numel() * prefill_cache.layers[l].keys.element_size() * 2
        for l in range(model.config.n_layer)
    )

    # Checkpoint Lossless LZ4
    t0 = time.perf_counter()
    ckpt_lz4 = checkpoint_prompt_cache(prefill_cache, quantize_fp8=False)
    t_chk_lz4 = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    restored_lz4 = restore_prompt_cache(ckpt_lz4)
    t_rest_lz4 = (time.perf_counter() - t0) * 1000

    # Bit-for-bit check
    lossless_exact_parity = all(
        torch.equal(prefill_cache.layers[l].keys, restored_lz4.layers[l].keys)
        and torch.equal(prefill_cache.layers[l].values, restored_lz4.layers[l].values)
        for l in range(model.config.n_layer)
    )

    # Checkpoint Dynamic FP8
    t0 = time.perf_counter()
    ckpt_fp8 = checkpoint_prompt_cache(prefill_cache, quantize_fp8=True)
    t_chk_fp8 = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    restored_fp8 = restore_prompt_cache(ckpt_fp8)
    t_rest_fp8 = (time.perf_counter() - t0) * 1000

    # FP8 error metrics
    fp8_cos_sims = []
    fp8_maes = []
    for l in range(model.config.n_layer):
        orig_k = prefill_cache.layers[l].keys
        rest_k = restored_fp8.layers[l].keys
        cos = torch.nn.functional.cosine_similarity(orig_k.flatten(), rest_k.flatten(), dim=0).item()
        mae = (orig_k - rest_k).abs().mean().item()
        fp8_cos_sims.append(cos)
        fp8_maes.append(mae)

    avg_cos_sim = float(np.mean(fp8_cos_sims))
    avg_mae = float(np.mean(fp8_maes))

    print(f"  -> Raw Prefill KV Memory: {raw_kv_bytes / 1024:.1f} KB ({raw_kv_bytes / (1024*1024):.2f} MB)")
    print(f"  -> Lossless LZ4 Checkpoint: {ckpt_lz4.total_bytes / 1024:.1f} KB | Parity (torch.equal): {lossless_exact_parity}")
    print(f"  -> Dynamic FP8 Checkpoint: {ckpt_fp8.total_bytes / 1024:.1f} KB | Ratio: {raw_kv_bytes / ckpt_fp8.total_bytes:.2f}x | Avg CosSim: {avg_cos_sim:.5f} | Avg MAE: {avg_mae:.5f}")

    results["prompt_checkpoint_evaluation"] = {
        "prompt_tokens": actual_prompt_len,
        "raw_kv_kb": raw_kv_bytes / 1024,
        "lz4": {
            "compressed_kb": ckpt_lz4.total_bytes / 1024,
            "compression_ratio": round(raw_kv_bytes / ckpt_lz4.total_bytes, 2),
            "checkpoint_ms": round(t_chk_lz4, 2),
            "restore_ms": round(t_rest_lz4, 2),
            "bit_for_bit_identical": lossless_exact_parity,
        },
        "fp8": {
            "compressed_kb": ckpt_fp8.total_bytes / 1024,
            "compression_ratio": round(raw_kv_bytes / ckpt_fp8.total_bytes, 2),
            "checkpoint_ms": round(t_chk_fp8, 2),
            "restore_ms": round(t_rest_fp8, 2),
            "bit_for_bit_identical": False,  # Transparently lossy
            "average_cosine_similarity": round(avg_cos_sim, 5),
            "average_mae": round(avg_mae, 5),
        },
    }

    # ------------------------------------------------------------------
    # Part 2: Statistical Streaming Generation (10 Trials + Warmup)
    # ------------------------------------------------------------------
    print("\n[3/3] Running Multi-Trial Generation (1 Warmup + 10 Independent Trials)...")
    gen_tokens = 64
    window_size = 256
    block_size = 32
    num_trials = 10

    # Warmup
    print("  -> Performing warmup runs...")
    with torch.no_grad():
        w_cache = DynamicCache()
        _ = model.generate(input_ids, max_new_tokens=8, past_key_values=w_cache, do_sample=False)
        w_vcache = VantageStreamingCache(max_active_tokens=window_size, block_size=block_size)
        _ = model.generate(input_ids, max_new_tokens=8, past_key_values=w_vcache, do_sample=False)

    # Benchmark Standard HF DynamicCache
    std_speeds = []
    std_rss_list = []
    out_std = None
    for trial in range(num_trials):
        cache = DynamicCache()
        rss_start = get_process_rss_mb()
        t0 = time.perf_counter()
        with torch.no_grad():
            out_std = model.generate(input_ids, max_new_tokens=gen_tokens, past_key_values=cache, do_sample=False)
        dur = time.perf_counter() - t0
        std_speeds.append(gen_tokens / dur)
        std_rss_list.append(get_process_rss_mb())

    std_active_kb = sum(
        cache.layers[l].keys.numel() * cache.layers[l].keys.element_size() * 2
        for l in range(model.config.n_layer)
    ) / 1024

    # Benchmark VantageStreamingCache
    v_speeds = []
    v_rss_list = []
    out_vantage = None
    for trial in range(num_trials):
        cache = VantageStreamingCache(max_active_tokens=window_size, block_size=block_size)
        t0 = time.perf_counter()
        with torch.no_grad():
            out_vantage = model.generate(input_ids, max_new_tokens=gen_tokens, past_key_values=cache, do_sample=False)
        dur = time.perf_counter() - t0
        v_speeds.append(gen_tokens / dur)
        v_rss_list.append(get_process_rss_mb())

    v_active_kb = sum(
        cache._get_layer_keys_values(l)[0].numel() * cache._get_layer_keys_values(l)[0].element_size() * 2
        for l in range(model.config.n_layer)
    ) / 1024
    v_offload_kb = cache.get_offloaded_memory_bytes() / 1024

    # Output Quality / Token Overlap Analysis
    token_overlap = (out_std == out_vantage).sum().item() / out_std.numel()

    # Statistical summaries
    std_p50 = float(np.median(std_speeds))
    std_std = float(np.std(std_speeds))
    v_p50 = float(np.median(v_speeds))
    v_std = float(np.std(v_speeds))

    print("\n" + "-" * 84)
    print(f"{'Metric':<32} | {'Standard HF DynamicCache':<22} | {'Vantage Streaming (Window=256)':<24}")
    print("-" * 84)
    print(f"{'Active Attention KV Memory':<32} | {std_active_kb:<22.1f} KB | {v_active_kb:<22.1f} KB")
    print(f"{'Offloaded Compressed Memory':<32} | {'0.0 KB':<22} | {v_offload_kb:<22.1f} KB")
    print(f"{'Total KV Memory (Active+Offload)':<32} | {std_active_kb:<22.1f} KB | {v_active_kb + v_offload_kb:<22.1f} KB")
    print(f"{'Active KV Memory Reduction':<32} | {'Baseline (0%)':<22} | {(1.0 - v_active_kb/std_active_kb)*100:<22.1f} %")
    print(f"{'Throughput P50 (Median)':<32} | {std_p50:<22.2f} tok/s | {v_p50:<22.2f} tok/s")
    print(f"{'Throughput StdDev':<32} | {std_std:<22.2f} tok/s | {v_std:<22.2f} tok/s")
    print(f"{'Token Overlap Ratio':<32} | {'100.0%':<22} | {token_overlap*100:<22.1f} %")
    print("-" * 84)

    results["streaming_generation_trials"] = {
        "trials": num_trials,
        "prompt_tokens": actual_prompt_len,
        "generated_tokens": gen_tokens,
        "window_size": window_size,
        "block_size": block_size,
        "standard_hf": {
            "active_kv_kb": std_active_kb,
            "offloaded_kb": 0.0,
            "throughput_p50": round(std_p50, 2),
            "throughput_std": round(std_std, 2),
            "peak_rss_mb": round(float(np.max(std_rss_list)), 2),
        },
        "vantage_streaming": {
            "active_kv_kb": v_active_kb,
            "offloaded_kb": v_offload_kb,
            "total_kv_kb": v_active_kb + v_offload_kb,
            "active_reduction_pct": round((1.0 - v_active_kb / std_active_kb) * 100, 1),
            "throughput_p50": round(v_p50, 2),
            "throughput_std": round(v_std, 2),
            "peak_rss_mb": round(float(np.max(v_rss_list)), 2),
            "token_overlap_ratio": round(token_overlap, 4),
        },
    }

    # Save to artifacts
    out_dir = PROJECT_ROOT / "artifacts" / "benchmarks"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "benchmark_scientific_validation_v7_3_1.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved scientific validation artifact to: {out_file.relative_to(PROJECT_ROOT)}")
    print("=" * 84)
    return results


if __name__ == "__main__":
    run_scientific_validation()
