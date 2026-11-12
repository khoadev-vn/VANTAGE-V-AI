"""
Long-Context KV-Cache Scaling & Parity Benchmark (256 -> 4096 tokens).

Evaluates:
1. Standard HuggingFace DynamicCache memory growth (O(N) linear memory).
2. VantageStreamingCache (Bounded Window + Block-level Offload):
   - Active KV tensor memory capping.
   - Offloaded memory footprint (LZ4 vs FP8).
   - Throughput (tokens/sec).
3. VantagePromptCheckpoint (100% Bit-for-Bit Lossless Parity vs FP8):
   - Serialization + compression throughput.
   - Decompression & restoration latency.
   - Exact numerical equivalence verification (torch.equal == True).
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

from core.adapters.transformers import (
    VantagePromptCheckpoint,
    VantageStreamingCache,
    checkpoint_prompt_cache,
    restore_prompt_cache,
)
from core.engine import CodecType


def get_process_rss_mb() -> float:
    return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)


def run_long_context_benchmark():
    print("=" * 80)
    print("  VANTAGE V-AI v7.3.0: Long-Context KV-Cache Scaling & Parity Benchmark")
    print("=" * 80)

    torch.manual_seed(42)
    vocab_size = 1000
    n_embd = 256
    n_head = 4
    n_layer = 4

    config = AutoConfig.for_model(
        "gpt2",
        n_layer=n_layer,
        n_head=n_head,
        n_embd=n_embd,
        vocab_size=vocab_size,
        n_positions=8192,
        bos_token_id=0,
        eos_token_id=1,
    )
    model = AutoModelForCausalLM.from_config(config)
    model.eval()

    context_lengths = [256, 512, 1024, 2048, 4096]
    results = {
        "hardware": {
            "cpu": "Intel64 Family 6 Model 165 Stepping 3",
            "cores": "6C / 12T",
            "python": sys.version.split()[0],
            "torch": torch.__version__,
        },
        "model_architecture": {
            "n_layer": n_layer,
            "n_head": n_head,
            "n_embd": n_embd,
            "head_dim": n_embd // n_head,
        },
        "prompt_checkpoint_scaling": [],
        "streaming_generation_scaling": [],
    }

    # -------------------------------------------------------------
    # Part 1: Prompt Checkpoint Scaling (100% Bit-for-Bit Lossless Parity)
    # -------------------------------------------------------------
    print("\n--- Part 1: Prompt Cache Checkpointing & Exact Parity Verification ---")
    print(f"{'Context':>8} | {'Raw KV (KB)':>12} | {'LZ4 (KB)':>10} | {'FP8 (KB)':>10} | {'LZ4 Ratio':>10} | {'FP8 Ratio':>10} | {'Lossless Parity':>16}")
    print("-" * 86)

    for ctx_len in context_lengths:
        # Generate synthetic prompt prefill KV
        prompt_cache = DynamicCache()
        for l in range(n_layer):
            k = torch.randn(1, n_head, ctx_len, n_embd // n_head, dtype=torch.float32)
            v = torch.randn(1, n_head, ctx_len, n_embd // n_head, dtype=torch.float32)
            prompt_cache.update(k, v, layer_idx=l)

        raw_bytes = sum(
            prompt_cache.layers[l].keys.numel() * prompt_cache.layers[l].keys.element_size() * 2
            for l in range(n_layer)
        )

        # Checkpoint Lossless
        t0 = time.perf_counter()
        ckpt_lz4 = checkpoint_prompt_cache(prompt_cache, quantize_fp8=False)
        t_lz4_chk = (time.perf_counter() - t0) * 1000

        # Restore Lossless
        t0 = time.perf_counter()
        restored_lz4 = restore_prompt_cache(ckpt_lz4)
        t_lz4_rest = (time.perf_counter() - t0) * 1000

        # Verify bit-for-bit parity
        parity = all(
            torch.equal(prompt_cache.layers[l].keys, restored_lz4.layers[l].keys)
            and torch.equal(prompt_cache.layers[l].values, restored_lz4.layers[l].values)
            for l in range(n_layer)
        )

        # Checkpoint FP8
        ckpt_fp8 = checkpoint_prompt_cache(prompt_cache, quantize_fp8=True)

        ratio_lz4 = raw_bytes / ckpt_lz4.total_bytes
        ratio_fp8 = raw_bytes / ckpt_fp8.total_bytes

        print(
            f"{ctx_len:>8} | {raw_bytes/1024:>12.1f} | {ckpt_lz4.total_bytes/1024:>10.1f} | "
            f"{ckpt_fp8.total_bytes/1024:>10.1f} | {ratio_lz4:>9.2f}x | {ratio_fp8:>9.2f}x | "
            f"{str(parity):>16}"
        )

        results["prompt_checkpoint_scaling"].append({
            "context_tokens": ctx_len,
            "raw_kv_kb": raw_bytes / 1024,
            "lz4_compressed_kb": ckpt_lz4.total_bytes / 1024,
            "fp8_compressed_kb": ckpt_fp8.total_bytes / 1024,
            "lz4_compression_ratio": round(ratio_lz4, 2),
            "fp8_compression_ratio": round(ratio_fp8, 2),
            "checkpoint_time_ms": round(t_lz4_chk, 2),
            "restore_time_ms": round(t_lz4_rest, 2),
            "bit_for_bit_lossless_parity": parity,
        })

    # -------------------------------------------------------------
    # Part 2: Long-Context Streaming Generation Scaling
    # -------------------------------------------------------------
    print("\n--- Part 2: Streaming Generation with Block-Level Offload ---")
    gen_tokens = 128
    window_size = 256
    block_size = 32

    print(f"Testing generation of {gen_tokens} tokens with window={window_size}, block={block_size}...")
    print(f"{'Prompt':>8} | {'Cache Type':>24} | {'Active KV (KB)':>15} | {'Offloaded (KB)':>15} | {'Throughput (tok/s)':>20}")
    print("-" * 92)

    for prompt_len in [256, 1024, 2048]:
        input_ids = torch.randint(2, vocab_size, (1, prompt_len))

        # Standard HF DynamicCache
        std_cache = DynamicCache()
        t0 = time.perf_counter()
        with torch.no_grad():
            model.generate(input_ids, max_new_tokens=gen_tokens, past_key_values=std_cache, do_sample=False)
        std_time = time.perf_counter() - t0
        std_speed = gen_tokens / std_time
        std_active_kb = sum(
            std_cache.layers[l].keys.numel() * std_cache.layers[l].keys.element_size() * 2
            for l in range(n_layer)
        ) / 1024

        print(f"{prompt_len:>8} | {'Standard HF DynamicCache':>24} | {std_active_kb:>15.1f} | {0.0:>15.1f} | {std_speed:>20.2f}")

        # Vantage StreamingCache (Lossless LZ4, block_size=32)
        vantage_cache = VantageStreamingCache(
            max_active_tokens=window_size,
            block_size=block_size,
            quantize_fp8=False,
        )
        t0 = time.perf_counter()
        with torch.no_grad():
            model.generate(input_ids, max_new_tokens=gen_tokens, past_key_values=vantage_cache, do_sample=False)
        v_time = time.perf_counter() - t0
        v_speed = gen_tokens / v_time
        v_active_kb = sum(
            vantage_cache._get_layer_keys_values(l)[0].numel() * vantage_cache._get_layer_keys_values(l)[0].element_size() * 2
            for l in range(n_layer)
        ) / 1024
        v_offloaded_kb = vantage_cache.get_offloaded_memory_bytes() / 1024

        print(f"{prompt_len:>8} | {'Vantage Streaming (Block=32)':>24} | {v_active_kb:>15.1f} | {v_offloaded_kb:>15.1f} | {v_speed:>20.2f}")

        results["streaming_generation_scaling"].append({
            "prompt_tokens": prompt_len,
            "generated_tokens": gen_tokens,
            "window_size": window_size,
            "block_size": block_size,
            "standard_cache": {
                "active_kv_kb": std_active_kb,
                "throughput_tok_s": round(std_speed, 2),
            },
            "vantage_streaming_cache": {
                "active_kv_kb": v_active_kb,
                "offloaded_compressed_kb": v_offloaded_kb,
                "active_kv_reduction_pct": round((1.0 - v_active_kb / std_active_kb) * 100, 1),
                "throughput_tok_s": round(v_speed, 2),
                "throughput_retention_pct": round((v_speed / std_speed) * 100, 1),
            },
        })

    # Save artifact
    out_dir = PROJECT_ROOT / "artifacts" / "benchmarks"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "benchmark_long_context_v7_3.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 80)
    print(f"  Benchmark complete! Saved artifact to: {out_file.relative_to(PROJECT_ROOT)}")
    print("=" * 80)
    return results


if __name__ == "__main__":
    run_long_context_benchmark()
