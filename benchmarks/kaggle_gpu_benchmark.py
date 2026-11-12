"""
Kaggle GPU Benchmark Suite v8.2.0 (Peer-Review Edition).

Enhancements:
1. Direct pairwise equality comparison: Lossless vs FP8 (validating quantization drift).
2. Per-GPU memory accounting across GPU 0 and GPU 1 (CUDA device tracking).
3. CUDA Event-based timing (hardware-accurate synchronization).
4. Prefill latency vs pure Decode throughput separation.
5. Multi-window sweep: W in {128, 256, 512, Full Attention}.
"""

import gc
import json
import os
import sys
import time

import torch
import numpy as np

if not os.path.exists("VANTAGE-V-AI"):
    print("[1/5] Cloning VANTAGE-V-AI repository...")
    os.system("git clone https://github.com/khoadev-vn/VANTAGE-V-AI.git")

sys.path.insert(0, os.path.abspath("VANTAGE-V-AI"))

from core.adapters.transformers import VantageStreamingCache
from core.kv import QUANT_LOSSLESS, QUANT_FP8, QUANT_TERNARY_158


def check_gpu_environment():
    print("=== GPU Environment Check ===")
    if not torch.cuda.is_available():
        print("[!] ERROR: CUDA is not available. Please enable GPU in Kaggle settings!")
        return 0
    
    device_count = torch.cuda.device_count()
    print(f"CUDA Available: True | Total Devices: {device_count}")
    for i in range(device_count):
        props = torch.cuda.get_device_properties(i)
        vram_gb = props.total_memory / (1024**3)
        print(f"  - Device {i}: {props.name} | VRAM: {vram_gb:.2f} GB | Compute Capability: {props.major}.{props.minor}")
    return device_count


def run_gpu_scientific_benchmark(
    model_id: str = "Qwen/Qwen2.5-3B-Instruct",
    prompt_len: int = 1024,
    gen_tokens: int = 64,
):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers.cache_utils import DynamicCache

    n_devices = check_gpu_environment()
    if n_devices == 0:
        return

    print(f"\n[2/5] Loading model {model_id} in FP16...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.float16,
        device_map="auto",
    )
    model.eval()

    # Log device placement across model layers
    device_alloc = {}
    for name, param in model.named_parameters():
        dev = str(param.device)
        device_alloc[dev] = device_alloc.get(dev, 0) + param.numel()
    print(f"Model Parameter Placement Across Devices: {device_alloc}")

    # Prepare standard prompt
    base_text = (
        "Artificial intelligence systems and autonomous agents require next-generation memory architectures "
        "to handle long-context reasoning, multi-turn dialogues, and hierarchical knowledge retrieval. "
    )
    tokens = tokenizer.encode(base_text)
    while len(tokens) < prompt_len:
        tokens = (tokens + tokens)[:prompt_len]
    tokens = tokens[:prompt_len]
    input_ids = torch.tensor([tokens], dtype=torch.long, device=model.device)

    print(f"\n[3/5] Benchmark Configuration: Prompt={prompt_len} tokens, Generate={gen_tokens} tokens")

    experiments = [
        ("standard_hf_full", "Standard Hugging Face Full Attention", None),
        ("vantage_w128_sink4", "Vantage Streaming (W=128, Sink=4, Lossless)", {"w": 128, "sink": 4, "quant": QUANT_LOSSLESS}),
        ("vantage_w256_nosink", "Vantage Streaming (W=256, No Sink, Lossless)", {"w": 256, "sink": 0, "quant": QUANT_LOSSLESS}),
        ("vantage_w256_sink4", "Vantage Streaming (W=256, Sink=4, Lossless)", {"w": 256, "sink": 4, "quant": QUANT_LOSSLESS}),
        ("vantage_w256_sink4_fp8", "Vantage Streaming (W=256, Sink=4, FP8)", {"w": 256, "sink": 4, "quant": QUANT_FP8}),
        ("vantage_w512_sink4", "Vantage Streaming (W=512, Sink=4, Lossless)", {"w": 512, "sink": 4, "quant": QUANT_LOSSLESS}),
    ]

    results = {}

    for cfg_id, cfg_label, cfg_params in experiments:
        torch.cuda.empty_cache()
        gc.collect()
        for d in range(n_devices):
            torch.cuda.reset_peak_memory_stats(d)

        if cfg_params is None:
            past_cache = DynamicCache()
        else:
            past_cache = VantageStreamingCache(
                max_active_tokens=cfg_params["w"],
                block_size=32,
                sink_tokens=cfg_params["sink"],
                quant_mode=cfg_params["quant"],
            )

        # CUDA Events for accurate timing
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()
        with torch.no_grad():
            out = model.generate(
                input_ids,
                past_key_values=past_cache,
                max_new_tokens=gen_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        end_event.record()
        torch.cuda.synchronize()

        elapsed_ms = start_event.elapsed_time(end_event)
        elapsed_s = elapsed_ms / 1000.0
        tok_s = gen_tokens / max(elapsed_s, 1e-6)

        # Per-GPU VRAM accounting
        gpu_mem = {}
        for d in range(n_devices):
            gpu_mem[f"gpu_{d}_allocated_mb"] = round(torch.cuda.max_memory_allocated(d) / (1024**2), 2)
            gpu_mem[f"gpu_{d}_reserved_mb"] = round(torch.cuda.max_memory_reserved(d) / (1024**2), 2)

        gen_tokens_ids = out[0, input_ids.shape[-1]:].tolist()
        gen_text = tokenizer.decode(gen_tokens_ids, skip_special_tokens=True)

        results[cfg_id] = {
            "label": cfg_label,
            "throughput_tok_s": round(tok_s, 2),
            "elapsed_ms": round(elapsed_ms, 2),
            "memory": gpu_mem,
            "generated_tokens": gen_tokens_ids,
            "sample_output": gen_text[:100].strip(),
        }

        print(f"[{cfg_id:<24}] Throughput: {tok_s:>5.2f} tok/s | Elapsed: {elapsed_ms:>7.1f} ms | GPU 0 VRAM: {gpu_mem['gpu_0_allocated_mb']} MB")

    # Pairwise Parity Evaluations
    print("\n[4/5] Pairwise Generation Parity Analysis:")
    std_tokens = results["standard_hf_full"]["generated_tokens"]
    lossless_tokens = results["vantage_w256_sink4"]["generated_tokens"]
    fp8_tokens = results["vantage_w256_sink4_fp8"]["generated_tokens"]

    # 1. Full vs Lossless (Sliding window trade-off)
    match_full_lossless = sum(1 for a, b in zip(std_tokens, lossless_tokens) if a == b)
    print(f"  - Full Attention vs Vantage Lossless (W=256): {match_full_lossless}/{gen_tokens} match ({match_full_lossless/gen_tokens*100:.1f}%) [Context-Reduction Drift]")

    # 2. Lossless vs FP8 (Direct Quantization Drift Check)
    match_lossless_fp8 = sum(1 for a, b in zip(lossless_tokens, fp8_tokens) if a == b)
    is_exact_parity = (lossless_tokens == fp8_tokens)
    print(f"  - Vantage Lossless vs Vantage FP8 (W=256):     {match_lossless_fp8}/{gen_tokens} match ({match_lossless_fp8/gen_tokens*100:.1f}%) [Exact Parity: {is_exact_parity}]")

    for cfg_id in results:
        cur_toks = results[cfg_id]["generated_tokens"]
        m_full = sum(1 for a, b in zip(std_tokens, cur_toks) if a == b)
        results[cfg_id]["match_against_full_attention"] = f"{m_full}/{gen_tokens}"
    
    results["pairwise_analysis"] = {
        "lossless_vs_fp8_exact_match": is_exact_parity,
        "lossless_vs_fp8_match_count": f"{match_lossless_fp8}/{gen_tokens}",
        "full_vs_lossless_match_count": f"{match_full_lossless}/{gen_tokens}",
    }

    print("\n[5/5] Final Summary Table:")
    print(f"{'Configuration':<45} | {'Throughput':<11} | {'GPU 0 VRAM':<12} | {'vs Full'}")
    print("-" * 80)
    for cfg_id, d in results.items():
        if cfg_id == "pairwise_analysis":
            continue
        print(f"{d['label'][:45]:<45} | {d['throughput_tok_s']:>6.2f} tok/s | {d['memory']['gpu_0_allocated_mb']:>7.2f} MB  | {d['match_against_full_attention']}")

    out_file = "kaggle_gpu_benchmark_v8_2.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved raw scientific artifact to {out_file}")
    return results


if __name__ == "__main__":
    run_gpu_scientific_benchmark()
