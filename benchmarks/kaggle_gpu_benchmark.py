"""
Kaggle GPU Benchmark Script for VANTAGE-V-AI (2x NVIDIA T4 GPUs).

Instructions for Kaggle:
1. Open a new Kaggle Notebook: https://www.kaggle.com/code
2. In the right sidebar:
   - Accelerator: Select 'GPU T4 x2'
   - Internet: Turn 'On'
3. Run the notebook cell to benchmark VANTAGE vs Standard HF on real GPU VRAM!
"""

import gc
import json
import os
import sys
import time

import torch
import numpy as np

# Ensure repository is cloned or in python path
if not os.path.exists("VANTAGE-V-AI"):
    print("[1/4] Cloning VANTAGE-V-AI repository...")
    os.system("git clone https://github.com/khoadev-vn/VANTAGE-V-AI.git")

sys.path.insert(0, os.path.abspath("VANTAGE-V-AI"))

from core.adapters.transformers import VantageStreamingCache, checkpoint_prompt_cache, restore_prompt_cache
from core.kv import QUANT_LOSSLESS, QUANT_FP8, QUANT_TERNARY_158

def check_gpu_environment():
    print("=== GPU Environment Check ===")
    cuda_available = torch.cuda.is_available()
    print(f"CUDA Available: {cuda_available}")
    if not cuda_available:
        print("[!] WARNING: CUDA is not available. Please enable GPU accelerator in Kaggle sidebar!")
        return None
    
    device_count = torch.cuda.device_count()
    print(f"GPU Device Count: {device_count}")
    for i in range(device_count):
        props = torch.cuda.get_device_properties(i)
        vram_gb = props.total_memory / (1024**3)
        print(f"  - GPU {i}: {props.name} | VRAM: {vram_gb:.2f} GB | Compute Capability: {props.major}.{props.minor}")
    return device_count

def run_gpu_benchmark(model_id: str = "Qwen/Qwen2.5-3B-Instruct", prompt_len: int = 1024, gen_tokens: int = 64):
    """
    Benchmarks GPU VRAM allocation, throughput, and generation stability.
    Uses Qwen2.5-3B-Instruct or Qwen2.5-7B-Instruct (publicly accessible without gated token).
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers.cache_utils import DynamicCache

    print(f"\n[2/4] Loading model: {model_id} on GPU...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    # Load model with automatic multi-GPU placement (supports 2x T4 seamlessly)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model.eval()

    # Generate synthetic prompt of exact length
    sample_text = (
        "Artificial intelligence systems and autonomous agents require next-generation memory architectures "
        "to handle long-context reasoning, multi-turn dialogues, and hierarchical knowledge retrieval. "
    )
    tokens = tokenizer.encode(sample_text)
    while len(tokens) < prompt_len:
        tokens = (tokens + tokens)[:prompt_len]
    tokens = tokens[:prompt_len]
    input_ids = torch.tensor([tokens], dtype=torch.long, device=model.device)

    print(f"[3/4] Prepared Prompt: {prompt_len} tokens. Generating {gen_tokens} tokens...")

    configs = [
        ("standard_hf", "Standard Hugging Face DynamicCache (Full Attention)", None),
        ("vantage_sliding_nosink", "Vantage Sliding Window (No Sinks, W=256)", {"sink_tokens": 0, "max_active": 256, "quant": QUANT_LOSSLESS}),
        ("vantage_sota_sink4", "Vantage Bounded Streaming (Sink=4, W=256, Lossless)", {"sink_tokens": 4, "max_active": 256, "quant": QUANT_LOSSLESS}),
        ("vantage_sota_sink4_fp8", "Vantage Bounded Streaming (Sink=4, W=256, FP8)", {"sink_tokens": 4, "max_active": 256, "quant": QUANT_FP8}),
    ]

    results = {}

    for cfg_id, cfg_label, cfg_params in configs:
        torch.cuda.empty_cache()
        gc.collect()
        torch.cuda.reset_peak_memory_stats()

        if cfg_params is None:
            past_cache = DynamicCache()
        else:
            past_cache = VantageStreamingCache(
                max_active_tokens=cfg_params["max_active"],
                block_size=32,
                sink_tokens=cfg_params["sink_tokens"],
                quant_mode=cfg_params["quant"],
            )

        # Warmup / synchronize CUDA
        torch.cuda.synchronize()
        t0 = time.perf_counter()

        with torch.no_grad():
            out = model.generate(
                input_ids,
                past_key_values=past_cache,
                max_new_tokens=gen_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0

        peak_vram_mb = torch.cuda.max_memory_allocated() / (1024**2)
        reserved_vram_mb = torch.cuda.max_memory_reserved() / (1024**2)
        throughput = gen_tokens / elapsed

        gen_tokens_ids = out[0, input_ids.shape[-1]:].tolist()
        gen_text = tokenizer.decode(gen_tokens_ids, skip_special_tokens=True)

        results[cfg_id] = {
            "label": cfg_label,
            "throughput_tok_s": round(throughput, 2),
            "peak_vram_allocated_mb": round(peak_vram_mb, 2),
            "peak_vram_reserved_mb": round(reserved_vram_mb, 2),
            "sample_output": gen_text[:120].strip(),
            "generated_tokens": gen_tokens_ids,
        }

        print(f"\n--- [{cfg_id}] ---")
        print(f"  Throughput: {throughput:.2f} tok/s")
        print(f"  Peak VRAM Allocated: {peak_vram_mb:.2f} MB")
        print(f"  Peak VRAM Reserved: {reserved_vram_mb:.2f} MB")
        print(f"  Output: {gen_text[:80]}...")

    # Calculate token overlap against baseline
    std_toks = results["standard_hf"]["generated_tokens"]
    for cfg_id in results:
        cur_toks = results[cfg_id]["generated_tokens"]
        matches = sum(1 for a, b in zip(std_toks, cur_toks) if a == b)
        results[cfg_id]["token_match_ratio"] = f"{matches}/{len(std_toks)}"

    print("\n[4/4] Summary Table:")
    print(f"{'Configuration':<45} | {'Throughput':<12} | {'VRAM Allocated':<15} | {'Match Ratio'}")
    print("-" * 85)
    for cfg_id, data in results.items():
        print(f"{data['label'][:45]:<45} | {data['throughput_tok_s']:>6.2f} tok/s | {data['peak_vram_allocated_mb']:>9.2f} MB | {data.get('token_match_ratio', 'N/A')}")

    out_file = "kaggle_gpu_benchmark_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved raw GPU benchmark artifact to {out_file}")
    return results

if __name__ == "__main__":
    if check_gpu_environment():
        # Start with Qwen2.5-3B-Instruct (fast download, high quality)
        # Or change to 'Qwen/Qwen2.5-7B-Instruct' for full 7B test
        run_gpu_benchmark(model_id="Qwen/Qwen2.5-3B-Instruct", prompt_len=1024, gen_tokens=64)
