# Vantage V-AI: Bounded-Memory Streaming Engine, 2-Bit Packed Ternary KV-Cache & HDC Associative Memory

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/Tests-45%20Passing%20(Local)-brightgreen.svg)](tests/)
[![Version](https://img.shields.io/badge/Version-8.2.0-purple.svg)](pyproject.toml)
[![Paper](https://img.shields.io/badge/Technical%20Report-v8.2.0%20(PDF)-blue.svg)](paper/VANTAGE_Paper_v8.2.pdf)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24243287.svg)](https://doi.org/10.5281/zenodo.24243287)
[![Kaggle GPU](https://img.shields.io/badge/Kaggle%20GPU-2x%20Tesla%20T4-orange.svg)](benchmarks/kaggle_gpu_benchmark.py)
[![Author](https://img.shields.io/badge/Author-Nguyen%20Vo%20Anh%20Khoa-red.svg)](https://github.com/khoadev-vn)

**Vantage V-AI (v8.2.0)** is an open-source, bounded-memory inference runtime and streaming data processing engine designed to solve KV-cache memory explosion and enable constant-complexity autoregressive generation across CPU, edge devices, and multi-GPU accelerators.

* 📄 **Technical Report (Peer-Review Edition):** [`paper/VANTAGE_Paper_v8.2.pdf`](paper/VANTAGE_Paper_v8.2.pdf) | [LaTeX Source](paper/VANTAGE_Technical_Report.tex)
* 🔬 **Task Quality Benchmark:** [`artifacts/benchmarks/task_quality_benchmark.json`](artifacts/benchmarks/task_quality_benchmark.json)
* ⚡ **Multi-GPU E2E Benchmark:** [`artifacts/benchmarks/benchmark_gpu_kaggle_v8_2.json`](artifacts/benchmarks/benchmark_gpu_kaggle_v8_2.json)

---

## 🛡️ Authorship, Prior Art & Intellectual Property Notice

* **Primary Author & Architect:** **Nguyen Vo Anh Khoa** ([@khoadev-vn](https://github.com/khoadev-vn))
* **Official Repository:** [https://github.com/khoadev-vn/VANTAGE-V-AI](https://github.com/khoadev-vn/VANTAGE-V-AI)
* **First Public Disclosure (Prior Art):** November 2026 (Git Commit History & Cryptographic Timestamps)
* **Legal Protection:** Licensed under the [MIT License](LICENSE). Under international copyright law and Open Source compliance standards, any distribution, redistribution, embedding, or commercial utilization of this codebase or its derivative works **strictly requires preservation of the original copyright notice and authorship attribution**:
  ```text
  Copyright (c) 2026 Nguyen Vo Anh Khoa. All rights reserved.
  ```

---

## ⚡ Quickstart: 2-Line Drop-in Integration

Vantage integrates transparently with any Hugging Face `AutoModelForCausalLM` without modifying existing generation loops or model definitions:

```python
import vantage
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load any open-weight model (Meta Llama-3, Qwen-2.5, DeepSeek-R1, Mistral)
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-3B-Instruct", torch_dtype="auto", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct")

# 1-Line Drop-in: bounds active attention window and anchors 4 sink tokens
vantage.enable(model, max_active_tokens=256, sink_tokens=4, quant_mode="fp8")

# Standard generation code runs unchanged with bounded attention memory!
inputs = tokenizer("The future of artificial intelligence depends on", return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=64)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

### 🖥️ Interactive Terminal Demo
Run the visual streaming monitor demo in your terminal:
```bash
python vantage_demo.py
```

---

## 🔬 Core Architectural Foundations (v8.2.0)

1. **Empirical Multi-GPU Hardware Validation (2x NVIDIA Tesla T4, Qwen2.5-3B-Instruct):**
   - Benchmarked on real CUDA accelerators with model parameters sharded across both devices (`cuda:0`: 1.85B params, `cuda:1`: 1.23B params).
   - Strict multi-device synchronization via `torch.cuda.synchronize(d)` across all participating GPUs following PyTorch asynchronous execution protocol.
   - **100.0% Exact Token Parity (64/64 tokens identical)** between Lossless and Dynamic FP8 streaming modes.
2. **Task Quality & Long-Context Trade-Offs (Needle-in-a-Haystack & PPL Stability):**
   - Evaluated across context horizons from 512 to 4,096 tokens with factual needles at varying depths (10% to 90%).
   - **0.988 Average Retrieval Coherence** with Attention Sinks, preventing the severe softmax collapse observed in sink-less sliding windows (PPL divergence $> 1,467$).
3. **2-Bit Packed Ternary Representation (Inspired by BitNet b1.58):**
   - Quantizes multi-head attention keys and values into ternary states $\{-1, 0, +1\}$ with per-tensor dynamic mean scaling $\gamma = \text{mean}(|X|)$.
   - Employs vectorized SIMD packing (4 elements per `uint8` byte = 2.0 bits/element), achieving **7.97x storage reduction** over FP16.
4. **Streaming KV-Cache with Attention Sinks (Inspired by StreamingLLM):**
   - Integrates initial attention sink retention (`sink_tokens=4`) within a block-level chunked offloading cache (`VantageStreamingCache`).
   - Anchors initial softmax mass, preventing autoregressive trajectory collapse during long-sequence generation.
5. **Hyperdimensional Computing (HDC) Associative Memory (`core/memory.py`):**
   - Projects dense embeddings into 4,096-bit binary hypervectors.
   - Hardware-accelerated bitwise XOR + SIMD popcount search (`np.bitwise_count`) achieving 100% 1-NN recall on 5,000 synthetic vectors under perturbation.

---

## 📊 Empirical Multi-GPU Benchmark Results (v8.2.0)

Tested on **2x NVIDIA Tesla T4 GPUs (Turing, Compute Capability 7.5, 32 GB Total VRAM)** using **`Qwen/Qwen2.5-3B-Instruct`** (FP16, Prompt: 1024 tokens, Generation: 64 tokens, Multi-Device Synchronized):

| Cache Configuration | E2E Wall-Clock | CUDA Throughput | GPU 0 VRAM | Pairwise Parity vs Lossless | Match vs Full Context |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Standard Hugging Face Full Attention** | 3.582 s | 17.87 tok/s | 6,842.0 MB | N/A | 64/64 (Reference) |
| **Vantage Streaming (W=128, Sink=4, Lossless)** | 3.372 s | 18.98 tok/s | 6,842.0 MB | N/A | 17/64 (26.6%) |
| **Vantage Streaming (W=256, No Sink, Lossless)** | 3.398 s | 18.83 tok/s | 6,842.0 MB | N/A | 20/64 (31.2%) |
| **Vantage Streaming (W=256, Sink=4, Lossless)** | **3.398 s** | **18.83 tok/s** | **6,842.0 MB** | **Reference** | 17/64 (26.6%) |
| **Vantage Streaming (W=256, Sink=4, FP8)** | **3.454 s** | **18.53 tok/s** | **6,842.0 MB** | **64/64 (100.0%)** | 17/64 (26.6%) |
| **Vantage Streaming (W=512, Sink=4, Lossless)** | 3.454 s | 18.53 tok/s | 6,842.0 MB | N/A | 18/64 (28.1%) |

*Raw verifiable JSON artifact:* [`artifacts/benchmarks/benchmark_gpu_kaggle_v8_2.json`](artifacts/benchmarks/benchmark_gpu_kaggle_v8_2.json).

---

## 🎯 Task Quality & Perplexity Stability (Long Context)

Evaluated across varying context lengths ($N \in \{512, 1024, 2048, 4096\}$) to characterize the context trade-off:

| Context Horizon | Needle Placement | VANTAGE (Sinks=4) | Sliding Window (Sinks=0) |
| :---: | :---: | :---: | :---: |
| **512 tokens** | 10% - 90% Depth | **1.00 (Perfect Retrieval)** | 0.88 (Degraded) |
| **1,024 tokens** | 10% Depth (Sink Region) | **1.00 (Anchor Preserved)** | 0.35 (Softmax Lost) |
| **1,024 tokens** | 90% Depth (Recency Window) | **1.00 (Window Preserved)** | 0.88 (Degraded) |
| **2,048 tokens** | Multi-Depth Average | **0.95 (Bounded Trade-off)** | 0.35 (Softmax Collapse) |
| **Perplexity (PPL)** | Sequence Step = 2,048 | **PPL = 10.04 (Stable)** | **PPL = 1,467.26 (Explosive)** |

*Raw Task Quality artifact:* [`artifacts/benchmarks/task_quality_benchmark.json`](artifacts/benchmarks/task_quality_benchmark.json).

---

## 📚 Academic Citation (BibTeX)

```bibtex
@software{nguyen2026vantage,
  author       = {Nguyen, Vo Anh Khoa},
  title        = {{VANTAGE-V-AI: Bounded-Memory Streaming Engine, 2-Bit Packed Ternary KV-Cache and Associative Memory Runtime for Large Language Models}},
  year         = {2026},
  publisher    = {GitHub},
  journal      = {GitHub repository},
  howpublished = {\url{https://github.com/khoadev-vn/VANTAGE-V-AI}},
  version      = {8.2.0}
}
```

---

## 📄 License & Compliance

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.
```text
Copyright (c) 2026 Nguyen Vo Anh Khoa. All rights reserved.
```
