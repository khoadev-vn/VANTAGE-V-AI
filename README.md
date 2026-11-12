# Vantage V-AI: Stateless Streaming Data Processing Engine for Resource-Constrained AI Systems

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/Tests-11%20Passing-brightgreen.svg)](tests/test_engine.py)

**Vantage V-AI (v6.0)** is a **stateless, bounded-memory incremental streaming engine** for resource-constrained AI/edge data pipelines.

> **Scientific Positioning:**
> Vantage V-AI is a **data processing and streaming compression layer**, not an LLM inference runtime. Earlier v4.0 claims (`4.3B tokens/s`, `<2MB RAM`) were retracted. Version 6.0 introduces **true incremental streaming (`process_iter`)**, **`FileSink`**, **adaptive chunk sizing (`AdaptiveStreamProcessor`)**, and **isolated-subprocess verification** of the `256 KiB` vs `512 KiB` throughput cliff.

---

## Architecture Overview

```text
Input Buffer (bytes / bytearray / memoryview)
   │
   ▼  memoryview zero-copy slicing (chunk_bytes = 256 KiB, L2-cache-local)
StatelessStreamProcessor  [No retained payload state; optional CRC32]
   │
   ▼  Pluggable Codec (lz4_frame | lz4_block | zstd | none)
OutputSink
   ├── NullSink              (streaming passthrough / throughput measurement)
   ├── BoundedMemorySink     (hard max_bytes cap with eviction or backpressure)
   ├── CallbackSink          (downstream consumer hook)
   └── CollectingSink        (round-trip verification & testing)
```

---

## Verified Benchmark Results (v5.1 — 30 Trials, Codec Parity)

Measured on **Windows 11 (AMD64), Intel64 Family 6 Model 165 Stepping 3 (6C / 12T), 15.9 GB RAM, Python 3.12.10** (`lz4 4.4.5`, `pyarrow 25.0.1`, `tiktoken 0.14.0`, `psutil 7.2.0`).  
Protocol: **8 MiB per dataset**, **2 warm-ups + 30 independent trials**, **CRC32 + round-trip verified (`PASS`)**.  
Raw trial data: [`benchmarks/results/benchmark_results.csv`](benchmarks/results/benchmark_results.csv) · Full log: [`benchmarks/VANTAGE_VALIDATION.log`](benchmarks/VANTAGE_VALIDATION.log).

### 1. Codec-Parity Streaming Compression (8 MiB Payload, 30 Trials)

- `vantage_lz4_block_raw` uses `lz4.block.compress(..., store_size=True)` with `256 KiB` `memoryview` slicing and **no** CRC32 (exact codec parity with `python_lz4_block_naive`).
- `vantage_lz4_block_1t` adds per-chunk `zlib.crc32` integrity verification on top.

| Dataset | Baseline | P50 MB/s | P50 (ms) | IQR (ms) | Std (ms) | P95 (ms) | P99 (ms) | Ratio | PyPeak (MB) |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `repeated` | `python_lz4_frame_naive` | 1289.89 | 6.202 | 0.563 | 1.186 | 8.122 | 10.826 | 242.19x | 8.034 |
| `repeated` | `python_lz4_block_naive` | 1328.68 | 6.021 | 0.697 | 0.698 | 7.240 | 8.466 | 254.45x | 8.063 |
| `repeated` | `pyarrow_lz4` | 1343.97 | 5.952 | 0.605 | 0.838 | 6.946 | 9.277 | 242.25x | 8.001 |
| `repeated` | `vantage_lz4_frame_1t` | 1069.93 | 7.477 | 2.650 | 1.702 | 11.280 | 11.755 | 225.99x | **0.253** |
| `repeated` | `vantage_lz4_block_1t` | 1238.89 | 6.457 | 1.478 | 2.574 | 12.188 | 16.325 | 238.75x | **0.254** |
| `repeated` | `vantage_lz4_block_raw` | **6180.45** | **1.294** | 0.517 | 0.777 | 3.487 | 3.761 | 238.75x | **0.254** |
| `json_records` | `python_lz4_frame_naive` | 725.58 | 11.026 | 3.695 | 3.011 | 17.527 | 20.303 | 6.95x | 9.153 |
| `json_records` | `python_lz4_block_naive` | 838.95 | 9.536 | 0.344 | 0.268 | 10.096 | 10.383 | 6.95x | 9.183 |
| `json_records` | `pyarrow_lz4` | 875.91 | 9.134 | 0.231 | 0.837 | 11.583 | 12.093 | 6.95x | 8.001 |
| `json_records` | `vantage_lz4_frame_1t` | 780.77 | 10.246 | 0.052 | 0.121 | 10.496 | 10.701 | 6.90x | **0.324** |
| `json_records` | `vantage_lz4_block_1t` | 867.65 | 9.220 | 0.262 | 0.358 | 9.624 | 10.623 | 6.90x | **0.325** |
| `json_records` | `vantage_lz4_block_raw` | **1761.33** | **4.542** | 0.100 | 0.173 | 4.919 | 5.223 | 6.90x | **0.325** |
| `ai_corpus` | `python_lz4_frame_naive` | 1088.63 | 7.349 | 0.099 | 0.095 | 7.577 | 7.664 | 13.24x | 8.605 |
| `ai_corpus` | `python_lz4_block_naive` | 1083.48 | 7.384 | 0.181 | 0.261 | 8.035 | 8.276 | 13.26x | 8.635 |
| `ai_corpus` | `pyarrow_lz4` | 1087.28 | 7.358 | 0.128 | 0.263 | 8.067 | 8.258 | 13.24x | 8.001 |
| `ai_corpus` | `vantage_lz4_frame_1t` | 900.65 | 8.883 | 0.888 | 0.961 | 11.629 | 12.051 | 12.97x | **0.290** |
| `ai_corpus` | `vantage_lz4_block_1t` | 1045.35 | 7.653 | 0.395 | 0.222 | 8.033 | 8.084 | 12.99x | **0.291** |
| `ai_corpus` | `vantage_lz4_block_raw` | **2777.60** | **2.880** | 0.159 | 0.117 | 3.162 | 3.254 | 12.99x | **0.291** |
| `random` | `python_lz4_frame_naive` | 683.08 | 11.712 | 1.714 | 3.026 | 18.342 | 23.699 | 1.00x | 16.001 |
| `random` | `python_lz4_block_naive` | 728.12 | 10.987 | 2.323 | 1.723 | 14.697 | 15.944 | 1.00x | 16.063 |
| `random` | `pyarrow_lz4` | 842.38 | 9.497 | 1.849 | 2.412 | 14.481 | 18.974 | 1.00x | 8.001 |
| `random` | `vantage_lz4_frame_1t` | 840.57 | 9.517 | 1.170 | 8.250 | 35.657 | 37.121 | 1.00x | **0.751** |
| `random` | `vantage_lz4_block_1t` | 1185.76 | 6.747 | 0.650 | 0.699 | 8.432 | 8.816 | 1.00x | **0.754** |
| `random` | `vantage_lz4_block_raw` | **2601.75** | **3.075** | 1.189 | 0.904 | 4.864 | 5.116 | 1.00x | **0.754** |

**Root-Cause Validation (v5.2, [`benchmarks/v52_root_cause.py`](benchmarks/v52_root_cause.py)):**
1. **Chunking Parity:** Plain Python `memoryview` + `256 KiB` `lz4.block.compress` (`python_lz4_block_chunked`) matches `vantage_lz4_block_raw` (`1494–7754 MB/s` vs `1636–6816 MB/s`, identical `0.254–0.754 MB` `PyPeak`). Thus, the throughput and heap-allocation gains come from the **chunking strategy**, while Vantage's contribution is packaging it into a stateless, bounded-sink streaming architecture.
2. **Chunk-Size Sweep (`64 KiB`–`8 MiB`):** On `random`, throughput peaks at **`256 KiB` (`3915.6 MB/s`)** and drops sharply at **`512 KiB` (`917.9 MB/s`)**. Slicing data into `256 KiB` chunks via `memoryview` reduces the processing unit size and limits concurrent transient allocations; cache-locality effects require hardware counter profiling to confirm.
3. **CRC32 Cost Isolation:** `CRC32` introduces substantial additional processing overhead, accounting for approximately **48–81%** of the combined execution time under the tested workloads (`crc32_only` `1356–1658 MB/s` vs `compress_plus_crc32` `735–1327 MB/s`). This is an empirical comparison, not an isolated attribution of CPU cost.

### 2. v6.0 Production Core Validation ([`benchmarks/v60_validation.py`](benchmarks/v60_validation.py))
- **Isolated `256 KiB` → `512 KiB` Cliff Re-Verification (`random`, 3 seeds, randomized order, 10 trials):** Confirmed reproducible across all seeds (`256 KiB`: `1874.81–4537.08 MB/s` vs `512 KiB`: `740.79–1240.68 MB/s` vs `1024 KiB`: `771.30–1140.62 MB/s`).
- **62.16 MiB Incremental Streaming (`process_iter`, `256 KiB` chunks):**
  - `NullSink`: **`3120.14 MB/s`**, **`0.747 MB` Python heap peak** (`0 B` retained).
  - `BoundedMemorySink (2 MiB cap)`: **`3188.01 MB/s`**, **`0.993 MB` Python heap peak** (`268,543 B` retained).
- **`AdaptiveStreamProcessor` vs Fixed Chunk Size (8 MiB, 15 trials):**
  - `repeated`: `fixed_256k` `10207.99 MB/s`, `fixed_512k` `11936.74 MB/s`, **`adaptive` `11778.56 MB/s`** (automatically selects `512 KiB`).
  - `random`: `fixed_256k` `5909.29 MB/s`, `fixed_512k` `1946.90 MB/s`, **`adaptive` `5683.43 MB/s`** (automatically selects `256 KiB`).

### 2. Isolated Subprocess Full-Lifecycle Peak RSS (`8 MiB random`)

Measured in a fresh Python process per baseline ([`benchmarks/isolated_memory_bench.py`](benchmarks/isolated_memory_bench.py)):
- `RSS_Start`: `~21.2–21.4 MB` · `RSS_Input` (after 8 MiB dataset): `~30.1–30.4 MB`
- `Peak_RSS`: `64.14–64.98 MB` across all baselines (`~34.1–34.9 MB` processing/runtime working set).

### 3. Separate Real BPE Tokenization Baseline (`tiktoken cl100k_base`)
Measured on a **1 MiB slice** of `ai_corpus` (`1,048,576` bytes → `251,940` actual tokens):
- **Median Time:** `88.24 ms` → **`2,855,145.1 tokens/s`** (`11.33 MB/s`)
- *Note:* This is a standalone tokenizer baseline—not LLM inference, not token generation, and not accelerated by Vantage.

---

## Limitations & Next Steps

1. **No End-to-End LLM Runtime Integration Yet:** Following guidance from *MLSysBook Vol. I*, proving AI-system impact requires integrating a real quantized edge runtime (e.g., `llama.cpp` / ONNX Runtime) to measure TTFT, end-to-end tokens/s, and total working memory.
2. **Thread-Pool Overhead at 8 MiB:** `ParallelStreamProcessor` was slower than single-threaded execution on 8 MiB payloads due to `ThreadPoolExecutor` dispatch overhead.
3. **Cross-Machine Validation:** Results above are from a single Intel 6C/12T Windows 11 machine; Linux/ARM edge validation is pending.

---

## How to Reproduce

```bash
git clone https://github.com/khoadev-vn/VANTAGE-V-AI.git
cd VANTAGE-V-AI
pip install -r requirements.txt

pytest tests/test_engine.py -v
python benchmarks/vantage_validation.py
```

For full methodology and answers to systems-review questions, see [`docs/TECHNICAL_REPORT.md`](docs/TECHNICAL_REPORT.md).
