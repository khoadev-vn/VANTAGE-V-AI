# Vantage V-AI: Stateless Streaming Data Processing Engine for Resource-Constrained AI Systems

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/Tests-11%20Passing-brightgreen.svg)](tests/test_engine.py)

**Vantage V-AI (v5.0)** is a research prototype for **stateless, bounded-memory streaming data processing and compression** in resource-constrained AI/edge pipelines.

> **Scientific Positioning & Correction Notice (v5.0):**
> Earlier v4.0 documentation reported `4.3B tokens/s` and `<2MB RAM` using a fixed `20 tokens/record` heuristic on repeated LZ4-compressed bytes and an unbounded `self.vault` list. In v5.0, those claims have been retracted and replaced with a strictly stateless architecture, bounded sinks, CRC32 + round-trip verification, multi-entropy datasets, fair baselines, and separate real BPE tokenization measurements.

---

## Architecture Overview

```text
Input Buffer (bytes / bytearray / memoryview)
   │
   ▼  memoryview zero-copy slicing (chunk_bytes = 256 KiB)
StatelessStreamProcessor  [No retained payload state; optional CRC32]
   │
   ▼  Pluggable Codec (lz4_frame | lz4_block | zstd | none)
OutputSink
   ├── NullSink              (streaming passthrough / throughput measurement)
   ├── BoundedMemorySink     (hard max_bytes cap with eviction or backpressure)
   ├── CallbackSink          (downstream consumer hook)
   └── CollectingSink        (round-trip verification & testing)
```

Key engineering properties in [`core/engine.py`](core/engine.py):
- **Strictly Stateless Execution:** [`StatelessStreamProcessor`](core/engine.py) holds configuration only—no `self.vault` accumulation across calls.
- **Bounded Memory:** [`BoundedMemorySink`](core/engine.py) enforces a hard `max_bytes` ceiling with configurable eviction or `MemoryError` backpressure.
- **Buffer-Protocol Slicing:** Slices `bytes`, `bytearray`, and `memoryview` inputs without intermediate Python-side input copies.
- **Verifiable Integrity:** Every run supports `zlib.crc32` checksums and full round-trip decompression via `verify_roundtrip()`.

---

## Verified Benchmark Results (v5.0)

Measured on **Windows 11 (AMD64), Intel64 Family 6 Model 165 Stepping 3 (6C / 12T), 15.9 GB RAM, Python 3.12.10** (`lz4 4.4.5`, `pyarrow 25.0.1`, `tiktoken 0.14.0`, `psutil 7.2.0`).  
Protocol: **8 MiB per dataset**, **1 warm-up + 7 independent trials**, **CRC32 + round-trip verified (`PASS`)**.  
Raw trial data: [`benchmarks/results/benchmark_results.csv`](benchmarks/results/benchmark_results.csv) · Full log: [`benchmarks/VANTAGE_VALIDATION.log`](benchmarks/VANTAGE_VALIDATION.log).

### 1. Streaming Compression & Memory Footprint (8 MiB Payload)

| Dataset | Baseline | P50 Throughput (MB/s) | P50 (ms) | P95 (ms) | P99 (ms) | Ratio | Py Heap Peak (MB) | Process Peak RSS (MB) | Verified |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: |
| `repeated` | `python_lz4_naive` | 1380.95 | 5.793 | 6.238 | 6.309 | 242.19x | 8.034 | 94.11 | PASS |
| `repeated` | `pyarrow_lz4` | 1337.72 | 5.980 | 6.239 | 6.253 | 242.25x | 8.001 | 95.12 | PASS |
| `repeated` | `vantage_lz4_frame_1t` | 1138.83 | 7.025 | 8.393 | 8.701 | 225.99x | **0.254** | 103.53 | PASS |
| `repeated` | `vantage_lz4_block_1t` | 1274.00 | 6.279 | 6.988 | 7.067 | 238.75x | **0.254** | 103.53 | PASS |
| `json_records` | `python_lz4_naive` | 775.83 | 10.312 | 12.742 | 12.801 | 6.95x | 9.153 | 104.80 | PASS |
| `json_records` | `pyarrow_lz4` | 854.47 | 9.363 | 9.612 | 9.658 | 6.95x | 8.001 | 104.80 | PASS |
| `json_records` | `vantage_lz4_block_1t` | **867.28** | **9.224** | **9.557** | 9.660 | 6.90x | **0.325** | 105.09 | PASS |
| `ai_corpus` | `python_lz4_naive` | 1020.19 | 7.842 | 8.012 | 8.036 | 13.24x | 8.605 | 105.55 | PASS |
| `ai_corpus` | `pyarrow_lz4` | 1077.14 | 7.427 | 7.792 | 7.842 | 13.24x | 8.001 | 105.55 | PASS |
| `ai_corpus` | `vantage_lz4_block_1t` | 1040.56 | 7.688 | 8.280 | 8.379 | 12.99x | **0.291** | 105.55 | PASS |
| `random` | `python_lz4_naive` | 598.60 | 13.365 | 16.470 | 17.406 | 1.00x | 16.001 | 119.69 | PASS |
| `random` | `pyarrow_lz4` | 783.49 | 10.211 | 10.650 | 10.670 | 1.00x | 8.001 | 119.69 | PASS |
| `random` | `vantage_lz4_block_1t` | **1188.14** | **6.733** | **7.001** | **7.008** | 1.00x | **0.754** | 119.69 | PASS |

### 2. Separate Real BPE Tokenization Baseline (`tiktoken cl100k_base`)
Measured on a **1 MiB slice** of `ai_corpus` (`1,048,576` bytes → `251,940` actual tokens):
- **Median Latency:** `87.03 ms`
- **Actual Tokenization Speed:** `2,894,804.0 tokens/s` (`11.49 MB/s`)
- **Key Takeaway:** BPE tokenization is **~80x–110x slower** than LZ4 byte-stream processing, confirming that byte compression cannot be equated with LLM token throughput.

---

## Limitations & Non-Claims

1. **Not an LLM Inference Engine:** Vantage optimizes byte-stream transport, chunking, and compression—not transformer attention or token generation.
2. **Python Interpreter RSS:** While Python heap allocation during streaming is `0.25–0.75 MB`, full-process Peak RSS is `94–120 MB` due to the Python runtime, loaded libraries, and pre-generated 8 MiB datasets.
3. **Thread-Pool Overhead at 8 MiB:** `ParallelStreamProcessor` (`632–847 MB/s`) is slower than single-threaded `StatelessStreamProcessor` at 8 MiB because `ThreadPoolExecutor` dispatch overhead exceeds the ~7–10 ms compute time.
4. **Entropy Sensitivity:** Compression ratios drop from `238x` (`repeated`) to `6.9x` (`json_records`) and `1.00x` (`random`).

---

## How to Reproduce

```bash
git clone https://github.com/khoadev-vn/VANTAGE-V-AI.git
cd VANTAGE-V-AI
pip install -r requirements.txt

# Run unit tests
pytest tests/test_engine.py -v

# Run full scientific benchmark suite
python benchmarks/vantage_validation.py
```

For full methodology and answers to systems-review questions, see [`docs/TECHNICAL_REPORT.md`](docs/TECHNICAL_REPORT.md).
