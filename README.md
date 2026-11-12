# Vantage V-AI: A Bounded-Memory Streaming Engine for Predictable Resource Usage

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/Tests-17%20Passing-brightgreen.svg)](tests/test_engine.py)

**Vantage V-AI (v6.3.0)** is a **bounded-memory streaming engine designed for predictable resource usage under variable workloads**.

> **Scientific Positioning:**
> Vantage V-AI is **not** a new compression algorithm and **not** an LLM inference runtime. Its contribution is a **bounded-memory streaming architecture** (`process_iter`, `BoundedMemorySink`, `FileSink` with partial-write safety and length-prefixed framing, `decompress_stream`) that keeps OS `Peak RSS` flat across `16 GiB` streams (`~23.8–27.7 MB`) and reduces End-to-End AI tokenization `Peak RSS` by **8.2x** (`65.16 MB` vs `534.28 MB`).

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

### 3. v6.1 Large-Scale Bounded Memory & Heterogeneous Adaptive Validation ([`benchmarks/v61_large_scale.py`](benchmarks/v61_large_scale.py))

#### A. Isolated Subprocess Bounded-Memory Scaling (`64 MiB → 4 GiB`, `BoundedMemorySink` 2 MiB cap)
Each scale runs in a fresh Python process streaming a pre-allocated `256 KiB` unit via `process_iter` (isolating engine + sink memory from dataset storage):

| Stream Size | Throughput (MB/s) | Time (s) | OS Peak RSS (MB) | PyPeak (MB) | Retained in Sink (B) |
| :--- | ---: | ---: | ---: | ---: | ---: |
| `64 MiB` | 2739.46 | 0.023 | **22.52** | 0.777 | 277,760 |
| `256 MiB` | 2602.41 | 0.098 | **23.61** | 1.602 | 1,111,040 |
| `1 GiB` | 3834.27 | 0.267 | **24.86** | 2.579 | 2,096,220 |
| `4 GiB` | 3916.45 | 1.046 | **24.76** | 2.579 | 2,096,220 |

A **64x increase in stream volume** (`64 MiB` → `4 GiB`) keeps OS `Peak RSS` constant at **`22.52–24.86 MB`** and `PyPeak` capped at **`2.579 MB`**.

#### B. Dynamic Adaptive vs Fixed Chunk Sizes on Heterogeneous Entropy Shifts (`16 MiB`, 10 trials, MB/s P50)
| Dataset | `64K` | `128K` | `256K` | `512K` | `1024K` | `Adaptive (Dyn)` |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| `repeated` | 9063.6 | 10078.1 | 9487.4 | 7770.4 | 4383.1 | 7547.5 |
| `random` | 4092.2 | 5402.0 | 5163.8 | 1360.9 | 1434.2 | **5214.9** |
| `low_then_high` (`64K` comp → `16M` rand) | 5153.6 | 5487.0 | 5023.6 | 1371.9 | 1436.5 | **4702.3** |
| `high_then_low` (`64K` rand → `16M` comp) | 9030.1 | 9741.5 | 10663.8 | 11301.8 | 9923.7 | **11221.4** |
| `alternating` (`1M` comp / `1M` rand) | 6594.7 | 7027.7 | 3992.9 | 2271.4 | 2331.3 | 3890.4 |

#### C. Sink Backpressure (`8 MiB random`)
- `NullSink`: `1.85 ms` (`4332.29 MB/s`)
- `BoundedMemorySink (2 MiB)`: `1.80 ms` (`4453.35 MB/s`)
- `SlowCallbackSink (200 µs/frame)`: `23.50 ms` (`340.41 MB/s`)

### 2. Isolated Subprocess Full-Lifecycle Peak RSS (`8 MiB random`)

Measured in a fresh Python process per baseline ([`benchmarks/isolated_memory_bench.py`](benchmarks/isolated_memory_bench.py)):
- `RSS_Start`: `~21.2–21.4 MB` · `RSS_Input` (after 8 MiB dataset): `~30.1–30.4 MB`
- `Peak_RSS`: `64.14–64.98 MB` across all baselines (`~34.1–34.9 MB` processing/runtime working set).

### 3. Separate Real BPE Tokenization Baseline (`tiktoken cl100k_base`)
Measured on a **1 MiB slice** of `ai_corpus` (`1,048,576` bytes → `251,940` actual tokens):
- **Median Time:** `88.24 ms` → **`2,855,145.1 tokens/s`** (`11.33 MB/s`)
- *Note:* This is a standalone tokenizer baseline—not LLM inference, not token generation, and not accelerated by Vantage.

### 4. v6.2 Stress Testing (`4 GiB → 16 GiB`), Backpressure RSS & Adaptive Ablation ([`benchmarks/v62_stress_and_ablation.py`](benchmarks/v62_stress_and_ablation.py))

#### A. Extreme Bounded-Memory Stress Test (`4 GiB → 16 GiB`, Isolated Subprocesses)
| Stream Size | Sink | Throughput (MB/s) | Time (s) | OS Peak RSS (MB) | PyPeak (MB) | Retained (B) |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: |
| `4 GiB` | `BoundedMemorySink (2 MiB)` | 4079.00 | 1.004 | **24.96** | 2.579 | 2,096,576 |
| `8 GiB` | `BoundedMemorySink (2 MiB)` | 3857.59 | 2.124 | **25.13** | 2.579 | 2,096,576 |
| `16 GiB` | `BoundedMemorySink (2 MiB)` | 4095.34 | 4.001 | **24.82** | 2.579 | 2,096,576 |
| `16 GiB` | `NullSink` | **4828.67** | **3.393** | **22.18** | **0.504** | 0 |

#### B. Backpressure + Bounded Memory Combined Proof (`32 MiB random`, Isolated Subprocesses)
- `NullSink`: `795.68 MB/s` (`0.040 s`), `Peak RSS = 23.32 MB`, `PyPeak = 1.004 MB`
- `BoundedMemorySink (2 MiB)`: `775.44 MB/s` (`0.041 s`), `Peak RSS = 24.50 MB`, `PyPeak = 2.511 MB`
- `SlowCallbackSink (100 µs/frame)`: `296.99 MB/s` (`0.108 s`), **`Peak RSS = 22.90 MB`**, **`PyPeak = 1.004 MB`** (throttled producer, zero RAM growth)

#### C. Adaptive Hysteresis Ablation & Adversarial Jitter (`16 MiB`, 10 trials, MB/s P50)
| Dataset | `fixed_128k` | `fixed_256k` | `fixed_512k` | `adapt_h1` | `adapt_h3` |
| :--- | ---: | ---: | ---: | ---: | ---: |
| `repeated` | 7828.4 | 8524.0 | 9806.0 | 8746.5 | **10862.2** |
| `random` | **5285.8** | 4681.4 | 1769.2 | 4863.1 | 5175.1 |
| `low_then_high` | 4983.3 | **5606.0** | 1963.6 | 4153.9 | 4135.0 |
| `high_then_low` | 9513.6 | 10070.5 | **11157.2** | 10179.4 | 9470.5 |
| `alternating` | **6887.5** | 6737.1 | 2435.1 | 3034.1 | 2363.9 |
| `adversarial_jitter` | 5594.4 | **6208.1** | 1908.6 | 2029.5 | 1944.1 |

> **Design Recommendation:** Because rapid entropy shifts (`alternating`, `adversarial_jitter`) penalize any `512 KiB` chunk that straddles incompressible bytes, `StatelessStreamProcessor(chunk_bytes=256 * 1024)` remains the **recommended default**, while `AdaptiveStreamProcessor(hysteresis_chunks=3)` is an opt-in policy for coarse-grained workloads.

### 5. v6.3 Long-Duration RSS Time-Series, Quantile Ablation & End-to-End AI Pipeline ([`benchmarks/v63_repro_and_e2e.py`](benchmarks/v63_repro_and_e2e.py))

#### A. 16 GiB Continuous RSS Time-Series (`BoundedMemorySink` 2 MiB cap, sampled every 1 GiB)
- **RSS (MB) at 1..16 GiB:** `23.8, 25.6, 24.9, 27.6, 27.5, 27.3, 27.7, 26.8, 27.6, 27.3, 27.3, 26.9, 27.4, 27.6, 27.6, 26.5`
- **Peak RSS:** Plateaus at **`27.7 MB`** from `5 GiB` through `16 GiB`.

#### B. Adaptive Policy Ablation with `P25/P50/P75`, `StdDev` & `switch_count` (`16 MiB`, 15 trials)
| Dataset | Policy | MB/s (P50) | P25 (ms) | P50 (ms) | P75 (ms) | Std (ms) | Switches |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| `repeated` | `fixed_256k` | 8413.08 | 1.705 | 1.902 | 2.140 | 0.292 | 0 |
| `repeated` | `adapt_h1` | **10145.20** | 1.505 | 1.577 | 1.988 | 0.333 | 0 |
| `random` | `fixed_256k` | 4607.76 | 3.321 | 3.472 | 3.667 | 0.370 | 0 |
| `random` | `adapt_h1` | **4906.02** | 3.154 | 3.261 | 3.522 | 0.269 | 0 |
| `low_then_high` | `fixed_256k` | **4688.92** | 3.234 | 3.412 | 3.540 | 0.325 | 0 |
| `low_then_high` | `adapt_h1` | 4209.97 | 3.532 | 3.801 | 4.313 | 0.578 | 1 |
| `alternating` | `fixed_256k` | **2968.35** | 5.118 | 5.390 | 5.908 | 0.543 | 0 |
| `alternating` | `adapt_h1` | 2479.97 | 5.962 | 6.452 | 6.871 | 0.553 | 15 |
| `alternating` | `adapt_h3` | 2340.04 | 6.605 | 6.838 | 7.528 | 0.564 | 0 |

#### C. End-to-End AI Preprocessing Pipeline (`16 MiB ai_corpus` → `tiktoken cl100k_base`)
| Mode | File Size (MB) | Tokens | Time (ms) | Tokens/s | PyPeak (MB) | OS Peak RSS (MB) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| `full_ram_baseline` | 16.000 | 4,031,088 | 4468.18 | 902,176.8 | 135.068 | 534.28 |
| `vantage_bounded_stream` | **1.233** | 4,031,149 | **4215.93** | **956,170.5** | **3.285** | **65.16** |

Streaming decompressed `256 KiB` chunks lazily into `tiktoken` reduces **OS Peak RSS by 8.2x** (`65.16 MB` vs `534.28 MB`), **Python heap peak by 41.1x** (`3.285 MB` vs `135.068 MB`), and **disk footprint by 12.98x**, while running **5.6% faster end-to-end**. *(Note: chunk-boundary tokenization shifts token count by `+61` tokens / `0.0015%`; line-aligned framing can eliminate this if exact token boundaries are required.)*

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
