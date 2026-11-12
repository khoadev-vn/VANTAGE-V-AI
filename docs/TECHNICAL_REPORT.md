# Vantage V-AI v6.4 Technical Report: 100% Token Parity & Bounded-Memory Streaming

**Author:** Nguyen Vo Anh Khoa  
**Version:** 6.4.0 (Launch-Ready Scientific Baseline)  
**Repository:** `khoadev-vn/VANTAGE-V-AI`  
**Academic Reference:** *MLSysBook Vol. I* (`https://mlsysbook.ai/vol1/`)

---

## 1. Abstract & Problem Statement

In edge and resource-constrained AI pipelines, data preprocessing and transport contend with tight memory budgets and Python heap allocation overhead. Earlier versions of Vantage V-AI (v4.0) conflated LZ4 byte-stream compression with LLM token throughput (`4.3B tokens/s`), retained unbounded state in `self.vault`, and omitted pre-allocated input buffers from memory measurements.

Vantage V-AI v5.1 resolves these flaws with a strictly stateless streaming core ([`core/engine.py`](../core/engine.py)), bounded output sinks, mandatory `CRC32` + round-trip verification, 30-trial codec-parity benchmarks ([`benchmarks/vantage_validation.py`](../benchmarks/vantage_validation.py)), and isolated-subprocess full-lifecycle memory profiling ([`benchmarks/isolated_memory_bench.py`](../benchmarks/isolated_memory_bench.py)).

---

## 2. Architecture & Design

- **`StatelessStreamProcessor`:** Holds only immutable configuration (`chunk_bytes`, `codec`, `compression_level`, `compute_checksum`). Slices `bytes`, `bytearray`, and `memoryview` inputs via `memoryview` into `256 KiB` chunks without intermediate Python-side input copies.
- **`OutputSink` Abstraction:** Decouples compute from retention via `NullSink`, `BoundedMemorySink` (enforcing `max_bytes`), `CallbackSink`, and `CollectingSink`.
- **Correctness Verification:** Every supported codec (`lz4_frame`, `lz4_block`, `zstd`, `none`) implements `decompress_chunk()` and `verify_roundtrip()` with `zlib.crc32` validation.

---

## 3. Key Experimental Findings (30 Trials, 8 MiB/Dataset)

1. **Root Cause of Throughput Advantage (Codec Parity):** Comparing `python_lz4_block_naive` (one-shot 8 MiB `lz4.block.compress`, no CRC32) against `vantage_lz4_block_raw` (`256 KiB` `memoryview` chunks, no CRC32) isolates the effect of chunked L2-cache-local processing:
   - `repeated`: **`6180.45 MB/s`** vs `1328.68 MB/s` (**4.65x faster**)
   - `json_records`: **`1761.33 MB/s`** vs `838.95 MB/s` (**2.10x faster**)
   - `ai_corpus`: **`2777.60 MB/s`** vs `1083.48 MB/s` (**2.56x faster**)
   - `random`: **`2601.75 MB/s`** vs `728.12 MB/s` (**3.57x faster**)
   Adding per-chunk `zlib.crc32` (`vantage_lz4_block_1t`) shifts the bottleneck to CRC32 computation (`1045–1239 MB/s`), while still outperforming `python_lz4_block_naive` on `random` (`1185.76` vs `728.12 MB/s`) and `json_records` (`867.65` vs `838.95 MB/s`).
2. **10x–31x Lower Python Heap Allocation:** `StatelessStreamProcessor` peaks at **`0.253–0.754 MB`** on the Python heap (`tracemalloc`), versus **`8.0–16.06 MB`** for one-shot LZ4 and Arrow baselines.
3. **Full-Lifecycle Isolated Peak RSS:** In fresh subprocesses per baseline (`8 MiB random`), total process Peak RSS is **`64.14–64.98 MB`** across all implementations (`~34.1–34.9 MB` runtime/processing working set above the `~30.1 MB` post-input RSS).
4. **Negative Result — Tokenization & Parallelism:**
   - Real BPE tokenization (`tiktoken cl100k_base`) on a 1 MiB `ai_corpus` slice runs at **`2,855,145.1 tokens/s` (`11.33 MB/s`)**—confirming that tokenization, not byte compression, dominates CPU preprocessing cost.
   - `ParallelStreamProcessor` (tested in v5.0) was slower than single-threaded execution on 8 MiB payloads due to `ThreadPoolExecutor` dispatch overhead.

---

## 4. v5.2 Root-Cause Experiments (15 Trials, 8 MiB/Dataset)

1. **Chunking Parity (`python_lz4_block_chunked` vs `vantage_lz4_block_raw`):** Both use `memoryview` `256 KiB` slices + `lz4.block.compress` without CRC32 and achieve comparable throughput (`7754` vs `6816 MB/s` on `repeated`; `1494` vs `1636 MB/s` on `json_records`; `2267` vs `2064 MB/s` on `ai_corpus`; `2220` vs `2799 MB/s` on `random`) and identical `PyPeak` (`0.254–0.754 MB`). This confirms the speedup stems from the chunking strategy rather than a novel compression algorithm.
2. **Chunk-Size Sweep (`64 KiB`–`8 MiB`):** On `random`, throughput peaks at `256 KiB` (`3915.6 MB/s`) before dropping to `917.9 MB/s` at `512 KiB`. Slicing data into `256 KiB` chunks via `memoryview` reduces processing unit size and limits concurrent transient allocations; hardware counter profiling is needed to confirm cache-locality effects.
3. **CRC32 Cost Isolation:** `CRC32` introduces substantial additional processing overhead, accounting for approximately **48–81%** of the combined execution time under the tested workloads (`crc32_only` `1356–1658 MB/s` vs `compress_plus_crc32` `735–1327 MB/s`). This is an empirical comparison, not an isolated attribution of CPU cost.

---

## 5. v6.0 Production Core Validation

1. **Isolated Subprocess Re-Check of `256 KiB` → `512 KiB` Cliff:** Across 3 random seeds in randomized order (`10` trials each), `256 KiB` (`1874.81–4537.08 MB/s`) consistently outperforms `512 KiB` (`740.79–1240.68 MB/s`) and `1024 KiB` (`771.30–1140.62 MB/s`), confirming the transition is not an ordering or single-seed artifact.
2. **62.16 MiB Incremental Streaming (`process_iter`):** Streams `256 × 256 KiB` chunks at **`3120.14 MB/s`** (`NullSink`, `0.747 MB` Python heap peak) and **`3188.01 MB/s`** (`BoundedMemorySink` 2 MiB cap, `0.993 MB` Python heap peak).
3. **`AdaptiveStreamProcessor`:** Probes the first `64 KiB` to select `512 KiB` for compressible payloads (`11778.56 MB/s` on `repeated`) and `256 KiB` for incompressible payloads (`5683.43 MB/s` on `random`).

---

## 6. v6.1 Large-Scale & Heterogeneous Streaming Validation

1. **O(1) Bounded Memory (`64 MiB → 4 GiB` in Isolated Subprocesses):** Streaming a pre-allocated `256 KiB` chunk via `process_iter` into `BoundedMemorySink(max_bytes=2 MiB)` keeps OS `Peak RSS` at `22.52 MB` (`64 MiB`), `23.61 MB` (`256 MiB`), `24.86 MB` (`1 GiB`), and `24.76 MB` (`4 GiB`, `3916.45 MB/s`, `PyPeak = 2.579 MB`, `Retained = 2,096,220 B`).
2. **Dynamic Per-Chunk Adaptation (`AdaptiveStreamProcessor`):** Updating chunk size per emitted frame avoids the `512 KiB` collapse on `low_then_high` (`4702.3 MB/s` vs `1371.9 MB/s` for `fixed_512k`) while reaching `11221.4 MB/s` on `high_then_low`.
3. **Production `FileSink` & Framed Container:** `FileSink(framed=True)` handles partial writes safely, writes `<I` length-prefixed frames, and round-trips via `read_framed_stream()` with `CRC32` verification.

---

## 7. v6.2 Stress Testing (`4–16 GiB`), Backpressure RSS & Adaptive Ablation

1. **Bounded-Memory Scaling (`64 MiB → 16 GiB`):** Under fixed chunk size (`256 KiB`) and `BoundedMemorySink(2 MiB)`, `PyPeak` grows from `0.777 MB` (`64 MiB`) to `2.579 MB` (`1 GiB`) as the sink fills, then remains constant at `2.579 MB` for `4 GiB`, `8 GiB`, and `16 GiB` (`OS Peak RSS = 24.82–25.13 MB`; `16 GiB NullSink = 22.18 MB`, `4828.67 MB/s`).
2. **Backpressure + Bounded Memory:** Under `SlowCallbackSink (100 µs/frame)` on a `32 MiB` stream, throughput drops to `296.99 MB/s` while `OS Peak RSS` remains `22.90 MB` (`PyPeak = 1.004 MB`).
3. **Adaptive Hysteresis Ablation:** `adapt_h3` (`hysteresis_chunks=3`) reaches `10862.2 MB/s` on `repeated` and `5175.1 MB/s` on `random`, but under `alternating` (`2363.9 MB/s`) and `adversarial_jitter` (`1944.1 MB/s`), `fixed_256k` (`6737.1` / `6208.1 MB/s`) remains superior.

---

## 8. v6.3–v6.4 Long-Duration RSS Time-Series & 100% Token-Parity E2E Pipeline

1. **16 GiB RSS Time-Series:** During a `16 GiB` streaming run, sampled process RSS remained within `23.8–27.7 MB`, with no observable growth proportional to the amount of data processed.
2. **Quantile Adaptive Ablation:** `adapt_h1` performs `15` chunk-size switches on `alternating` (`2479.97 MB/s`), whereas `adapt_h3` suppresses switching (`0` switches) yet still trails `fixed_256k` (`2968.35 MB/s`), confirming `fixed_256k` as the robust default.
3. **100% Token-ID-Parity E2E AI Preprocessing (`ai_corpus` → `tiktoken cl100k_base`):** Using `iter_record_chunks(..., delimiter=b"\n")` guarantees `stream_token_ids == full_token_ids` (`1,007,785` tokens at `4 MiB`; `4,031,088` tokens at `16 MiB`). Across `15` trials at `4 MiB`, `vantage_aligned_stream` achieves `P50 = 334.18 ms` (`3,015,669.9 tokens/s`) vs `340.25 ms` (`2,961,857.4 tokens/s`). In isolated `16 MiB` subprocesses, Vantage reduces **OS Peak RSS by 8.06x** (`66.27 MB` vs `534.37 MB`), **Python heap peak by 35.7x** (`3.786 MB` vs `135.067 MB`), and **disk size by 12.98x** (`1.233 MB` vs `16.000 MB`).

---

## 9. Next Phase: End-to-End Edge AI Inference Runtime Integration

Following *MLSysBook Vol. I*, the next milestone is integrating Vantage's bounded-memory streaming layer with a real quantized edge inference runtime (e.g., `llama.cpp` or ONNX Runtime) to evaluate **TTFT**, **end-to-end tokens/s**, and **total Peak RSS** on identical prompts and model weights.
