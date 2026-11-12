# Vantage V-AI v6.0 Technical Report: Production Streaming Core & Anomaly Verification

**Author:** Nguyen Vo Anh Khoa  
**Version:** 6.0.0 (Production Streaming Core)  
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

## 6. Next Phase: End-to-End Edge AI Integration

Following *MLSysBook Vol. I*, the next milestone is integrating Vantage's bounded-memory streaming layer with a real quantized edge inference runtime (e.g., `llama.cpp` or ONNX Runtime) to evaluate **TTFT**, **end-to-end tokens/s**, and **total Peak RSS** on identical prompts and model weights.
