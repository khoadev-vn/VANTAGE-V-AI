# Vantage V-AI v5.0 Technical Report: Stateless Streaming & Memory-Bounded Processing

**Author:** Nguyen Vo Anh Khoa  
**Version:** 5.0 (Scientific Baseline)  
**Repository:** `khoadev-vn/VANTAGE-V-AI`

---

## 1. Abstract & Problem Statement

In edge and resource-constrained AI pipelines, data preprocessing, serialization, and transport frequently contend with limited memory budgets and Python heap allocation overhead. Earlier versions of Vantage V-AI (v4.0) suffered from three methodological flaws:
1. Conflating LZ4 byte-stream compression with LLM token generation (`4.3B tokens/s` via a hardcoded `20 tokens/record` multiplier).
2. Retaining all compressed blocks in an unbounded `self.vault` list while claiming statelessness.
3. Reporting RSS deltas that excluded pre-allocated input buffers.

Vantage V-AI v5.0 resolves these flaws with a strictly stateless streaming core ([`core/engine.py`](../core/engine.py)), bounded output sinks, mandatory CRC32 + round-trip verification, and a reproducible multi-baseline benchmark suite ([`benchmarks/vantage_validation.py`](../benchmarks/vantage_validation.py)).

---

## 2. Architecture & Design

- **`StatelessStreamProcessor`:** Holds only immutable configuration (`chunk_bytes`, `codec`, `compression_level`, `compute_checksum`). Accepts `bytes`, `bytearray`, or `memoryview` and slices chunks via `memoryview` without intermediate Python-side input copies.
- **`OutputSink` Abstraction:** Decouples compute from retention via `NullSink`, `BoundedMemorySink` (enforcing `max_bytes`), `CallbackSink`, and `CollectingSink`.
- **Correctness Verification:** Every supported codec (`lz4_frame`, `lz4_block`, `zstd`, `none`) implements `decompress_chunk()` and `verify_roundtrip()` with `zlib.crc32` validation.

---

## 3. Experimental Findings

Across 8 MiB workloads on Windows 11 (Intel 6C/12T, Python 3.12.10, 7 trials after warm-up):

1. **10x–31x Lower Python Peak Heap Allocation:** `StatelessStreamProcessor` (`256 KiB` chunks) peaks at **`0.254–0.754 MB`** on the Python heap (`tracemalloc`), compared to **`8.0–16.0 MB`** for `python_lz4_naive` and **`8.001 MB`** for `pyarrow_lz4`.
2. **Incompressible & JSON Throughput Advantage:** On `random` (incompressible) and `json_records`, `vantage_lz4_block_1t` achieves **`1188.14 MB/s`** and **`867.28 MB/s`** respectively, outperforming both `python_lz4_naive` (`598.60` / `775.83 MB/s`) and `pyarrow_lz4` (`783.49` / `854.47 MB/s`) due to cache-local 256 KiB chunking.
3. **Negative Result — Thread-Pool Overhead at 8 MiB:** `ParallelStreamProcessor` (`632–847 MB/s`) is slower than single-threaded `StatelessStreamProcessor` on 8 MiB payloads because `ThreadPoolExecutor` setup/synchronization exceeds the ~7–10 ms compression window.
4. **Tokenization Bottleneck:** Real BPE tokenization (`tiktoken cl100k_base`) on `ai_corpus` runs at **`2.89M tokens/s` (`11.49 MB/s`)**—roughly two orders of magnitude slower than LZ4 byte-stream compression.

---

## 4. Responses to the 10 Systems-Review Questions

1. **What problem does Vantage solve beyond raw LZ4?** Bounded-memory streaming (`0.25–0.75 MB` Python heap peak vs `8–16 MB` for one-shot LZ4/Arrow) with explicit backpressure sinks and zero-copy `memoryview` chunking.
2. **How is statelessness defined?** `StatelessStreamProcessor` stores zero payload state between `process()` calls; all output flows to an explicit `OutputSink`.
3. **Which tokenizer produced 4.3B tokens/s?** None—it was a synthetic `20 tokens/record` multiplier in v4.0 and has been retracted. Real `tiktoken cl100k_base` speed is measured separately at `2.89M tokens/s`.
4. **What is the actual peak memory?** Python heap peak during streaming is `0.254–0.754 MB`; total process Peak RSS (including Python runtime, libraries, and 8 MiB input buffers) is `94.11–119.69 MB`.
5. **Why 25,000 records for L1/L2 cache?** Retracted. v5.0 uses explicit byte-bounded chunks (`256 KiB`) sized for L2/L3 cache locality rather than record-count heuristics.
6. **Is Vantage faster than Python + LZ4?** Yes on `random` (`1188 vs 599 MB/s`) and `json_records` (`867 vs 776 MB/s`) with `lz4_block`; slightly slower on `repeated` (`1274 vs 1381 MB/s`) due to CRC32 and chunk framing overhead.
7. **How does entropy affect performance?** Compression ratio spans `238.75x` (`repeated`), `12.99x` (`ai_corpus`), `6.90x` (`json_records`), and `1.00x` (`random`).
8. **Is data integrity verified?** Yes—every baseline and dataset passes `zlib.crc32` + full round-trip decompression (`PASS`).
9. **Does Vantage improve end-to-end LLM inference?** Not yet demonstrated; integrating a quantized edge runtime (e.g., `llama.cpp` / ONNX Runtime) to measure TTFT and Peak RSS is the next phase.
10. **Where is the contribution?** Systems engineering and experimental methodology: bounded-memory streaming abstractions, reproducible multi-entropy benchmarking, and honest bottleneck attribution.
