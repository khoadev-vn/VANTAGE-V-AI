# Changelog

All notable changes to Vantage V-AI are documented in this file.

## [7.2.0] - 2026-11-12

### Added
- **Hugging Face Transformers Adapter (`core.adapters.transformers.VantageDynamicCache`):**
  - Drop-in replacement for `transformers.cache_utils.DynamicCache` compatible with `model.generate()`.
  - Supports modern Hugging Face Transformers v5.x (`layers[i].keys/values`) and legacy list-based structures.
  - Bounded active token sliding window with automatic offload to `KVCacheStreamEngine`.
  - Supports both Lossless LZ4 and Dynamic FP8 Quantized compression.
  - Provides `cache.restore_full_cache()` for prompt cache checkpointing and full sequence recovery.
- **End-to-End Hugging Face Benchmark (`benchmarks/bench_hf_generate.py`):**
  - Causal LM generation testing 64 prompt tokens + 64 generated tokens.
  - Demonstrates 74.8% active KV tensor memory reduction (`128.0 KB` vs `508.0 KB`) and 4.0x offloaded compression in FP8 mode.
  - Automated JSON artifact persistence at `artifacts/benchmarks/benchmark_hf_transformers_v7_2.json`.
- **Unit Tests:** `tests/test_transformers_adapter.py` (3 test cases covering basic updates, bounded offload/restore, and FP8 quantization). Total test suite expanded to 29 passing tests.

## [7.1.0] - 2026-11-12

### Added
- **Attention KV-Cache Streaming Engine (`core/kv.py`):**
  - Direct offloading and on-demand streaming reload for LLM Attention Key and Value tensors.
  - Dual modes: Lossless Zero-Drift (bit-for-bit identical roundtrip) and Dynamic FP8 Quantization (~2.0x memory reduction, MAE < 0.0077, Cosine similarity 1.0000).
  - Support for 3D and 4D tensor representations across PyTorch and NumPy.
- **KV Benchmark Suite (`benchmarks/bench_kv_cache.py`):**
  - Simulated Llama-3-8B attention slice (8 layers, 8 heads, 128 head dim) across 4K and 8K context lengths.

## [6.4.1] - 2026-11-12

### Added
- **Record-boundary realignment (`iter_record_chunks`):** Guarantees 100% exact bit-for-bit token ID parity against full-file tokenization while preserving bounded streaming memory.
- **Robustness Tests:**
  - `test_utf8_multibyte_split_across_chunks`: Verifies multibyte CJK, Vietnamese, and emoji tokens across 1 KiB chunk boundaries.
  - `test_crlf_and_no_trailing_newline_token_parity`: Tests mixed `\r\n` / `\n` and streams lacking a final trailing newline.
  - `test_corrupted_and_truncated_frames_raise`: Verifies graceful failure (`EOFError` / `ValueError`) on truncated headers, truncated frame payloads, and CRC32 mismatches.
- **Open-Source Governance & Launch Assets:**
  - `CODE_OF_CONDUCT.md`: Contributor Covenant 2.1.
  - `.github/ISSUE_TEMPLATE/`: Standardized templates for bug reports, benchmark reports, and feature requests.
  - `examples/quickstart.py`: End-to-end executable examples covering bounded streaming, framed file round-trips, and record-aligned token parity.

### Changed
- **E2E Benchmark Methodology:** Documented the trade-off in 16 MiB AI preprocessing: Vantage reduces OS Peak RSS by 8.06x (`66.27 MB` vs `534.37 MB`) and disk footprint by 12.98x, with a moderate execution time trade-off (`4223.63 ms` vs `3767.14 ms`).
- **Packaging:** Updated version to `6.4.1` with explicit PEP 621 metadata.
