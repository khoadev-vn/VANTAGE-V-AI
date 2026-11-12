# Changelog

All notable changes to Vantage V-AI are documented in this file.

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
