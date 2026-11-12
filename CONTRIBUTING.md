# Contributing to Vantage V-AI

We welcome reproducible bug fixes, cross-platform benchmark reports (Linux/macOS/ARM/AMD), and streaming sink/codec improvements.

## Development Setup

```bash
git clone https://github.com/khoadev-vn/VANTAGE-V-AI.git
cd VANTAGE-V-AI
pip install -e ".[bench,dev]"
pytest tests/test_engine.py -v
```

## Research & Benchmark Standards

1. All correctness changes must include unit tests with CRC32 and round-trip verification.
2. Any performance claim must include the benchmark script, environment metadata, multiple trials (`P25/P50/P75`, `StdDev`), and isolated-process memory (`Peak RSS` + `PyPeak`).
3. Negative results and workload trade-offs must be documented alongside gains.
