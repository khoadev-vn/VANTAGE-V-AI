"""
Vantage V-AI Scientific Validation & Benchmark Suite (v5.1)

Upgrades over v5.0:
- 30 independent trials per (dataset, baseline) with IQR & StdDev dispersion
- Strict codec parity: python_lz4_frame_naive vs vantage_lz4_frame_1t, and
  python_lz4_block_naive vs vantage_lz4_block_raw (no CRC32) & vantage_lz4_block_1t (with CRC32)
- Isolated subprocess Peak RSS profiling per baseline (full lifecycle)
- Separate real BPE tokenization baseline (tiktoken cl100k_base, 1 MiB slice)
"""

from __future__ import annotations

import csv
import datetime
import gc
import json
import platform
import statistics
import subprocess
import sys
import time
import zlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import lz4.block
import lz4.frame
import numpy as np
import psutil

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from benchmarks.datasets import DatasetSpec, generate_datasets
from benchmarks.isolated_memory_bench import run_isolated_memory_suite
from benchmarks.memory_profiler import measure_memory
from core.engine import CodecType, CollectingSink, NullSink, StatelessStreamProcessor

try:
    import pyarrow as pa
except ImportError:
    pa = None

try:
    import tiktoken
except ImportError:
    tiktoken = None


RESULTS_DIR = REPO_ROOT / "benchmarks" / "results"
LOG_PATH = REPO_ROOT / "benchmarks" / "VANTAGE_VALIDATION.log"

DATASET_BYTES = 8 * 1024 * 1024  # 8 MiB per dataset
TRIALS = 30
WARMUP = 2


@dataclass
class SummaryRow:
    dataset: str
    baseline: str
    input_mb: float
    output_mb: float
    compression_ratio: float
    throughput_p50_mbs: float
    latency_p50_ms: float
    latency_iqr_ms: float
    latency_stddev_ms: float
    latency_p95_ms: float
    latency_p99_ms: float
    py_peak_alloc_mb: float
    verified: bool


def _git_sha() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(REPO_ROOT),
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return out.strip()
    except Exception:
        return "unknown"


def _collect_env_info() -> Dict[str, object]:
    return {
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "git_sha": _git_sha(),
        "os": f"{platform.system()} {platform.release()} ({platform.machine()})",
        "cpu": platform.processor() or "unknown",
        "physical_cores": psutil.cpu_count(logical=False),
        "logical_threads": psutil.cpu_count(logical=True),
        "ram_gb": round(psutil.virtual_memory().total / (1024 ** 3), 2),
        "python_version": platform.python_version(),
        "packages": {
            "lz4": getattr(lz4, "__version__", "unknown"),
            "psutil": psutil.__version__,
            "numpy": np.__version__,
            "pyarrow": getattr(pa, "__version__", "not-installed"),
            "tiktoken": getattr(tiktoken, "__version__", "not-installed"),
        },
        "dataset_bytes": DATASET_BYTES,
        "warmup_runs": WARMUP,
        "measured_trials": TRIALS,
    }


def _run_python_lz4_frame_naive(data: bytes) -> Tuple[int, Callable[[], bool]]:
    compressed = lz4.frame.compress(data)
    expected_crc = zlib.crc32(data) & 0xFFFFFFFF

    def verify() -> bool:
        restored = lz4.frame.decompress(compressed)
        return (zlib.crc32(restored) & 0xFFFFFFFF) == expected_crc

    return len(compressed), verify


def _run_python_lz4_block_naive(data: bytes) -> Tuple[int, Callable[[], bool]]:
    compressed = lz4.block.compress(data, store_size=True)
    expected_crc = zlib.crc32(data) & 0xFFFFFFFF

    def verify() -> bool:
        restored = lz4.block.decompress(compressed)
        return (zlib.crc32(restored) & 0xFFFFFFFF) == expected_crc

    return len(compressed), verify


def _run_pyarrow_lz4(data: bytes) -> Tuple[int, Callable[[], bool]]:
    if pa is None:
        raise RuntimeError("pyarrow not installed")
    buf = pa.py_buffer(data)
    compressed = pa.compress(buf, codec="lz4", asbytes=True)
    expected_crc = zlib.crc32(data) & 0xFFFFFFFF

    def verify() -> bool:
        restored = pa.decompress(compressed, decompressed_size=len(data), codec="lz4").to_pybytes()
        return (zlib.crc32(restored) & 0xFFFFFFFF) == expected_crc

    return len(compressed), verify


def _run_vantage(
    data: bytes,
    processor: StatelessStreamProcessor,
) -> Tuple[int, Callable[[], bool]]:
    sink = NullSink()
    stats = processor.process(data, sink)

    def verify() -> bool:
        csink = CollectingSink()
        cstats = processor.process(data, csink)
        expected = cstats.crc32 if processor.compute_checksum else None
        restored = processor.verify_roundtrip(csink.frames(), expected_crc32=expected)
        return restored == data

    return stats.output_bytes, verify


def _percentile(values: List[float], pct: float) -> float:
    return float(np.percentile(values, pct))


def benchmark_pair(
    ds: DatasetSpec,
    baseline_name: str,
    runner: Callable[[bytes], Tuple[int, Callable[[], bool]]],
    raw_rows: List[Dict[str, object]],
) -> SummaryRow:
    for _ in range(WARMUP):
        out_bytes, verify_fn = runner(ds.data)
    verified = verify_fn()

    latencies_ms: List[float] = []
    py_peaks_mb: List[float] = []

    for trial in range(1, TRIALS + 1):
        gc.collect()
        with measure_memory() as mem:
            t0 = time.perf_counter()
            out_bytes, _ = runner(ds.data)
            elapsed_s = time.perf_counter() - t0

        lat_ms = elapsed_s * 1000.0
        latencies_ms.append(lat_ms)
        py_peaks_mb.append(mem.python_peak_alloc_mb)

        raw_rows.append(
            {
                "dataset": ds.name,
                "baseline": baseline_name,
                "trial": trial,
                "input_bytes": len(ds.data),
                "output_bytes": out_bytes,
                "latency_ms": round(lat_ms, 4),
                "throughput_mbs": round((len(ds.data) / (1024 * 1024)) / elapsed_s, 2),
                "py_peak_alloc_mb": round(mem.python_peak_alloc_mb, 3),
                "verified": verified,
            }
        )

    p50_ms = statistics.median(latencies_ms)
    iqr_ms = _percentile(latencies_ms, 75) - _percentile(latencies_ms, 25)
    std_ms = statistics.stdev(latencies_ms)
    p95_ms = _percentile(latencies_ms, 95)
    p99_ms = _percentile(latencies_ms, 99)
    input_mb = len(ds.data) / (1024 * 1024)
    output_mb = out_bytes / (1024 * 1024)
    throughput_p50 = input_mb / (p50_ms / 1000.0)
    ratio = len(ds.data) / out_bytes if out_bytes > 0 else 0.0

    return SummaryRow(
        dataset=ds.name,
        baseline=baseline_name,
        input_mb=round(input_mb, 2),
        output_mb=round(output_mb, 3),
        compression_ratio=round(ratio, 2),
        throughput_p50_mbs=round(throughput_p50, 2),
        latency_p50_ms=round(p50_ms, 3),
        latency_iqr_ms=round(iqr_ms, 3),
        latency_stddev_ms=round(std_ms, 3),
        latency_p95_ms=round(p95_ms, 3),
        latency_p99_ms=round(p99_ms, 3),
        py_peak_alloc_mb=round(statistics.median(py_peaks_mb), 3),
        verified=verified,
    )


def benchmark_real_tokenization(ai_corpus: DatasetSpec) -> Optional[Dict[str, object]]:
    """Measure actual BPE tokenization (tiktoken cl100k_base) on a 1 MiB text slice."""
    if tiktoken is None:
        return None
    enc = tiktoken.get_encoding("cl100k_base")
    text_slice = ai_corpus.data[: 1024 * 1024].decode("utf-8", errors="ignore")
    tokens = enc.encode(text_slice)
    times: List[float] = []
    for _ in range(10):
        t0 = time.perf_counter()
        tokens = enc.encode(text_slice)
        times.append(time.perf_counter() - t0)
    med_s = statistics.median(times)
    return {
        "tokenizer": "tiktoken (cl100k_base)",
        "input_bytes": len(text_slice.encode("utf-8")),
        "actual_tokens": len(tokens),
        "median_ms": round(med_s * 1000.0, 2),
        "tokens_per_sec": round(len(tokens) / med_s, 1),
        "mb_per_sec": round((len(text_slice.encode("utf-8")) / (1024 * 1024)) / med_s, 2),
    }


def render_log(
    env: Dict[str, object],
    summaries: List[SummaryRow],
    isolated_mem: List[Dict[str, object]],
    tok_result: Optional[Dict[str, object]],
) -> str:
    lines = [
        "=" * 104,
        "VANTAGE V-AI v5.1 — SCIENTIFIC VALIDATION & CODEC-PARITY BENCHMARK LOG",
        "=" * 104,
        f"Timestamp (UTC): {env['timestamp_utc']}",
        f"Git SHA:         {env['git_sha']}",
        f"OS / Arch:       {env['os']}",
        f"CPU:             {env['cpu']} ({env['physical_cores']}C / {env['logical_threads']}T)",
        f"System RAM:      {env['ram_gb']} GB",
        f"Python:          {env['python_version']}",
        f"Packages:        {env['packages']}",
        f"Protocol:        {DATASET_BYTES // (1024 * 1024)} MiB/dataset, {WARMUP} warmups, {TRIALS} trials, CRC32 + roundtrip verified",
        "-" * 104,
        f"{'Dataset':<13} {'Baseline':<23} {'MB/s(P50)':>9} {'P50(ms)':>8} {'IQR':>7} {'Std':>7} {'P95(ms)':>8} {'P99(ms)':>8} {'Ratio':>7} {'PyPeak':>7} {'OK':>4}",
        "-" * 104,
    ]
    for r in summaries:
        lines.append(
            f"{r.dataset:<13} {r.baseline:<23} {r.throughput_p50_mbs:>9.2f} "
            f"{r.latency_p50_ms:>8.3f} {r.latency_iqr_ms:>7.3f} {r.latency_stddev_ms:>7.3f} "
            f"{r.latency_p95_ms:>8.3f} {r.latency_p99_ms:>8.3f} "
            f"{r.compression_ratio:>6.2f}x {r.py_peak_alloc_mb:>7.3f} {'PASS' if r.verified else 'FAIL':>4}"
        )

    lines.extend(
        [
            "-" * 104,
            "ISOLATED SUBPROCESS FULL-LIFECYCLE MEMORY (8 MiB random, fresh process per baseline):",
            f"{'Baseline':<25} {'RSS_Start(MB)':>13} {'RSS_Input(MB)':>13} {'Peak_RSS(MB)':>13} {'Proc_Overhead(MB)':>18}",
        ]
    )
    for m in isolated_mem:
        lines.append(
            f"{m['baseline']:<25} {m['rss_start_mb']:>13.2f} {m['rss_after_input_mb']:>13.2f} "
            f"{m['isolated_peak_rss_mb']:>13.2f} {m['processing_rss_overhead_mb']:>18.2f}"
        )

    if tok_result:
        lines.extend(
            [
                "-" * 104,
                "SEPARATE REAL TOKENIZATION BASELINE (1 MiB ai_corpus slice; NOT LLM inference or Vantage-accelerated):",
                f"  Tokenizer:     {tok_result['tokenizer']}",
                f"  Input Slice:   {tok_result['input_bytes']} bytes -> {tok_result['actual_tokens']} actual tokens",
                f"  Median Time:   {tok_result['median_ms']} ms",
                f"  Real Speed:    {tok_result['tokens_per_sec']:,.1f} tokens/s ({tok_result['mb_per_sec']} MB/s)",
            ]
        )

    lines.append("=" * 104)
    return "\n".join(lines) + "\n"


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    env = _collect_env_info()
    datasets = generate_datasets(DATASET_BYTES)

    vantage_frame_1t = StatelessStreamProcessor(
        chunk_bytes=256 * 1024, codec=CodecType.LZ4_FRAME, compute_checksum=True, frame_checksum=True
    )
    vantage_block_1t = StatelessStreamProcessor(
        chunk_bytes=256 * 1024, codec=CodecType.LZ4_BLOCK, compute_checksum=True
    )
    vantage_block_raw = StatelessStreamProcessor(
        chunk_bytes=256 * 1024, codec=CodecType.LZ4_BLOCK, compute_checksum=False
    )

    baselines: List[Tuple[str, Callable[[bytes], Tuple[int, Callable[[], bool]]]]] = [
        ("python_lz4_frame_naive", _run_python_lz4_frame_naive),
        ("python_lz4_block_naive", _run_python_lz4_block_naive),
    ]
    if pa is not None:
        baselines.append(("pyarrow_lz4", _run_pyarrow_lz4))
    baselines.extend(
        [
            ("vantage_lz4_frame_1t", lambda d: _run_vantage(d, vantage_frame_1t)),
            ("vantage_lz4_block_1t", lambda d: _run_vantage(d, vantage_block_1t)),
            ("vantage_lz4_block_raw", lambda d: _run_vantage(d, vantage_block_raw)),
        ]
    )

    raw_rows: List[Dict[str, object]] = []
    summaries: List[SummaryRow] = []

    for ds_name in ("repeated", "json_records", "ai_corpus", "random"):
        ds = datasets[ds_name]
        for b_name, runner in baselines:
            summaries.append(benchmark_pair(ds, b_name, runner, raw_rows))

    isolated_mem = run_isolated_memory_suite([b[0] for b in baselines], "random", DATASET_BYTES)
    tok_result = benchmark_real_tokenization(datasets["ai_corpus"])

    csv_path = RESULTS_DIR / "benchmark_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(raw_rows[0].keys()))
        writer.writeheader()
        writer.writerows(raw_rows)

    json_path = RESULTS_DIR / "benchmark_summary.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "environment": env,
                "summary": [asdict(s) for s in summaries],
                "isolated_memory": isolated_mem,
                "tokenization_baseline": tok_result,
            },
            f,
            indent=2,
        )

    log_text = render_log(env, summaries, isolated_mem, tok_result)
    LOG_PATH.write_text(log_text, encoding="utf-8")
    print(log_text)


if __name__ == "__main__":
    main()
