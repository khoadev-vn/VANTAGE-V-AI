"""
Vantage V-AI v6.1 — Large-Scale Streaming & Heterogeneous Entropy Validation

1. Isolated Subprocess Bounded-Memory Scaling: 64 MiB -> 256 MiB -> 1 GiB -> 4 GiB
   with BoundedMemorySink(2 MiB cap), measuring OS Peak RSS, PyPeak, and Retained.
2. Dynamic Adaptive vs Fixed (64K, 128K, 256K, 512K, 1024K) on homogeneous and
   heterogeneous entropy shifts (low_then_high, high_then_low, alternating).
3. Backpressure Sink Comparison: NullSink vs BoundedMemorySink vs SlowCallbackSink.
"""

from __future__ import annotations

import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from core.engine import (
    AdaptiveStreamProcessor,
    BoundedMemorySink,
    CallbackSink,
    CodecType,
    NullSink,
    StatelessStreamProcessor,
)

RESULTS_DIR = REPO_ROOT / "benchmarks" / "results"
LOG_PATH = REPO_ROOT / "benchmarks" / "VANTAGE_V61_LARGE_SCALE.log"

_SCALE_WORKER = """
import json, os, psutil, sys, time, tracemalloc
sys.path.insert(0, sys.argv[1])
total_mib = int(sys.argv[2])

from core.engine import BoundedMemorySink, CodecType, StatelessStreamProcessor

proc_os = psutil.Process(os.getpid())
mib = 1024 * 1024
chunk_bytes = 256 * 1024
num_chunks = (total_mib * mib) // chunk_bytes

# Pre-allocated 256 KiB input unit streamed repeatedly (isolates engine + sink memory)
unit = (b"ID:999|Node:V-AI|LargeScale:Streaming_Unit|" * 6400)[:chunk_bytes]
engine = StatelessStreamProcessor(chunk_bytes=chunk_bytes, codec=CodecType.LZ4_BLOCK, compute_checksum=False)
sink = BoundedMemorySink(max_bytes=2 * mib, drop_oldest=True)

tracemalloc.start()
t0 = time.perf_counter()
stats = engine.process_iter((unit for _ in range(num_chunks)), sink)
dt = time.perf_counter() - t0
_, py_peak = tracemalloc.get_traced_memory()
tracemalloc.stop()

info = proc_os.memory_info()
peak_rss = (info.peak_wset if hasattr(info, "peak_wset") else info.rss) / mib
print(json.dumps({
    "dataset_mib": total_mib,
    "throughput_mbs": round((stats.input_bytes / mib) / dt, 2),
    "elapsed_s": round(dt, 3),
    "peak_rss_mb": round(peak_rss, 2),
    "py_peak_mb": round(py_peak / mib, 3),
    "retained_bytes": sink.retained_bytes,
    "evicted_frames": sink.evicted_frames,
}))
"""


def run_large_scale_isolated() -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for total_mib in (64, 256, 1024, 4096):
        out = subprocess.check_output(
            [sys.executable, "-c", _SCALE_WORKER, str(REPO_ROOT), str(total_mib)],
            text=True,
        )
        rows.append(json.loads(out.strip()))
    return rows


def _build_heterogeneous_datasets() -> Dict[str, bytes]:
    rng = np.random.default_rng(2026)
    comp_1m = (b"VANTAGE_LOW_ENTROPY_STREAM_RECORD_" * 32768)[: 1024 * 1024]
    rand_1m = rng.bytes(1024 * 1024)
    return {
        "repeated": comp_1m * 16,
        "random": rng.bytes(16 * 1024 * 1024),
        "low_then_high": comp_1m[: 64 * 1024] + rng.bytes(16 * 1024 * 1024 - 64 * 1024),
        "high_then_low": rng.bytes(64 * 1024) + (comp_1m * 16)[: 16 * 1024 * 1024 - 64 * 1024],
        "alternating": b"".join(comp_1m if i % 2 == 0 else rand_1m for i in range(16)),
    }


def run_adaptive_heterogeneous() -> List[Dict[str, object]]:
    datasets = _build_heterogeneous_datasets()
    policies = [
        ("fixed_64k", StatelessStreamProcessor(chunk_bytes=64 * 1024, codec=CodecType.LZ4_BLOCK, compute_checksum=False)),
        ("fixed_128k", StatelessStreamProcessor(chunk_bytes=128 * 1024, codec=CodecType.LZ4_BLOCK, compute_checksum=False)),
        ("fixed_256k", StatelessStreamProcessor(chunk_bytes=256 * 1024, codec=CodecType.LZ4_BLOCK, compute_checksum=False)),
        ("fixed_512k", StatelessStreamProcessor(chunk_bytes=512 * 1024, codec=CodecType.LZ4_BLOCK, compute_checksum=False)),
        ("fixed_1024k", StatelessStreamProcessor(chunk_bytes=1024 * 1024, codec=CodecType.LZ4_BLOCK, compute_checksum=False)),
        ("adaptive_dyn", AdaptiveStreamProcessor(codec=CodecType.LZ4_BLOCK, compute_checksum=False)),
    ]

    rows: List[Dict[str, object]] = []
    for ds_name, data in datasets.items():
        mib = len(data) / (1024 * 1024)
        for p_name, proc in policies:
            for _ in range(2):
                proc.process(data, NullSink())
            lats = []
            for _ in range(10):
                t0 = time.perf_counter()
                proc.process(data, NullSink())
                lats.append((time.perf_counter() - t0) * 1000.0)
            p50 = statistics.median(lats)
            rows.append(
                {
                    "dataset": ds_name,
                    "policy": p_name,
                    "p50_mbs": round(mib / (p50 / 1000.0), 2),
                    "p50_ms": round(p50, 3),
                    "p95_ms": round(float(np.percentile(lats, 95)), 3),
                    "p99_ms": round(float(np.percentile(lats, 99)), 3),
                }
            )
    return rows


def run_backpressure_benchmark() -> List[Dict[str, object]]:
    data = os.urandom(8 * 1024 * 1024)
    proc = StatelessStreamProcessor(chunk_bytes=256 * 1024, codec=CodecType.LZ4_BLOCK, compute_checksum=False)
    results: List[Dict[str, object]] = []

    def slow_cb(_frame: bytes) -> None:
        time.sleep(0.0002)  # 200 us simulated slow downstream I/O per frame

    for name, sink in [
        ("NullSink", NullSink()),
        ("BoundedMemorySink_2MiB", BoundedMemorySink(max_bytes=2 * 1024 * 1024, drop_oldest=True)),
        ("SlowCallbackSink_200us", CallbackSink(slow_cb)),
    ]:
        t0 = time.perf_counter()
        proc.process(data, sink)
        dt = time.perf_counter() - t0
        results.append(
            {
                "sink": name,
                "elapsed_ms": round(dt * 1000.0, 2),
                "throughput_mbs": round(8.0 / dt, 2),
            }
        )
    return results


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    scale_rows = run_large_scale_isolated()
    het_rows = run_adaptive_heterogeneous()
    bp_rows = run_backpressure_benchmark()

    payload = {
        "large_scale_isolated": scale_rows,
        "heterogeneous_adaptive": het_rows,
        "backpressure_sinks": bp_rows,
    }
    (RESULTS_DIR / "v61_large_scale.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "=" * 92,
        "VANTAGE V-AI v6.1 — LARGE-SCALE BOUNDED STREAMING & ADAPTIVE VALIDATION",
        "=" * 92,
        "1. ISOLATED SUBPROCESS BOUNDED-MEMORY SCALING (64 MiB -> 4 GiB, BoundedMemorySink 2 MiB)",
        f"{'Dataset':<10} {'Throughput(MB/s)':>16} {'Time(s)':>9} {'PeakRSS(MB)':>12} {'PyPeak(MB)':>11} {'Retained(B)':>12}",
        "-" * 92,
    ]
    for r in scale_rows:
        label = f"{r['dataset_mib']} MiB" if r["dataset_mib"] < 1024 else f"{r['dataset_mib'] // 1024} GiB"
        lines.append(
            f"{label:<10} {r['throughput_mbs']:>16.2f} {r['elapsed_s']:>9.3f} "
            f"{r['peak_rss_mb']:>12.2f} {r['py_peak_mb']:>11.3f} {r['retained_bytes']:>12}"
        )

    lines.extend([
        "-" * 92,
        "2. HETEROGENEOUS ENTROPY: DYNAMIC ADAPTIVE vs FIXED CHUNK SIZES (16 MiB, 10 trials, MB/s P50)",
        f"{'Dataset':<15} {'64K':>9} {'128K':>9} {'256K':>9} {'512K':>9} {'1024K':>9} {'Adaptive':>10}",
        "-" * 92,
    ])
    for ds_name in ("repeated", "random", "low_then_high", "high_then_low", "alternating"):
        vals = [r["p50_mbs"] for r in het_rows if r["dataset"] == ds_name]
        lines.append(f"{ds_name:<15} " + " ".join(f"{v:>9.1f}" for v in vals))

    lines.extend([
        "-" * 92,
        "3. SINK BACKPRESSURE COMPARISON (8 MiB random)",
        f"{'Sink':<26} {'Elapsed(ms)':>12} {'Throughput(MB/s)':>17}",
        "-" * 92,
    ])
    for r in bp_rows:
        lines.append(f"{r['sink']:<26} {r['elapsed_ms']:>12.2f} {r['throughput_mbs']:>17.2f}")
    lines.append("=" * 92)

    log_text = "\n".join(lines) + "\n"
    LOG_PATH.write_text(log_text, encoding="utf-8")
    print(log_text)


if __name__ == "__main__":
    main()
