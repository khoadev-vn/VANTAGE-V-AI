"""
Vantage V-AI v6.0 — Production Core & Anomaly Verification Suite

1. Isolated Subprocess Re-Verification of the 256 KiB -> 512 KiB drop on
   high-entropy random data across 3 seeds and randomized order.
2. 64 MiB True Incremental Streaming (process_iter) with BoundedMemorySink
   (2 MiB cap) vs NullSink, proving O(1) bounded memory.
3. AdaptiveStreamProcessor vs Fixed 256 KiB / 512 KiB on repeated & random.
"""

from __future__ import annotations

import json
import os
import random
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from benchmarks.datasets import generate_datasets
from benchmarks.memory_profiler import measure_memory
from core.engine import (
    AdaptiveStreamProcessor,
    BoundedMemorySink,
    CodecType,
    NullSink,
    StatelessStreamProcessor,
)

RESULTS_DIR = REPO_ROOT / "benchmarks" / "results"
LOG_PATH = REPO_ROOT / "benchmarks" / "VANTAGE_V60_VALIDATION.log"

_ISOLATED_CHUNK_WORKER = """
import json, os, statistics, sys, time
import numpy as np
sys.path.insert(0, sys.argv[1])
chunk_kib = int(sys.argv[2])
seed = int(sys.argv[3])
trials = int(sys.argv[4])

from core.engine import CodecType, NullSink, StatelessStreamProcessor
rng = np.random.default_rng(seed)
data = rng.bytes(8 * 1024 * 1024)
proc = StatelessStreamProcessor(chunk_bytes=chunk_kib * 1024, codec=CodecType.LZ4_BLOCK, compute_checksum=False)

for _ in range(2):
    proc.process(data, NullSink())

lats = []
for _ in range(trials):
    t0 = time.perf_counter()
    proc.process(data, NullSink())
    lats.append((time.perf_counter() - t0) * 1000.0)

p50 = statistics.median(lats)
print(json.dumps({
    "chunk_kib": chunk_kib,
    "seed": seed,
    "p50_ms": round(p50, 3),
    "p50_mbs": round(8.0 / (p50 / 1000.0), 2),
    "trials_ms": [round(x, 3) for x in lats],
}))
"""


def run_isolated_chunk_anomaly_check() -> List[Dict[str, object]]:
    configs = [(sz, seed) for seed in (101, 202, 303) for sz in (256, 512, 1024)]
    random.Random(42).shuffle(configs)
    rows: List[Dict[str, object]] = []
    for sz, seed in configs:
        out = subprocess.check_output(
            [sys.executable, "-c", _ISOLATED_CHUNK_WORKER, str(REPO_ROOT), str(sz), str(seed), "10"],
            text=True,
        )
        rows.append(json.loads(out.strip()))
    rows.sort(key=lambda r: (r["chunk_kib"], r["seed"]))
    return rows


def run_64mib_streaming_benchmark() -> List[Dict[str, object]]:
    # Stream 256 x 256 KiB chunks = 64 MiB without holding 64 MiB in one buffer
    unit = (b"ID:999|Node:V-AI|Streaming:64MiB_Test|" * 6700)[: 256 * 1024]
    total_bytes = len(unit) * 256
    proc = StatelessStreamProcessor(
        chunk_bytes=256 * 1024, codec=CodecType.LZ4_BLOCK, compute_checksum=False
    )

    results: List[Dict[str, object]] = []
    for sink_name, sink_factory in [
        ("NullSink", NullSink),
        ("BoundedMemorySink_2MiB", lambda: BoundedMemorySink(max_bytes=2 * 1024 * 1024, drop_oldest=True)),
    ]:
        sink = sink_factory()
        with measure_memory() as mem:
            t0 = time.perf_counter()
            stats = proc.process_iter((unit for _ in range(256)), sink)
            dt = time.perf_counter() - t0
        results.append(
            {
                "sink": sink_name,
                "stream_mib": round(total_bytes / (1024 * 1024), 2),
                "output_mib": round(stats.output_bytes / (1024 * 1024), 3),
                "elapsed_ms": round(dt * 1000.0, 2),
                "throughput_mbs": round((total_bytes / (1024 * 1024)) / dt, 2),
                "py_peak_mb": round(mem.python_peak_alloc_mb, 3),
                "peak_rss_mb": round(mem.peak_rss_mb, 2),
                "retained_bytes": getattr(sink, "retained_bytes", 0),
            }
        )
    return results


def run_adaptive_benchmark() -> List[Dict[str, object]]:
    datasets = generate_datasets(8 * 1024 * 1024)
    adaptive = AdaptiveStreamProcessor(codec=CodecType.LZ4_BLOCK, compute_checksum=False)
    fixed_256 = StatelessStreamProcessor(chunk_bytes=256 * 1024, codec=CodecType.LZ4_BLOCK, compute_checksum=False)
    fixed_512 = StatelessStreamProcessor(chunk_bytes=512 * 1024, codec=CodecType.LZ4_BLOCK, compute_checksum=False)

    rows: List[Dict[str, object]] = []
    for ds_name in ("repeated", "random"):
        data = datasets[ds_name].data
        for name, proc in [("fixed_256k", fixed_256), ("fixed_512k", fixed_512), ("adaptive", adaptive)]:
            for _ in range(2):
                proc.process(data, NullSink())
            lats = []
            for _ in range(15):
                t0 = time.perf_counter()
                proc.process(data, NullSink())
                lats.append((time.perf_counter() - t0) * 1000.0)
            p50 = statistics.median(lats)
            rows.append(
                {
                    "dataset": ds_name,
                    "policy": name,
                    "p50_mbs": round(8.0 / (p50 / 1000.0), 2),
                    "p50_ms": round(p50, 3),
                }
            )
    return rows


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    anomaly = run_isolated_chunk_anomaly_check()
    stream64 = run_64mib_streaming_benchmark()
    adaptive = run_adaptive_benchmark()

    payload = {
        "isolated_random_chunk_check": anomaly,
        "streaming_64mib": stream64,
        "adaptive_vs_fixed": adaptive,
    }
    (RESULTS_DIR / "v60_validation.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "=" * 88,
        "VANTAGE V-AI v6.0 — PRODUCTION CORE & ANOMALY VERIFICATION LOG",
        "=" * 88,
        "1. ISOLATED SUBPROCESS RE-CHECK ON RANDOM (8 MiB, 3 seeds, randomized order, 10 trials)",
        f"{'Chunk(KiB)':>10} {'Seed':>6} {'P50(ms)':>9} {'MB/s(P50)':>10}",
        "-" * 88,
    ]
    for r in anomaly:
        lines.append(f"{r['chunk_kib']:>10} {r['seed']:>6} {r['p50_ms']:>9.3f} {r['p50_mbs']:>10.2f}")

    lines.extend([
        "-" * 88,
        "2. 64 MiB TRUE INCREMENTAL STREAMING (process_iter, 256 KiB chunks)",
        f"{'Sink':<24} {'Stream(MiB)':>11} {'MB/s':>9} {'PyPeak(MB)':>11} {'Retained(B)':>12}",
        "-" * 88,
    ])
    for r in stream64:
        lines.append(
            f"{r['sink']:<24} {r['stream_mib']:>11.2f} {r['throughput_mbs']:>9.2f} {r['py_peak_mb']:>11.3f} {r['retained_bytes']:>12}"
        )

    lines.extend([
        "-" * 88,
        "3. ADAPTIVE STREAM PROCESSOR vs FIXED CHUNK SIZE (8 MiB, 15 trials)",
        f"{'Dataset':<12} {'Policy':<14} {'MB/s(P50)':>10} {'P50(ms)':>9}",
        "-" * 88,
    ])
    for r in adaptive:
        lines.append(f"{r['dataset']:<12} {r['policy']:<14} {r['p50_mbs']:>10.2f} {r['p50_ms']:>9.3f}")
    lines.append("=" * 88)

    log_text = "\n".join(lines) + "\n"
    LOG_PATH.write_text(log_text, encoding="utf-8")
    print(log_text)


if __name__ == "__main__":
    main()
