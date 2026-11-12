"""
Vantage V-AI v6.2 — Memory Stress Testing (4-16 GiB), Backpressure RSS Proof,
and Adaptive Policy Ablation (Hysteresis & Adversarial Entropy Transitions).
"""

from __future__ import annotations

import json
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
    CodecType,
    NullSink,
    StatelessStreamProcessor,
)

RESULTS_DIR = REPO_ROOT / "benchmarks" / "results"
LOG_PATH = REPO_ROOT / "benchmarks" / "VANTAGE_V62_STRESS_ABLATION.log"

_STRESS_WORKER = """
import json, os, psutil, sys, time, tracemalloc
sys.path.insert(0, sys.argv[1])
total_gib = int(sys.argv[2])
sink_mode = sys.argv[3]

from core.engine import BoundedMemorySink, CallbackSink, CodecType, NullSink, StatelessStreamProcessor

proc_os = psutil.Process(os.getpid())
mib = 1024 * 1024
chunk_bytes = 256 * 1024
num_chunks = (total_gib * 1024 * mib) // chunk_bytes

unit = (b"ID:999|Node:V-AI|StressTest:Bounded_Streaming|" * 6000)[:chunk_bytes]
engine = StatelessStreamProcessor(chunk_bytes=chunk_bytes, codec=CodecType.LZ4_BLOCK, compute_checksum=False)
sink = NullSink() if sink_mode == "null" else BoundedMemorySink(max_bytes=2 * mib, drop_oldest=True)

tracemalloc.start()
t0 = time.perf_counter()
stats = engine.process_iter((unit for _ in range(num_chunks)), sink)
dt = time.perf_counter() - t0
_, py_peak = tracemalloc.get_traced_memory()
tracemalloc.stop()

info = proc_os.memory_info()
peak_rss = (info.peak_wset if hasattr(info, "peak_wset") else info.rss) / mib
print(json.dumps({
    "stream_gib": total_gib,
    "sink_mode": sink_mode,
    "throughput_mbs": round((stats.input_bytes / mib) / dt, 2),
    "elapsed_s": round(dt, 3),
    "peak_rss_mb": round(peak_rss, 2),
    "py_peak_mb": round(py_peak / mib, 3),
    "retained_bytes": getattr(sink, "retained_bytes", 0),
}))
"""

_BP_WORKER = """
import json, os, psutil, sys, time, tracemalloc
sys.path.insert(0, sys.argv[1])
sink_mode = sys.argv[2]

from core.engine import BoundedMemorySink, CallbackSink, CodecType, NullSink, StatelessStreamProcessor

proc_os = psutil.Process(os.getpid())
mib = 1024 * 1024
chunk_bytes = 256 * 1024
num_chunks = 128  # 32 MiB stream
unit = os.urandom(chunk_bytes)

engine = StatelessStreamProcessor(chunk_bytes=chunk_bytes, codec=CodecType.LZ4_BLOCK, compute_checksum=False)
if sink_mode == "null":
    sink = NullSink()
elif sink_mode == "bounded_2mib":
    sink = BoundedMemorySink(max_bytes=2 * mib, drop_oldest=True)
else:
    sink = CallbackSink(lambda _f: time.sleep(0.0001))

tracemalloc.start()
t0 = time.perf_counter()
stats = engine.process_iter((unit for _ in range(num_chunks)), sink)
dt = time.perf_counter() - t0
_, py_peak = tracemalloc.get_traced_memory()
tracemalloc.stop()

info = proc_os.memory_info()
peak_rss = (info.peak_wset if hasattr(info, "peak_wset") else info.rss) / mib
print(json.dumps({
    "sink": sink_mode,
    "stream_mib": 32,
    "elapsed_s": round(dt, 3),
    "throughput_mbs": round((stats.input_bytes / mib) / dt, 2),
    "peak_rss_mb": round(peak_rss, 2),
    "py_peak_mb": round(py_peak / mib, 3),
}))
"""


def run_stress_suite() -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for gib, mode in [(4, "bounded"), (8, "bounded"), (16, "bounded"), (16, "null")]:
        out = subprocess.check_output(
            [sys.executable, "-c", _STRESS_WORKER, str(REPO_ROOT), str(gib), mode],
            text=True,
        )
        rows.append(json.loads(out.strip()))
    return rows


def run_backpressure_rss_suite() -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for mode in ("null", "bounded_2mib", "slow_100us"):
        out = subprocess.check_output(
            [sys.executable, "-c", _BP_WORKER, str(REPO_ROOT), mode],
            text=True,
        )
        rows.append(json.loads(out.strip()))
    return rows


def _build_ablation_datasets() -> Dict[str, bytes]:
    rng = np.random.default_rng(2026)
    comp_1m = (b"VANTAGE_LOW_ENTROPY_STREAM_RECORD_" * 32768)[: 1024 * 1024]
    rand_1m = rng.bytes(1024 * 1024)

    # Adversarial jitter: short 192 KiB - 320 KiB bursts switching entropy continuously
    jitter_parts: List[bytes] = []
    total = 0
    step = 0
    while total < 16 * 1024 * 1024:
        sz = (192 + (step % 3) * 64) * 1024
        if step % 2 == 0:
            jitter_parts.append(comp_1m[:sz])
        else:
            jitter_parts.append(rng.bytes(sz))
        total += sz
        step += 1

    return {
        "repeated": comp_1m * 16,
        "random": rng.bytes(16 * 1024 * 1024),
        "low_then_high": comp_1m[: 64 * 1024] + rng.bytes(16 * 1024 * 1024 - 64 * 1024),
        "high_then_low": rng.bytes(64 * 1024) + (comp_1m * 16)[: 16 * 1024 * 1024 - 64 * 1024],
        "alternating": b"".join(comp_1m if i % 2 == 0 else rand_1m for i in range(16)),
        "adversarial_jitter": b"".join(jitter_parts)[: 16 * 1024 * 1024],
    }


def run_adaptive_ablation() -> List[Dict[str, object]]:
    datasets = _build_ablation_datasets()
    policies = [
        ("fixed_128k", StatelessStreamProcessor(chunk_bytes=128 * 1024, codec=CodecType.LZ4_BLOCK, compute_checksum=False)),
        ("fixed_256k", StatelessStreamProcessor(chunk_bytes=256 * 1024, codec=CodecType.LZ4_BLOCK, compute_checksum=False)),
        ("fixed_512k", StatelessStreamProcessor(chunk_bytes=512 * 1024, codec=CodecType.LZ4_BLOCK, compute_checksum=False)),
        ("adaptive_h1", AdaptiveStreamProcessor(hysteresis_chunks=1, codec=CodecType.LZ4_BLOCK, compute_checksum=False)),
        ("adaptive_h3", AdaptiveStreamProcessor(hysteresis_chunks=3, codec=CodecType.LZ4_BLOCK, compute_checksum=False)),
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
                }
            )
    return rows


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stress = run_stress_suite()
    bp = run_backpressure_rss_suite()
    ablation = run_adaptive_ablation()

    payload = {
        "stress_4_to_16_gib": stress,
        "backpressure_rss": bp,
        "adaptive_ablation": ablation,
    }
    (RESULTS_DIR / "v62_stress_ablation.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "=" * 94,
        "VANTAGE V-AI v6.2 — MEMORY STRESS (4-16 GiB), BACKPRESSURE RSS & ADAPTIVE ABLATION",
        "=" * 94,
        "1. EXTREME BOUNDED-MEMORY STRESS TEST (Isolated subprocesses, 4 GiB -> 16 GiB)",
        f"{'Stream':<8} {'Sink':<10} {'MB/s':>10} {'Time(s)':>9} {'PeakRSS(MB)':>12} {'PyPeak(MB)':>11} {'Retained(B)':>12}",
        "-" * 94,
    ]
    for r in stress:
        lines.append(
            f"{r['stream_gib']} GiB   {r['sink_mode']:<10} {r['throughput_mbs']:>10.2f} {r['elapsed_s']:>9.3f} "
            f"{r['peak_rss_mb']:>12.2f} {r['py_peak_mb']:>11.3f} {r['retained_bytes']:>12}"
        )

    lines.extend([
        "-" * 94,
        "2. BACKPRESSURE + BOUNDED MEMORY COMBINED PROOF (Isolated subprocess, 32 MiB random)",
        f"{'Sink':<16} {'MB/s':>10} {'Time(s)':>9} {'PeakRSS(MB)':>12} {'PyPeak(MB)':>11}",
        "-" * 94,
    ])
    for r in bp:
        lines.append(
            f"{r['sink']:<16} {r['throughput_mbs']:>10.2f} {r['elapsed_s']:>9.3f} {r['peak_rss_mb']:>12.2f} {r['py_peak_mb']:>11.3f}"
        )

    lines.extend([
        "-" * 94,
        "3. ADAPTIVE POLICY ABLATION & ADVERSARIAL JITTER (16 MiB, 10 trials, MB/s P50)",
        f"{'Dataset':<19} {'fixed_128k':>11} {'fixed_256k':>11} {'fixed_512k':>11} {'adapt_h1':>11} {'adapt_h3':>11}",
        "-" * 94,
    ])
    for ds_name in ("repeated", "random", "low_then_high", "high_then_low", "alternating", "adversarial_jitter"):
        vals = [r["p50_mbs"] for r in ablation if r["dataset"] == ds_name]
        lines.append(f"{ds_name:<19} " + " ".join(f"{v:>11.1f}" for v in vals))
    lines.append("=" * 94)

    log_text = "\n".join(lines) + "\n"
    LOG_PATH.write_text(log_text, encoding="utf-8")
    print(log_text)


if __name__ == "__main__":
    main()
