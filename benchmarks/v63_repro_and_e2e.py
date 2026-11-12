"""
Vantage V-AI v6.3 — Long-Duration RSS Time-Series, P25/P50/P75 Adaptive Ablation,
and End-to-End AI Preprocessing Pipeline (Vantage -> tiktoken).
"""

from __future__ import annotations

import json
import os
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, List

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from benchmarks.datasets import generate_datasets
from core.engine import (
    AdaptiveStreamProcessor,
    BoundedMemorySink,
    CodecType,
    FileSink,
    NullSink,
    StatelessStreamProcessor,
)

RESULTS_DIR = REPO_ROOT / "benchmarks" / "results"
LOG_PATH = REPO_ROOT / "benchmarks" / "VANTAGE_V63_VALIDATION.log"

_RSS_TIMESERIES_WORKER = """
import json, os, psutil, sys, time
sys.path.insert(0, sys.argv[1])
from core.engine import BoundedMemorySink, CodecType, StatelessStreamProcessor

proc_os = psutil.Process(os.getpid())
mib = 1024 * 1024
chunk_bytes = 256 * 1024
chunks_per_gib = (1024 * mib) // chunk_bytes
unit = (b"ID:999|Node:V-AI|TimeSeries:16GiB_RSS|" * 6700)[:chunk_bytes]

engine = StatelessStreamProcessor(chunk_bytes=chunk_bytes, codec=CodecType.LZ4_BLOCK, compute_checksum=False)
sink = BoundedMemorySink(max_bytes=2 * mib, drop_oldest=True)

samples = []
t_start = time.perf_counter()
for gib in range(1, 17):
    engine.process_iter((unit for _ in range(chunks_per_gib)), sink)
    info = proc_os.memory_info()
    rss_mb = info.rss / mib
    peak_mb = (info.peak_wset if hasattr(info, "peak_wset") else info.rss) / mib
    samples.append({
        "gib_processed": gib,
        "elapsed_s": round(time.perf_counter() - t_start, 3),
        "rss_mb": round(rss_mb, 2),
        "peak_rss_mb": round(peak_mb, 2),
    })
print(json.dumps(samples))
"""

_E2E_WORKER = """
import json, os, psutil, sys, time, tracemalloc
import tiktoken
sys.path.insert(0, sys.argv[1])
mode = sys.argv[2]
path = sys.argv[3]

from core.engine import CodecType, decompress_stream, read_framed_stream

proc_os = psutil.Process(os.getpid())
mib = 1024 * 1024
enc = tiktoken.get_encoding("cl100k_base")

tracemalloc.start()
t0 = time.perf_counter()
total_tokens = 0
if mode == "full_ram_baseline":
    with open(path, "rb") as f:
        text = f.read().decode("utf-8")
    total_tokens = len(enc.encode(text))
elif mode == "vantage_bounded_stream":
    with open(path, "rb") as f:
        for chunk in decompress_stream(read_framed_stream(f), codec=CodecType.LZ4_BLOCK):
            total_tokens += len(enc.encode(chunk.decode("utf-8")))
else:
    raise ValueError(mode)
dt = time.perf_counter() - t0
_, py_peak = tracemalloc.get_traced_memory()
tracemalloc.stop()

info = proc_os.memory_info()
peak_rss = (info.peak_wset if hasattr(info, "peak_wset") else info.rss) / mib
print(json.dumps({
    "mode": mode,
    "total_tokens": total_tokens,
    "elapsed_ms": round(dt * 1000.0, 2),
    "tokens_per_sec": round(total_tokens / dt, 1),
    "py_peak_mb": round(py_peak / mib, 3),
    "peak_rss_mb": round(peak_rss, 2),
}))
"""


def run_rss_timeseries() -> List[Dict[str, object]]:
    out = subprocess.check_output(
        [sys.executable, "-c", _RSS_TIMESERIES_WORKER, str(REPO_ROOT)],
        text=True,
    )
    return json.loads(out.strip())


def run_adaptive_full_ablation() -> List[Dict[str, object]]:
    rng = np.random.default_rng(2026)
    comp_1m = (b"VANTAGE_LOW_ENTROPY_STREAM_RECORD_" * 32768)[: 1024 * 1024]
    rand_1m = rng.bytes(1024 * 1024)
    datasets = {
        "repeated": comp_1m * 16,
        "random": rng.bytes(16 * 1024 * 1024),
        "low_then_high": comp_1m[: 64 * 1024] + rng.bytes(16 * 1024 * 1024 - 64 * 1024),
        "alternating": b"".join(comp_1m if i % 2 == 0 else rand_1m for i in range(16)),
    }
    policies = [
        ("fixed_256k", StatelessStreamProcessor(chunk_bytes=256 * 1024, codec=CodecType.LZ4_BLOCK, compute_checksum=False)),
        ("adapt_h1", AdaptiveStreamProcessor(hysteresis_chunks=1, codec=CodecType.LZ4_BLOCK, compute_checksum=False)),
        ("adapt_h3", AdaptiveStreamProcessor(hysteresis_chunks=3, codec=CodecType.LZ4_BLOCK, compute_checksum=False)),
    ]

    rows: List[Dict[str, object]] = []
    for ds_name, data in datasets.items():
        mib = len(data) / (1024 * 1024)
        for p_name, proc in policies:
            for _ in range(2):
                proc.process(data, NullSink())
            lats = []
            switches = 0
            for _ in range(15):
                t0 = time.perf_counter()
                stats = proc.process(data, NullSink())
                lats.append((time.perf_counter() - t0) * 1000.0)
                switches = stats.switch_count
            p25, p50, p75 = np.percentile(lats, [25, 50, 75])
            rows.append(
                {
                    "dataset": ds_name,
                    "policy": p_name,
                    "p50_mbs": round(mib / (p50 / 1000.0), 2),
                    "p25_ms": round(float(p25), 3),
                    "p50_ms": round(float(p50), 3),
                    "p75_ms": round(float(p75), 3),
                    "std_ms": round(statistics.stdev(lats), 3),
                    "switches": switches,
                }
            )
    return rows


def run_e2e_ai_pipeline() -> List[Dict[str, object]]:
    # 16 MiB of ai_corpus (~4M real tokens)
    ds = generate_datasets(16 * 1024 * 1024)["ai_corpus"]
    proc = StatelessStreamProcessor(chunk_bytes=256 * 1024, codec=CodecType.LZ4_BLOCK, compute_checksum=False)

    with tempfile.TemporaryDirectory() as tmp:
        raw_path = Path(tmp) / "corpus.txt"
        vnt_path = Path(tmp) / "corpus.vnt"
        raw_path.write_bytes(ds.data)
        with vnt_path.open("wb") as f:
            proc.process(ds.data, FileSink(f, framed=True))

        results: List[Dict[str, object]] = []
        for mode, p in [("full_ram_baseline", raw_path), ("vantage_bounded_stream", vnt_path)]:
            out = subprocess.check_output(
                [sys.executable, "-c", _E2E_WORKER, str(REPO_ROOT), mode, str(p)],
                text=True,
            )
            row = json.loads(out.strip())
            row["file_size_mb"] = round(p.stat().st_size / (1024 * 1024), 3)
            results.append(row)
        return results


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = run_rss_timeseries()
    abl = run_adaptive_full_ablation()
    e2e = run_e2e_ai_pipeline()

    payload = {
        "rss_timeseries_16gib": ts,
        "adaptive_ablation_quantiles": abl,
        "e2e_ai_pipeline": e2e,
    }
    (RESULTS_DIR / "v63_validation.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "=" * 94,
        "VANTAGE V-AI v6.3 — RSS TIME-SERIES, P25/P50/P75 ABLATION & E2E AI PIPELINE",
        "=" * 94,
        "1. LONG-DURATION RSS TIME-SERIES (16 GiB continuous stream, BoundedMemorySink 2 MiB)",
        "   GiB:  " + "  ".join(f"{r['gib_processed']:>5}" for r in ts),
        "   RSS:  " + "  ".join(f"{r['rss_mb']:>5.1f}" for r in ts) + "  (MB)",
        "   Peak: " + "  ".join(f"{r['peak_rss_mb']:>5.1f}" for r in ts) + "  (MB)",
        "-" * 94,
        "2. ADAPTIVE ABLATION WITH P25/P50/P75, STDDEV & SWITCH COUNT (16 MiB, 15 trials)",
        f"{'Dataset':<14} {'Policy':<11} {'MB/s(P50)':>10} {'P25(ms)':>8} {'P50(ms)':>8} {'P75(ms)':>8} {'Std(ms)':>8} {'Switches':>9}",
        "-" * 94,
    ]
    for r in abl:
        lines.append(
            f"{r['dataset']:<14} {r['policy']:<11} {r['p50_mbs']:>10.2f} "
            f"{r['p25_ms']:>8.3f} {r['p50_ms']:>8.3f} {r['p75_ms']:>8.3f} {r['std_ms']:>8.3f} {r['switches']:>9}"
        )

    lines.extend([
        "-" * 94,
        "3. END-TO-END AI PREPROCESSING PIPELINE (16 MiB ai_corpus -> tiktoken cl100k_base)",
        f"{'Mode':<24} {'File(MB)':>9} {'Tokens':>10} {'Time(ms)':>10} {'Tokens/s':>12} {'PyPeak(MB)':>11} {'PeakRSS(MB)':>12}",
        "-" * 94,
    ])
    for r in e2e:
        lines.append(
            f"{r['mode']:<24} {r['file_size_mb']:>9.3f} {r['total_tokens']:>10} "
            f"{r['elapsed_ms']:>10.2f} {r['tokens_per_sec']:>12,.1f} {r['py_peak_mb']:>11.3f} {r['peak_rss_mb']:>12.2f}"
        )
    lines.append("=" * 94)

    log_text = "\n".join(lines) + "\n"
    LOG_PATH.write_text(log_text, encoding="utf-8")
    print(log_text)


if __name__ == "__main__":
    main()
