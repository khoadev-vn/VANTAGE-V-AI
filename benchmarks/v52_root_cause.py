"""
Vantage V-AI v5.2 — Root-Cause Validation Suite

Implements the three controlled experiments required to isolate the source
of chunked throughput gains:
1. Chunking Parity: plain Python memoryview+chunked lz4.block vs Vantage
2. Chunk-Size Sweep: 64 KiB .. 8 MiB across all 4 entropy datasets
3. CRC32 Cost Isolation: crc32_only vs compress_only vs compress_plus_crc32
"""

from __future__ import annotations

import csv
import gc
import json
import statistics
import sys
import time
import zlib
from pathlib import Path
from typing import Callable, Dict, List

import lz4.block
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from benchmarks.datasets import generate_datasets
from benchmarks.memory_profiler import measure_memory
from core.engine import CodecType, NullSink, StatelessStreamProcessor

RESULTS_DIR = REPO_ROOT / "benchmarks" / "results"
LOG_PATH = REPO_ROOT / "benchmarks" / "VANTAGE_V52_ROOT_CAUSE.log"

DATASET_BYTES = 8 * 1024 * 1024
TRIALS = 15
WARMUP = 2


def _bench(fn: Callable[[], int], input_bytes: int) -> Dict[str, float]:
    for _ in range(WARMUP):
        fn()
    latencies_ms: List[float] = []
    py_peaks: List[float] = []
    for _ in range(TRIALS):
        gc.collect()
        with measure_memory() as mem:
            t0 = time.perf_counter()
            fn()
            dt = time.perf_counter() - t0
        latencies_ms.append(dt * 1000.0)
        py_peaks.append(mem.python_peak_alloc_mb)

    p50 = statistics.median(latencies_ms)
    iqr = float(np.percentile(latencies_ms, 75) - np.percentile(latencies_ms, 25))
    p95 = float(np.percentile(latencies_ms, 95))
    mbs = (input_bytes / (1024 * 1024)) / (p50 / 1000.0)
    return {
        "p50_mbs": round(mbs, 2),
        "p50_ms": round(p50, 3),
        "iqr_ms": round(iqr, 3),
        "p95_ms": round(p95, 3),
        "py_peak_mb": round(statistics.median(py_peaks), 3),
    }


def _plain_python_chunked(data: bytes, chunk_bytes: int = 256 * 1024) -> int:
    view = memoryview(data)
    total = 0
    for offset in range(0, len(view), chunk_bytes):
        out = lz4.block.compress(view[offset : offset + chunk_bytes], store_size=True)
        total += len(out)
    return total


def _crc32_only_chunked(data: bytes, chunk_bytes: int = 256 * 1024) -> int:
    view = memoryview(data)
    crc = 0
    for offset in range(0, len(view), chunk_bytes):
        crc = zlib.crc32(view[offset : offset + chunk_bytes], crc)
    return crc & 0xFFFFFFFF


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    datasets = generate_datasets(DATASET_BYTES)

    # Experiment 1: Chunking Parity (256 KiB, no CRC32)
    vantage_raw = StatelessStreamProcessor(
        chunk_bytes=256 * 1024, codec=CodecType.LZ4_BLOCK, compute_checksum=False
    )
    exp1: List[Dict[str, object]] = []
    for ds_name in ("repeated", "json_records", "ai_corpus", "random"):
        ds = datasets[ds_name]
        r_plain = _bench(lambda d=ds.data: _plain_python_chunked(d, 256 * 1024), len(ds.data))
        r_vantage = _bench(lambda d=ds.data: vantage_raw.process(d, NullSink()).output_bytes, len(ds.data))
        exp1.append({"dataset": ds_name, "impl": "python_lz4_block_chunked", **r_plain})
        exp1.append({"dataset": ds_name, "impl": "vantage_lz4_block_raw", **r_vantage})

    # Experiment 2: Chunk-Size Sweep (64 KiB .. 8 MiB)
    sweep_sizes = [
        64 * 1024,
        128 * 1024,
        256 * 1024,
        512 * 1024,
        1024 * 1024,
        2 * 1024 * 1024,
        8 * 1024 * 1024,
    ]
    exp2: List[Dict[str, object]] = []
    for ds_name in ("repeated", "json_records", "ai_corpus", "random"):
        ds = datasets[ds_name]
        for sz in sweep_sizes:
            proc = StatelessStreamProcessor(
                chunk_bytes=sz, codec=CodecType.LZ4_BLOCK, compute_checksum=False
            )
            res = _bench(lambda d=ds.data, p=proc: p.process(d, NullSink()).output_bytes, len(ds.data))
            exp2.append({"dataset": ds_name, "chunk_kib": sz // 1024, **res})

    # Experiment 3: CRC32 Cost Isolation (256 KiB chunks)
    proc_no_crc = StatelessStreamProcessor(
        chunk_bytes=256 * 1024, codec=CodecType.LZ4_BLOCK, compute_checksum=False
    )
    proc_with_crc = StatelessStreamProcessor(
        chunk_bytes=256 * 1024, codec=CodecType.LZ4_BLOCK, compute_checksum=True
    )
    exp3: List[Dict[str, object]] = []
    for ds_name in ("repeated", "json_records", "ai_corpus", "random"):
        ds = datasets[ds_name]
        r_crc = _bench(lambda d=ds.data: _crc32_only_chunked(d, 256 * 1024), len(ds.data))
        r_comp = _bench(lambda d=ds.data: proc_no_crc.process(d, NullSink()).output_bytes, len(ds.data))
        r_both = _bench(lambda d=ds.data: proc_with_crc.process(d, NullSink()).output_bytes, len(ds.data))
        exp3.append({"dataset": ds_name, "mode": "crc32_only", **r_crc})
        exp3.append({"dataset": ds_name, "mode": "compress_only", **r_comp})
        exp3.append({"dataset": ds_name, "mode": "compress_plus_crc32", **r_both})

    payload = {"exp1_chunking_parity": exp1, "exp2_chunk_sweep": exp2, "exp3_crc_isolation": exp3}
    (RESULTS_DIR / "v52_root_cause.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    with (RESULTS_DIR / "v52_chunk_sweep.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(exp2[0].keys()))
        writer.writeheader()
        writer.writerows(exp2)

    # Render readable log
    lines = [
        "=" * 88,
        "VANTAGE V-AI v5.2 — ROOT-CAUSE VALIDATION LOG (15 trials, 8 MiB/dataset)",
        "=" * 88,
        "EXPERIMENT 1: CHUNKING PARITY (256 KiB memoryview + lz4.block, no CRC32)",
        f"{'Dataset':<13} {'Implementation':<26} {'MB/s(P50)':>10} {'P50(ms)':>9} {'IQR(ms)':>9} {'PyPeak(MB)':>11}",
        "-" * 88,
    ]
    for r in exp1:
        lines.append(
            f"{r['dataset']:<13} {r['impl']:<26} {r['p50_mbs']:>10.2f} {r['p50_ms']:>9.3f} {r['iqr_ms']:>9.3f} {r['py_peak_mb']:>11.3f}"
        )

    lines.extend([
        "-" * 88,
        "EXPERIMENT 2: CHUNK-SIZE SWEEP (MB/s P50, lz4.block, no CRC32)",
        f"{'Dataset':<13} {'64K':>8} {'128K':>8} {'256K':>8} {'512K':>8} {'1024K':>8} {'2048K':>8} {'8192K':>8}",
        "-" * 88,
    ])
    for ds_name in ("repeated", "json_records", "ai_corpus", "random"):
        vals = [r["p50_mbs"] for r in exp2 if r["dataset"] == ds_name]
        lines.append(f"{ds_name:<13} " + " ".join(f"{v:>8.1f}" for v in vals))

    lines.extend([
        "-" * 88,
        "EXPERIMENT 3: CRC32 COST ISOLATION (256 KiB chunks)",
        f"{'Dataset':<13} {'Mode':<22} {'MB/s(P50)':>10} {'P50(ms)':>9} {'IQR(ms)':>9}",
        "-" * 88,
    ])
    for r in exp3:
        lines.append(
            f"{r['dataset']:<13} {r['mode']:<22} {r['p50_mbs']:>10.2f} {r['p50_ms']:>9.3f} {r['iqr_ms']:>9.3f}"
        )
    lines.append("=" * 88)

    log_text = "\n".join(lines) + "\n"
    LOG_PATH.write_text(log_text, encoding="utf-8")
    print(log_text)


if __name__ == "__main__":
    main()
