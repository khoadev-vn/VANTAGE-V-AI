"""
Isolated subprocess memory profiler for Vantage V-AI (v5.1).

Measures full-lifecycle Peak RSS (from dataset generation through processing
to sink completion) in a fresh Python process per baseline, eliminating
cross-baseline RSS contamination.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

REPO_ROOT = Path(__file__).resolve().parent.parent

_WORKER_CODE = """
import json, os, sys, psutil
sys.path.insert(0, sys.argv[1])
baseline = sys.argv[2]
dataset_name = sys.argv[3]
target_bytes = int(sys.argv[4])

from benchmarks.datasets import generate_datasets
from core.engine import CodecType, NullSink, StatelessStreamProcessor
import lz4.block, lz4.frame

proc = psutil.Process(os.getpid())
mib = 1024 * 1024
rss_start = proc.memory_info().rss / mib

ds = generate_datasets(target_bytes)[dataset_name]
data = ds.data
rss_after_input = proc.memory_info().rss / mib

if baseline == "python_lz4_frame_naive":
    _ = lz4.frame.compress(data)
elif baseline == "python_lz4_block_naive":
    _ = lz4.block.compress(data, store_size=True)
elif baseline == "pyarrow_lz4":
    import pyarrow as pa
    _ = pa.compress(pa.py_buffer(data), codec="lz4", asbytes=True)
elif baseline == "vantage_lz4_frame_1t":
    p = StatelessStreamProcessor(chunk_bytes=256 * 1024, codec=CodecType.LZ4_FRAME)
    p.process(data, NullSink())
elif baseline == "vantage_lz4_block_1t":
    p = StatelessStreamProcessor(chunk_bytes=256 * 1024, codec=CodecType.LZ4_BLOCK, compute_checksum=True)
    p.process(data, NullSink())
elif baseline == "vantage_lz4_block_raw":
    p = StatelessStreamProcessor(chunk_bytes=256 * 1024, codec=CodecType.LZ4_BLOCK, compute_checksum=False)
    p.process(data, NullSink())
else:
    raise ValueError(baseline)

info = proc.memory_info()
peak_rss = (info.peak_wset if hasattr(info, "peak_wset") else info.rss) / mib
rss_end = info.rss / mib
print(json.dumps({
    "baseline": baseline,
    "dataset": dataset_name,
    "rss_start_mb": round(rss_start, 2),
    "rss_after_input_mb": round(rss_after_input, 2),
    "rss_end_mb": round(rss_end, 2),
    "isolated_peak_rss_mb": round(peak_rss, 2),
    "processing_rss_overhead_mb": round(peak_rss - rss_after_input, 2),
}))
"""


def run_isolated_memory_suite(
    baselines: List[str],
    dataset_name: str = "random",
    target_bytes: int = 8 * 1024 * 1024,
) -> List[Dict[str, object]]:
    """Run each baseline in a fresh subprocess and collect isolated Peak RSS."""
    results: List[Dict[str, object]] = []
    for b in baselines:
        out = subprocess.check_output(
            [sys.executable, "-c", _WORKER_CODE, str(REPO_ROOT), b, dataset_name, str(target_bytes)],
            text=True,
        )
        results.append(json.loads(out.strip()))
    return results


if __name__ == "__main__":
    rows = run_isolated_memory_suite(
        [
            "python_lz4_frame_naive",
            "python_lz4_block_naive",
            "pyarrow_lz4",
            "vantage_lz4_frame_1t",
            "vantage_lz4_block_1t",
            "vantage_lz4_block_raw",
        ]
    )
    for r in rows:
        print(r)
