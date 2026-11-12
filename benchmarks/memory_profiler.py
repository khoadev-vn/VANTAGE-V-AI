"""
Memory measurement utilities for Vantage V-AI benchmarks.

Tracks both Python-level heap allocations (tracemalloc) and OS-level process
working set / Peak RSS (psutil / resource) so memory claims include total
working memory rather than hiding pre-allocated buffers.
"""

from __future__ import annotations

import os
import tracemalloc
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator

import psutil


@dataclass
class MemorySnapshot:
    """Memory metrics captured around a benchmarked operation."""

    rss_before_mb: float
    rss_after_mb: float
    rss_delta_mb: float
    peak_rss_mb: float
    python_peak_alloc_mb: float


def _get_peak_rss_bytes(proc: psutil.Process) -> int:
    info = proc.memory_info()
    if hasattr(info, "peak_wset"):
        return int(info.peak_wset)
    try:
        import resource  # POSIX fallback

        ru_maxrss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # Linux reports KiB; macOS reports bytes
        return int(ru_maxrss * 1024) if os.uname().sysname == "Linux" else int(ru_maxrss)
    except Exception:
        return int(info.rss)


@contextmanager
def measure_memory() -> Iterator[MemorySnapshot]:
    """Context manager that populates a MemorySnapshot on exit."""
    proc = psutil.Process(os.getpid())
    mib = 1024 * 1024
    rss_before = proc.memory_info().rss / mib

    tracemalloc.start()
    snapshot = MemorySnapshot(
        rss_before_mb=rss_before,
        rss_after_mb=rss_before,
        rss_delta_mb=0.0,
        peak_rss_mb=rss_before,
        python_peak_alloc_mb=0.0,
    )
    try:
        yield snapshot
    finally:
        _, peak_py = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        rss_after = proc.memory_info().rss / mib
        peak_rss = _get_peak_rss_bytes(proc) / mib
        snapshot.rss_after_mb = rss_after
        snapshot.rss_delta_mb = rss_after - rss_before
        snapshot.peak_rss_mb = peak_rss
        snapshot.python_peak_alloc_mb = peak_py / mib
