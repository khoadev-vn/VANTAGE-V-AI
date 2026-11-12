"""
Dataset generators for Vantage V-AI scientific benchmarks.

Provides four distinct entropy/structure profiles so compression and streaming
engines are evaluated across best-case, realistic, and worst-case workloads.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class DatasetSpec:
    """Container for a generated benchmark payload and its metadata."""

    name: str
    description: str
    data: bytes
    record_count: int


def _build_repeated(target_bytes: int) -> DatasetSpec:
    rec = b"ID:999 | Node:V-AI_NODE | Data:Stateless_Logic_System|||"
    repeats = max(1, target_bytes // len(rec))
    data = rec * repeats
    return DatasetSpec(
        name="repeated",
        description="Low-entropy repeated telemetry records (best-case compressible)",
        data=data,
        record_count=repeats,
    )


def _build_random(target_bytes: int) -> DatasetSpec:
    data = os.urandom(target_bytes)
    return DatasetSpec(
        name="random",
        description="High-entropy cryptographic random bytes (incompressible worst-case)",
        data=data,
        record_count=max(1, target_bytes // 64),
    )


def _build_json_records(target_bytes: int) -> DatasetSpec:
    chunks = []
    total = 0
    idx = 0
    while total < target_bytes:
        row = json.dumps(
            {
                "req_id": f"req-{idx:07d}",
                "node": f"edge-node-{idx % 16:02d}",
                "model": "tiny-llama-1.1b-q4",
                "temp": round(0.1 + (idx % 9) * 0.1, 2),
                "max_tokens": 128 + (idx % 4) * 64,
                "prompt": f"Summarize telemetry batch {idx} for anomaly status code {idx % 503}.",
            },
            separators=(",", ":"),
        ).encode("utf-8") + b"\n"
        chunks.append(row)
        total += len(row)
        idx += 1
    return DatasetSpec(
        name="json_records",
        description="Structured JSON inference request records (medium-low entropy)",
        data=b"".join(chunks),
        record_count=idx,
    )


def _build_ai_corpus(target_bytes: int) -> DatasetSpec:
    templates = [
        "Explain how memory bandwidth limits autoregressive LLM decoding on edge CPUs.",
        "Compare zero-copy buffer slicing against object-per-record allocation in Python.",
        "Design a bounded-memory streaming pipeline with backpressure and CRC32 checks.",
        "Analyze P99 tail latency spikes caused by garbage collection pauses under load.",
        "Evaluate LZ4 frame compression ratios across structured JSON and natural language.",
    ]
    chunks = []
    total = 0
    idx = 0
    while total < target_bytes:
        line = f"[{idx:06d}] {templates[idx % len(templates)]} (shard={idx % 32})\n".encode("utf-8")
        chunks.append(line)
        total += len(line)
        idx += 1
    return DatasetSpec(
        name="ai_corpus",
        description="Natural-language AI prompt corpus (realistic text entropy)",
        data=b"".join(chunks),
        record_count=idx,
    )


def generate_datasets(target_bytes: int = 8 * 1024 * 1024) -> Dict[str, DatasetSpec]:
    """Generate all four benchmark datasets of approximately ``target_bytes`` each."""
    return {
        "repeated": _build_repeated(target_bytes),
        "json_records": _build_json_records(target_bytes),
        "ai_corpus": _build_ai_corpus(target_bytes),
        "random": _build_random(target_bytes),
    }
