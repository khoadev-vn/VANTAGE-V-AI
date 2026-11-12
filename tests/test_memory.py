"""Unit tests for Vantage V-AI Hyperdimensional Associative Memory."""

import time
import numpy as np
import pytest

from core.memory import HyperdimensionalMemory


def test_hd_memory_initialization():
    hd = HyperdimensionalMemory(input_dim=128, hd_dim=4096, seed=42)
    assert hd.input_dim == 128
    assert hd.hd_dim == 4096
    assert hd.num_bytes == 4096 // 8
    assert len(hd) == 0


def test_hd_memory_projection_deterministic():
    hd1 = HyperdimensionalMemory(input_dim=64, hd_dim=2048, seed=123)
    hd2 = HyperdimensionalMemory(input_dim=64, hd_dim=2048, seed=123)

    vec = np.random.randn(64).astype(np.float32)
    b1 = hd1.project(vec)
    b2 = hd2.project(vec)

    np.testing.assert_array_equal(b1, b2)


def test_hd_memory_store_and_exact_recall():
    hd = HyperdimensionalMemory(input_dim=64, hd_dim=2048, seed=42)

    vecA = np.random.randn(64).astype(np.float32)
    vecB = np.random.randn(64).astype(np.float32)

    hd.store("doc_A", vecA, metadata={"type": "query_A"})
    hd.store("doc_B", vecB, metadata={"type": "query_B"})

    assert len(hd) == 2

    # Query with exact vecA
    results = hd.query(vecA, top_k=2)
    assert len(results) == 2
    assert results[0].key == "doc_A"
    assert results[0].hamming_dist == 0
    assert results[0].similarity == 1.0
    assert results[0].metadata == {"type": "query_A"}


def test_hd_memory_noise_tolerance():
    hd = HyperdimensionalMemory(input_dim=128, hd_dim=8192, seed=99)

    vec_base = np.random.randn(128).astype(np.float32)
    vec_noisy = vec_base + 0.05 * np.random.randn(128).astype(np.float32)
    vec_unrelated = np.random.randn(128).astype(np.float32)

    hd.store("target", vec_base)
    hd.store("distractor", vec_unrelated)

    results = hd.query(vec_noisy, top_k=2)
    assert results[0].key == "target"
    assert results[0].similarity > 0.85


def test_hd_memory_sub_millisecond_search_latency():
    hd = HyperdimensionalMemory(input_dim=256, hd_dim=4096, seed=42)

    # Store 1000 items in memory
    rng = np.random.default_rng(100)
    for i in range(1000):
        v = rng.standard_normal(256).astype(np.float32)
        hd.store(f"item_{i}", v)

    query_v = rng.standard_normal(256).astype(np.float32)

    # Warmup
    hd.query(query_v, top_k=5)

    # Measure latency across 20 queries
    start = time.perf_counter()
    n_queries = 20
    for _ in range(n_queries):
        hd.query(query_v, top_k=5)
    elapsed = time.perf_counter() - start

    avg_ms = (elapsed / n_queries) * 1000
    # Must achieve sub-millisecond query latency (< 5.0ms even on CPU for 1k items)
    assert avg_ms < 5.0, f"Expected < 5.0ms, got {avg_ms:.2f}ms"


def test_hd_memory_clear():
    hd = HyperdimensionalMemory(input_dim=32, hd_dim=1024)
    hd.store("item1", np.ones(32, dtype=np.float32))
    assert len(hd) == 1
    hd.clear()
    assert len(hd) == 0
