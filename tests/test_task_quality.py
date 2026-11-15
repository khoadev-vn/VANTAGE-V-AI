"""Unit tests for Task Quality & Context Trade-Off Suite (v8.2.0)."""

import pytest
from benchmarks.task_quality_benchmark import (
    generate_synthetic_haystack,
    evaluate_needle_retrieval,
    compute_streaming_perplexity_profile,
    run_task_quality_suite,
)


def test_synthetic_haystack_generation():
    haystack, query = generate_synthetic_haystack(
        total_tokens=512,
        needle_text="The secret code is OMEGA-99",
        needle_depth_pct=0.25,
    )
    assert "OMEGA-99" in haystack
    assert "target key" in query
    assert len(haystack.split()) > 30


def test_needle_retrieval_boundary():
    needle = "ALPHA-TEST-123"
    short_haystack = f"Header info. Important notice: {needle} followed by recent tokens."
    
    # Within sliding window
    res_within = evaluate_needle_retrieval(short_haystack, needle, window_size=512, sink_tokens=4, use_sinks=True)
    assert res_within["is_retained"] is True
    assert res_within["coherence_score"] == 1.0


def test_streaming_perplexity_stability():
    profile = compute_streaming_perplexity_profile(seq_length=2048, window_size=256, sink_tokens=4)
    assert profile["vantage_stability_preserved"] is True
    assert profile["softmax_collapse_observed_without_sinks"] is True
    # Verify Vantage bounded PPL remains within normal limits (< 15.0)
    assert all(ppl < 15.0 for ppl in profile["ppl_vantage_bounded_sinks"])
    # Verify sink-less sliding window diverges exponentially (> 100.0)
    assert profile["ppl_sliding_window_no_sinks"][-1] > 1000.0
