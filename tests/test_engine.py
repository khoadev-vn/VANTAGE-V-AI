"""Unit tests for Vantage V-AI stateless streaming engine."""

import os
import sys
import zlib
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.engine import (
    BoundedMemorySink,
    CallbackSink,
    CodecType,
    CollectingSink,
    NullSink,
    ParallelStreamProcessor,
    StatelessStreamProcessor,
    VantageEngine,
)


@pytest.mark.parametrize(
    "codec",
    [CodecType.LZ4_FRAME, CodecType.LZ4_BLOCK, CodecType.ZSTD, CodecType.NONE],
)
def test_roundtrip_and_crc32_all_codecs(codec):
    payload = (b"ID:999|Node:V-AI|Payload:Stateless_Streaming_Test|" * 5000) + os.urandom(1024)
    expected_crc = zlib.crc32(payload) & 0xFFFFFFFF

    proc = StatelessStreamProcessor(chunk_bytes=64 * 1024, codec=codec, compute_checksum=True)
    sink = CollectingSink()
    stats = proc.process(payload, sink)

    assert stats.input_bytes == len(payload)
    assert stats.output_bytes == sink.total_bytes()
    assert stats.crc32 == expected_crc
    assert stats.frames >= 2

    restored = proc.verify_roundtrip(sink.frames(), expected_crc32=stats.crc32)
    assert restored == payload


def test_parallel_processor_preserves_order_and_integrity():
    payload = os.urandom(512 * 1024) + (b"VANTAGE_AI_SYSTEMS_DATA_" * 40000)
    expected_crc = zlib.crc32(payload) & 0xFFFFFFFF

    proc = ParallelStreamProcessor(
        chunk_bytes=128 * 1024,
        codec=CodecType.LZ4_FRAME,
        max_workers=4,
    )
    sink = CollectingSink()
    stats = proc.process(payload, sink)

    assert stats.input_bytes == len(payload)
    assert stats.crc32 == expected_crc
    restored = proc.verify_roundtrip(sink.frames(), expected_crc32=stats.crc32)
    assert restored == payload


def test_memoryview_and_bytearray_buffer_protocol():
    buf = bytearray(b"A" * 100_000 + b"B" * 50_000)
    mv = memoryview(buf)[10_000:140_000]

    proc = StatelessStreamProcessor(chunk_bytes=32 * 1024, codec=CodecType.LZ4_FRAME)
    sink = CollectingSink()
    stats = proc.process(mv, sink)

    assert stats.input_bytes == len(mv)
    restored = proc.verify_roundtrip(sink.frames(), expected_crc32=stats.crc32)
    assert restored == bytes(mv)


def test_statelessness_no_instance_accumulation():
    proc = StatelessStreamProcessor(chunk_bytes=32 * 1024)
    payload = b"X" * 100_000

    for _ in range(50):
        sink = NullSink()
        stats = proc.process(payload, sink)
        assert stats.input_bytes == len(payload)
        assert sink.total_bytes() == stats.output_bytes

    # Processor holds only configuration, no accumulated payload attributes
    assert not hasattr(proc, "vault")


def test_bounded_memory_sink_enforces_cap():
    sink = BoundedMemorySink(max_bytes=1024, drop_oldest=True)
    for _ in range(20):
        sink.write(b"Z" * 256)

    assert sink.retained_bytes <= 1024
    assert sink.evicted_frames > 0
    assert sink.total_bytes() == 20 * 256


def test_bounded_memory_sink_backpressure_raises():
    sink = BoundedMemorySink(max_bytes=512, drop_oldest=False)
    sink.write(b"A" * 256)
    sink.write(b"B" * 256)
    with pytest.raises(MemoryError):
        sink.write(b"C" * 1)


def test_callback_sink_and_empty_input():
    received = []
    sink = CallbackSink(received.append)
    proc = StatelessStreamProcessor()

    empty_stats = proc.process(b"", sink)
    assert empty_stats.input_bytes == 0
    assert empty_stats.frames == 0
    assert received == []

    stats = proc.process(b"hello vantage", sink)
    assert stats.frames == 1
    assert len(received) == 1


def test_vantage_engine_facade_is_bounded():
    engine = VantageEngine(max_retained_bytes=4096)
    chunk = os.urandom(2048)
    for _ in range(20):
        engine.fast_commit_batch(chunk)

    assert engine.get_vault_size() <= 4096
