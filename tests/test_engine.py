"""Unit tests for Vantage V-AI stateless streaming engine."""

import os
import sys
import zlib
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.engine import (
    AdaptiveStreamProcessor,
    BoundedMemorySink,
    CallbackSink,
    CodecType,
    CollectingSink,
    FileSink,
    NullSink,
    ParallelStreamProcessor,
    StatelessStreamProcessor,
    VantageEngine,
    decompress_stream,
    iter_record_chunks,
    read_framed_stream,
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


def test_incremental_process_iter_and_file_sink():
    import io

    parts = [b"VANTAGE_STREAM_" * 5000, os.urandom(70_000), b"TAIL_DATA" * 1000]
    full_payload = b"".join(parts)
    expected_crc = zlib.crc32(full_payload) & 0xFFFFFFFF

    proc = StatelessStreamProcessor(chunk_bytes=32 * 1024, codec=CodecType.LZ4_FRAME)
    csink = CollectingSink()
    stats = proc.process_iter(parts, csink)

    assert stats.input_bytes == len(full_payload)
    assert stats.crc32 == expected_crc
    assert proc.verify_roundtrip(csink.frames(), expected_crc32=stats.crc32) == full_payload

    bio = io.BytesIO()
    fsink = FileSink(bio, framed=True)
    fstats = proc.process_iter(parts, fsink)
    assert fsink.total_bytes() == fstats.output_bytes + 4 * fstats.frames
    assert len(bio.getvalue()) == fsink.total_bytes()


def test_adaptive_stream_processor_selects_chunk_and_verifies():
    proc = AdaptiveStreamProcessor(
        low_entropy_chunk_bytes=512 * 1024,
        high_entropy_chunk_bytes=256 * 1024,
        codec=CodecType.LZ4_BLOCK,
        compute_checksum=True,
    )
    compressible = b"A" * (1024 * 1024)
    incompressible = os.urandom(1024 * 1024)

    assert proc.select_chunk_bytes(memoryview(compressible)) == 512 * 1024
    assert proc.select_chunk_bytes(memoryview(incompressible)) == 256 * 1024

    sink = CollectingSink()
    stats = proc.process(incompressible, sink)
    assert proc.verify_roundtrip(sink.frames(), expected_crc32=stats.crc32) == incompressible


def test_file_sink_partial_writes_and_framed_roundtrip(tmp_path):
    import io

    class PartialStream(io.BytesIO):
        def write(self, b):
            mv = memoryview(b)
            return super().write(mv[:7])  # Force partial writes of at most 7 bytes

    payload = (b"VANTAGE_FRAMED_" * 4000) + os.urandom(50_000)
    expected_crc = zlib.crc32(payload) & 0xFFFFFFFF

    proc = StatelessStreamProcessor(chunk_bytes=32 * 1024, codec=CodecType.LZ4_FRAME)
    raw_stream = PartialStream()
    fsink = FileSink(raw_stream, framed=True)
    stats = proc.process(payload, fsink)

    raw_stream.seek(0)
    frames = list(read_framed_stream(raw_stream))
    assert len(frames) == stats.frames
    assert proc.verify_roundtrip(frames, expected_crc32=expected_crc) == payload

    # Also test real disk file round-trip
    disk_file = tmp_path / "stream.vnt"
    with disk_file.open("wb") as f:
        proc.process(payload, FileSink(f, framed=True))
    with disk_file.open("rb") as f:
        disk_frames = list(read_framed_stream(f))
    assert proc.verify_roundtrip(disk_frames, expected_crc32=expected_crc) == payload


def test_file_sink_stalled_stream_raises_oserror():
    class StalledStream:
        def write(self, b):
            return 0

    sink = FileSink(StalledStream(), framed=True)  # type: ignore[arg-type]
    with pytest.raises(OSError):
        sink.write(b"abc")


def test_adaptive_dynamic_heterogeneous_streams():
    for hyst in (1, 3):
        proc = AdaptiveStreamProcessor(
            low_entropy_chunk_bytes=512 * 1024,
            high_entropy_chunk_bytes=256 * 1024,
            hysteresis_chunks=hyst,
            codec=CodecType.LZ4_BLOCK,
            compute_checksum=True,
        )
        low_then_high = (b"R" * (600 * 1024)) + os.urandom(900 * 1024)
        high_then_low = os.urandom(300 * 1024) + (b"S" * (1100 * 1024))

        for payload in (low_then_high, high_then_low):
            sink = CollectingSink()
            stats = proc.process(payload, sink)
            assert proc.verify_roundtrip(sink.frames(), expected_crc32=stats.crc32) == payload

            isink = CollectingSink()
            chunks = [payload[i : i + 100_000] for i in range(0, len(payload), 100_000)]
            istats = proc.process_iter(chunks, isink)
            assert proc.verify_roundtrip(isink.frames(), expected_crc32=istats.crc32) == payload
            assert stats.switch_count >= 0


def test_decompress_stream_lazy_roundtrip(tmp_path):
    payload = b"LAZY_DECOMPRESS_STREAM_" * 20_000
    proc = StatelessStreamProcessor(chunk_bytes=64 * 1024, codec=CodecType.LZ4_FRAME)
    out_file = tmp_path / "lazy.vnt"
    with out_file.open("wb") as f:
        proc.process(payload, FileSink(f, framed=True))

    with out_file.open("rb") as f:
        restored = b"".join(decompress_stream(read_framed_stream(f), codec=CodecType.LZ4_FRAME))
    assert restored == payload


def test_iter_record_chunks_exact_token_parity(tmp_path):
    tiktoken = pytest.importorskip("tiktoken")
    enc = tiktoken.get_encoding("cl100k_base")

    lines = [
        f"[{i:05d}] Bounded-memory streaming preserves exact token boundaries.\n".encode("utf-8")
        for i in range(4000)
    ]
    payload = b"".join(lines)
    full_tokens = enc.encode(payload.decode("utf-8"))

    proc = StatelessStreamProcessor(chunk_bytes=16 * 1024, codec=CodecType.LZ4_BLOCK)
    out_file = tmp_path / "parity.vnt"
    with out_file.open("wb") as f:
        proc.process(payload, FileSink(f, framed=True))

    stream_tokens = []
    with out_file.open("rb") as f:
        decompressed = decompress_stream(read_framed_stream(f), codec=CodecType.LZ4_BLOCK)
        for aligned in iter_record_chunks(decompressed, delimiter=b"\n"):
            stream_tokens.extend(enc.encode(aligned.decode("utf-8")))

    assert stream_tokens == full_tokens


def test_utf8_multibyte_split_across_chunks(tmp_path):
    tiktoken = pytest.importorskip("tiktoken")
    enc = tiktoken.get_encoding("cl100k_base")

    # Multibyte UTF-8 (Vietnamese + emoji + CJK) forced across tiny 1 KiB chunk boundaries
    line = "Xin chào Vantage 🚀 日本語テスト — streaming bounded memory.\n".encode("utf-8")
    payload = line * 300
    full_tokens = enc.encode(payload.decode("utf-8"))

    proc = StatelessStreamProcessor(chunk_bytes=1024, codec=CodecType.LZ4_BLOCK)
    out_file = tmp_path / "utf8.vnt"
    with out_file.open("wb") as f:
        proc.process(payload, FileSink(f, framed=True))

    stream_tokens = []
    with out_file.open("rb") as f:
        decompressed = decompress_stream(read_framed_stream(f), codec=CodecType.LZ4_BLOCK)
        for aligned in iter_record_chunks(decompressed, delimiter=b"\n"):
            stream_tokens.extend(enc.encode(aligned.decode("utf-8")))

    assert stream_tokens == full_tokens


def test_crlf_and_no_trailing_newline_token_parity(tmp_path):
    tiktoken = pytest.importorskip("tiktoken")
    enc = tiktoken.get_encoding("cl100k_base")

    payload = (
        b"line one with crlf\r\n" * 500
        + b"line two with lf\n" * 500
        + b"final line without trailing newline"
    )
    full_tokens = enc.encode(payload.decode("utf-8"))

    proc = StatelessStreamProcessor(chunk_bytes=2048, codec=CodecType.LZ4_FRAME)
    out_file = tmp_path / "crlf.vnt"
    with out_file.open("wb") as f:
        proc.process(payload, FileSink(f, framed=True))

    stream_tokens = []
    with out_file.open("rb") as f:
        decompressed = decompress_stream(read_framed_stream(f), codec=CodecType.LZ4_FRAME)
        for aligned in iter_record_chunks(decompressed, delimiter=b"\n"):
            stream_tokens.extend(enc.encode(aligned.decode("utf-8")))

    assert stream_tokens == full_tokens


def test_corrupted_and_truncated_frames_raise():
    import io
    import struct

    # Truncated header (< 4 bytes)
    with pytest.raises(EOFError):
        list(read_framed_stream(io.BytesIO(b"\x01\x02")))

    # Truncated payload (header claims 100 bytes, stream has 3)
    bad_payload = struct.pack("<I", 100) + b"abc"
    with pytest.raises(EOFError):
        list(read_framed_stream(io.BytesIO(bad_payload)))

    # CRC32 mismatch on verify_roundtrip
    proc = StatelessStreamProcessor(codec=CodecType.LZ4_FRAME, compute_checksum=True)
    sink = CollectingSink()
    proc.process(b"integrity test payload", sink)
    with pytest.raises(ValueError, match="CRC32 mismatch"):
        proc.verify_roundtrip(sink.frames(), expected_crc32=0xDEADBEEF)
