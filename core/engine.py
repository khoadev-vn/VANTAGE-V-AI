"""
Vantage V-AI Core Engine (v5.0)
Stateless, bounded-memory streaming data processing and compression engine.

Design principles:
- Strictly stateless execution: processors hold configuration only, never
  accumulating processed payloads inside the engine instance.
- Buffer-protocol slicing: accepts bytes, bytearray, or memoryview and slices
  chunks via memoryview to avoid redundant Python-side input copies.
- Pluggable bounded sinks: output frames flow to an explicit OutputSink
  (NullSink, BoundedMemorySink, CallbackSink, or CollectingSink).
- Verifiable correctness: optional CRC32 checksums and round-trip
  decompression for every supported codec.
"""

from __future__ import annotations

from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from enum import Enum
import zlib
from typing import BinaryIO, Callable, Deque, Iterable, List, Optional, Protocol, Union

import lz4.block
import lz4.frame

try:
    import zstandard as zstd
except ImportError:  # pragma: no cover - optional dependency
    zstd = None


BufferLike = Union[bytes, bytearray, memoryview]


class CodecType(str, Enum):
    """Supported compression / passthrough codecs."""

    LZ4_FRAME = "lz4_frame"
    LZ4_BLOCK = "lz4_block"
    ZSTD = "zstd"
    NONE = "none"


@dataclass(frozen=True)
class StreamStats:
    """Immutable summary of a single stateless stream execution."""

    input_bytes: int
    output_bytes: int
    frames: int
    crc32: int

    @property
    def compression_ratio(self) -> float:
        """Ratio of input_bytes to output_bytes (>1.0 means size reduction)."""
        if self.output_bytes == 0:
            return 0.0
        return self.input_bytes / self.output_bytes


class OutputSink(Protocol):
    """Protocol for downstream consumers of processed frames."""

    def write(self, frame: bytes) -> None:
        """Consume a single processed frame."""

    def total_bytes(self) -> int:
        """Return total bytes written to this sink."""


class NullSink:
    """Zero-retention sink for streaming throughput benchmarks."""

    def __init__(self) -> None:
        self._bytes_written = 0
        self._frames_written = 0

    def write(self, frame: bytes) -> None:
        self._bytes_written += len(frame)
        self._frames_written += 1

    def total_bytes(self) -> int:
        return self._bytes_written

    @property
    def frames_written(self) -> int:
        return self._frames_written


class BoundedMemorySink:
    """
    Bounded-capacity ring sink that guarantees memory usage never exceeds
    ``max_bytes`` (for stored frame payloads).
    """

    def __init__(self, max_bytes: int = 16 * 1024 * 1024, drop_oldest: bool = True) -> None:
        if max_bytes <= 0:
            raise ValueError("max_bytes must be positive")
        self.max_bytes = max_bytes
        self.drop_oldest = drop_oldest
        self._frames: Deque[bytes] = deque()
        self._current_bytes = 0
        self._total_bytes = 0
        self._evicted_frames = 0

    def write(self, frame: bytes) -> None:
        frame_len = len(frame)
        if frame_len > self.max_bytes:
            raise MemoryError(
                f"Single frame ({frame_len} B) exceeds sink max_bytes ({self.max_bytes} B)"
            )

        while self._current_bytes + frame_len > self.max_bytes:
            if not self.drop_oldest:
                raise MemoryError("BoundedMemorySink capacity exceeded (backpressure)")
            evicted = self._frames.popleft()
            self._current_bytes -= len(evicted)
            self._evicted_frames += 1

        self._frames.append(frame)
        self._current_bytes += frame_len
        self._total_bytes += frame_len

    def total_bytes(self) -> int:
        return self._total_bytes

    @property
    def retained_bytes(self) -> int:
        return self._current_bytes

    @property
    def evicted_frames(self) -> int:
        return self._evicted_frames

    def frames(self) -> List[bytes]:
        return list(self._frames)


class CollectingSink:
    """Unbounded sink used for round-trip correctness verification and tests."""

    def __init__(self) -> None:
        self._frames: List[bytes] = []
        self._total_bytes = 0

    def write(self, frame: bytes) -> None:
        self._frames.append(frame)
        self._total_bytes += len(frame)

    def total_bytes(self) -> int:
        return self._total_bytes

    def frames(self) -> List[bytes]:
        return self._frames


class CallbackSink:
    """Forwards each processed frame to a caller-supplied callback."""

    def __init__(self, callback: Callable[[bytes], None]) -> None:
        self._callback = callback
        self._total_bytes = 0

    def write(self, frame: bytes) -> None:
        self._callback(frame)
        self._total_bytes += len(frame)

    def total_bytes(self) -> int:
        return self._total_bytes


class FileSink:
    """Streams processed frames directly to a binary file-like object."""

    def __init__(self, stream: BinaryIO) -> None:
        self._stream = stream
        self._total_bytes = 0

    def write(self, frame: bytes) -> None:
        self._stream.write(frame)
        self._total_bytes += len(frame)

    def total_bytes(self) -> int:
        return self._total_bytes


def compress_chunk(
    chunk: BufferLike,
    codec: CodecType = CodecType.LZ4_FRAME,
    compression_level: int = 0,
    content_checksum: bool = True,
) -> bytes:
    """Compress a single buffer-like chunk using the selected codec."""
    if codec == CodecType.LZ4_FRAME:
        return lz4.frame.compress(
            chunk,
            compression_level=compression_level,
            content_checksum=content_checksum,
        )
    if codec == CodecType.LZ4_BLOCK:
        return lz4.block.compress(chunk, store_size=True)
    if codec == CodecType.ZSTD:
        if zstd is None:
            raise RuntimeError("zstandard package is not installed")
        level = compression_level if compression_level > 0 else 3
        return zstd.ZstdCompressor(level=level).compress(chunk)
    if codec == CodecType.NONE:
        return bytes(chunk)
    raise ValueError(f"Unsupported codec: {codec}")


def decompress_chunk(frame: BufferLike, codec: CodecType = CodecType.LZ4_FRAME) -> bytes:
    """Decompress a single frame produced by ``compress_chunk``."""
    if codec == CodecType.LZ4_FRAME:
        return lz4.frame.decompress(frame)
    if codec == CodecType.LZ4_BLOCK:
        return lz4.block.decompress(frame)
    if codec == CodecType.ZSTD:
        if zstd is None:
            raise RuntimeError("zstandard package is not installed")
        return zstd.ZstdDecompressor().decompress(frame)
    if codec == CodecType.NONE:
        return bytes(frame)
    raise ValueError(f"Unsupported codec: {codec}")


class StatelessStreamProcessor:
    """
    Strictly stateless streaming processor.

    No payload data is retained on ``self`` across calls. Input buffers are
    sliced via ``memoryview`` into ``chunk_bytes`` segments, optionally
    checksummed with CRC32, compressed, and pushed to ``sink``.
    """

    def __init__(
        self,
        chunk_bytes: int = 256 * 1024,
        codec: CodecType = CodecType.LZ4_FRAME,
        compression_level: int = 0,
        compute_checksum: bool = True,
        frame_checksum: bool = True,
    ) -> None:
        if chunk_bytes <= 0:
            raise ValueError("chunk_bytes must be positive")
        self.chunk_bytes = chunk_bytes
        self.codec = CodecType(codec)
        self.compression_level = compression_level
        self.compute_checksum = compute_checksum
        self.frame_checksum = frame_checksum

    def process(self, data: BufferLike, sink: OutputSink) -> StreamStats:
        """Process ``data`` in ``chunk_bytes`` slices and stream frames to ``sink``."""
        view = memoryview(data)
        total_len = len(view)
        if total_len == 0:
            return StreamStats(input_bytes=0, output_bytes=0, frames=0, crc32=0)

        out_bytes = 0
        frames = 0
        crc = 0

        for offset in range(0, total_len, self.chunk_bytes):
            chunk = view[offset : offset + self.chunk_bytes]
            if self.compute_checksum:
                crc = zlib.crc32(chunk, crc)
            frame = compress_chunk(
                chunk,
                codec=self.codec,
                compression_level=self.compression_level,
                content_checksum=self.frame_checksum,
            )
            sink.write(frame)
            out_bytes += len(frame)
            frames += 1

        return StreamStats(
            input_bytes=total_len,
            output_bytes=out_bytes,
            frames=frames,
            crc32=crc & 0xFFFFFFFF,
        )

    def verify_roundtrip(self, frames: List[bytes], expected_crc32: Optional[int] = None) -> bytes:
        """Decompress ``frames`` in order and optionally verify CRC32 integrity."""
        restored = b"".join(decompress_chunk(f, codec=self.codec) for f in frames)
        if expected_crc32 is not None:
            actual_crc = zlib.crc32(restored) & 0xFFFFFFFF
            if actual_crc != expected_crc32:
                raise ValueError(
                    f"CRC32 mismatch: expected {expected_crc32:#010x}, got {actual_crc:#010x}"
                )
        return restored

    def process_iter(self, chunks: Iterable[BufferLike], sink: OutputSink) -> StreamStats:
        """
        True incremental streaming over an iterable of buffer-like chunks.

        Memory remains bounded by ``chunk_bytes`` + ``sink`` capacity regardless
        of total stream length (e.g., 64 MiB, 1 GiB+).
        """
        total_in = 0
        total_out = 0
        frames = 0
        crc = 0
        buf = bytearray()

        for item in chunks:
            view = memoryview(item)
            if len(view) == 0:
                continue
            total_in += len(view)
            buf.extend(view)

            while len(buf) >= self.chunk_bytes:
                slice_view = memoryview(buf)[: self.chunk_bytes]
                if self.compute_checksum:
                    crc = zlib.crc32(slice_view, crc)
                frame = compress_chunk(
                    slice_view,
                    codec=self.codec,
                    compression_level=self.compression_level,
                    content_checksum=self.frame_checksum,
                )
                slice_view.release()
                sink.write(frame)
                total_out += len(frame)
                frames += 1
                del buf[: self.chunk_bytes]

        if len(buf) > 0:
            tail_view = memoryview(buf)
            if self.compute_checksum:
                crc = zlib.crc32(tail_view, crc)
            frame = compress_chunk(
                tail_view,
                codec=self.codec,
                compression_level=self.compression_level,
                content_checksum=self.frame_checksum,
            )
            sink.write(frame)
            total_out += len(frame)
            frames += 1

        return StreamStats(
            input_bytes=total_in,
            output_bytes=total_out,
            frames=frames,
            crc32=crc & 0xFFFFFFFF,
        )


class AdaptiveStreamProcessor(StatelessStreamProcessor):
    """
    Adaptive chunk-sizing processor (v6.0).

    Selects 256 KiB chunks for low-compressibility payloads (ratio < 2.0x)
    and 512 KiB chunks for highly compressible payloads (ratio >= 2.0x) based
    on a lightweight probe of the first chunk.
    """

    def __init__(
        self,
        low_entropy_chunk_bytes: int = 512 * 1024,
        high_entropy_chunk_bytes: int = 256 * 1024,
        codec: CodecType = CodecType.LZ4_BLOCK,
        compression_level: int = 0,
        compute_checksum: bool = False,
        frame_checksum: bool = True,
    ) -> None:
        super().__init__(
            chunk_bytes=high_entropy_chunk_bytes,
            codec=codec,
            compression_level=compression_level,
            compute_checksum=compute_checksum,
            frame_checksum=frame_checksum,
        )
        self.low_entropy_chunk_bytes = low_entropy_chunk_bytes
        self.high_entropy_chunk_bytes = high_entropy_chunk_bytes

    def select_chunk_bytes(self, sample: memoryview) -> int:
        probe_len = min(len(sample), 64 * 1024)
        if probe_len == 0:
            return self.high_entropy_chunk_bytes
        probe_out = compress_chunk(
            sample[:probe_len],
            codec=self.codec,
            compression_level=self.compression_level,
            content_checksum=self.frame_checksum,
        )
        ratio = probe_len / max(1, len(probe_out))
        return self.low_entropy_chunk_bytes if ratio >= 2.0 else self.high_entropy_chunk_bytes

    def process(self, data: BufferLike, sink: OutputSink) -> StreamStats:
        view = memoryview(data)
        total_len = len(view)
        if total_len == 0:
            return StreamStats(input_bytes=0, output_bytes=0, frames=0, crc32=0)

        chosen_chunk = self.select_chunk_bytes(view)
        out_bytes = 0
        frames = 0
        crc = 0

        for offset in range(0, total_len, chosen_chunk):
            chunk = view[offset : offset + chosen_chunk]
            if self.compute_checksum:
                crc = zlib.crc32(chunk, crc)
            frame = compress_chunk(
                chunk,
                codec=self.codec,
                compression_level=self.compression_level,
                content_checksum=self.frame_checksum,
            )
            sink.write(frame)
            out_bytes += len(frame)
            frames += 1

        return StreamStats(
            input_bytes=total_len,
            output_bytes=out_bytes,
            frames=frames,
            crc32=crc & 0xFFFFFFFF,
        )


class ParallelStreamProcessor(StatelessStreamProcessor):
    """
    Multi-threaded stateless stream processor.

    Leverages C-extension GIL release in lz4/zstd/zlib to compress independent
    ``memoryview`` chunks across a worker thread pool while writing frames to
    ``sink`` in deterministic input order.
    """

    def __init__(
        self,
        chunk_bytes: int = 512 * 1024,
        codec: CodecType = CodecType.LZ4_FRAME,
        compression_level: int = 0,
        compute_checksum: bool = True,
        frame_checksum: bool = True,
        max_workers: Optional[int] = None,
    ) -> None:
        super().__init__(
            chunk_bytes=chunk_bytes,
            codec=codec,
            compression_level=compression_level,
            compute_checksum=compute_checksum,
            frame_checksum=frame_checksum,
        )
        self.max_workers = max_workers

    def _compress_one(self, chunk: memoryview) -> bytes:
        return compress_chunk(
            chunk,
            codec=self.codec,
            compression_level=self.compression_level,
            content_checksum=self.frame_checksum,
        )

    def process(self, data: BufferLike, sink: OutputSink) -> StreamStats:
        view = memoryview(data)
        total_len = len(view)
        if total_len == 0:
            return StreamStats(input_bytes=0, output_bytes=0, frames=0, crc32=0)

        chunks = [
            view[offset : offset + self.chunk_bytes]
            for offset in range(0, total_len, self.chunk_bytes)
        ]

        if len(chunks) == 1:
            return super().process(view, sink)

        crc = zlib.crc32(view) & 0xFFFFFFFF if self.compute_checksum else 0

        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            frames = list(pool.map(self._compress_one, chunks))

        out_bytes = 0
        for frame in frames:
            sink.write(frame)
            out_bytes += len(frame)

        return StreamStats(
            input_bytes=total_len,
            output_bytes=out_bytes,
            frames=len(frames),
            crc32=crc,
        )


class VantageEngine:
    """
    Backwards-compatible facade over StatelessStreamProcessor + BoundedMemorySink.

    Unlike v4.0's unbounded ``self.vault``, the default sink is bounded to
    ``max_retained_bytes`` (16 MiB) so repeated calls cannot exhaust RAM.
    """

    def __init__(
        self,
        batch_size: int = 25000,
        chunk_bytes: int = 256 * 1024,
        max_retained_bytes: int = 16 * 1024 * 1024,
        codec: CodecType = CodecType.LZ4_FRAME,
    ) -> None:
        self.batch_size = batch_size
        self.processor = StatelessStreamProcessor(
            chunk_bytes=chunk_bytes,
            codec=codec,
            compute_checksum=True,
            frame_checksum=True,
        )
        self.sink = BoundedMemorySink(max_bytes=max_retained_bytes, drop_oldest=True)

    def fast_commit_batch(self, batch_data_bytes: BufferLike) -> int:
        """Process a batch through the stateless processor into the bounded sink."""
        stats = self.processor.process(batch_data_bytes, self.sink)
        return stats.output_bytes

    def get_vault_size(self) -> int:
        """Return currently retained bytes in the bounded sink."""
        return self.sink.retained_bytes
