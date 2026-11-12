"""
Vantage V-AI v6.4 Quickstart — Three Runnable Use Cases:
1. Bounded-memory incremental streaming with BoundedMemorySink
2. Framed FileSink write + lazy decompress_stream round-trip
3. Record-aligned streaming into tiktoken with 100% token ID parity
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import (
    BoundedMemorySink,
    CodecType,
    FileSink,
    StatelessStreamProcessor,
    decompress_stream,
    iter_record_chunks,
    read_framed_stream,
)


def main() -> None:
    # 1. Bounded-memory incremental streaming
    proc = StatelessStreamProcessor(chunk_bytes=256 * 1024, codec=CodecType.LZ4_BLOCK)
    sink = BoundedMemorySink(max_bytes=1024 * 1024, drop_oldest=True)
    chunks = (b"ID:101|Node:edge-01|Status:OK\n" * 10_000 for _ in range(10))
    stats = proc.process_iter(chunks, sink)
    print(f"[1] Streamed {stats.input_bytes / (1024*1024):.2f} MiB -> retained {sink.retained_bytes} B")

    # 2. Framed FileSink + lazy decompress_stream
    sample = b"Prompt: Summarize telemetry batch.\n" * 5_000
    with tempfile.TemporaryDirectory() as tmp:
        vnt_path = Path(tmp) / "sample.vnt"
        with vnt_path.open("wb") as f:
            proc.process(sample, FileSink(f, framed=True))

        with vnt_path.open("rb") as f:
            restored = b"".join(decompress_stream(read_framed_stream(f), codec=CodecType.LZ4_BLOCK))
        assert restored == sample
        print(f"[2] FileSink round-trip verified ({len(sample)} B -> {vnt_path.stat().st_size} B on disk)")

        # 3. Record-aligned streaming for 100% token ID parity (optional tiktoken)
        try:
            import tiktoken

            enc = tiktoken.get_encoding("cl100k_base")
            full_ids = enc.encode(sample.decode("utf-8"))
            stream_ids = []
            with vnt_path.open("rb") as f:
                raw_chunks = decompress_stream(read_framed_stream(f), codec=CodecType.LZ4_BLOCK)
                for aligned in iter_record_chunks(raw_chunks, delimiter=b"\n"):
                    stream_ids.extend(enc.encode(aligned.decode("utf-8")))
            assert stream_ids == full_ids
            print(f"[3] 100% Token ID parity verified ({len(stream_ids)} identical tokens)")
        except ImportError:
            print("[3] Skipped tiktoken example (install with: pip install tiktoken)")


if __name__ == "__main__":
    main()
