"""Vantage V-AI core streaming data processing package (v6.4.0)."""

from core.engine import (
    AdaptiveStreamProcessor,
    BoundedMemorySink,
    CallbackSink,
    CodecType,
    CollectingSink,
    FileSink,
    NullSink,
    OutputSink,
    ParallelStreamProcessor,
    StatelessStreamProcessor,
    StreamStats,
    VantageEngine,
    decompress_stream,
    iter_record_chunks,
    read_framed_stream,
)

__version__ = "6.4.1"

__all__ = [
    "__version__",
    "AdaptiveStreamProcessor",
    "BoundedMemorySink",
    "CallbackSink",
    "CodecType",
    "CollectingSink",
    "FileSink",
    "NullSink",
    "OutputSink",
    "ParallelStreamProcessor",
    "StatelessStreamProcessor",
    "StreamStats",
    "VantageEngine",
    "decompress_stream",
    "iter_record_chunks",
    "read_framed_stream",
]
