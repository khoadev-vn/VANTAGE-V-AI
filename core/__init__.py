"""Vantage V-AI core streaming data processing package (v6.2.0)."""

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
    read_framed_stream,
)

__version__ = "6.2.0"

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
    "read_framed_stream",
]
