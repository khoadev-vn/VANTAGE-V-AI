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
from core.kv import (
    KVCacheMetadata,
    KVCacheStreamEngine,
    KVQuantizationError,
    compute_quantization_error,
)

__version__ = "7.1.0"

__all__ = [
    "__version__",
    "AdaptiveStreamProcessor",
    "BoundedMemorySink",
    "CallbackSink",
    "CodecType",
    "CollectingSink",
    "FileSink",
    "KVCacheMetadata",
    "KVCacheStreamEngine",
    "KVQuantizationError",
    "NullSink",
    "OutputSink",
    "ParallelStreamProcessor",
    "StatelessStreamProcessor",
    "StreamStats",
    "VantageEngine",
    "compute_quantization_error",
    "decompress_stream",
    "iter_record_chunks",
    "read_framed_stream",
]
