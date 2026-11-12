"""Vantage V-AI core streaming data processing & KV-cache package (v7.2.0)."""

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

try:
    from core.adapters.transformers import VantageDynamicCache
except ImportError:
    VantageDynamicCache = None

__version__ = "7.2.0"

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
    "VantageDynamicCache",
    "VantageEngine",
    "compute_quantization_error",
    "decompress_stream",
    "iter_record_chunks",
    "read_framed_stream",
]
