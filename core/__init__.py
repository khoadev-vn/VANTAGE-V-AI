"""Vantage V-AI core streaming data processing & KV-cache package (v8.0.0)."""

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
    QUANT_FP8,
    QUANT_LOSSLESS,
    QUANT_TERNARY_158,
    KVCacheMetadata,
    KVCacheStreamEngine,
    KVQuantizationError,
    compute_quantization_error,
    pack_ternary_2bit,
    unpack_ternary_2bit,
)
from core.memory import HyperdimensionalMemory

try:
    from core.adapters.transformers import (
        VantageDynamicCache,
        VantagePromptCheckpoint,
        VantageStreamingCache,
        checkpoint_prompt_cache,
        restore_prompt_cache,
    )
except ImportError:
    VantageDynamicCache = None
    VantagePromptCheckpoint = None
    VantageStreamingCache = None
    checkpoint_prompt_cache = None
    restore_prompt_cache = None

__version__ = "8.2.0"

__all__ = [
    "__version__",
    "AdaptiveStreamProcessor",
    "BoundedMemorySink",
    "CallbackSink",
    "CodecType",
    "CollectingSink",
    "FileSink",
    "HyperdimensionalMemory",
    "KVCacheMetadata",
    "KVCacheStreamEngine",
    "KVQuantizationError",
    "NullSink",
    "OutputSink",
    "ParallelStreamProcessor",
    "QUANT_FP8",
    "QUANT_LOSSLESS",
    "QUANT_TERNARY_158",
    "StatelessStreamProcessor",
    "StreamStats",
    "VantageDynamicCache",
    "VantageEngine",
    "VantagePromptCheckpoint",
    "VantageStreamingCache",
    "checkpoint_prompt_cache",
    "compute_quantization_error",
    "decompress_stream",
    "iter_record_chunks",
    "pack_ternary_2bit",
    "read_framed_stream",
    "restore_prompt_cache",
    "unpack_ternary_2bit",
]
