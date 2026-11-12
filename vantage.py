"""
VANTAGE-V-AI: High-Performance Bounded-Memory KV-Cache & Streaming Runtime.

Quickstart:
    import vantage
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-3B-Instruct")
    vantage.enable(model, max_active_tokens=256, sink_tokens=4, quant_mode="fp8")

    out = model.generate(input_ids, max_new_tokens=64)
"""

from core import (
    AdaptiveStreamProcessor,
    BoundedMemorySink,
    CallbackSink,
    CodecType,
    CollectingSink,
    FileSink,
    HyperdimensionalMemory,
    KVCacheMetadata,
    KVCacheStreamEngine,
    KVQuantizationError,
    NullSink,
    OutputSink,
    ParallelStreamProcessor,
    QUANT_FP8,
    QUANT_LOSSLESS,
    QUANT_TERNARY_158,
    StatelessStreamProcessor,
    StreamStats,
    VantageDynamicCache,
    VantageEngine,
    VantagePromptCheckpoint,
    VantageStreamingCache,
    checkpoint_prompt_cache,
    compute_quantization_error,
    decompress_stream,
    iter_record_chunks,
    pack_ternary_2bit,
    read_framed_stream,
    restore_prompt_cache,
    unpack_ternary_2bit,
    __version__,
)
from core.patch import enable, disable, patch, unpatch
from core.kv import pack_ternary_torch, unpack_ternary_torch

__all__ = [
    "__version__",
    "AdaptiveStreamProcessor",
    "BoundedMemorySink",
    "CallbackSink",
    "CodecType",
    "CollectingSink",
    "disable",
    "enable",
    "FileSink",
    "HyperdimensionalMemory",
    "KVCacheMetadata",
    "KVCacheStreamEngine",
    "KVQuantizationError",
    "NullSink",
    "OutputSink",
    "ParallelStreamProcessor",
    "patch",
    "pack_ternary_2bit",
    "pack_ternary_torch",
    "QUANT_FP8",
    "QUANT_LOSSLESS",
    "QUANT_TERNARY_158",
    "read_framed_stream",
    "restore_prompt_cache",
    "StatelessStreamProcessor",
    "StreamStats",
    "unpack_ternary_2bit",
    "unpack_ternary_torch",
    "unpatch",
    "VantageDynamicCache",
    "VantageEngine",
    "VantagePromptCheckpoint",
    "VantageStreamingCache",
    "checkpoint_prompt_cache",
    "compute_quantization_error",
    "decompress_stream",
    "iter_record_chunks",
]
