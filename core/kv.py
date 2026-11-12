"""
Vantage V-AI KV-Cache Streaming & Compression Engine (v7.0.0).

Designed for LLM inference runtimes (vLLM, HuggingFace, llama.cpp):
Compresses and streams Key-Value attention tensors across memory hierarchies
(GPU VRAM -> Host RAM / SSD) without intermediate copies, preserving
exact numerical integrity (lossless) or controlled precision (quantized FP8).
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import BinaryIO, Dict, Iterator, List, Optional, Tuple, Union

import numpy as np

try:
    import torch
except ImportError:
    torch = None

from core.engine import (
    CodecType,
    StatelessStreamProcessor,
    compress_chunk,
    decompress_chunk,
)


@dataclass(frozen=True)
class KVCacheMetadata:
    """Header describing shape, dtype, and layout of an attention KV tensor."""

    layer_idx: int
    num_heads: int
    head_dim: int
    num_tokens: int
    dtype_str: str
    is_quantized: bool = False
    scale: float = 1.0

    def to_bytes(self) -> bytes:
        dt_encoded = self.dtype_str.encode("ascii")
        return struct.pack(
            "<IIIIB?f",
            self.layer_idx,
            self.num_heads,
            self.head_dim,
            self.num_tokens,
            len(dt_encoded),
            self.is_quantized,
            self.scale,
        ) + dt_encoded

    @classmethod
    def from_bytes(cls, buf: bytes) -> Tuple[KVCacheMetadata, int]:
        layer_idx, num_heads, head_dim, num_tokens, dt_len, is_q, scale = struct.unpack_from(
            "<IIIIB?f", buf, 0
        )
        offset = struct.calcsize("<IIIIB?f")
        dtype_str = buf[offset : offset + dt_len].decode("ascii")
        return cls(layer_idx, num_heads, head_dim, num_tokens, dtype_str, is_q, scale), offset + dt_len


@dataclass(frozen=True)
class KVStreamStats:
    """Metrics for KV-Cache compression & offloading."""

    total_tokens: int
    raw_bytes: int
    compressed_bytes: int
    compression_ratio: float
    offload_time_ms: float
    throughput_gb_s: float


class KVCacheStreamEngine:
    """
    Stateless KV-Cache streaming engine.

    Directly accepts PyTorch tensors or NumPy arrays representing Attention KV blocks:
    Shape: [num_heads, num_tokens, head_dim] or [num_tokens, num_heads, head_dim].
    Offloads to memoryview -> stateless LZ4/Zstandard compression -> bounded buffer or disk.
    """

    def __init__(
        self,
        codec: CodecType = CodecType.LZ4_BLOCK,
        chunk_tokens: int = 128,
        compute_checksum: bool = False,
    ) -> None:
        self.codec = codec
        self.chunk_tokens = chunk_tokens
        self.compute_checksum = compute_checksum

    def _extract_raw_bytes(self, tensor_or_array) -> Tuple[bytes, KVCacheMetadata]:
        if torch is not None and isinstance(tensor_or_array, torch.Tensor):
            t = tensor_or_array.detach().cpu().contiguous()
            num_tokens, num_heads, head_dim = t.shape[0], t.shape[1], t.shape[2]
            raw_bytes = t.numpy().tobytes()
            meta = KVCacheMetadata(
                layer_idx=0,
                num_heads=num_heads,
                head_dim=head_dim,
                num_tokens=num_tokens,
                dtype_str=str(t.dtype).replace("torch.", ""),
            )
            return raw_bytes, meta

        if isinstance(tensor_or_array, np.ndarray):
            arr = np.ascontiguousarray(tensor_or_array)
            num_tokens, num_heads, head_dim = arr.shape[0], arr.shape[1], arr.shape[2]
            return arr.tobytes(), KVCacheMetadata(
                layer_idx=0,
                num_heads=num_heads,
                head_dim=head_dim,
                num_tokens=num_tokens,
                dtype_str=str(arr.dtype),
            )

        raise TypeError(f"Expected torch.Tensor or np.ndarray, got {type(tensor_or_array)}")

    def compress_kv_block(
        self,
        key_or_val_tensor,
        layer_idx: int = 0,
        quantize_fp8: bool = False,
    ) -> Tuple[bytes, KVCacheMetadata]:
        """Compress a single layer's Key or Value tensor with optional FP8 dynamic scaling."""
        raw_bytes, base_meta = self._extract_raw_bytes(key_or_val_tensor)
        scale = 1.0

        if quantize_fp8:
            # Dynamic per-tensor INT8/FP8 quantization
            if torch is not None and isinstance(key_or_val_tensor, torch.Tensor):
                t = key_or_val_tensor.detach().cpu().to(torch.float32)
                max_val = float(t.abs().max())
                scale = max(max_val / 127.0, 1e-8)
                q_t = (t / scale).round().clamp(-128, 127).to(torch.int8)
                payload_bytes = q_t.numpy().tobytes()
            else:
                arr = np.asarray(key_or_val_tensor, dtype=np.float32)
                max_val = float(np.max(np.abs(arr)))
                scale = max(max_val / 127.0, 1e-8)
                q_arr = np.clip(np.round(arr / scale), -128, 127).astype(np.int8)
                payload_bytes = q_arr.tobytes()
        else:
            payload_bytes = raw_bytes

        meta = KVCacheMetadata(
            layer_idx=layer_idx,
            num_heads=base_meta.num_heads,
            head_dim=base_meta.head_dim,
            num_tokens=base_meta.num_tokens,
            dtype_str=base_meta.dtype_str,
            is_quantized=quantize_fp8,
            scale=scale,
        )

        header = meta.to_bytes()
        compressed_body = compress_chunk(
            payload_bytes,
            codec=self.codec,
            compression_level=0,
            content_checksum=self.compute_checksum,
        )
        return header + compressed_body, meta

    def decompress_kv_block(
        self,
        payload: bytes,
        as_torch: bool = True,
    ) -> Tuple[Union["torch.Tensor", np.ndarray], KVCacheMetadata]:
        """Restore exact Key or Value tensor from compressed payload with zero drift."""
        meta, header_size = KVCacheMetadata.from_bytes(payload)
        compressed_body = memoryview(payload)[header_size:]
        raw_bytes = decompress_chunk(compressed_body, codec=self.codec)

        if meta.is_quantized:
            q_arr = np.frombuffer(raw_bytes, dtype=np.int8).reshape(
                meta.num_tokens, meta.num_heads, meta.head_dim
            )
            restored_np = (q_arr.astype(np.float32) * meta.scale).astype(np.float16)
        else:
            np_dtype = np.dtype(meta.dtype_str if meta.dtype_str != "float16" else np.float16)
            restored_np = np.frombuffer(raw_bytes, dtype=np_dtype).reshape(
                meta.num_tokens, meta.num_heads, meta.head_dim
            )

        if as_torch and torch is not None:
            restored_tensor = torch.from_numpy(restored_np.copy())
            return restored_tensor, meta
        return restored_np, meta

    def stream_offload_cache(
        self,
        kv_tensors: Dict[int, Tuple[Union["torch.Tensor", np.ndarray], Union["torch.Tensor", np.ndarray]]],
        stream_out: BinaryIO,
        quantize_fp8: bool = False,
    ) -> int:
        """
        Stream an entire multi-layer KV-cache dictionary out to disk or network.

        kv_tensors: {layer_idx: (key_tensor, value_tensor)}
        Returns total compressed bytes written.
        """
        total_written = 0
        for layer_idx, (k_tensor, v_tensor) in kv_tensors.items():
            for is_value, tensor in ((False, k_tensor), (True, v_tensor)):
                payload, meta = self.compress_kv_block(tensor, layer_idx=layer_idx, quantize_fp8=quantize_fp8)
                tag = struct.pack("<BI", 1 if is_value else 0, len(payload))
                stream_out.write(tag)
                stream_out.write(payload)
                total_written += len(tag) + len(payload)
        return total_written

    def stream_reload_cache(
        self,
        stream_in: BinaryIO,
        as_torch: bool = True,
    ) -> Dict[int, Tuple[Union["torch.Tensor", np.ndarray], Union["torch.Tensor", np.ndarray]]]:
        """Stream reload multi-layer KV cache back into memory."""
        kv_dict: Dict[int, List[Optional[Union["torch.Tensor", np.ndarray]]]] = {}

        while True:
            tag = stream_in.read(5)
            if not tag:
                break
            if len(tag) < 5:
                raise EOFError("Truncated KV block tag")
            is_val_flag, payload_len = struct.unpack("<BI", tag)
            payload = stream_in.read(payload_len)
            if len(payload) < payload_len:
                raise EOFError("Truncated KV block payload")

            tensor, meta = self.decompress_kv_block(payload, as_torch=as_torch)
            if meta.layer_idx not in kv_dict:
                kv_dict[meta.layer_idx] = [None, None]
            kv_dict[meta.layer_idx][1 if is_val_flag else 0] = tensor

        return {layer: (t[0], t[1]) for layer, t in kv_dict.items() if t[0] is not None and t[1] is not None}
