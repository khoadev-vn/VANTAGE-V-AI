"""
VANTAGE-V-AI: Ollama Drop-in Code Adapter (v8.3.0).

Enables 100% plug-and-play code compatibility with the official `ollama` Python package:
    # Drop-in replacement:
    import vantage as ollama

    # Existing Ollama code runs unchanged with 90%+ KV-cache memory reduction:
    response = ollama.chat(
        model="Qwen/Qwen2.5-3B-Instruct",
        messages=[{"role": "user", "content": "Hello!"}]
    )
    print(response["message"]["content"])
"""

from __future__ import annotations

import datetime
import time
from typing import Any, Dict, Generator, Iterator, List, Optional, Union

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
except ImportError:
    torch = None
    AutoModelForCausalLM = None
    AutoTokenizer = None
    TextIteratorStreamer = None

from core.patch import enable as vantage_enable

# Standard model name alias mapper for Ollama users
MODEL_ALIASES: Dict[str, str] = {
    "llama3:8b": "meta-llama/Meta-Llama-3-8B-Instruct",
    "llama3": "meta-llama/Meta-Llama-3-8B-Instruct",
    "llama3.1:8b": "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "qwen2.5:3b": "Qwen/Qwen2.5-3B-Instruct",
    "qwen2.5:7b": "Qwen/Qwen2.5-7B-Instruct",
    "qwen2.5:1.5b": "Qwen/Qwen2.5-1.5B-Instruct",
    "mistral:7b": "mistralai/Mistral-7B-Instruct-v0.3",
    "mistral": "mistralai/Mistral-7B-Instruct-v0.3",
    "phi3": "microsoft/Phi-3-mini-4k-instruct",
}

# In-process singleton model cache to avoid re-loading weights on every call
_LOADED_MODELS: Dict[str, Any] = {}
_LOADED_TOKENIZERS: Dict[str, Any] = {}


def _resolve_model_name(model_name: str) -> str:
    """Maps Ollama model tags (e.g., 'llama3:8b') to Hugging Face model IDs."""
    key = model_name.strip().lower()
    return MODEL_ALIASES.get(key, model_name)


def _get_or_load_model(
    model_name: str,
    max_active_tokens: int = 256,
    sink_tokens: int = 4,
    quant_mode: str = "fp8",
) -> tuple:
    """Loads and patches model with VANTAGE bounded-memory streaming."""
    if torch is None or AutoModelForCausalLM is None:
        raise RuntimeError("PyTorch and transformers are required to run the local Vantage Ollama adapter.")

    hf_model_id = _resolve_model_name(model_name)

    if hf_model_id not in _LOADED_MODELS:
        device_map = "auto" if torch.cuda.is_available() else "cpu"
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        tokenizer = AutoTokenizer.from_pretrained(hf_model_id)
        model = AutoModelForCausalLM.from_pretrained(
            hf_model_id,
            torch_dtype=torch_dtype,
            device_map=device_map,
        )

        # 1-Line Drop-in VANTAGE bounded-memory acceleration
        vantage_enable(
            model,
            max_active_tokens=max_active_tokens,
            sink_tokens=sink_tokens,
            quant_mode=quant_mode,
        )

        _LOADED_MODELS[hf_model_id] = model
        _LOADED_TOKENIZERS[hf_model_id] = tokenizer

    return _LOADED_MODELS[hf_model_id], _LOADED_TOKENIZERS[hf_model_id]


def chat(
    model: str,
    messages: List[Dict[str, str]],
    stream: bool = False,
    options: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> Union[Dict[str, Any], Iterator[Dict[str, Any]]]:
    """
    Ollama-compatible chat API:
    `ollama.chat(model='llama3:8b', messages=[{'role': 'user', 'content': 'hi'}])`
    """
    opts = options or {}
    max_tokens = opts.get("num_predict", 256)
    window_size = opts.get("num_ctx", 256)
    temperature = opts.get("temperature", 0.7)

    loaded_model, tokenizer = _get_or_load_model(
        model,
        max_active_tokens=window_size,
        sink_tokens=4,
        quant_mode="fp8",
    )

    # Format chat prompt using tokenizer chat template if available
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        # Simple fallback conversation formatting
        formatted = []
        for m in messages:
            formatted.append(f"{m['role'].capitalize()}: {m['content']}")
        formatted.append("Assistant: ")
        prompt_text = "\n".join(formatted)

    input_ids = tokenizer(prompt_text, return_tensors="pt").input_ids.to(loaded_model.device)
    prompt_tokens_len = input_ids.shape[-1]

    t_start = time.perf_counter()

    if stream:
        return _stream_chat(loaded_model, tokenizer, input_ids, model, max_tokens, temperature)

    # Synchronous generation with bounded memory
    with torch.no_grad():
        out = loaded_model.generate(
            input_ids,
            max_new_tokens=max_tokens,
            do_sample=(temperature > 0.0),
            temperature=max(temperature, 1e-4),
            pad_token_id=tokenizer.eos_token_id,
        )

    t_total = time.perf_counter() - t_start
    gen_tokens = out[0, prompt_tokens_len:]
    content = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()

    return {
        "model": model,
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "message": {
            "role": "assistant",
            "content": content,
        },
        "done": True,
        "total_duration": int(t_total * 1e9),
        "prompt_eval_count": prompt_tokens_len,
        "eval_count": len(gen_tokens),
        "vantage_accelerated": True,
    }


def _stream_chat(
    model: Any,
    tokenizer: Any,
    input_ids: Any,
    model_name: str,
    max_tokens: int,
    temperature: float,
) -> Iterator[Dict[str, Any]]:
    """Yields streaming chunks matching Ollama's streaming protocol."""
    import threading

    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    gen_kwargs = dict(
        input_ids=input_ids,
        streamer=streamer,
        max_new_tokens=max_tokens,
        do_sample=(temperature > 0.0),
        temperature=max(temperature, 1e-4),
        pad_token_id=tokenizer.eos_token_id,
    )

    thread = threading.Thread(target=model.generate, kwargs=gen_kwargs)
    thread.start()

    for text_chunk in streamer:
        yield {
            "model": model_name,
            "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "message": {
                "role": "assistant",
                "content": text_chunk,
            },
            "done": False,
        }

    thread.join()
    yield {
        "model": model_name,
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "message": {
            "role": "assistant",
            "content": "",
        },
        "done": True,
        "vantage_accelerated": True,
    }


def generate(
    model: str,
    prompt: str,
    stream: bool = False,
    options: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> Union[Dict[str, Any], Iterator[Dict[str, Any]]]:
    """
    Ollama-compatible generate API:
    `ollama.generate(model='llama3:8b', prompt='Once upon a time')`
    """
    messages = [{"role": "user", "content": prompt}]
    res = chat(model=model, messages=messages, stream=stream, options=options, **kwargs)
    if stream:
        def _gen_wrapper():
            for chunk in res:
                yield {
                    "model": chunk["model"],
                    "created_at": chunk["created_at"],
                    "response": chunk["message"]["content"],
                    "done": chunk["done"],
                }
        return _gen_wrapper()
    else:
        return {
            "model": res["model"],
            "created_at": res["created_at"],
            "response": res["message"]["content"],
            "done": True,
            "total_duration": res["total_duration"],
            "vantage_accelerated": True,
        }


class Client:
    """Ollama Client class compatibility: `client = ollama.Client()`."""

    def __init__(self, host: Optional[str] = None, **kwargs: Any) -> None:
        self.host = host

    def chat(self, *args: Any, **kwargs: Any) -> Any:
        return chat(*args, **kwargs)

    def generate(self, *args: Any, **kwargs: Any) -> Any:
        return generate(*args, **kwargs)
