"""Unit tests for Ollama Drop-in Adapter."""

import pytest
from core.adapters.ollama import _resolve_model_name, Client


def test_ollama_model_resolution():
    assert "llama" in _resolve_model_name("llama3:8b").lower()
    assert "qwen" in _resolve_model_name("qwen2.5:3b").lower()
    assert "mistral" in _resolve_model_name("mistral:7b").lower()
    # Non-aliased model passes through
    assert _resolve_model_name("custom/my-model") == "custom/my-model"


def test_ollama_client_interface():
    client = Client(host="http://localhost:11434")
    assert hasattr(client, "chat")
    assert hasattr(client, "generate")
