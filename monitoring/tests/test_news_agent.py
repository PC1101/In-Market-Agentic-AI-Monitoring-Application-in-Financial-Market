"""Tests for the model factory (Task 2) and News Context Agent (Task 10)."""
import pytest

from agentic.model import default_model, OfflineStubModel, OllamaModel


def test_default_model_stub():
    assert isinstance(default_model("stub"), OfflineStubModel)


def test_default_model_ollama_with_name():
    m = default_model("ollama:qwen2.5:3b")
    assert isinstance(m, OllamaModel)
    assert m.model == "qwen2.5:3b"


def test_default_model_ollama_default_is_qwen():
    m = default_model("ollama")
    assert isinstance(m, OllamaModel)
    assert m.model == "qwen2.5:3b"


def test_default_model_env_fallback(monkeypatch):
    monkeypatch.delenv("MONITOR_MODEL", raising=False)
    assert isinstance(default_model(None), OfflineStubModel)
    monkeypatch.setenv("MONITOR_MODEL", "ollama:qwen2.5:3b")
    m = default_model(None)
    assert isinstance(m, OllamaModel)


def test_default_model_rejects_unknown():
    with pytest.raises(ValueError):
        default_model("gpt-4")
