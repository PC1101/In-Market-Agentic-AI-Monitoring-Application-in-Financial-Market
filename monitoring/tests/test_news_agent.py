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
    assert m.model == "qwen2.5:1.5b"


def test_default_model_env_fallback(monkeypatch):
    monkeypatch.delenv("MONITOR_MODEL", raising=False)
    assert isinstance(default_model(None), OfflineStubModel)
    monkeypatch.setenv("MONITOR_MODEL", "ollama:qwen2.5:3b")
    m = default_model(None)
    assert isinstance(m, OllamaModel)


def test_default_model_rejects_unknown():
    with pytest.raises(ValueError):
        default_model("gpt-4")


from agentic.guardrails import scrub_future_dated
from agentic.news_agent import run_news_agent
from agentic.schemas import NewsFlags


def _articles():
    return [
        {"date": "2007-08-08", "ticker": "GS",
         "title": "Quant meltdown: funds face margin calls",
         "summary": "Widespread liquidation reported."},
        {"date": "2007-08-09", "ticker": "",
         "title": "Crisis talk grows as selloff continues", "summary": ""},
        {"date": "2007-08-09", "ticker": "MS",
         "title": "Fed meets on 2007-09-18 to discuss rates",   # future date in TEXT
         "summary": ""},
    ]


def test_scrub_future_dated_drops_future_text_mentions():
    kept = scrub_future_dated(_articles(), "2007-08-09")
    assert len(kept) == 2
    assert all("2007-09-18" not in a["title"] for a in kept)


def test_run_news_agent_returns_validated_flags():
    flags = run_news_agent("2007-08-09", _articles(), OfflineStubModel())
    assert isinstance(flags, NewsFlags)
    assert flags.as_of == "2007-08-09"
    assert flags.overall_risk in ("LOW", "ELEVATED", "HIGH", "SEVERE")


def test_run_news_agent_refuses_future_articles():
    future = [{"date": "2007-08-20", "ticker": "", "title": "tomorrow", "summary": ""}]
    flags = run_news_agent("2007-08-09", future, OfflineStubModel())
    assert flags.n_articles == 0   # timestamp filter removed everything, fail-closed
