"""Supervisor v3 tests: news in context/prompt, stub escalation, backward compat."""

import numpy as np
import pandas as pd
import pytest

from agentic import (
    as_of_context, run_supervisor, OfflineStubModel, RiskFlag, NewsIntensity,
    State, Action,
)
from agentic.guardrails import LookaheadError
from agentic.prompts import build_supervisor_prompt, SUPERVISOR_PROMPT_VERSION


@pytest.fixture
def returns():
    idx = pd.bdate_range("2010-01-04", periods=120)
    rng = np.random.default_rng(7)
    return pd.Series(rng.normal(0.0003, 0.006, len(idx)), index=idx)


AS_OF = "2010-06-15"


def _news(intensity=NewsIntensity.HIGH, flags=(RiskFlag.MARKET_SELLOFF,)):
    return {
        "as_of": AS_OF,
        "risk_flags": list(flags),
        "narrative": "Stress headlines dominate the visible window.",
        "news_intensity": intensity,
        "confidence": 0.8,
        "headlines_cited": [],
        "triage_mode": "CHEAP_MODEL",
    }


# ---- guardrails -------------------------------------------------------------------

def test_context_carries_news(returns):
    ctx = as_of_context(returns, AS_OF, news=_news())
    assert ctx["news"]["news_intensity"] == NewsIntensity.HIGH


def test_context_without_news_unchanged(returns):
    ctx = as_of_context(returns, AS_OF)
    assert "news" not in ctx  # Week-2 baseline shape preserved


def test_future_dated_news_rejected(returns):
    bad = _news()
    bad["narrative"] = "References an event on 2010-06-20."
    with pytest.raises(LookaheadError):
        as_of_context(returns, AS_OF, news=bad)


# ---- prompt ------------------------------------------------------------------------

def test_prompt_version_is_v3():
    assert SUPERVISOR_PROMPT_VERSION == "supervisor-v3"


def test_prompt_includes_news_block(returns):
    ctx = as_of_context(returns, AS_OF, news=_news())
    _, user = build_supervisor_prompt(ctx)
    assert "News context summary:" in user
    assert "MARKET_SELLOFF" in user


def test_prompt_without_news_says_none(returns):
    ctx = as_of_context(returns, AS_OF)
    _, user = build_supervisor_prompt(ctx)
    assert "News context summary: none available" in user


# ---- stub behavior ------------------------------------------------------------------

def test_stub_bumps_severity_on_hot_news(returns):
    stub = OfflineStubModel()
    base = run_supervisor(as_of_context(returns, AS_OF), stub)
    hot = run_supervisor(as_of_context(returns, AS_OF, news=_news()), stub)
    order = [State.NORMAL, State.WATCH, State.ALERT, State.CRITICAL]
    assert order.index(hot.state) > order.index(base.state)
    assert "news_context" in hot.detectors_cited


def test_stub_ignores_quiet_and_low_news(returns):
    stub = OfflineStubModel()
    base = run_supervisor(as_of_context(returns, AS_OF), stub)
    quiet = run_supervisor(
        as_of_context(returns, AS_OF, news={"status": "quiet",
                                            "triage_mode": "SKIP"}), stub)
    low = run_supervisor(
        as_of_context(returns, AS_OF,
                      news=_news(intensity=NewsIntensity.LOW)), stub)
    none_flag = run_supervisor(
        as_of_context(returns, AS_OF,
                      news=_news(flags=(RiskFlag.NONE,))), stub)
    for a in (quiet, low, none_flag):
        assert a.state == base.state
        assert "news_context" not in a.detectors_cited


def test_stub_elevated_news_at_least_watch(returns):
    stub = OfflineStubModel()
    a = run_supervisor(
        as_of_context(returns, AS_OF,
                      news=_news(intensity=NewsIntensity.ELEVATED)), stub)
    assert a.state != State.NORMAL
    assert a.action != Action.HOLD
