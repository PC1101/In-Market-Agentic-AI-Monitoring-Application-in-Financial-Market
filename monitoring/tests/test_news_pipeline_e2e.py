"""End-to-end offline test: fixture -> pipeline -> news agent -> supervisor -> log.

Mirrors the --news path of run_classical.py at one decision date, entirely
offline (JSONL fixture + FakeScorer + OfflineStubModel).
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from agentic import (
    as_of_context, run_news_agent, run_supervisor, OfflineStubModel,
    RunLogger, RiskFlag, State,
)
from news import NewsConfig, build_news_block, FakeScorer, TriageMode
from news.records import load_jsonl_news

FIXTURE = Path(__file__).parent / "fixtures" / "news_fixture.jsonl"
AS_OF = "2010-06-15"


@pytest.fixture
def returns():
    """Synthetic strategy curve that turns loss-making alongside the news onset."""
    idx = pd.bdate_range("2010-01-04", "2010-06-15")
    rng = np.random.default_rng(3)
    x = rng.normal(0.0004, 0.006, len(idx))
    x[idx >= pd.Timestamp("2010-06-09")] = rng.normal(
        -0.02, 0.01, int((idx >= pd.Timestamp("2010-06-09")).sum()))
    return pd.Series(x, index=idx)


def test_full_news_path(returns, tmp_path):
    logger = RunLogger(tmp_path / "agent_log.jsonl")
    model = OfflineStubModel()

    # 1. causal news block for the decision date
    records = load_jsonl_news(FIXTURE)
    cfg = NewsConfig(scorer=FakeScorer())
    block = build_news_block(AS_OF, cfg, records=records,
                             n_recent_alarms=2, aggregate_alarm=True)
    mode = TriageMode(block["triage_mode"])
    assert mode is TriageMode.THINKING_MODEL

    # 2. news agent (thinking variant, as run_classical would pick for this mode)
    summary = run_news_agent(block, model, logger=logger,
                             prompt_variant="thinking")
    assert summary.as_of == AS_OF
    assert RiskFlag.MARKET_SELLOFF in summary.risk_flags

    # 3. supervisor sees telemetry + detectors + news
    news_ctx = dict(summary.to_dict(), triage_mode=mode.value,
                    signal=block["signal"])
    ctx = as_of_context(returns, AS_OF,
                        detector_alarms={"page_hinkley": ["2010-06-11"],
                                         "bocpd": ["2010-06-14"]},
                        news=news_ctx)
    assessment = run_supervisor(ctx, model, logger=logger)
    assert assessment.state in (State.ALERT, State.CRITICAL)
    assert "news_context" in assessment.detectors_cited

    # 4. both invocations logged with valid JSON payloads
    log = logger.read_all()
    assert [r["agent"] for r in log] == ["news_context", "performance_supervisor"]
    assert all(r["error"] is None for r in log)
    assert log[0]["assessment"]["risk_flags"]
    assert log[1]["prompt_version"] == "supervisor-v3"


def test_quiet_date_skips_llm(returns, tmp_path):
    records = load_jsonl_news(FIXTURE)
    cfg = NewsConfig(scorer=FakeScorer())
    block = build_news_block("2010-06-05", cfg, records=records)
    assert TriageMode(block["triage_mode"]) is TriageMode.SKIP

    # run_classical passes a quiet status block instead of calling the LLM
    ctx = as_of_context(returns, "2010-06-05",
                        news={"status": "quiet", "triage_mode": "SKIP",
                              "signal": block["signal"]})
    assessment = run_supervisor(ctx, OfflineStubModel())
    assert assessment.state == State.NORMAL


def test_escalation_when_no_coverage_but_alarms(returns):
    cfg = NewsConfig(scorer=FakeScorer())
    # no records at all visible -> zero coverage while detectors alarm
    block = build_news_block(AS_OF, cfg, records=[], n_recent_alarms=2,
                             aggregate_alarm=True)
    assert TriageMode(block["triage_mode"]) is TriageMode.CLASSICAL_ESCALATION
