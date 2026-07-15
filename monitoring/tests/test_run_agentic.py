"""Daily agentic loop (run_agentic.run_window) end-to-end, fully offline.

Uses the JSONL news fixture + FakeScorer + OfflineStubModel — no network, no
ML dependencies, mirroring the offline discipline of the rest of the suite.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from agentic import OfflineStubModel, RunLogger
from agentic.schemas import validate_assessment
from news import NewsConfig, FakeScorer, TriageMode
from news.records import load_jsonl_news
from run_agentic import run_window
from windows import Window

FIXTURE = Path(__file__).parent / "fixtures" / "news_fixture.jsonl"

WINDOW = Window(name="synthetic_event", kind="event",
                start="2010-06-01", end="2010-06-15", onset="2010-06-09",
                description="synthetic crash aligned with the news fixture")


@pytest.fixture
def returns():
    """Calm curve that turns loss-making alongside the fixture's news onset."""
    idx = pd.bdate_range("2010-01-04", "2010-06-15")
    rng = np.random.default_rng(3)
    x = rng.normal(0.0004, 0.006, len(idx))
    crash = idx >= pd.Timestamp("2010-06-09")
    x[crash] = rng.normal(-0.02, 0.01, int(crash.sum()))
    return pd.Series(x, index=idx)


def test_run_window_end_to_end_stub(returns, tmp_path):
    logger = RunLogger(tmp_path / "log.jsonl")
    model = OfflineStubModel()
    cfg = NewsConfig(scorer=FakeScorer())
    records = load_jsonl_news(FIXTURE)

    days = run_window(returns, WINDOW, cfg, supervisor_model=model,
                      news_model=model, logger=logger, records=records,
                      label="TEST:synthetic_event")

    # one record per trading day in the window
    n_days = int(((returns.index >= WINDOW.start_ts)
                  & (returns.index <= WINDOW.end_ts)).sum())
    assert len(days) == n_days and n_days > 5

    modes = {r["triage_mode"] for r in days}
    assert TriageMode.SKIP.value in modes           # quiet pre-onset days skip
    assert modes & {TriageMode.CHEAP_MODEL.value,   # crash days escalate
                    TriageMode.THINKING_MODEL.value,
                    TriageMode.CLASSICAL_ESCALATION.value}

    # SKIP days spend no LLM call; every other day gets a supervisor assessment
    for r in days:
        if r["triage_mode"] == TriageMode.SKIP.value:
            assert "assessment" not in r
        else:
            validate_assessment(r["assessment"])    # raises on invalid
            assert r["assessment"]["as_of"] <= WINDOW.end

    # the JSONL log is replayable and contains both agents
    lines = [json.loads(l) for l in
             (tmp_path / "log.jsonl").read_text(encoding="utf-8").splitlines()]
    agents = {rec["agent"] for rec in lines}
    assert "performance_supervisor" in agents
    assert "news_context" in agents
    # supervisor entries use the news-aware v3 prompt
    sup = [rec for rec in lines if rec["agent"] == "performance_supervisor"]
    assert all(rec["prompt_version"] == "supervisor-v3" for rec in sup)
