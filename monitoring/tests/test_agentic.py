import numpy as np
import pandas as pd
import pytest

from agentic import (
    validate_assessment, State, Action, ASSESSMENT_JSON_SCHEMA,
    as_of_context, assert_no_lookahead, filter_news_by_timestamp,
    OfflineStubModel, RunLogger, run_supervisor,
)
from agentic.schemas import SchemaError
from agentic.guardrails import LookaheadError


# ---- schema ------------------------------------------------------------------

def _valid():
    return {
        "state": State.WATCH,
        "action": Action.INVESTIGATE,
        "root_cause": "detectors fired",
        "confidence": 0.7,
        "as_of": "2008-09-15",
        "detectors_cited": ["page_hinkley"],
    }


def test_valid_assessment_parses():
    a = validate_assessment(_valid())
    assert a.state == State.WATCH
    assert a.confidence == 0.7


@pytest.mark.parametrize("mutate", [
    lambda d: d.pop("state"),
    lambda d: d.update(state="PANIC"),
    lambda d: d.update(action="SELL_EVERYTHING"),
    lambda d: d.update(confidence=1.5),
    lambda d: d.update(confidence="high"),
    lambda d: d.update(root_cause=""),
    lambda d: d.update(as_of="not-a-date"),
    lambda d: d.update(detectors_cited="page_hinkley"),
])
def test_invalid_assessment_rejected(mutate):
    d = _valid()
    mutate(d)
    with pytest.raises(SchemaError):
        validate_assessment(d)


def test_confidence_bool_rejected():
    d = _valid()
    d["confidence"] = True
    with pytest.raises(SchemaError):
        validate_assessment(d)


# ---- guardrails --------------------------------------------------------------

@pytest.fixture
def returns():
    idx = pd.bdate_range("2008-01-01", periods=250)
    rng = np.random.default_rng(0)
    return pd.Series(rng.normal(0, 0.01, 250), index=idx, name="port_ret")


def test_as_of_context_is_causal(returns):
    as_of = returns.index[150]
    alarms = {"page_hinkley": [returns.index[100], returns.index[200]]}  # one future
    ctx = as_of_context(returns, as_of, alarms)
    assert ctx["as_of"] == str(as_of.date())
    # the future alarm (index 200) must be dropped
    assert len(ctx["detector_alarms"]["page_hinkley"]) == 1
    assert ctx["telemetry"]["last_date"] == str(as_of.date())


def test_assert_no_lookahead_catches_future_date():
    ctx = {"as_of": "2008-06-01", "note": "2009-01-01 event"}
    with pytest.raises(LookaheadError):
        assert_no_lookahead(ctx, "2008-06-01")


def test_news_filter_drops_future_and_undated():
    recs = [
        {"timestamp": "2008-09-01", "h": "a"},
        {"timestamp": "2008-09-20", "h": "b"},   # future
        {"h": "no-date"},                         # undated -> dropped fail-closed
    ]
    kept = filter_news_by_timestamp(recs, "2008-09-15")
    assert [r["h"] for r in kept] == ["a"]


# ---- end-to-end runner with the offline stub --------------------------------

def test_run_supervisor_offline_produces_valid_json(returns, tmp_path):
    as_of = returns.index[-1]
    ctx = as_of_context(returns, as_of, {"page_hinkley": [returns.index[-5]]})
    logger = RunLogger(tmp_path / "agent_log.jsonl")
    a = run_supervisor(ctx, OfflineStubModel(), logger=logger)
    assert a.state in State.ALL
    assert a.action in Action.ALL
    assert 0.0 <= a.confidence <= 1.0
    assert a.as_of == str(as_of.date())
    # logged exactly one invocation
    rows = logger.read_all()
    assert len(rows) == 1
    assert rows[0]["assessment"]["state"] == a.state


def test_stub_escalates_with_more_detectors(returns):
    as_of = returns.index[-1]
    calm = as_of_context(returns, as_of, {})
    stressed = as_of_context(returns, as_of, {
        "page_hinkley": [as_of], "bocpd": [as_of], "hmm": [as_of],
    })
    a_calm = run_supervisor(calm, OfflineStubModel())
    a_stressed = run_supervisor(stressed, OfflineStubModel())
    order = {s: i for i, s in enumerate(State.ALL)}
    assert order[a_stressed.state] > order[a_calm.state]
