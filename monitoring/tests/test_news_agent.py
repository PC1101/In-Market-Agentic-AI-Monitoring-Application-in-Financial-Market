"""News Context Agent tests: schema validation, prompts, stub run, repair path."""

from pathlib import Path

import pytest

from agentic import (
    validate_news_context, RiskFlag, NewsIntensity, SchemaError,
    OfflineStubModel, RunLogger, run_news_agent, make_model,
)
from agentic.prompts import (
    build_news_prompt, NEWS_PROMPT_VERSION, NEWS_PROMPT_VERSION_THINKING,
)
from news import NewsConfig, build_news_block, FakeScorer
from news.records import load_jsonl_news

FIXTURE = Path(__file__).parent / "fixtures" / "news_fixture.jsonl"


# ---- schema ------------------------------------------------------------------

def _valid():
    return {
        "as_of": "2010-06-15",
        "risk_flags": [RiskFlag.MARKET_SELLOFF, RiskFlag.LIQUIDITY_STRESS],
        "narrative": "Broad sell-off with funding stress in the visible headlines.",
        "news_intensity": NewsIntensity.HIGH,
        "confidence": 0.8,
        "headlines_cited": ["Stocks plunge in panic sell-off"],
    }


def test_valid_news_context_parses():
    s = validate_news_context(_valid())
    assert s.news_intensity == NewsIntensity.HIGH
    assert RiskFlag.MARKET_SELLOFF in s.risk_flags


def test_none_flag_alone_is_valid():
    d = _valid()
    d["risk_flags"] = [RiskFlag.NONE]
    d["headlines_cited"] = []
    assert validate_news_context(d).risk_flags == [RiskFlag.NONE]


@pytest.mark.parametrize("mutate", [
    lambda d: d.pop("risk_flags"),
    lambda d: d.update(risk_flags=[]),
    lambda d: d.update(risk_flags=["PANIC"]),
    lambda d: d.update(risk_flags=[RiskFlag.NONE, RiskFlag.MARKET_SELLOFF]),
    lambda d: d.update(narrative=""),
    lambda d: d.update(narrative="x" * 601),
    lambda d: d.update(news_intensity="EXTREME"),
    lambda d: d.update(confidence=1.5),
    lambda d: d.update(confidence=True),
    lambda d: d.update(as_of="not-a-date"),
    lambda d: d.update(headlines_cited="a headline"),
])
def test_invalid_news_context_rejected(mutate):
    d = _valid()
    mutate(d)
    with pytest.raises(SchemaError):
        validate_news_context(d)


def test_duplicate_flags_deduped():
    d = _valid()
    d["risk_flags"] = [RiskFlag.FUND_STRESS, RiskFlag.FUND_STRESS]
    assert validate_news_context(d).risk_flags == [RiskFlag.FUND_STRESS]


# ---- pipeline block (fixture) ---------------------------------------------------

@pytest.fixture
def block():
    records = load_jsonl_news(FIXTURE)
    cfg = NewsConfig(scorer=FakeScorer())
    return build_news_block("2010-06-15", cfg, records=records,
                            n_recent_alarms=2, aggregate_alarm=True)


def test_block_is_causal(block):
    assert block["cutoff"] == "2010-06-14"
    heads = [h["headline"] for h in block["headlines"]]
    assert all("2010-06-15" not in h["date"] for h in block["headlines"])
    assert "Decision-day story must be excluded by the T-1 cutoff" not in heads
    assert "Post-decision-date story must never be visible" not in heads
    assert "Undated story dropped fail-closed" not in heads


def test_block_drops_future_iso_date_in_text(block):
    # the 2010-06-13 record whose text mentions 2010-06-18 is dropped, loudly
    heads = [h["headline"] for h in block["headlines"]]
    assert not any("2010-06-18" in h for h in heads)
    assert "dropped by lookahead text guard" in block["coverage_note"]


def test_block_triage_thinking(block):
    # stress spike + aggregate alarm on the fixture escalation
    assert block["triage_mode"] == "THINKING_MODEL"
    assert block["signal"]["n_hits_3d"] >= 4


# ---- prompts --------------------------------------------------------------------

def test_prompt_markers_and_variants(block):
    system, user = build_news_prompt(block)
    assert "Decision date (as_of): 2010-06-15" in user
    assert "Aggregate news signal:" in user
    assert "Filtered headlines" in user
    assert "risk_flags" in system

    sys_thinking, _ = build_news_prompt(block, variant="thinking")
    assert len(sys_thinking) > len(system)
    with pytest.raises(ValueError):
        build_news_prompt(block, variant="bogus")


# ---- runner + stub ---------------------------------------------------------------

def test_stub_run_end_to_end(block, tmp_path):
    logger = RunLogger(tmp_path / "log.jsonl")
    summary = run_news_agent(block, OfflineStubModel(), logger=logger)
    assert summary.as_of == "2010-06-15"
    assert RiskFlag.NONE not in summary.risk_flags
    assert RiskFlag.MARKET_SELLOFF in summary.risk_flags
    assert summary.news_intensity in (NewsIntensity.ELEVATED, NewsIntensity.HIGH)

    (rec,) = logger.read_all()
    assert rec["agent"] == "news_context"
    assert rec["prompt_version"] == NEWS_PROMPT_VERSION
    assert rec["error"] is None
    assert rec["triage_mode"] == "THINKING_MODEL"


def test_stub_quiet_block_returns_none_flag():
    records = load_jsonl_news(FIXTURE)
    cfg = NewsConfig(scorer=FakeScorer())
    quiet = build_news_block("2010-06-05", cfg, records=records)
    assert quiet["triage_mode"] == "SKIP"
    summary = run_news_agent(quiet, OfflineStubModel())
    assert summary.risk_flags == [RiskFlag.NONE]
    assert summary.news_intensity == NewsIntensity.LOW


class _FlakyModel(OfflineStubModel):
    """Returns garbage once, then defers to the stub — exercises the repair retry."""

    def __init__(self):
        self.calls = 0

    def complete(self, system, user):
        self.calls += 1
        if self.calls == 1:
            return {"risk_flags": ["PANIC"]}  # missing fields + bad enum
        return super().complete(system, user)


def test_repair_retry_logged(block, tmp_path):
    logger = RunLogger(tmp_path / "log.jsonl")
    model = _FlakyModel()
    summary = run_news_agent(block, model, logger=logger,
                             prompt_variant="thinking")
    assert model.calls == 2
    assert summary.as_of == "2010-06-15"
    (rec,) = logger.read_all()
    assert rec["error"] is not None  # first failure recorded
    assert rec["prompt_version"] == NEWS_PROMPT_VERSION_THINKING


class _AlwaysBadModel(OfflineStubModel):
    def complete(self, system, user):
        return {"nope": 1}


def test_unrepairable_output_raises(block):
    with pytest.raises(SchemaError):
        run_news_agent(block, _AlwaysBadModel())


# ---- make_model -----------------------------------------------------------------

def test_make_model_agent_schemas():
    from agentic import NEWS_CONTEXT_JSON_SCHEMA, ASSESSMENT_JSON_SCHEMA
    m_news = make_model("ollama:x", agent="news_context")
    m_sup = make_model("ollama:x")  # default preserves Week-2 behavior
    assert m_news.json_schema is NEWS_CONTEXT_JSON_SCHEMA
    assert m_sup.json_schema is ASSESSMENT_JSON_SCHEMA
    with pytest.raises(ValueError):
        make_model("ollama:x", agent="mystery")
