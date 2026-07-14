from agentic.prompts import (build_supervisor_prompt_v2, SUPERVISOR_PROMPT_VERSION_V2,
                             build_supervisor_prompt)
from agentic.runner import run_supervisor
from agentic.model import OfflineStubModel
from agentic.schemas import AgentAssessment


def _ctx(with_news=True, with_macro=True):
    ctx = {
        "as_of": "2007-08-09",
        "telemetry": {"last_date": "2007-08-09", "n_obs": 500,
                      "recent_mean_daily_return": -0.004, "recent_daily_vol": 0.012,
                      "recent_cum_return": -0.05, "current_drawdown": -0.06,
                      "recent_worst_day": -0.047},
        "detector_alarms": {"page_hinkley": ["2007-08-08"], "bocpd": []},
    }
    if with_macro:
        ctx["macro"] = {"vixcls": {"value": 26.25, "obs_date": "2007-08-08"},
                        "yield_curve_10y_2y": 0.15}
    if with_news:
        ctx["news"] = {"overall_risk": "HIGH",
                       "risk_flags": [{"flag": "forced_deleveraging",
                                       "evidence": "margin call headlines"}],
                       "narrative": "Quant funds unwinding.", "confidence": 0.7,
                       "as_of": "2007-08-09", "n_articles": 12}
    return ctx


def test_v2_prompt_includes_news_and_macro():
    system, user = build_supervisor_prompt_v2(_ctx())
    assert "forced_deleveraging" in user
    assert "vixcls" in user
    assert "2007-08-09" in user


def test_v2_prompt_omits_absent_blocks():
    _, user = build_supervisor_prompt_v2(_ctx(with_news=False, with_macro=False))
    assert "News Context Agent" not in user and "Macro" not in user


def test_run_supervisor_accepts_custom_builder_and_version():
    a = run_supervisor(_ctx(), OfflineStubModel(),
                       prompt_builder=build_supervisor_prompt_v2,
                       prompt_version=SUPERVISOR_PROMPT_VERSION_V2)
    assert isinstance(a, AgentAssessment)
    assert a.as_of == "2007-08-09"


def test_run_supervisor_default_builder_unchanged():
    ctx = _ctx(with_news=False, with_macro=False)
    a = run_supervisor(ctx, OfflineStubModel())
    assert isinstance(a, AgentAssessment)   # Week-2 path still works
