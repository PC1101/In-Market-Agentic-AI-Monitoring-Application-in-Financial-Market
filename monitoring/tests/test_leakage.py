"""Tests for Phase 3 leakage controls (preregistration §8)."""
import copy
import numpy as np
import pandas as pd
import pytest

from agentic.guardrails import mask_dates_in_context, mask_dates_in_articles
from leakage.synthetic import (
    generate_synthetic_windows, SyntheticWindow,
    _measure_crash_template, _block_bootstrap, _inject_crash,
)
from run_leakage import compute_leakage_bound
from windows import EVENT_WINDOWS


# ---------------------------------------------------------------------------
# 3A: Date masking — mask_dates_in_context
# ---------------------------------------------------------------------------

def _make_context(as_of="2008-09-15"):
    return {
        "as_of": as_of,
        "telemetry": {
            "last_date": as_of,
            "recent_mean_daily_return": -0.025,
            "recent_daily_vol": 0.018,
            "current_drawdown": -0.15,
            "recent_worst_day": -0.05,
            "n_obs": 300,
            "recent_cum_return": -0.12,
        },
        "detector_alarms": {
            "page_hinkley": ["2008-09-12", "2008-09-15"],
            "bocpd": [],
        },
    }


def test_mask_dates_replaces_as_of():
    ctx = _make_context()
    masked = mask_dates_in_context(ctx)
    assert masked["as_of"] == "XXXX-XX-XX"
    assert ctx["as_of"] == "2008-09-15"  # original unchanged


def test_mask_dates_replaces_telemetry_last_date():
    ctx = _make_context()
    masked = mask_dates_in_context(ctx)
    assert masked["telemetry"]["last_date"] == "XXXX-XX-XX"


def test_mask_dates_replaces_alarm_dates():
    ctx = _make_context()
    masked = mask_dates_in_context(ctx)
    assert all(d == "XXXX-XX-XX"
               for d in masked["detector_alarms"]["page_hinkley"])


def test_mask_dates_preserves_numeric_telemetry():
    ctx = _make_context()
    masked = mask_dates_in_context(ctx)
    tel = masked["telemetry"]
    assert tel["recent_mean_daily_return"] == pytest.approx(-0.025)
    assert tel["current_drawdown"] == pytest.approx(-0.15)
    assert tel["n_obs"] == 300


def test_mask_dates_no_year_in_masked_context():
    ctx = _make_context("2008-09-15")
    masked = mask_dates_in_context(ctx)
    # Stringify and check no real years leaked through
    as_str = str(masked)
    assert "2008" not in as_str


def test_mask_dates_deep_copy():
    ctx = _make_context()
    masked = mask_dates_in_context(ctx)
    # Mutating masked should not affect original
    masked["as_of"] = "changed"
    assert ctx["as_of"] == "2008-09-15"


def test_mask_dates_nested_macro_block():
    ctx = _make_context()
    ctx["macro"] = {"date": "2008-09-15", "vix": 40.2, "obs": [{"date": "2008-09-12"}]}
    masked = mask_dates_in_context(ctx)
    assert masked["macro"]["date"] == "XXXX-XX-XX"
    assert masked["macro"]["obs"][0]["date"] == "XXXX-XX-XX"
    assert masked["macro"]["vix"] == pytest.approx(40.2)


# ---------------------------------------------------------------------------
# 3A: Date masking — mask_dates_in_articles
# ---------------------------------------------------------------------------

def _make_articles():
    return [
        {"date": "2008-09-14", "ticker": "GS", "title": "Banks under stress on 2008-09-14",
         "summary": "Markets fell sharply after news on September 14, 2008."},
        {"date": "2008-09-15", "ticker": "AIG", "title": "AIG bailout",
         "summary": "Government steps in on 2008-09-15."},
    ]


def test_mask_articles_date_field():
    arts = _make_articles()
    masked = mask_dates_in_articles(arts)
    for a in masked:
        assert a["date"] == "XXXX-XX-XX"


def test_mask_articles_date_in_title():
    arts = _make_articles()
    masked = mask_dates_in_articles(arts)
    assert "2008" not in masked[0]["title"]
    assert "XXXX-XX-XX" in masked[0]["title"]


def test_mask_articles_date_in_summary():
    arts = _make_articles()
    masked = mask_dates_in_articles(arts)
    assert "2008-09-15" not in masked[1]["summary"]


def test_mask_articles_preserves_non_date_fields():
    arts = _make_articles()
    masked = mask_dates_in_articles(arts)
    assert masked[0]["ticker"] == "GS"
    assert "Banks under stress" in masked[0]["title"]


def test_mask_articles_does_not_mutate_original():
    arts = _make_articles()
    _ = mask_dates_in_articles(arts)
    assert arts[0]["date"] == "2008-09-14"


def test_mask_articles_empty_list():
    assert mask_dates_in_articles([]) == []


# ---------------------------------------------------------------------------
# 3B: Synthetic window generation
# ---------------------------------------------------------------------------

def _make_calm_series(n=300, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2004-01-02", periods=n)
    return pd.Series(rng.normal(0, 0.008, size=n), index=dates, name="port_ret")


def _make_dev_event_returns():
    """Build tiny synthetic dev event returns for template measurement."""
    rng = np.random.default_rng(1)
    returns = {}
    for w in EVENT_WINDOWS:
        dates = pd.bdate_range(w.start, periods=80)
        rets = rng.normal(0, 0.01, size=80)
        # Inject a crash at day 20
        rets[20:35] = rng.normal(-0.03, 0.025, size=15)
        returns[w.name] = pd.Series(rets, index=dates, name="port_ret")
    return returns


def test_generate_returns_m_windows():
    calm = _make_calm_series()
    sw_list = generate_synthetic_windows(
        calm, list(EVENT_WINDOWS), _make_dev_event_returns(), M=3, rng_seed=42
    )
    assert len(sw_list) == 3


def test_synthetic_window_has_correct_kind():
    calm = _make_calm_series()
    sw_list = generate_synthetic_windows(calm, list(EVENT_WINDOWS), _make_dev_event_returns(), M=2)
    for sw in sw_list:
        assert sw.window.kind == "event"
        assert sw.window.onset is not None


def test_synthetic_returns_length():
    calm = _make_calm_series()
    sw_list = generate_synthetic_windows(
        calm, list(EVENT_WINDOWS), _make_dev_event_returns(), M=2, window_len=80
    )
    for sw in sw_list:
        assert len(sw.returns) == 80


def test_synthetic_dates_are_far_future():
    calm = _make_calm_series()
    sw_list = generate_synthetic_windows(calm, list(EVENT_WINDOWS), _make_dev_event_returns(), M=2)
    for sw in sw_list:
        assert sw.window.start[:4] == "2099"
        assert sw.window.onset[:4] == "2099"


def test_synthetic_has_articles():
    calm = _make_calm_series()
    sw_list = generate_synthetic_windows(calm, list(EVENT_WINDOWS), _make_dev_event_returns(), M=2)
    for sw in sw_list:
        assert isinstance(sw.articles, list)
        # Should have some articles (stress headlines near onset)
        assert len(sw.articles) > 0


def test_synthetic_no_real_dates_in_articles():
    calm = _make_calm_series()
    sw_list = generate_synthetic_windows(calm, list(EVENT_WINDOWS), _make_dev_event_returns(), M=2)
    for sw in sw_list:
        for art in sw.articles:
            # Articles should be dated in 2099
            assert art["date"].startswith("2099")


def test_synthetic_crash_params_present():
    calm = _make_calm_series()
    sw_list = generate_synthetic_windows(calm, list(EVENT_WINDOWS), _make_dev_event_returns(), M=2)
    for sw in sw_list:
        assert "drawdown_pct" in sw.crash_params
        assert "vol_mult" in sw.crash_params
        assert "duration_days" in sw.crash_params


def test_block_bootstrap_returns_correct_length():
    rng = np.random.default_rng(0)
    arr = pd.Series(np.random.normal(0, 0.01, 200))
    result = _block_bootstrap(arr, target_len=100, block_size=20, rng=rng)
    assert len(result) == 100


def test_inject_crash_changes_returns():
    baseline = np.zeros(60)
    template = {"drawdown_pct": -0.15, "vol_mult": 2.0, "duration_days": 10}
    rng = np.random.default_rng(0)
    result = _inject_crash(baseline, template, onset_idx=20, rng=rng)
    # The crash region should no longer be all zeros
    assert not np.allclose(result[20:30], 0)
    # The pre-crash region should still be zeros
    assert np.allclose(result[:20], 0)


# ---------------------------------------------------------------------------
# run_leakage: compute_leakage_bound
# ---------------------------------------------------------------------------

def test_leakage_bound_evidence_driven():
    score_a = {"detected": True}
    score_b = {"detected": True}
    score_c = {"recall_mean": 0.7}
    bound = compute_leakage_bound(score_a, score_b, score_c)
    assert bound["evidence_skill_lower_bound"] == pytest.approx(0.7)
    assert bound["memorisation_upper_bound"] == pytest.approx(0.3)
    assert bound["conclusion"] == "evidence-driven"


def test_leakage_bound_memorisation_dominated():
    score_a = {"detected": True}
    score_b = {"detected": False}
    score_c = {"recall_mean": 0.1}
    bound = compute_leakage_bound(score_a, score_b, score_c)
    assert bound["memorisation_upper_bound"] > 0
    assert bound["conclusion"] == "memorisation-dominated"


def test_leakage_bound_no_detection():
    score_a = {"detected": False}
    score_b = {"detected": False}
    score_c = {"recall_mean": 0.0}
    bound = compute_leakage_bound(score_a, score_b, score_c)
    assert bound["perf_A"] == pytest.approx(0.0)
    assert bound["memorisation_upper_bound"] == pytest.approx(0.0)


def test_leakage_bound_memorisation_non_negative():
    # memorisation_upper_bound must always be >= 0
    score_a = {"detected": False}
    score_b = {"detected": True}
    score_c = {"recall_mean": 0.8}
    bound = compute_leakage_bound(score_a, score_b, score_c)
    assert bound["memorisation_upper_bound"] >= 0.0


def test_leakage_bound_keys():
    score_a = {"detected": True}
    score_b = {"detected": True}
    score_c = {"recall_mean": 0.5}
    bound = compute_leakage_bound(score_a, score_b, score_c)
    for key in ("evidence_skill_lower_bound", "memorisation_upper_bound",
                "perf_A", "perf_B", "perf_C_mean", "conclusion"):
        assert key in bound
