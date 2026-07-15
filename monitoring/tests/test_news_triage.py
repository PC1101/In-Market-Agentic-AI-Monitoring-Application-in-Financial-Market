"""Triage-mode tests: all four modes reachable from crafted signal frames."""

import math

import pandas as pd
import pytest

from news.triage import (
    triage, TriageMode, STRESS_CHEAP, STRESS_THINKING, HIT_Z_THINKING,
)


def _frame(n_articles, n_hits, stress, hit_z):
    """Build a signal frame from equal-length lists (last row = most recent)."""
    days = pd.date_range("2010-06-01", periods=len(n_articles))
    return pd.DataFrame({
        "n_articles": n_articles,
        "n_hits": n_hits,
        "hit_rate": [h / a if a else math.nan for h, a in zip(n_hits, n_articles)],
        "stress": stress,
        "hit_z": hit_z,
    }, index=days)


QUIET = _frame([5] * 10, [0] * 10, [math.nan] * 10, [0.0] * 10)


def test_skip_when_all_quiet():
    assert triage(QUIET) is TriageMode.SKIP


def test_cheap_on_recent_hits():
    f = _frame([5] * 10, [0] * 9 + [1], [math.nan] * 9 + [0.1], [0.5] * 10)
    assert triage(f) is TriageMode.CHEAP_MODEL


def test_cheap_on_stress_threshold():
    f = _frame([5] * 10, [0] * 10, [math.nan] * 9 + [STRESS_CHEAP], [0.0] * 10)
    assert triage(f) is TriageMode.CHEAP_MODEL


def test_cheap_on_detector_alarm_even_if_news_quiet():
    assert triage(QUIET, n_recent_alarms=1) is TriageMode.CHEAP_MODEL


def test_old_hits_outside_recent_window_do_not_trigger():
    # hits 5+ days ago, nothing in the trailing 3 days
    f = _frame([5] * 10, [3] * 5 + [0] * 5, [0.8] * 5 + [math.nan] * 5, [0.0] * 10)
    assert triage(f) is TriageMode.SKIP


def test_thinking_needs_spike_AND_aggregate_alarm():
    spike = _frame([5] * 10, [0] * 9 + [4], [math.nan] * 9 + [STRESS_THINKING],
                   [0.0] * 9 + [HIT_Z_THINKING])
    # spike alone -> only CHEAP
    assert triage(spike, aggregate_alarm=False) is TriageMode.CHEAP_MODEL
    # spike + aggregate alarm -> THINKING
    assert triage(spike, aggregate_alarm=True) is TriageMode.THINKING_MODEL


def test_thinking_via_hit_z_alone():
    f = _frame([5] * 10, [0] * 9 + [4], [math.nan] * 9 + [0.1],
               [0.0] * 9 + [HIT_Z_THINKING + 0.5])
    assert triage(f, aggregate_alarm=True) is TriageMode.THINKING_MODEL


def test_classical_escalation_on_zero_coverage_with_alarm():
    empty = _frame([0] * 10, [0] * 10, [math.nan] * 10, [math.nan] * 10)
    assert triage(empty, aggregate_alarm=True) is TriageMode.CLASSICAL_ESCALATION
    assert triage(empty, n_recent_alarms=2) is TriageMode.CLASSICAL_ESCALATION


def test_zero_coverage_without_alarm_is_skip():
    empty = _frame([0] * 10, [0] * 10, [math.nan] * 10, [math.nan] * 10)
    assert triage(empty) is TriageMode.SKIP


@pytest.mark.parametrize("mode", list(TriageMode))
def test_modes_are_json_serialisable(mode):
    assert isinstance(mode.value, str)
    assert TriageMode(mode.value) is mode
