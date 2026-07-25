"""Tests for agentic alarm extraction (preregistration §5)."""
import pandas as pd
import pytest

from agentic.alarm_extraction import (
    extract_agentic_alarms, cluster_starts, count_runtime_failures,
    evaluate_agentic_window, reconstruct_day_records,
)
from windows import get_window


def _make_record(as_of, state=None, triage_mode="classical_escalation", error=None):
    """Helper: build a minimal agentic run record."""
    rec = {"as_of": as_of, "triage_mode": triage_mode}
    if state is not None:
        rec["assessment"] = {"state": state, "action": "REDUCE",
                             "root_cause": "test", "confidence": 0.9}
    if error is not None:
        rec["error"] = error
    return rec


# ---------------------------------------------------------------------------
# extract_agentic_alarms
# ---------------------------------------------------------------------------

def test_extracts_alert_and_critical():
    records = [
        _make_record("2010-05-03", "NORMAL"),
        _make_record("2010-05-04", "ALERT"),
        _make_record("2010-05-05", "WATCH"),
        _make_record("2010-05-06", "CRITICAL"),
    ]
    alarms = extract_agentic_alarms(records)
    assert len(alarms) == 2
    assert pd.Timestamp("2010-05-04") in alarms
    assert pd.Timestamp("2010-05-06") in alarms


def test_skip_records_not_alarms():
    records = [
        {"as_of": "2010-05-03", "triage_mode": "skip"},
        _make_record("2010-05-04", "ALERT"),
    ]
    alarms = extract_agentic_alarms(records)
    assert len(alarms) == 1


def test_runtime_failure_not_alarm():
    # No assessment dict → runtime failure, not an alarm
    records = [
        {"as_of": "2010-05-04", "triage_mode": "classical_escalation", "error": "schema_invalid"},
    ]
    alarms = extract_agentic_alarms(records)
    assert len(alarms) == 0


def test_empty_records():
    assert extract_agentic_alarms([]) == []


# ---------------------------------------------------------------------------
# cluster_starts (5-day dedup)
# ---------------------------------------------------------------------------

def _bdays(start, n):
    return pd.bdate_range(start, periods=n)


def test_cluster_starts_deduplicates():
    trading = _bdays("2012-01-02", 30)
    # Three alarms on consecutive trading days → one cluster start
    alarms = [trading[0], trading[1], trading[2]]
    clusters = cluster_starts(alarms, trading, dedup_days=5)
    assert len(clusters) == 1
    assert clusters[0] == trading[0]


def test_cluster_starts_two_clusters():
    trading = _bdays("2012-01-02", 30)
    # Alarm at day 0 and day 8 (>5 trading days apart) → two clusters
    alarms = [trading[0], trading[8]]
    clusters = cluster_starts(alarms, trading, dedup_days=5)
    assert len(clusters) == 2


def test_cluster_starts_empty():
    trading = _bdays("2012-01-02", 20)
    assert cluster_starts([], trading) == []


# ---------------------------------------------------------------------------
# count_runtime_failures
# ---------------------------------------------------------------------------

def test_counts_escalation_failures():
    records = [
        _make_record("2010-05-03", "ALERT"),               # valid → not a failure
        {"as_of": "2010-05-04", "triage_mode": "classical_escalation"},  # no assessment → failure
        {"as_of": "2010-05-05", "triage_mode": "skip"},    # skip → not a failure
        {"as_of": "2010-05-06", "triage_mode": "thinking"},# no assessment → failure
    ]
    assert count_runtime_failures(records) == 2


# ---------------------------------------------------------------------------
# evaluate_agentic_window — event window
# ---------------------------------------------------------------------------

def test_event_window_detection_within_tolerance():
    w = get_window("flash_crash_2010")  # onset 2010-05-06
    trading = pd.bdate_range(w.start_ts, w.end_ts)
    records = [
        _make_record(str(w.onset_ts.date()), "ALERT"),  # alarm ON onset → detected, latency=0
    ]
    result = evaluate_agentic_window(records, w, trading)
    assert result.detected is True
    assert result.latency_days == 0


def test_event_window_no_detection():
    w = get_window("flash_crash_2010")
    trading = pd.bdate_range(w.start_ts, w.end_ts)
    # Alarm before onset → not in zone
    records = [_make_record(str(w.start_ts.date()), "CRITICAL")]
    result = evaluate_agentic_window(records, w, trading)
    assert result.detected is False
    assert result.latency_days is None


# ---------------------------------------------------------------------------
# evaluate_agentic_window — calm window
# ---------------------------------------------------------------------------

def test_calm_window_fpr_with_cluster_dedup():
    w = get_window("calm_2012")
    trading = pd.bdate_range(w.start_ts, w.end_ts)
    # Two alarms 3 trading days apart → one cluster; one alarm 10 days later → second cluster
    d0 = trading[50]
    d1 = trading[53]   # 3 days later → same cluster
    d2 = trading[65]   # 12 days later → new cluster
    records = [
        _make_record(str(d0.date()), "ALERT"),
        _make_record(str(d1.date()), "CRITICAL"),
        _make_record(str(d2.date()), "ALERT"),
    ]
    result = evaluate_agentic_window(records, w, trading)
    assert result.kind == "calm"
    assert result.n_false_positives == 2  # 2 cluster starts
    assert result.fpr is not None and result.fpr > 0


# ---------------------------------------------------------------------------
# reconstruct_day_records — JSONL log format → per-day records
# ---------------------------------------------------------------------------

def _make_jsonl_entries(as_of, triage_mode="thinking", state=None):
    """Build realistic JSONL log entries for one trading day."""
    entries = [
        {"agent": "triage", "as_of": as_of, "triage_mode": triage_mode,
         "triage_reason": "stress high", "finbert_stress": 0.45, "window": "test"},
    ]
    if triage_mode != "skip" and state is not None:
        entries.append({
            "agent": "news_context", "as_of": as_of,
            "overall_risk": "HIGH", "confidence": 0.8,
        })
        entries.append({
            "agent": "performance_supervisor", "as_of": as_of,
            "assessment": {"state": state, "action": "REDUCE",
                           "root_cause": "stress", "confidence": 0.85},
            "latency_s": 12.3,
        })
    return entries


def test_reconstruct_merges_triage_and_supervisor():
    entries = _make_jsonl_entries("2008-09-15", "thinking", "ALERT")
    day_recs = reconstruct_day_records(entries)
    assert len(day_recs) == 1
    assert day_recs[0]["as_of"] == "2008-09-15"
    assert day_recs[0]["triage_mode"] == "thinking"
    assert day_recs[0]["assessment"]["state"] == "ALERT"


def test_reconstruct_skip_day_has_no_assessment():
    entries = _make_jsonl_entries("2008-09-15", "skip", state=None)
    day_recs = reconstruct_day_records(entries)
    assert len(day_recs) == 1
    assert day_recs[0]["triage_mode"] == "skip"
    assert "assessment" not in day_recs[0]


def test_reconstruct_multiple_days_sorted():
    entries = (
        _make_jsonl_entries("2008-09-16", "thinking", "WATCH")
        + _make_jsonl_entries("2008-09-15", "thinking", "ALERT")
    )
    day_recs = reconstruct_day_records(entries)
    assert len(day_recs) == 2
    assert day_recs[0]["as_of"] == "2008-09-15"
    assert day_recs[1]["as_of"] == "2008-09-16"


def test_reconstruct_runtime_failure_no_assessment():
    """Triage says non-skip but supervisor never logged → runtime failure."""
    entries = [
        {"agent": "triage", "as_of": "2008-09-15", "triage_mode": "thinking",
         "finbert_stress": 0.5, "window": "test"},
        # No performance_supervisor entry → assessment missing
    ]
    day_recs = reconstruct_day_records(entries)
    assert len(day_recs) == 1
    assert "assessment" not in day_recs[0]
    # Should be counted as runtime failure by count_runtime_failures
    assert count_runtime_failures(day_recs) == 1


def test_reconstruct_zero_runtime_failures_with_valid_assessment():
    entries = _make_jsonl_entries("2008-09-15", "thinking", "NORMAL")
    day_recs = reconstruct_day_records(entries)
    assert count_runtime_failures(day_recs) == 0


def test_reconstruct_empty_input():
    assert reconstruct_day_records([]) == []


# ---------------------------------------------------------------------------
# evaluate_agentic_window — override_onset
# ---------------------------------------------------------------------------

def test_override_onset_shifts_agentic_latency():
    w = get_window("flash_crash_2010")  # onset 2010-05-06
    trading = pd.bdate_range(w.start_ts, w.end_ts)

    # Alarm 3 days after hardcoded onset → latency=3
    alarm_date = pd.Timestamp("2010-05-11")
    records = [_make_record(str(alarm_date.date()), "ALERT")]

    m_default = evaluate_agentic_window(records, w, trading)
    assert m_default.latency_days == int((alarm_date - w.onset_ts).days)

    # Override onset 2 days later → latency increases by 2 (alarm is 1 day after new onset? no: 2010-05-06 + 2 = 2010-05-10 + 1 day = 2010-05-11, so latency = 1)
    override = pd.Timestamp("2010-05-10")
    m_override = evaluate_agentic_window(records, w, trading, override_onset=override)
    expected_lat = int((alarm_date - override).days)
    assert m_override.latency_days == expected_lat
    assert m_override.detected is True


def test_override_onset_none_reproduces_default():
    w = get_window("flash_crash_2010")
    trading = pd.bdate_range(w.start_ts, w.end_ts)
    records = [_make_record(str(w.onset_ts.date()), "ALERT")]

    m_default = evaluate_agentic_window(records, w, trading)
    m_none = evaluate_agentic_window(records, w, trading, override_onset=None)

    assert m_default.detected == m_none.detected
    assert m_default.latency_days == m_none.latency_days
