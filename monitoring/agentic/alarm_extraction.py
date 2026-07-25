"""Agentic arm alarm extraction and scoring (preregistration §5).

Converts JSONL run records produced by ``run_agentic.py`` into alarm timestamps,
then scores them against evaluation windows using the same ``WindowMetrics``
dataclass as the classical arm — so H1/H2/H3 comparisons are apples-to-apples.

Scoring rules (verbatim from preregistration §5):

  Agentic detection
    First day in ``[onset, onset+21 trading days]`` whose supervisor assessment
    has ``state ∈ {ALERT, CRITICAL}``.

  False positive (calm windows)
    A detection-cluster start: an ALERT/CRITICAL day with no ALERT/CRITICAL in
    the prior 5 trading days.  FPR/day = cluster_starts ÷ trading_days.

  Runtime failures
    Schema-invalid after one repair retry, or CLASSICAL_ESCALATION day where
    the LLM failed, count as *no detection* for the agentic arm on event days
    and are reported separately.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from windows import Window
from metrics import WindowMetrics, aggregate_metrics
from .schemas import State


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def _is_alarm(record: dict) -> bool:
    """True iff the record has a schema-valid supervisor assessment with ALERT or CRITICAL."""
    assessment = record.get("assessment")
    if not isinstance(assessment, dict):
        return False
    return assessment.get("state") in (State.ALERT, State.CRITICAL)


def _is_runtime_failure(record: dict) -> bool:
    """True if an LLM was called but the assessment is invalid/missing.

    Specifically: triage_mode is not "skip" (the model was invoked) but there
    is no valid assessment dict.
    """
    if record.get("triage_mode") == "skip":
        return False
    return not isinstance(record.get("assessment"), dict)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_agentic_alarms(records: list[dict]) -> list[pd.Timestamp]:
    """Return sorted alarm timestamps from agentic run records.

    An alarm day = any trading day with a schema-valid supervisor assessment
    whose ``state ∈ {ALERT, CRITICAL}``.  Days with ``triage_mode == "skip"``
    or runtime failures are not alarms.
    """
    alarms: list[pd.Timestamp] = []
    for rec in records:
        if _is_alarm(rec):
            alarms.append(pd.Timestamp(rec["as_of"]))
    return sorted(alarms)


def cluster_starts(
    alarm_days: list[pd.Timestamp],
    trading_dates: pd.DatetimeIndex,
    dedup_days: int = 5,
) -> list[pd.Timestamp]:
    """Reduce alarm list to cluster starts (5 trading-day dedup, preregistration §5).

    A cluster start is an alarm day with no prior alarm in the previous
    ``dedup_days`` *trading* days (not calendar days).

    Args:
        alarm_days:    Sorted alarm timestamps.
        trading_dates: Full DatetimeIndex of trading days in the relevant window.
        dedup_days:    Trading-day cooldown (default 5, matches aggregation rule).

    Returns:
        Subset of alarm_days that are cluster starts, in chronological order.
    """
    if not alarm_days:
        return []
    sorted_alarms = sorted(alarm_days)
    starts: list[pd.Timestamp] = [sorted_alarms[0]]
    for alarm in sorted_alarms[1:]:
        prev = starts[-1]
        # Count trading days between prev cluster start and this alarm.
        n_td = int(np.busday_count(prev.date(), alarm.date()))
        if n_td > dedup_days:
            starts.append(alarm)
    return starts


def count_runtime_failures(records: list[dict]) -> int:
    """Count records where the LLM was invoked but produced no valid assessment."""
    return sum(1 for r in records if _is_runtime_failure(r))


def evaluate_agentic_window(
    records: list[dict],
    window: Window,
    trading_dates: pd.DatetimeIndex,
    tolerance_days: int = 21,
    dedup_days: int = 5,
    override_onset: "pd.Timestamp | None" = None,
) -> "AgenticWindowMetrics":
    """Score agentic performance on one evaluation window.

    Event windows:
        Detection = first ALERT/CRITICAL in [onset, onset+21 trading days].
        Runtime failures on event days count as non-detection (they are tallied).

    Calm windows:
        FPR = cluster_starts / trading_days using ``dedup_days``-trading-day dedup.

    Returns:
        An AgenticWindowMetrics instance (extends WindowMetrics with failure counts).
    """
    window_records = [
        r for r in records
        if window.start_ts <= pd.Timestamp(r["as_of"]) <= window.end_ts
    ]
    n_td = int(trading_dates[(trading_dates >= window.start_ts) & (trading_dates <= window.end_ts)].shape[0])
    if n_td == 0:
        n_td = int(np.busday_count(
            window.start_ts.date(), (window.end_ts + pd.Timedelta(days=1)).date()
        ))

    alarm_days = [pd.Timestamp(r["as_of"]) for r in window_records if _is_alarm(r)]
    n_failures = sum(1 for r in window_records if _is_runtime_failure(r))

    if window.kind == "event":
        onset = override_onset if override_onset is not None else window.onset_ts
        zone_end = onset + pd.Timedelta(days=tolerance_days)
        in_zone = [a for a in alarm_days if onset <= a <= zone_end]
        detected = len(in_zone) > 0
        latency = int((in_zone[0] - onset).days) if detected else None
        tp = len(in_zone)
        fp = len([a for a in alarm_days if a < onset or a > zone_end])
        wm = WindowMetrics(
            window=window.name, kind="event", n_alarms=len(alarm_days),
            detected=detected, latency_days=latency,
            n_true_positives=tp, n_false_positives=fp,
            n_trading_days=n_td, fpr=None,
        )
        return AgenticWindowMetrics(base=wm, n_runtime_failures=n_failures)

    # Calm: FPR via cluster starts
    window_td = trading_dates[(trading_dates >= window.start_ts) & (trading_dates <= window.end_ts)]
    clusters = cluster_starts(alarm_days, window_td, dedup_days=dedup_days)
    fp = len(clusters)
    fpr = fp / n_td if n_td > 0 else None
    wm = WindowMetrics(
        window=window.name, kind="calm", n_alarms=len(alarm_days),
        detected=None, latency_days=None,
        n_true_positives=0, n_false_positives=fp,
        n_trading_days=n_td, fpr=fpr,
    )
    return AgenticWindowMetrics(base=wm, n_runtime_failures=n_failures)


@dataclass
class AgenticWindowMetrics:
    """WindowMetrics + agentic-specific runtime-failure count."""

    base: WindowMetrics
    n_runtime_failures: int = 0

    # Delegate attribute access to base so this is duck-type compatible
    # with WindowMetrics in aggregate_metrics().
    def __getattr__(self, name: str):
        return getattr(self.base, name)

    def to_window_metrics(self) -> WindowMetrics:
        return self.base


def load_records(jsonl_path: str) -> list[dict]:
    """Load agentic run records from a JSONL file."""
    import json
    from pathlib import Path
    records = []
    for line in Path(jsonl_path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            records.append(json.loads(line))
    return records


def reconstruct_day_records(jsonl_records: list[dict]) -> list[dict]:
    """Merge per-agent JSONL entries into one record per trading day.

    The run_agentic.py JSONL logger writes multiple entries per day
    (agent="triage", agent="news_context", agent="performance_supervisor").
    This function groups them by ``as_of`` date and merges:
      - triage_mode, triage_reason, finbert_stress from the triage entry
      - assessment from the performance_supervisor entry (if present)

    Returns one dict per trading day, compatible with evaluate_agentic_window().
    """
    from collections import defaultdict
    by_date: dict[str, dict] = {}

    for r in jsonl_records:
        agent = r.get("agent", "")
        as_of = r.get("as_of")
        if not as_of:
            continue

        if as_of not in by_date:
            by_date[as_of] = {"as_of": as_of}

        if agent == "triage":
            by_date[as_of].update({
                "triage_mode": r.get("triage_mode"),
                "triage_reason": r.get("triage_reason"),
                "finbert_stress": r.get("finbert_stress"),
                "window": r.get("window"),
            })
        elif agent == "performance_supervisor":
            assessment = r.get("assessment")
            if isinstance(assessment, dict):
                by_date[as_of]["assessment"] = assessment

    return sorted(by_date.values(), key=lambda r: r["as_of"])
