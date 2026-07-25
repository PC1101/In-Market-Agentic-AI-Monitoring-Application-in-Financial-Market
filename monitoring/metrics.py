"""Evaluation metrics for change-point alarms against the six-window ground truth.

Scoring rule (clean and symmetric):

  * An alarm is a **true positive** iff it lies within ``[onset, onset + tolerance]`` of
    some event window's onset. Every other alarm is a **false positive**.
  * An event is **detected** (for recall) iff it has >= 1 alarm in its tolerance zone.
  * **Detection latency** = calendar days from an event's onset to its first in-zone alarm.
  * **False-positive rate** = false-positive alarms in calm windows per calm trading day.

Headline metrics (H1/H2/H3 in the VRI plan): mean detection latency, false-positive rate,
precision, recall.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from windows import Window, ALL_WINDOWS


@dataclass
class WindowMetrics:
    """Per-window scoring for one detector/condition."""

    window: str
    kind: str  # "event" | "calm"
    n_alarms: int
    detected: bool | None            # event windows only
    latency_days: int | None         # event windows only
    n_true_positives: int
    n_false_positives: int
    n_trading_days: int
    fpr: float | None                # false alarms per trading day (calm windows)


def evaluate_window(alarms, window: Window, tolerance_days: int = 21,
                    n_trading_days: int | None = None,
                    override_onset: "pd.Timestamp | None" = None) -> WindowMetrics:
    """Score a detector's alarms for a single window.

    Args:
        alarms: iterable of alarm timestamps (any that fall outside the window are ignored).
        window: the Window being scored.
        tolerance_days: detection horizon after an event onset for a TP.
        n_trading_days: number of trading days in the window (for the FPR denominator);
                        if None, uses the count of business days spanned.

    Returns:
        WindowMetrics.
    """
    alarms = sorted(pd.Timestamp(a) for a in alarms)
    in_window = [a for a in alarms if window.contains(a)]

    if n_trading_days is None:
        n_trading_days = int(np.busday_count(
            window.start_ts.date(), (window.end_ts + pd.Timedelta(days=1)).date()
        ))

    if window.kind == "event":
        onset = override_onset if override_onset is not None else window.onset_ts
        zone_end = onset + pd.Timedelta(days=tolerance_days)
        in_zone = [a for a in in_window if onset <= a <= zone_end]
        detected = len(in_zone) > 0
        latency = int((in_zone[0] - onset).days) if detected else None
        tp = len(in_zone)
        fp = len(in_window) - tp
        return WindowMetrics(
            window=window.name, kind="event", n_alarms=len(in_window),
            detected=detected, latency_days=latency,
            n_true_positives=tp, n_false_positives=fp,
            n_trading_days=n_trading_days, fpr=None,
        )

    # Calm window: every alarm is a false positive.
    fp = len(in_window)
    fpr = fp / n_trading_days if n_trading_days > 0 else None
    return WindowMetrics(
        window=window.name, kind="calm", n_alarms=fp,
        detected=None, latency_days=None,
        n_true_positives=0, n_false_positives=fp,
        n_trading_days=n_trading_days, fpr=fpr,
    )


@dataclass
class HeadlineMetrics:
    """Aggregate metrics across a set of windows for one detector/condition."""

    n_events: int
    n_detected: int
    recall: float
    precision: float
    mean_latency_days: float | None
    false_positive_rate: float        # calm false alarms per calm trading day
    n_false_positives_calm: int
    per_window: list[WindowMetrics]


def aggregate_metrics(window_metrics: list[WindowMetrics]) -> HeadlineMetrics:
    """Roll per-window metrics up into headline numbers."""
    events = [m for m in window_metrics if m.kind == "event"]
    calms = [m for m in window_metrics if m.kind == "calm"]

    n_events = len(events)
    n_detected = sum(1 for m in events if m.detected)
    recall = n_detected / n_events if n_events else float("nan")

    tp = sum(m.n_true_positives for m in window_metrics)
    fp = sum(m.n_false_positives for m in window_metrics)
    precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")

    latencies = [m.latency_days for m in events if m.latency_days is not None]
    mean_latency = float(np.mean(latencies)) if latencies else None

    fp_calm = sum(m.n_false_positives for m in calms)
    calm_days = sum(m.n_trading_days for m in calms)
    fpr = fp_calm / calm_days if calm_days > 0 else float("nan")

    return HeadlineMetrics(
        n_events=n_events, n_detected=n_detected, recall=recall,
        precision=precision, mean_latency_days=mean_latency,
        false_positive_rate=fpr, n_false_positives_calm=fp_calm,
        per_window=window_metrics,
    )
