"""Aggregation rule: an aggregated alarm fires when >= ``min_detectors`` *distinct*
detectors fire within a rolling ``window_days``-day window.

The aggregated alarm is timestamped at the moment the rule becomes satisfied (the firing
of the detector that completes the quorum) — the earliest causal time the ensemble could
raise the alarm. A cooldown prevents the same cluster of firings from producing repeated
aggregated alarms.
"""

from __future__ import annotations

import pandas as pd

from .base import DetectionResult


def aggregate_alarms(results, window_days: int = 5, min_detectors: int = 2,
                     cooldown_days: int | None = None) -> list[pd.Timestamp]:
    """Combine per-detector alarms into aggregated alarms.

    Args:
        results: an iterable of DetectionResult, or a mapping name -> list[Timestamp].
        window_days: rolling window (calendar days) within which firings are grouped.
        min_detectors: number of *distinct* detectors required to raise an alarm.
        cooldown_days: suppress new aggregated alarms for this many days after firing;
                       defaults to ``window_days``.

    Returns:
        Sorted list of aggregated-alarm timestamps.
    """
    if cooldown_days is None:
        cooldown_days = window_days

    # Normalise input into a flat list of (timestamp, detector_name) events.
    events: list[tuple[pd.Timestamp, str]] = []
    if isinstance(results, dict):
        for name, alarms in results.items():
            for a in alarms:
                events.append((pd.Timestamp(a), name))
    else:
        for res in results:
            if not isinstance(res, DetectionResult):
                raise TypeError("results must be DetectionResult objects or a name->alarms dict")
            for a in res.alarms:
                events.append((pd.Timestamp(a), res.detector))

    events.sort(key=lambda e: e[0])
    window = pd.Timedelta(days=window_days)
    cooldown = pd.Timedelta(days=cooldown_days)

    fired: list[pd.Timestamp] = []
    last_fire: pd.Timestamp | None = None

    for i, (ts, _det) in enumerate(events):
        lo = ts - window
        distinct = {d for (t, d) in events if lo <= t <= ts}
        if len(distinct) >= min_detectors:
            if last_fire is None or (ts - last_fire) > cooldown:
                fired.append(ts)
                last_fire = ts

    return fired
