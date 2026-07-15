"""Triage logic — decide how much (LLM) attention a decision date deserves.

Modes (VRI Week-3 wording: cheap-model / thinking-model / classical-escalation):

* ``SKIP``                 — quiet news + quiet detectors: no LLM call at all;
                             the supervisor receives a "quiet" news block.
* ``CHEAP_MODEL``          — something is stirring: run the News Context Agent
                             with the standard prompt on the local 3B model.
* ``THINKING_MODEL``       — news spike coincides with an aggregate detector
                             alarm: same local model with the extended
                             step-by-step reasoning prompt variant. (A genuinely
                             larger "thinking" model is documented future work —
                             the 4 GB VRAM cap binds.)
* ``CLASSICAL_ESCALATION`` — the news channel cannot be trusted for this date
                             (zero coverage while detectors alarm) or the LLM
                             failed at runtime: the supervisor must run on
                             classical outputs alone, and the escalation is
                             logged. Runtime LLM failures are mapped to this
                             mode by the caller (run_classical.py), not here.

Thresholds are fixed a priori (module constants) — NOT tuned on the six
evaluation windows (see WEEK2_STATUS open item 1 on in-sample tuning).
"""

from __future__ import annotations

import math
from enum import Enum

import pandas as pd

# Fixed a-priori thresholds.
STRESS_CHEAP = 0.30      # any-day stress at/above this → at least CHEAP_MODEL
STRESS_THINKING = 0.60   # stress at/above this (with aggregate alarm) → THINKING
HIT_Z_THINKING = 2.0     # news-spike z-score (with aggregate alarm) → THINKING
RECENT_DAYS = 3          # trailing window (on T-1 data) that triage looks at
COVERAGE_DAYS = 7        # days with zero articles that mean "no news channel"


class TriageMode(str, Enum):
    SKIP = "SKIP"
    CHEAP_MODEL = "CHEAP_MODEL"
    THINKING_MODEL = "THINKING_MODEL"
    CLASSICAL_ESCALATION = "CLASSICAL_ESCALATION"


def triage(signal: pd.DataFrame, n_recent_alarms: int = 0,
           aggregate_alarm: bool = False) -> TriageMode:
    """Pick a triage mode from the (already T-1-truncated) daily signal frame.

    Args:
        signal: output of ``aggregate.daily_signal`` whose last row is the most
                recent day the agent may see (T-1 relative to the decision date).
        n_recent_alarms: number of individual detector alarms in the last 5
                trading days (visible as of the decision date).
        aggregate_alarm: whether the classical aggregate rule (>=2 detectors
                within 5 days) is currently firing.
    """
    recent = signal.tail(RECENT_DAYS)
    hits_recent = int(recent["n_hits"].sum())
    stress_recent = float(recent["stress"].max()) if recent["stress"].notna().any() else math.nan
    hit_z_last = float(signal["hit_z"].iloc[-1]) if len(signal) else math.nan

    # No news coverage at all while detectors alarm → the news channel is
    # unusable for this date; fall back to classical-only, loudly.
    coverage = signal["n_articles"].tail(COVERAGE_DAYS)
    if int(coverage.sum()) == 0 and (n_recent_alarms > 0 or aggregate_alarm):
        return TriageMode.CLASSICAL_ESCALATION

    spike = (not math.isnan(hit_z_last) and hit_z_last >= HIT_Z_THINKING) or (
        not math.isnan(stress_recent) and stress_recent >= STRESS_THINKING
    )
    if spike and aggregate_alarm:
        return TriageMode.THINKING_MODEL

    if hits_recent > 0 or (
        not math.isnan(stress_recent) and stress_recent >= STRESS_CHEAP
    ) or n_recent_alarms > 0:
        return TriageMode.CHEAP_MODEL

    return TriageMode.SKIP
