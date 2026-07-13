"""Stage 3 of the news pipeline: triage — how much model do we spend today?

Modes (VRI Week 3):
  skip                 — no signal anywhere; log a heuristic no-op, no LLM call
  cheap                — mild signal: one LLM pass on a compact context
  thinking             — strong NEWS signal: extended-reasoning prompt
  classical_escalation — the classical aggregate fired: always run the full
                         agentic assessment regardless of news

Thresholds are fixed a priori (same no-in-sample-tuning discipline as the
classical detectors — see WEEK2_STATUS open item 1).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

Z_CHEAP = 1.0
Z_THINKING = 2.5

MODES = ("skip", "cheap", "thinking", "classical_escalation")


@dataclass(frozen=True)
class TriageDecision:
    mode: str
    reason: str


def decide(intensity_z: float, n_detectors_recent: int, aggregate_recent: bool,
           z_cheap: float = Z_CHEAP, z_thinking: float = Z_THINKING) -> TriageDecision:
    """Pick a triage mode for one decision day.

    Args:
        intensity_z: causal news-intensity z-score for the day (NaN = warm-up/no news).
        n_detectors_recent: individual detectors that alarmed within the last 5 days.
        aggregate_recent: whether the >=2-within-5-days aggregate fired in that span.
    """
    z = 0.0 if (intensity_z is None or math.isnan(intensity_z)) else intensity_z
    if aggregate_recent:
        return TriageDecision("classical_escalation",
                              "classical aggregate alarm within the last 5 days")
    if z >= z_thinking:
        return TriageDecision("thinking", f"news intensity z={z:.1f} >= {z_thinking}")
    if z >= z_cheap or n_detectors_recent >= 1:
        return TriageDecision("cheap",
                              f"news z={z:.1f} or {n_detectors_recent} detector(s) recent")
    return TriageDecision("skip", "no news or detector signal")
