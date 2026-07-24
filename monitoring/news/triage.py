"""Stage 3 of the news pipeline: triage — how much model do we spend today?

Modes (preregistration §6.2):
  skip                 — no signal anywhere; log a heuristic no-op, no LLM call
  cheap                — mild FinBERT stress signal (≥ STRESS_CHEAP)
  thinking             — strong stress signal (FinBERT ≥ STRESS_THINKING OR
                         news intensity z ≥ HIT_Z_THINKING), or one recent detector
  classical_escalation — the classical aggregate fired: always run the full
                         agentic assessment regardless of news

Threshold constants (preregistration §6.2, fixed a priori — no in-sample tuning):
  STRESS_CHEAP     = 0.30  FinBERT daily-max stress probability → cheap mode
  STRESS_THINKING  = 0.60  FinBERT daily-max stress probability → thinking mode
  HIT_Z_THINKING   = 2.0   Regex intensity z-score fallback → thinking mode
  RECENT_DAYS      = 3     Detector lookback window for "recent" check
  COVERAGE_DAYS    = 7     News-context window fed to agents

When FinBERT is unavailable (transformers not installed), stress_score defaults to
0.0 and the HIT_Z_THINKING pathway handles escalation based on regex z-scores.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# FinBERT stress probability thresholds (preregistration §6.2)
STRESS_CHEAP = 0.30
STRESS_THINKING = 0.60

# Fallback: regex-based news intensity z-score threshold for thinking mode
HIT_Z_THINKING = 2.0

# Detector lookback window in calendar days
RECENT_DAYS = 3

# News context window fed to the news agent
COVERAGE_DAYS = 7

MODES = ("skip", "cheap", "thinking", "classical_escalation")


@dataclass(frozen=True)
class TriageDecision:
    mode: str
    reason: str


def decide(
    intensity_z: float,
    n_detectors_recent: int,
    aggregate_recent: bool,
    stress_score: float = 0.0,
    *,
    stress_cheap: float = STRESS_CHEAP,
    stress_thinking: float = STRESS_THINKING,
    hit_z_thinking: float = HIT_Z_THINKING,
) -> TriageDecision:
    """Pick a triage mode for one decision day.

    Primary signal is ``stress_score`` (FinBERT daily-max stress probability).
    ``intensity_z`` (regex-based causal z-score) is the fallback pathway via
    ``HIT_Z_THINKING``; it no longer drives cheap mode on its own.

    Args:
        intensity_z:        Causal news-intensity z-score for the day
                            (NaN = warm-up / no news).
        n_detectors_recent: Individual detectors that alarmed within the last
                            RECENT_DAYS days.
        aggregate_recent:   Whether the ≥2-within-5-days aggregate fired recently.
        stress_score:       FinBERT daily-max negative-sentiment probability (0-1).
                            Defaults to 0.0 when FinBERT is unavailable.
        stress_cheap:       Threshold above which stress_score → cheap mode.
        stress_thinking:    Threshold above which stress_score → thinking mode.
        hit_z_thinking:     Intensity-z fallback threshold for thinking mode.
    """
    z = 0.0 if (intensity_z is None or math.isnan(intensity_z)) else float(intensity_z)
    s = 0.0 if (stress_score is None or math.isnan(stress_score)) else float(stress_score)

    if aggregate_recent:
        return TriageDecision(
            "classical_escalation",
            "classical aggregate alarm within the last 5 days",
        )

    # Strong news signal: FinBERT stress OR high z-score OR a recent detector
    if s >= stress_thinking or z >= hit_z_thinking or n_detectors_recent >= 1:
        reason_parts = []
        if s >= stress_thinking:
            reason_parts.append(f"FinBERT stress={s:.2f} >= {stress_thinking}")
        if z >= hit_z_thinking:
            reason_parts.append(f"news intensity z={z:.1f} >= {hit_z_thinking}")
        if n_detectors_recent >= 1:
            reason_parts.append(f"{n_detectors_recent} detector(s) recent")
        return TriageDecision("thinking", "; ".join(reason_parts))

    # Mild signal: FinBERT stress above cheap threshold
    if s >= stress_cheap:
        return TriageDecision("cheap", f"FinBERT stress={s:.2f} >= {stress_cheap}")

    return TriageDecision("skip", "no news or detector signal")
