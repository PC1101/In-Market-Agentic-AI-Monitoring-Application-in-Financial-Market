"""Tests for the triage module (preregistration §6.2 thresholds).

Triage constants (from triage.py):
  STRESS_CHEAP=0.30, STRESS_THINKING=0.60, HIT_Z_THINKING=2.0, RECENT_DAYS=3
"""
from news.triage import decide, TriageDecision, STRESS_CHEAP, STRESS_THINKING, HIT_Z_THINKING


def test_classical_escalation_beats_everything():
    d = decide(intensity_z=0.0, n_detectors_recent=3, aggregate_recent=True)
    assert d.mode == "classical_escalation"


def test_classical_escalation_beats_high_stress():
    d = decide(intensity_z=5.0, n_detectors_recent=0, aggregate_recent=True,
               stress_score=0.99)
    assert d.mode == "classical_escalation"


def test_high_finbert_stress_routes_to_thinking():
    d = decide(intensity_z=0.0, n_detectors_recent=0, aggregate_recent=False,
               stress_score=STRESS_THINKING + 0.1)
    assert d.mode == "thinking"


def test_high_z_score_routes_to_thinking():
    # intensity_z >= HIT_Z_THINKING triggers thinking even without FinBERT stress
    d = decide(intensity_z=HIT_Z_THINKING + 0.5, n_detectors_recent=0, aggregate_recent=False)
    assert d.mode == "thinking"


def test_single_detector_recent_routes_to_thinking():
    # A recent detector alarm is a strong enough signal for thinking mode
    d = decide(intensity_z=0.0, n_detectors_recent=1, aggregate_recent=False)
    assert d.mode == "thinking"


def test_mild_finbert_stress_routes_to_cheap():
    # stress >= STRESS_CHEAP but < STRESS_THINKING, no other signals → cheap
    d = decide(intensity_z=0.0, n_detectors_recent=0, aggregate_recent=False,
               stress_score=STRESS_CHEAP + 0.05)
    assert d.mode == "cheap"


def test_quiet_day_skips():
    d = decide(intensity_z=0.1, n_detectors_recent=0, aggregate_recent=False,
               stress_score=0.0)
    assert d.mode == "skip"


def test_nan_z_with_no_other_signal_skips():
    assert decide(float("nan"), 0, False).mode == "skip"


def test_nan_z_with_detector_routes_to_thinking():
    # nan intensity_z + recent detector → thinking (not cheap)
    assert decide(float("nan"), 1, False).mode == "thinking"


def test_decision_carries_reason():
    d = decide(3.0, 0, False)
    assert isinstance(d, TriageDecision) and d.reason


def test_stress_below_cheap_threshold_skips():
    d = decide(intensity_z=0.0, n_detectors_recent=0, aggregate_recent=False,
               stress_score=STRESS_CHEAP - 0.05)
    assert d.mode == "skip"


def test_stress_at_cheap_boundary():
    d = decide(intensity_z=0.0, n_detectors_recent=0, aggregate_recent=False,
               stress_score=STRESS_CHEAP)
    assert d.mode == "cheap"


def test_stress_at_thinking_boundary():
    d = decide(intensity_z=0.0, n_detectors_recent=0, aggregate_recent=False,
               stress_score=STRESS_THINKING)
    assert d.mode == "thinking"
