from news.triage import decide, TriageDecision


def test_classical_escalation_beats_everything():
    d = decide(intensity_z=0.0, n_detectors_recent=3, aggregate_recent=True)
    assert d.mode == "classical_escalation"


def test_high_news_z_routes_to_thinking():
    d = decide(intensity_z=3.0, n_detectors_recent=0, aggregate_recent=False)
    assert d.mode == "thinking"


def test_mild_news_or_single_detector_routes_to_cheap():
    assert decide(1.5, 0, False).mode == "cheap"
    assert decide(0.0, 1, False).mode == "cheap"


def test_quiet_day_skips():
    d = decide(intensity_z=0.1, n_detectors_recent=0, aggregate_recent=False)
    assert d.mode == "skip"


def test_nan_z_treated_as_no_news_signal():
    assert decide(float("nan"), 0, False).mode == "skip"
    assert decide(float("nan"), 1, False).mode == "cheap"


def test_decision_carries_reason():
    d = decide(3.0, 0, False)
    assert isinstance(d, TriageDecision) and d.reason
