import pandas as pd

from detectors.aggregate import aggregate_alarms
from detectors.base import DetectionResult


def _res(name, dates):
    return DetectionResult(name, [pd.Timestamp(d) for d in dates])


def test_two_detectors_within_window_fire():
    results = [
        _res("a", ["2010-01-04"]),
        _res("b", ["2010-01-06"]),  # 2 days later
    ]
    agg = aggregate_alarms(results, window_days=5)
    assert len(agg) == 1
    assert agg[0] == pd.Timestamp("2010-01-06")  # fires when quorum completes


def test_same_detector_twice_does_not_fire():
    results = [_res("a", ["2010-01-04", "2010-01-05"])]
    agg = aggregate_alarms(results, window_days=5, min_detectors=2)
    assert agg == []


def test_outside_window_does_not_fire():
    results = [
        _res("a", ["2010-01-04"]),
        _res("b", ["2010-01-20"]),  # far apart
    ]
    agg = aggregate_alarms(results, window_days=5)
    assert agg == []


def test_cooldown_suppresses_repeat():
    results = [
        _res("a", ["2010-01-04", "2010-01-07"]),
        _res("b", ["2010-01-05", "2010-01-08"]),
    ]
    agg = aggregate_alarms(results, window_days=5, cooldown_days=5)
    assert len(agg) == 1


def test_accepts_dict_input():
    agg = aggregate_alarms(
        {"a": [pd.Timestamp("2010-01-04")], "b": [pd.Timestamp("2010-01-05")]},
        window_days=5,
    )
    assert len(agg) == 1


def test_three_detectors_min_three():
    results = [
        _res("a", ["2010-01-04"]),
        _res("b", ["2010-01-05"]),
        _res("c", ["2010-01-06"]),
    ]
    assert len(aggregate_alarms(results, window_days=5, min_detectors=3)) == 1
    # only two distinct within window -> no fire at min_detectors=3
    two = [_res("a", ["2010-01-04"]), _res("b", ["2010-01-05"])]
    assert aggregate_alarms(two, window_days=5, min_detectors=3) == []
