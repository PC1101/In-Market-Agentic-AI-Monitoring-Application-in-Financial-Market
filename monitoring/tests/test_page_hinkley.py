import pandas as pd

from detectors import PageHinkley


def _split(alarms, change_ts):
    before = [a for a in alarms if a < change_ts]
    after = [a for a in alarms if a >= change_ts]
    return before, after


def test_fires_after_mean_shift(mean_shift, change_ts):
    res = PageHinkley().detect(mean_shift)
    before, after = _split(res.alarms, change_ts)
    assert after, "Page-Hinkley should fire after a mean shift"
    latency = (after[0] - change_ts).days
    assert latency <= 45, f"detection latency too high: {latency} days"
    assert len(before) <= 1, "should be quiet before the change"


def test_quiet_on_stationary(stationary):
    res = PageHinkley().detect(stationary)
    assert len(res.alarms) <= 2, f"too many false alarms: {len(res.alarms)}"


def test_statistic_series_aligned(mean_shift):
    res = PageHinkley().detect(mean_shift)
    assert isinstance(res.statistic, pd.Series)
    assert res.statistic.index.equals(mean_shift.index)


def test_warmup_blocks_early_alarm():
    # A short series shorter than min_samples must never alarm.
    idx = pd.bdate_range("2010-01-04", periods=10)
    s = pd.Series([0.5] * 10, index=idx, name="port_ret")
    res = PageHinkley(min_samples=20).detect(s)
    assert res.alarms == []
