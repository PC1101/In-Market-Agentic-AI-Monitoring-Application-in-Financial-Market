from detectors import DistributionalThreshold


def _split(alarms, change_ts):
    return ([a for a in alarms if a < change_ts],
            [a for a in alarms if a >= change_ts])


def test_fires_after_mean_shift(mean_shift, change_ts):
    res = DistributionalThreshold().detect(mean_shift)
    before, after = _split(res.alarms, change_ts)
    assert after, "should fire after a mean shift"
    assert (after[0] - change_ts).days <= 60
    assert len(before) <= 1


def test_fires_after_vol_shift(vol_shift, change_ts):
    res = DistributionalThreshold().detect(vol_shift)
    _, after = _split(res.alarms, change_ts)
    assert after, "should fire after a volatility jump"
    assert (after[0] - change_ts).days <= 60


def test_quiet_on_stationary(stationary):
    res = DistributionalThreshold().detect(stationary)
    assert len(res.alarms) <= 2, f"too many false alarms: {len(res.alarms)}"


def test_needs_enough_history():
    import pandas as pd
    idx = pd.bdate_range("2010-01-04", periods=50)
    s = pd.Series([0.01] * 50, index=idx, name="port_ret")
    res = DistributionalThreshold(baseline=120, recent=20).detect(s)
    assert res.alarms == []
