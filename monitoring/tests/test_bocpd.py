from detectors import BOCPD


def _split(alarms, change_ts):
    return ([a for a in alarms if a < change_ts],
            [a for a in alarms if a >= change_ts])


def test_fires_after_vol_shift(vol_shift, change_ts):
    res = BOCPD().detect(vol_shift)
    before, after = _split(res.alarms, change_ts)
    assert after, "BOCPD should fire after a volatility change"
    assert (after[0] - change_ts).days <= 70
    assert len(before) <= 2


def test_fires_after_mean_shift(mean_shift, change_ts):
    res = BOCPD().detect(mean_shift)
    _, after = _split(res.alarms, change_ts)
    assert after, "BOCPD should fire after a mean change"


def test_quiet_on_stationary(stationary):
    # Isolated BOCPD false alarms on calm data (~1 per 6 months) are expected and are
    # filtered downstream by the >=2-detector aggregation rule.
    res = BOCPD().detect(stationary)
    assert len(res.alarms) <= 5, f"too many false alarms: {len(res.alarms)}"


def test_map_run_length_grows_then_collapses(vol_shift):
    res = BOCPD().detect(vol_shift)
    # The MAP run length (the diagnostic series) should reach a large value in the
    # stable first regime before collapsing at the change.
    assert res.statistic.iloc[:300].max() >= 50
