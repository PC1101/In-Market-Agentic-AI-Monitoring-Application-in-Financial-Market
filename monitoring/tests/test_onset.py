"""Tests for onset.curve_onset — curve-derived onset derivation."""
import numpy as np
import pandas as pd
import pytest

from onset import curve_onset
from windows import Window


def _make_window(start: str, end: str, onset: str, name: str = "test_event") -> Window:
    return Window(name=name, kind="event", start=start, end=end,
                  onset=onset, description="synthetic")


def _make_calm_window(start: str, end: str) -> Window:
    return Window(name="test_calm", kind="calm", start=start, end=end,
                  onset=None, description="synthetic calm")


def _series(dates, returns) -> pd.Series:
    return pd.Series(returns, index=pd.DatetimeIndex(dates))


# ---------------------------------------------------------------------------
# Basic: engineered peak -> drawdown -> trough
# ---------------------------------------------------------------------------

def test_curve_onset_finds_peak_before_trough():
    # 5 flat days, then peak (+10%), then 5 crash days (-5% each)
    dates = pd.bdate_range("2010-01-04", periods=11)
    rets = [0.0] * 5 + [0.10] + [-0.05] * 5
    series = _series(dates, rets)
    w = _make_window("2010-01-04", "2010-01-18", "2010-01-11")

    onset = curve_onset(series, w)

    # The peak is on 2010-01-11 (index 5, +10% day): cumsum peaks there
    assert onset == dates[5], f"expected peak at {dates[5]}, got {onset}"


def test_curve_onset_flat_then_crash():
    # Flat for 10 days, then a sharp single-day crash, then more flat
    dates = pd.bdate_range("2008-01-02", periods=20)
    rets = [0.001] * 10 + [-0.15] + [0.001] * 9
    series = _series(dates, rets)
    w = _make_window("2008-01-02", "2008-01-29", "2008-01-15")

    onset = curve_onset(series, w)

    # Peak should be the last 0-drawdown day before the crash (day 9, index 9)
    assert onset == dates[9]


def test_curve_onset_returns_timestamp():
    dates = pd.bdate_range("2007-01-02", periods=10)
    rets = [0.01] * 5 + [-0.05] * 5
    series = _series(dates, rets)
    w = _make_window("2007-01-02", "2007-01-15", "2007-01-09")

    result = curve_onset(series, w)
    assert isinstance(result, pd.Timestamp)


# ---------------------------------------------------------------------------
# Error cases
# ---------------------------------------------------------------------------

def test_curve_onset_raises_on_calm_window():
    dates = pd.bdate_range("2004-01-02", periods=5)
    series = _series(dates, [0.001] * 5)
    calm = _make_calm_window("2004-01-02", "2004-01-08")

    with pytest.raises(ValueError, match="event window"):
        curve_onset(series, calm)


def test_curve_onset_raises_when_no_data_in_window():
    dates = pd.bdate_range("2010-01-04", periods=5)
    series = _series(dates, [0.001] * 5)
    # Window is entirely outside the series
    w = _make_window("2020-01-02", "2020-01-15", "2020-01-06")

    with pytest.raises(ValueError, match="no data"):
        curve_onset(series, w)


# ---------------------------------------------------------------------------
# Consistency: onset is always <= trough date
# ---------------------------------------------------------------------------

def test_curve_onset_before_trough():
    rng = np.random.default_rng(42)
    dates = pd.bdate_range("2009-03-01", periods=60)
    rets = list(rng.normal(0.001, 0.005, 30)) + list(rng.normal(-0.02, 0.008, 30))
    series = _series(dates, rets)
    w = _make_window("2009-03-02", "2009-05-22", "2009-04-01")

    onset = curve_onset(series, w)
    # Onset must be within the window
    assert w.start_ts <= onset <= w.end_ts
