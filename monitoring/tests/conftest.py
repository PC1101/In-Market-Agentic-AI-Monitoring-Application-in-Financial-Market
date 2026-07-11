"""Shared fixtures: synthetic daily-return series with known change points.

All series use a fixed RNG seed so tests are deterministic. The change point is always
at index ``CHANGE_IDX`` so detector tests can assert "fires after the change, quiet
before" against a known ground truth.
"""

import numpy as np
import pandas as pd
import pytest

N = 600
CHANGE_IDX = 300
SEED = 12345


def _dates(n: int) -> pd.DatetimeIndex:
    return pd.bdate_range("2010-01-04", periods=n)


@pytest.fixture
def dates():
    return _dates(N)


@pytest.fixture
def stationary(dates):
    """Single-regime N(0, 0.01) returns — no change point."""
    rng = np.random.default_rng(SEED)
    x = rng.normal(0.0003, 0.010, N)
    return pd.Series(x, index=dates, name="port_ret")


@pytest.fixture
def mean_shift(dates):
    """Downward mean shift at CHANGE_IDX (regime turns loss-making).

    Crisis-scale: a persistent ~-1%/day bleed, of the order seen in real strategy
    breaks (e.g. the Aug-2007 stat-arb meltdown), not a subtle sub-sigma drift.
    """
    rng = np.random.default_rng(SEED)
    x = np.empty(N)
    x[:CHANGE_IDX] = rng.normal(0.0006, 0.008, CHANGE_IDX)
    x[CHANGE_IDX:] = rng.normal(-0.010, 0.008, N - CHANGE_IDX)
    return pd.Series(x, index=dates, name="port_ret")


@pytest.fixture
def vol_shift(dates):
    """Volatility jump at CHANGE_IDX (calm -> stressed), mean ~ unchanged."""
    rng = np.random.default_rng(SEED)
    x = np.empty(N)
    x[:CHANGE_IDX] = rng.normal(0.0, 0.006, CHANGE_IDX)
    x[CHANGE_IDX:] = rng.normal(0.0, 0.028, N - CHANGE_IDX)
    return pd.Series(x, index=dates, name="port_ret")


@pytest.fixture
def change_ts(dates):
    return dates[CHANGE_IDX]
