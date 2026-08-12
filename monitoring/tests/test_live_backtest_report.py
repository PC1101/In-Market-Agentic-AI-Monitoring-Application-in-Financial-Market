"""Tests for live_backtest.report — metrics calculations."""
import numpy as np
import pandas as pd
import pytest

from live_backtest.report import compute_metrics


def _make_value(values, start="2020-01-01"):
    idx = pd.bdate_range(start, periods=len(values))
    return pd.Series(values, index=idx, dtype=float)


class TestComputeMetrics:
    def test_constant_value_zero_return(self):
        """Flat portfolio => 0 return, 0 vol, 0 drawdown."""
        v = _make_value([1_000_000] * 100)
        m = compute_metrics(v)
        assert m["total_return"] == pytest.approx(0.0)
        assert m["ann_vol"] == pytest.approx(0.0)
        assert m["max_dd"] == pytest.approx(0.0)
        assert m["terminal_value"] == pytest.approx(1_000_000.0)

    def test_known_total_return(self):
        """Portfolio doubles => total_return = 1.0."""
        # Need at least 3 values so pct_change().dropna() yields n_days >= 2
        v = _make_value([1_000_000, 1_500_000, 2_000_000])
        m = compute_metrics(v)
        assert m["total_return"] == pytest.approx(1.0)
        assert m["terminal_value"] == pytest.approx(2_000_000.0)

    def test_drawdown(self):
        """Peak at 1.1M, trough at 0.9M => DD = (0.9 - 1.1) / 1.1."""
        v = _make_value([1_000_000, 1_100_000, 900_000, 1_000_000])
        m = compute_metrics(v)
        expected_dd = (900_000 - 1_100_000) / 1_100_000  # ~ -0.1818
        assert m["max_dd"] == pytest.approx(expected_dd, abs=1e-4)

    def test_sharpe_positive_for_positive_returns(self):
        """Steadily increasing portfolio => positive Sharpe."""
        values = [1_000_000 * (1.0003 ** i) for i in range(252)]
        v = _make_value(values)
        m = compute_metrics(v)
        assert m["sharpe"] > 0
        assert m["cagr"] > 0

    def test_short_series(self):
        """Single value => NaN metrics."""
        v = _make_value([1_000_000])
        m = compute_metrics(v)
        assert np.isnan(m["cagr"])
