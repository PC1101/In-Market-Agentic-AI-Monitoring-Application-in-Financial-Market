"""Tests for live_backtest.engine — portfolio simulation accounting."""
import numpy as np
import pandas as pd
import pytest

from live_backtest.engine import EngineConfig, simulate


def _make_returns(values, start="2020-01-01"):
    idx = pd.bdate_range(start, periods=len(values))
    return pd.Series(values, index=idx, dtype=float)


class TestSimulate:
    def test_full_exposure_no_tcost_matches_cumprod(self):
        """exposure=1, tcost=0 => value = capital * cumprod(1+r)."""
        rets = _make_returns([0.01, -0.005, 0.02, 0.015, -0.01])
        exp = pd.Series(1.0, index=rets.index)
        cfg = EngineConfig(capital=1_000_000.0, tcost_bps=0.0)
        result = simulate(rets, exp, cfg)

        expected = 1_000_000.0 * np.cumprod(1 + rets.values)
        np.testing.assert_allclose(result["value"].values, expected, rtol=1e-10)

    def test_zero_exposure_stays_flat(self):
        """exposure=0, no cash rate => value stays constant."""
        rets = _make_returns([0.01, -0.005, 0.02])
        exp = pd.Series(0.0, index=rets.index)
        cfg = EngineConfig(capital=1_000_000.0, tcost_bps=0.0, cash_rate_annual=0.0)
        result = simulate(rets, exp, cfg)

        # First day has tcost=0 (initial position), so value should stay at 1M
        np.testing.assert_allclose(result["value"].values, 1_000_000.0, rtol=1e-10)

    def test_tcost_charged_on_exposure_change(self):
        """Transaction cost reduces value when exposure changes."""
        rets = _make_returns([0.0, 0.0, 0.0])  # zero returns to isolate tcost
        # Exposure changes from 1.0 to 0.0 on day 2
        exp = pd.Series([1.0, 0.0, 0.0], index=rets.index)
        cfg = EngineConfig(capital=1_000_000.0, tcost_bps=100.0)  # 1% = 100bps

        result = simulate(rets, exp, cfg)

        # Day 0: no prior, no tcost, value = 1M
        assert result["value"].iloc[0] == pytest.approx(1_000_000.0)
        # Day 1: exposure goes 1->0, delta=1.0, tcost = 1.0 * 1M * 100/10000 = 10000
        assert result["tcost"].iloc[1] == pytest.approx(10_000.0)
        assert result["value"].iloc[1] == pytest.approx(990_000.0)

    def test_pnl_is_value_diff(self):
        rets = _make_returns([0.01, -0.005, 0.02, 0.015])
        exp = pd.Series(1.0, index=rets.index)
        result = simulate(rets, exp, EngineConfig(capital=500_000.0, tcost_bps=0.0))
        expected_pnl = np.diff(np.concatenate([[500_000.0], result["value"].values]))
        np.testing.assert_allclose(result["pnl"].values, expected_pnl, rtol=1e-10)

    def test_drawdown_negative_or_zero(self):
        rets = _make_returns([0.01, -0.05, 0.02, -0.03])
        exp = pd.Series(1.0, index=rets.index)
        result = simulate(rets, exp)
        assert (result["drawdown"] <= 0).all()

    def test_cash_rate_accrual(self):
        """When exposure=0 and cash_rate > 0, uninvested cash earns interest."""
        rets = _make_returns([0.0, 0.0])
        exp = pd.Series(0.0, index=rets.index)
        annual_rate = 0.05  # 5%
        cfg = EngineConfig(capital=1_000_000.0, tcost_bps=0.0,
                           cash_rate_annual=annual_rate)
        result = simulate(rets, exp, cfg)
        daily_rate = (1.05) ** (1/252) - 1
        expected_day1 = 1_000_000.0 * (1 + daily_rate)
        assert result["value"].iloc[0] == pytest.approx(expected_day1, rel=1e-8)
