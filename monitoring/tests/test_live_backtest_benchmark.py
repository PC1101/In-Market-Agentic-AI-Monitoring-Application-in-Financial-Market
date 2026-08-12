"""Tests for live_backtest.benchmark — buy-and-hold math, cache loading."""
import numpy as np
import pandas as pd
import pytest

from live_backtest.benchmark import buy_and_hold


class TestBuyAndHold:
    def test_basic_compounding(self):
        """buy_and_hold = capital * cumprod(1+r)."""
        rets = pd.Series([0.01, -0.005, 0.02],
                         index=pd.bdate_range("2020-01-01", periods=3))
        result = buy_and_hold(rets, capital=1_000_000.0)

        expected = 1_000_000.0 * np.cumprod(1 + rets.values)
        np.testing.assert_allclose(result.values, expected, rtol=1e-10)

    def test_zero_returns_stay_flat(self):
        rets = pd.Series([0.0, 0.0, 0.0],
                         index=pd.bdate_range("2020-01-01", periods=3))
        result = buy_and_hold(rets, capital=500_000.0)
        np.testing.assert_allclose(result.values, 500_000.0, rtol=1e-10)

    def test_negative_return(self):
        rets = pd.Series([-0.50], index=pd.bdate_range("2020-01-01", periods=1))
        result = buy_and_hold(rets, capital=1_000_000.0)
        assert result.iloc[0] == pytest.approx(500_000.0)

    def test_preserves_index(self):
        idx = pd.bdate_range("2023-06-01", periods=5)
        rets = pd.Series([0.01] * 5, index=idx)
        result = buy_and_hold(rets)
        assert (result.index == idx).all()

    def test_cache_file_load(self, tmp_path):
        """Test fetch_benchmark with a pre-existing cache file (no yfinance)."""
        from live_backtest.benchmark import fetch_benchmark

        # Create a fake cache CSV
        cache = tmp_path / "benchmark_TEST.csv"
        dates = pd.bdate_range("2020-01-02", periods=5)
        df = pd.DataFrame({"Date": dates, "ret": [0.01, -0.005, 0.02, 0.015, -0.01]})
        df.to_csv(cache, index=False)

        result = fetch_benchmark("TEST", start="2020-01-01", end="2020-12-31",
                                 cache_dir=tmp_path, no_fetch=True)
        assert len(result) == 5
        assert result.iloc[0] == pytest.approx(0.01)
