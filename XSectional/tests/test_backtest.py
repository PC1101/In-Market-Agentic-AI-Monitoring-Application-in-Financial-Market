import pytest
import pandas as pd
import numpy as np
from backtest import run_backtest


@pytest.fixture
def prices_5y():
    """5 years of daily prices for 5 tickers."""
    np.random.seed(7)
    dates = pd.date_range("2000-01-01", periods=1825, freq="D")
    arr = 100.0 * np.cumprod(1 + np.random.normal(0.0003, 0.01, (1825, 5)), axis=0)
    return pd.DataFrame(arr, index=dates, columns=["A", "B", "C", "D", "E"])


@pytest.fixture
def flat_weights(prices_5y):
    """Simple long-only equal weights, monthly, covering part of the price range."""
    monthly_dates = pd.date_range("2001-01-31", periods=24, freq="ME")
    weights = pd.DataFrame(0.0, index=monthly_dates, columns=["A", "B", "C", "D", "E"])
    weights["A"] = 0.5
    weights["B"] = 0.5
    return weights


def test_run_backtest_returns_series(prices_5y, flat_weights):
    result = run_backtest(flat_weights, prices_5y)
    assert isinstance(result, pd.Series)


def test_run_backtest_no_nan(prices_5y, flat_weights):
    result = run_backtest(flat_weights, prices_5y)
    assert not result.isna().any(), "Monthly returns should not contain NaN"


def test_run_backtest_returns_not_empty(prices_5y, flat_weights):
    result = run_backtest(flat_weights, prices_5y)
    assert len(result) > 0


def test_run_backtest_monthly_returns_are_plausible(prices_5y, flat_weights):
    """Monthly returns should be within a realistic range (-50% to +50%)."""
    result = run_backtest(flat_weights, prices_5y)
    assert result.abs().max() < 0.50, (
        f"Implausibly large monthly return: {result.abs().max():.2%}"
    )


def test_run_backtest_zero_weights_gives_zero_returns(prices_5y):
    """All-zero weights should produce all-zero portfolio returns."""
    monthly_dates = pd.date_range("2001-01-31", periods=24, freq="ME")
    zero_weights = pd.DataFrame(
        0.0, index=monthly_dates, columns=["A", "B", "C", "D", "E"]
    )
    result = run_backtest(zero_weights, prices_5y)
    assert (result.abs() < 1e-9).all(), "Zero weights must produce zero returns"


def test_run_backtest_weights_applied_one_month_forward(prices_5y):
    """
    Weights formed at month t are applied to returns of month t+1.
    Verify by setting weights at month 0 and checking only month 1 return is non-zero.
    """
    monthly_dates = pd.date_range("2001-01-31", periods=12, freq="ME")
    weights = pd.DataFrame(0.0, index=monthly_dates, columns=["A", "B", "C", "D", "E"])
    weights.iloc[0]["A"] = 1.0   # only month 0 is non-zero

    result = run_backtest(weights, prices_5y)

    # Month at index 1 (formed using month-0 weights) should be non-zero
    non_zero = result[result.abs() > 1e-9]
    assert len(non_zero) == 1, (
        f"Expected 1 non-zero return period, got {len(non_zero)}"
    )
