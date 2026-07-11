import pytest
import pandas as pd
import numpy as np
from backtest import run_backtest, run_backtest_daily, write_daily_equity_curve


@pytest.fixture
def prices_5y():
    np.random.seed(7)
    dates = pd.date_range("2000-01-01", periods=1825, freq="D")
    arr = 100.0 * np.cumprod(1 + np.random.normal(0.0003, 0.01, (1825, 5)), axis=0)
    return pd.DataFrame(arr, index=dates, columns=["A", "B", "C", "D", "E"])


@pytest.fixture
def flat_weights():
    monthly_dates = pd.date_range("2001-01-31", periods=24, freq="ME")
    weights = pd.DataFrame(0.0, index=monthly_dates, columns=["A", "B", "C", "D", "E"])
    weights["A"] = 0.5
    weights["B"] = 0.5
    return weights


def test_daily_schema(prices_5y, flat_weights):
    daily = run_backtest_daily(flat_weights, prices_5y)
    assert list(daily.columns) == ["port_ret", "equity", "drawdown"]
    assert daily.index.name == "Date"
    assert isinstance(daily.index, pd.DatetimeIndex)
    assert not daily["port_ret"].isna().any()


def test_daily_equity_and_drawdown_consistent(prices_5y, flat_weights):
    daily = run_backtest_daily(flat_weights, prices_5y)
    # equity is the cumulative product of (1 + port_ret)
    recomputed = (1.0 + daily["port_ret"]).cumprod()
    assert np.allclose(daily["equity"].to_numpy(), recomputed.to_numpy())
    # drawdown is non-positive and 0 at the running peak
    assert (daily["drawdown"] <= 1e-12).all()
    assert daily["drawdown"].max() <= 1e-9


def test_daily_single_asset_matches_monthly_exactly(prices_5y):
    """For a single held asset, daily-compounded monthly returns equal the monthly
    backtest exactly — there are no cross-asset compounding terms."""
    monthly_dates = pd.date_range("2001-01-31", periods=24, freq="ME")
    w = pd.DataFrame(0.0, index=monthly_dates, columns=["A", "B", "C", "D", "E"])
    w["A"] = 1.0
    daily = run_backtest_daily(w, prices_5y)
    monthly = run_backtest(w, prices_5y)
    daily_by_month = (1.0 + daily["port_ret"]).resample("ME").prod() - 1.0
    common = monthly.index.intersection(daily_by_month.index)[1:-1]  # drop partial edges
    assert np.allclose(monthly.loc[common].to_numpy(),
                       daily_by_month.loc[common].to_numpy(), atol=1e-9)


def test_daily_multi_asset_tracks_monthly(prices_5y, flat_weights):
    """For a multi-asset portfolio the daily (constant-weight, daily-rebalanced) curve
    tracks the monthly single-period backtest closely, differing only by second-order
    intra-month compounding terms."""
    daily = run_backtest_daily(flat_weights, prices_5y)
    monthly = run_backtest(flat_weights, prices_5y)
    daily_by_month = (1.0 + daily["port_ret"]).resample("ME").prod() - 1.0
    common = monthly.index.intersection(daily_by_month.index)[1:-1]
    a, b = monthly.loc[common].to_numpy(), daily_by_month.loc[common].to_numpy()
    assert np.corrcoef(a, b)[0, 1] > 0.99
    assert np.max(np.abs(a - b)) < 0.01


def test_daily_no_lookahead_first_month_zero(prices_5y):
    monthly_dates = pd.date_range("2001-01-31", periods=12, freq="ME")
    weights = pd.DataFrame(0.0, index=monthly_dates, columns=["A", "B", "C", "D", "E"])
    weights.loc[weights.index[0], "A"] = 1.0
    daily = run_backtest_daily(weights, prices_5y)
    # No return should be booked on or before the first weight date (weights act next day).
    before = daily.loc[: monthly_dates[0], "port_ret"]
    assert (before.abs() < 1e-12).all()


def test_write_daily_equity_curve(prices_5y, flat_weights, tmp_path):
    daily = run_backtest_daily(flat_weights, prices_5y)
    out = tmp_path / "sub" / "equity_curve.csv"
    write_daily_equity_curve(daily, str(out))
    assert out.exists()
    reloaded = pd.read_csv(out)
    assert list(reloaded.columns) == ["Date", "port_ret", "equity", "drawdown"]
