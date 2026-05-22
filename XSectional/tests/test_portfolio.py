import pytest
import pandas as pd
import numpy as np
from portfolio import construct_portfolio


@pytest.fixture
def sample_scores():
    """50 tickers, 24 months, random scores (enough stocks for 20% quantile cuts)."""
    np.random.seed(42)
    dates = pd.date_range("2002-01-31", periods=24, freq="ME")
    tickers = [f"T{i:02d}" for i in range(50)]
    return pd.DataFrame(
        np.random.randn(24, 50),
        index=dates,
        columns=tickers,
    )


def test_construct_portfolio_returns_dataframe(sample_scores):
    weights = construct_portfolio(sample_scores)
    assert isinstance(weights, pd.DataFrame)


def test_construct_portfolio_same_shape_as_scores(sample_scores):
    weights = construct_portfolio(sample_scores)
    assert weights.shape == sample_scores.shape


def test_long_weights_are_positive(sample_scores):
    weights = construct_portfolio(sample_scores)
    assert (weights[weights > 0] > 0).all().all()


def test_short_weights_are_negative(sample_scores):
    weights = construct_portfolio(sample_scores)
    assert (weights[weights < 0] < 0).all().all()


def test_long_leg_sums_to_one_each_month(sample_scores):
    weights = construct_portfolio(sample_scores)
    for date in weights.index:
        row = weights.loc[date]
        long_sum = row[row > 0].sum()
        if long_sum > 0:
            assert abs(long_sum - 1.0) < 1e-9, (
                f"Long leg at {date} sums to {long_sum:.6f}, expected 1.0"
            )


def test_short_leg_sums_to_minus_one_each_month(sample_scores):
    weights = construct_portfolio(sample_scores)
    for date in weights.index:
        row = weights.loc[date]
        short_sum = row[row < 0].sum()
        if short_sum < 0:
            assert abs(short_sum + 1.0) < 1e-9, (
                f"Short leg at {date} sums to {short_sum:.6f}, expected -1.0"
            )


def test_top_20_percent_assigned_positive_weight(sample_scores):
    weights = construct_portfolio(sample_scores)
    date = sample_scores.index[0]
    row = sample_scores.loc[date].dropna()
    n_long = int(len(row) * 0.20)
    top_tickers = row.nlargest(n_long).index
    assert all(weights.loc[date, t] > 0 for t in top_tickers)


def test_bottom_20_percent_assigned_negative_weight(sample_scores):
    weights = construct_portfolio(sample_scores)
    date = sample_scores.index[0]
    row = sample_scores.loc[date].dropna()
    n_short = int(len(row) * 0.20)
    bottom_tickers = row.nsmallest(n_short).index
    assert all(weights.loc[date, t] < 0 for t in bottom_tickers)


def test_all_nan_row_produces_zero_weights(sample_scores):
    """A month where all scores are NaN should result in all-zero weights."""
    scores_with_nan = sample_scores.copy()
    scores_with_nan.iloc[5] = np.nan
    weights = construct_portfolio(scores_with_nan)
    assert (weights.iloc[5] == 0.0).all()


def test_insufficient_stocks_produces_zero_weights():
    """Fewer than MIN_STOCKS_PER_LEG qualifying stocks → zero weights for that month."""
    dates = pd.date_range("2002-01-31", periods=3, freq="ME")
    # Only 5 stocks — fewer than MIN_STOCKS_PER_LEG (10)
    tickers = [f"T{i}" for i in range(5)]
    scores = pd.DataFrame(
        np.random.randn(3, 5), index=dates, columns=tickers
    )
    weights = construct_portfolio(scores)
    assert (weights == 0.0).all().all()
