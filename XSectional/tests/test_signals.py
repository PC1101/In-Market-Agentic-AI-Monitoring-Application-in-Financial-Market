import pytest
import pandas as pd
import numpy as np
from signals import compute_monthly_returns, compute_momentum_scores


@pytest.fixture
def flat_prices():
    """All-flat prices — every return is zero, momentum should be zero."""
    dates = pd.date_range("2000-01-01", periods=500, freq="D")
    return pd.DataFrame(
        100.0,
        index=dates,
        columns=["A", "B", "C", "D", "E"],
    )


@pytest.fixture
def random_prices():
    """Random-walk prices with 5 tickers and ~2 years of daily data."""
    np.random.seed(42)
    dates = pd.date_range("2000-01-01", periods=500, freq="D")
    prices = 100.0 * np.cumprod(
        1 + np.random.normal(0.0003, 0.01, (500, 5)), axis=0
    )
    return pd.DataFrame(prices, index=dates, columns=["A", "B", "C", "D", "E"])


def test_compute_monthly_returns_columns_match(random_prices):
    result = compute_monthly_returns(random_prices)
    assert list(result.columns) == ["A", "B", "C", "D", "E"]


def test_compute_monthly_returns_first_row_is_nan(random_prices):
    result = compute_monthly_returns(random_prices)
    assert result.iloc[0].isna().all()


def test_compute_monthly_returns_has_fewer_rows_than_daily(random_prices):
    result = compute_monthly_returns(random_prices)
    assert len(result) < len(random_prices)


def test_compute_momentum_scores_columns_match(random_prices):
    scores = compute_momentum_scores(random_prices)
    assert list(scores.columns) == ["A", "B", "C", "D", "E"]


def test_compute_momentum_scores_produces_valid_values_after_warmup(random_prices):
    """After 13+ months of data, at least some momentum scores should be non-NaN."""
    scores = compute_momentum_scores(random_prices)
    valid_rows = scores.dropna(how="all")
    assert len(valid_rows) > 0


def test_compute_momentum_scores_flat_prices_are_zero(flat_prices):
    """When all prices are flat, every momentum score should be 0."""
    scores = compute_momentum_scores(flat_prices)
    valid = scores.dropna(how="all")
    assert len(valid) > 0
    assert (valid.fillna(0).abs() < 1e-9).all().all()


def test_compute_momentum_scores_needs_13_months_warmup(random_prices):
    """Momentum score rows before 13 months should all be NaN."""
    scores = compute_momentum_scores(random_prices)
    monthly_returns = compute_monthly_returns(random_prices)
    first_valid_idx = 13  # 12 lookback + 1 skip
    early_scores = scores.iloc[: first_valid_idx - 1]
    assert early_scores.dropna(how="all").empty
