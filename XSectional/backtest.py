# backtest.py — simulate monthly long-short portfolio returns

import pandas as pd

import config


def run_backtest(weights: pd.DataFrame, prices: pd.DataFrame) -> pd.Series:
    """
    Simulate monthly portfolio returns from portfolio weights and daily prices.

    Weights at month-end t are applied to the returns earned during month t+1
    (i.e., we form the portfolio at close of month t, hold through month t+1).

    Args:
        weights: DataFrame of portfolio weights (monthly dates × tickers).
                 Positive = long, negative = short, zero = not held.
        prices:  DataFrame of daily adjusted close prices (daily dates × tickers).

    Returns:
        Series of monthly portfolio returns, indexed by month-end dates.
        The first return corresponds to the month after the first weight observation.
    """
    # Step 1: compute monthly returns from daily prices
    monthly_prices = prices.resample(config.REBALANCE_FREQ).last()
    monthly_returns = monthly_prices.pct_change()

    # Step 2: align weights and returns on the same index and columns
    weights_aligned, returns_aligned = weights.align(monthly_returns, join="inner")

    # Step 3: apply weights from t to returns at t+1 (shift weights forward by 1 row)
    # After shift(1): at row t, we use the weights that were computed at row t-1
    portfolio_returns = (weights_aligned.shift(1) * returns_aligned).sum(axis=1)

    # Step 4: drop leading NaN introduced by the shift
    return portfolio_returns.dropna()
