# signals.py — compute monthly returns and 12-1 cross-sectional momentum scores

import numpy as np
import pandas as pd

import config


def compute_monthly_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Resample daily prices to month-end and compute simple period returns.

    Returns a DataFrame with the same columns as prices, indexed by month-end dates.
    The first row is always NaN (no prior month to compare against).
    """
    monthly = prices.resample(config.REBALANCE_FREQ).last()
    return monthly.pct_change()


def compute_momentum_scores(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Compute 12-1 cross-sectional momentum scores for all tickers.

    At each month-end t, the score is the cumulative return from
    t-13 to t-2 (12 months, skipping the most recent month t-1).

    Uses log returns summed over a rolling window for numerical stability,
    then converts back to simple returns.

    Returns a DataFrame aligned with monthly_returns index and prices columns.
    Scores before the 13-month warm-up period are NaN.
    """
    monthly_returns = compute_monthly_returns(prices)

    # Clip to avoid log(0) or log(negative) on extreme down-moves
    log_returns = np.log(1 + monthly_returns.clip(lower=-0.9999))

    # shift(SKIP_MONTHS) moves the window back — at time t, the most recent
    # value is now r_{t-1} (one month ago), so rolling sum skips month t.
    # rolling(LOOKBACK_MONTHS).sum() then accumulates the 12 months before that.
    momentum_log = (
        log_returns.shift(config.SKIP_MONTHS)
        .rolling(config.LOOKBACK_MONTHS)
        .sum()
    )

    return np.exp(momentum_log) - 1
