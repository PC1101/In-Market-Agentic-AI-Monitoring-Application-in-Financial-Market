# backtest.py — simulate monthly long-short portfolio returns

from pathlib import Path

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


def run_backtest_daily(weights: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    """Simulate *daily* long-short portfolio PnL from monthly weights and daily prices.

    The month-end weights are held through the following month: at each trading day we
    apply the weights formed at the most recent month-end strictly before that day (the
    monthly weights are forward-filled onto the daily grid and shifted one day). This is
    the daily analogue of ``run_backtest`` and introduces no look-ahead — the return on
    day *t* uses only weights decided on or before day *t-1*.

    Emitting a daily curve lets the classical monitoring layer treat the JT momentum
    strategy identically to the AL PCA stat-arb strategy (same ``equity_curve.csv``
    schema, same detection-latency clock).

    Args:
        weights: monthly portfolio weights (month-end dates x tickers); NA = not held.
        prices:  daily adjusted-close prices (daily dates x tickers).

    Returns:
        DataFrame indexed by trading date ("Date") with columns:
            port_ret  — daily portfolio return
            equity    — cumulative equity (starts at 1.0 on the first traded day)
            drawdown  — running drawdown from the equity peak (<= 0)
    """
    # fill_method=None: do not forward-fill stale/pre-listing prices into fake 0% returns.
    daily_returns = prices.pct_change(fill_method=None)

    # Align columns, coerce nullable weights to float (NA -> NaN -> 0 = not held).
    w = weights.astype(float).reindex(columns=daily_returns.columns)

    # Forward-fill monthly weights onto the daily grid, then shift one day so weights
    # formed at a month-end apply from the *next* trading day onward.
    w_daily = w.reindex(daily_returns.index, method="ffill").shift(1)

    port_ret = (w_daily.fillna(0.0) * daily_returns.fillna(0.0)).sum(axis=1)

    # Trim leading days before the first active (non-zero) weight row.
    active = (w_daily.fillna(0.0).abs().sum(axis=1) > 0)
    if active.any():
        port_ret = port_ret.loc[active.idxmax():]

    equity = (1.0 + port_ret).cumprod()
    drawdown = equity / equity.cummax() - 1.0

    out = pd.DataFrame({"port_ret": port_ret, "equity": equity, "drawdown": drawdown})
    out.index.name = "Date"
    return out


def write_daily_equity_curve(daily: pd.DataFrame, path: str) -> None:
    """Write a daily PnL DataFrame to CSV in the locked schema (Date,port_ret,equity,drawdown)."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    daily.to_csv(path)
