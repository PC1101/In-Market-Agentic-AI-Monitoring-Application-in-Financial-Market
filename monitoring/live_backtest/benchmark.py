"""Fetch and cache S&P 500 benchmark (SPY) for buy-and-hold comparison."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

_CACHE_DIR = Path(__file__).resolve().parent.parent / "data"


def fetch_benchmark(ticker: str = "SPY",
                    start: str = "2001-01-01",
                    end: str = "2025-12-31",
                    cache_dir: Path | None = None,
                    no_fetch: bool = False) -> pd.Series:
    """Fetch daily returns for a benchmark ticker via yfinance, with CSV caching.

    Args:
        ticker: benchmark ticker symbol (default SPY; ^GSPC supported but price-only).
        start, end: date range.
        cache_dir: directory for the cache CSV.
        no_fetch: if True, only load from cache (raise if not cached).

    Returns:
        pd.Series of daily simple returns indexed by DatetimeIndex.
    """
    if cache_dir is None:
        cache_dir = _CACHE_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"benchmark_{ticker}.csv"

    if cache_path.exists():
        df = pd.read_csv(cache_path, parse_dates=["Date"], index_col="Date")
        ret = df["ret"].astype(float)
        # Filter to requested range
        ret = ret.loc[start:end]
        if len(ret) > 0:
            return ret

    if no_fetch:
        raise FileNotFoundError(
            f"Benchmark cache not found at {cache_path} and --no-fetch is set."
        )

    import yfinance as yf
    data = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    if data.empty:
        raise RuntimeError(f"yfinance returned no data for {ticker}")

    close = data["Close"]
    # Handle MultiIndex columns from yfinance
    if hasattr(close, "columns"):
        close = close.iloc[:, 0]
    ret = close.pct_change().dropna()
    ret.name = "ret"
    ret.index.name = "Date"

    # Cache
    ret.to_frame().to_csv(cache_path)
    return ret


def buy_and_hold(returns: pd.Series, capital: float = 1_000_000.0) -> pd.Series:
    """Compute buy-and-hold value series from daily returns."""
    cumulative = (1 + returns).cumprod()
    return capital * cumulative
