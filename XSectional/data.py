# data.py — download and cache S&P 500 adjusted close prices

import os
import logging

import pandas as pd
import yfinance as yf

import config

logger = logging.getLogger(__name__)

SP500_CACHE = os.path.join(config.DATA_DIR, "sp500_prices.csv")


def get_sp500_tickers() -> list[str]:
    """Fetch current S&P 500 tickers from Wikipedia."""
    tables = pd.read_html(
        "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    )
    tickers = tables[0]["Symbol"].str.replace(".", "-", regex=False).tolist()
    return tickers


def download_prices(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    """Download adjusted close prices from yfinance for a list of tickers."""
    raw = yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
    )
    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"]
    else:
        prices = raw[["Close"]]
    return prices


def drop_missing(prices: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Drop columns (tickers) where the fraction of NaN values exceeds threshold."""
    if prices.empty:
        return prices
    missing_frac = prices.isna().mean()
    return prices.loc[:, missing_frac <= threshold]


def load_prices() -> pd.DataFrame:
    """
    Return a DataFrame of adjusted close prices (dates × tickers).

    Loads from local CSV cache if available, otherwise downloads from yfinance
    and saves to cache. Raises if download fails and no cache exists.
    """
    os.makedirs(config.DATA_DIR, exist_ok=True)

    if os.path.exists(SP500_CACHE):
        logger.info("Loading prices from cache: %s", SP500_CACHE)
        prices = pd.read_csv(SP500_CACHE, index_col=0, parse_dates=True)
        return prices

    logger.info("Cache not found — downloading S&P 500 prices from yfinance...")
    tickers = get_sp500_tickers()
    prices = download_prices(tickers, config.START_DATE, config.END_DATE)
    prices = drop_missing(prices, config.MISSING_DATA_THRESHOLD)
    prices.to_csv(SP500_CACHE)
    logger.info("Prices saved to cache: %s", SP500_CACHE)
    return prices
