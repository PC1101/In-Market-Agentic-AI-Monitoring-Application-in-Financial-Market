# data.py — download and cache S&P 500 adjusted close prices

import io
import os
import logging

import pandas as pd
import requests
import yfinance as yf

import config

logger = logging.getLogger(__name__)

SP500_CACHE = os.path.join(config.DATA_DIR, "sp500_prices.csv")
PIT_CACHE = os.path.join(config.DATA_DIR, "sp500_prices_pit.csv")
PIT_REPORT = os.path.join(config.DATA_DIR, "pit_download_report.csv")


def get_sp500_tickers() -> list[str]:
    """Fetch current S&P 500 tickers from Wikipedia."""
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    headers = {"User-Agent": "Mozilla/5.0 (compatible; XSectional/1.0; +https://github.com/PC1101)"}
    response = requests.get(url, headers=headers, timeout=15)
    response.raise_for_status()
    tables = pd.read_html(io.StringIO(response.text))
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


def _to_yahoo_symbol(ticker: str) -> str:
    """Map an index symbol to yfinance's convention (class shares use '-')."""
    return ticker.replace(".", "-")


def load_prices_pit() -> pd.DataFrame:
    """
    Return adjusted close prices for the POINT-IN-TIME universe: every ticker that
    was an S&P 500 member at any time in [START_DATE, END_DATE], per the
    sp500-master membership table — not just today's members.

    Downloads the union of historical members from yfinance (many delisted names
    are unavailable there; those failures are recorded in PIT_REPORT so residual
    survivorship bias is measurable, not silent). Cached to PIT_CACHE after the
    first download.

    Columns are named with the membership-table symbols so they align with
    universe.membership_mask.
    """
    import universe

    os.makedirs(config.DATA_DIR, exist_ok=True)

    if os.path.exists(PIT_CACHE):
        logger.info("Loading PIT prices from cache: %s", PIT_CACHE)
        return pd.read_csv(PIT_CACHE, index_col=0, parse_dates=True)

    membership = universe.load_membership()
    tickers = universe.tickers_active_between(
        config.START_DATE, config.END_DATE, membership
    )
    logger.info(
        "PIT universe: %d historical member tickers — downloading from yfinance...",
        len(tickers),
    )
    yahoo_map = {_to_yahoo_symbol(t): t for t in tickers}
    prices = download_prices(list(yahoo_map), config.START_DATE, config.END_DATE)
    # Restore membership-table symbols.
    prices = prices.rename(columns=yahoo_map)

    # A column that is entirely NaN means yfinance has no data (usually delisted).
    got = prices.columns[prices.notna().any()].tolist()
    missing = sorted(set(tickers) - set(got))
    report = pd.DataFrame(
        {"ticker": tickers,
         "downloaded": [t in set(got) for t in tickers]}
    )
    report.to_csv(PIT_REPORT, index=False)
    logger.info(
        "PIT download: %d/%d tickers have data; %d unavailable "
        "(mostly delisted; see %s)",
        len(got), len(tickers), len(missing), PIT_REPORT,
    )

    prices = prices[got]
    # NOTE: unlike load_prices, do NOT drop tickers for having a high missing
    # fraction — a name that traded 2001-2008 then died is exactly the point of
    # the PIT universe. The membership mask handles candidacy per date.
    prices.to_csv(PIT_CACHE)
    logger.info("PIT prices saved to cache: %s", PIT_CACHE)
    return prices
