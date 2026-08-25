"""
data_norgate.py — S&P 500 data loading layer via Norgate Data API.

Provides three cached datasets:
  load_prices()         — total-return adjusted daily close, all S&P 500 C&P symbols
  load_membership()     — point-in-time S&P 500 constituent mask (bool, trading days)
  compute_pca_factors() — rolling PCA eigen-portfolio returns for AL stat arb

All outputs are cached as CSV under USEquities/data/ so subsequent runs skip
the Norgate API calls. Delete the relevant CSV to force a refresh.
"""

import os
import logging
import numpy as np
import pandas as pd
import norgatedata as nd
from sklearn.decomposition import PCA

log = logging.getLogger(__name__)

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")

PRICES_CACHE     = os.path.join(DATA_DIR, "sp500_prices.csv")
MEMBERSHIP_CACHE = os.path.join(DATA_DIR, "sp500_membership.csv")
PCA_CACHE        = os.path.join(DATA_DIR, "sp500_pca_factors.csv")

WATCHLIST   = "S&P 500 Current & Past"
INDEX_NAME  = "S&P 500"
START       = "2024-08-01"
END         = "2026-08-24"

N_PCA_COMPONENTS = 15   # eigen-portfolios for AL stat arb defactoring
N_EST_WINDOW     = 60   # rolling estimation window (trading days)


# ─────────────────────────────────────────────────────────────────────────────
# Prices
# ─────────────────────────────────────────────────────────────────────────────

def load_prices(use_cache: bool = True) -> pd.DataFrame:
    """
    Total-return adjusted daily close prices for all S&P 500 C&P symbols.

    Returns
    -------
    DataFrame: index = trading dates, columns = Norgate ticker symbols (e.g. 'AAPL')
    """
    os.makedirs(DATA_DIR, exist_ok=True)

    if use_cache and os.path.exists(PRICES_CACHE):
        log.info("Prices: loading from cache (%s)", PRICES_CACHE)
        return pd.read_csv(PRICES_CACHE, index_col=0, parse_dates=True)

    symbols = nd.watchlist_symbols(WATCHLIST)
    log.info("Prices: downloading %d S&P 500 C&P series from Norgate...", len(symbols))

    frames: dict[str, pd.Series] = {}
    failed: list[str] = []
    for sym in symbols:
        try:
            rec = nd.price_timeseries(
                sym,
                stock_price_adjustment_setting=nd.StockPriceAdjustmentType.TOTALRETURN,
                start_date=START,
                end_date=END,
            )
            df = pd.DataFrame(rec)[["Date", "Close"]].copy()
            df["Date"] = pd.to_datetime(df["Date"])
            df = df.set_index("Date")
            if not df.empty:
                frames[sym] = df["Close"]
        except Exception as exc:
            log.debug("  skip %s: %s", sym, exc)
            failed.append(sym)

    if failed:
        log.warning("  %d symbols had no data: %s%s",
                    len(failed), failed[:8], " ..." if len(failed) > 8 else "")

    prices = pd.DataFrame(frames).sort_index()
    prices.index = pd.to_datetime(prices.index)
    prices.to_csv(PRICES_CACHE)
    log.info("Prices saved: %d days × %d tickers → %s", *prices.shape, PRICES_CACHE)
    return prices


# ─────────────────────────────────────────────────────────────────────────────
# Point-in-time S&P 500 membership mask
# ─────────────────────────────────────────────────────────────────────────────

def load_membership(prices: pd.DataFrame, use_cache: bool = True) -> pd.DataFrame:
    """
    Point-in-time S&P 500 constituent mask aligned to prices.index.

    Returns
    -------
    DataFrame: same shape as prices, dtype bool.
                True  = stock was in the S&P 500 on that trading day.
    """
    os.makedirs(DATA_DIR, exist_ok=True)

    if use_cache and os.path.exists(MEMBERSHIP_CACHE):
        log.info("Membership: loading from cache (%s)", MEMBERSHIP_CACHE)
        return pd.read_csv(MEMBERSHIP_CACHE, index_col=0, parse_dates=True).astype(bool)

    symbols = list(prices.columns)
    log.info("Membership: building PIT mask for %d symbols...", len(symbols))
    mask = pd.DataFrame(False, index=prices.index, columns=symbols, dtype=bool)

    for sym in symbols:
        try:
            rec = nd.index_constituent_timeseries(
                sym,
                INDEX_NAME,
                padding_setting=nd.PaddingType.ALLCALENDARDAYS,
                start_date=START,
                end_date=END,
            )
            df = pd.DataFrame(rec).copy()
            df["Date"] = pd.to_datetime(df["Date"])
            df = df.set_index("Date")
            series = (
                df["Index Constituent"]
                .reindex(prices.index, method="ffill")
                .fillna(0)
                .astype(bool)
            )
            mask[sym] = series
        except Exception as exc:
            log.debug("  membership skip %s: %s", sym, exc)

    mask.to_csv(MEMBERSHIP_CACHE)
    log.info("Membership saved: %d days × %d tickers → %s",
             mask.shape[0], mask.shape[1], MEMBERSHIP_CACHE)
    return mask


# ─────────────────────────────────────────────────────────────────────────────
# Rolling PCA eigen-portfolio returns
# ─────────────────────────────────────────────────────────────────────────────

def compute_pca_factors(
    prices: pd.DataFrame,
    membership: pd.DataFrame,
    n_window: int = N_EST_WINDOW,
    n_components: int = N_PCA_COMPONENTS,
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    Daily rolling PCA eigen-portfolio returns, point-in-time.
    """
    os.makedirs(DATA_DIR, exist_ok=True)

    if use_cache and os.path.exists(PCA_CACHE):
        log.info("PCA factors: loading from cache (%s)", PCA_CACHE)
        return pd.read_csv(PCA_CACHE, index_col=0, parse_dates=True)

    log.info("PCA factors: computing (%d components, %d-day window)...",
             n_components, n_window)

    ret = prices.astype(float).ffill().pct_change()
    n_days = len(ret)
    results: list[list] = []
    n_skipped = 0

    for i in range(n_window, n_days):
        dt = ret.index[i]

        if dt not in membership.index:
            results.append([dt] + [np.nan] * n_components)
            continue
        active = membership.loc[dt]
        active_tickers = active[active].index.tolist()

        window = ret.iloc[i - n_window: i][active_tickers]
        window = window.replace(0, np.nan).dropna(axis=1, how="any")

        vol = window.std()
        window = window.loc[:, vol > 1e-10]

        if window.shape[1] < n_components + 1:
            results.append([dt] + [np.nan] * n_components)
            continue

        rho = window.corr().dropna(axis=0, how="any").dropna(axis=1, how="any")
        sig_bar = window.std()[rho.columns]

        if rho.shape[0] < n_components + 1:
            results.append([dt] + [np.nan] * n_components)
            continue

        try:
            v = PCA(n_components=n_components).fit(rho).components_
            v = v / np.sum(np.abs(v))
            ret_dt = ret.iloc[i][rho.columns].fillna(0).values
            f = (v / sig_bar.values[np.newaxis, :]).dot(ret_dt)
            results.append([dt, *f])
        except np.linalg.LinAlgError:
            n_skipped += 1
            results.append([dt] + [np.nan] * n_components)

        if (i - n_window) % 100 == 0:
            log.info("  PCA: day %d / %d", i - n_window, n_days - n_window)

    cols = ["Date"] + [f"pca_{k}" for k in range(n_components)]
    pca_df = (
        pd.DataFrame(results, columns=cols)
        .set_index("Date")
        .dropna(how="all")
    )
    pca_df.to_csv(PCA_CACHE)
    log.info("PCA factors saved: %d rows → %s (skipped %d SVD failures)",
             len(pca_df), PCA_CACHE, n_skipped)
    return pca_df
