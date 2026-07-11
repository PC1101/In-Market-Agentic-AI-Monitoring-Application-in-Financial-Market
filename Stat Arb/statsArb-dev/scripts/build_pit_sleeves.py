"""Build POINT-IN-TIME sector sleeve price files for the Avellaneda-Lee backtest.

The stock universe of each SPDR sector sleeve is rebuilt from the sp500-master
membership table (all historical S&P 500 members, incl. names that later left the
index) instead of the 2020 holdings snapshot. Each ticker's prices are masked to its
membership interval(s), so a name delisted in 2009 exists in the 2007 backtest and
vanishes afterwards.

Inputs
  * ../../sp500-master/sp500_ticker_start_end.csv   — membership intervals
  * ../../XSectional/data/sp500_prices_pit.csv      — adjusted-close prices for all
    historical members yfinance carries (built by the JT PIT pipeline; reused here)
  * yfinance                                        — sector classification per ticker
    (cached to data/pit/sector_map.csv) and the 11 SPDR ETF price series

Outputs (data/pit/)
  * {etf}_pit.csv       — wide "TICKER Adj. Close" files in the bt engine's format,
                          ETF column first, constituents masked to membership
  * sector_map.csv      — ticker -> sleeve map (with 'unclassified' rows kept for audit)
  * pit_sleeve_report.csv — per-sleeve ticker counts and coverage

Classification caveats (documented):
  * Sector metadata comes from yfinance today; names with no metadata (mostly dead
    tickers) are reported as unclassified and excluded — a measured residual, not silent.
  * Communication-services names map to the XLC sleeve, which the runner skips for
    windows before XLC's 2018 inception — the same treatment the 2020-snapshot
    baseline applies, so PIT-vs-baseline comparisons are like-for-like.

Run:  python scripts/build_pit_sleeves.py
"""

import logging
import os
import sys
import time

import pandas as pd
import yfinance as yf

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)                       # statsArb-dev
ROOT = os.path.dirname(os.path.dirname(PROJ))      # repo root

MEMBERSHIP = os.path.join(ROOT, "sp500-master", "sp500_ticker_start_end.csv")
PIT_PRICES = os.path.join(ROOT, "XSectional", "data", "sp500_prices_pit.csv")
OUT_DIR = os.path.join(PROJ, "data", "pit")
SECTOR_MAP_CACHE = os.path.join(OUT_DIR, "sector_map.csv")

# Price span: covers calm_2004_2006 and baseline_2007_2015 with engine warm-up room.
PRICE_START, PRICE_END = "2003-01-01", "2016-12-31"

ETFS = ["xlf", "xlb", "xlc", "xle", "xli", "xlk", "xlp", "xlre", "xlu", "xlv", "xly"]

# yfinance sector string -> SPDR sleeve
SECTOR_TO_ETF = {
    "Financial Services": "xlf",
    "Basic Materials": "xlb",
    "Communication Services": "xlc",
    "Energy": "xle",
    "Industrials": "xli",
    "Technology": "xlk",
    "Consumer Defensive": "xlp",
    "Real Estate": "xlre",
    "Utilities": "xlu",
    "Healthcare": "xlv",
    "Consumer Cyclical": "xly",
}

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("build_pit_sleeves")


def yahoo_symbol(t: str) -> str:
    return t.replace(".", "-")


def load_membership() -> pd.DataFrame:
    m = pd.read_csv(MEMBERSHIP, parse_dates=["start_date", "end_date"])
    m["ticker"] = m["ticker"].str.strip().str.upper()
    return m


def build_sector_map(tickers: list) -> pd.DataFrame:
    """ticker -> sleeve via yfinance metadata; cached across runs."""
    if os.path.exists(SECTOR_MAP_CACHE):
        cached = pd.read_csv(SECTOR_MAP_CACHE)
        if set(tickers) <= set(cached["ticker"]):
            log.info("Sector map loaded from cache (%d tickers)", len(cached))
            return cached
    rows = []
    log.info("Classifying %d tickers via yfinance metadata...", len(tickers))
    for i, t in enumerate(tickers):
        sector = None
        try:
            info = yf.Ticker(yahoo_symbol(t)).info
            sector = info.get("sector")
        except Exception:
            sector = None
        rows.append({"ticker": t,
                     "sector": sector or "unknown",
                     "etf": SECTOR_TO_ETF.get(sector, "unclassified")})
        if (i + 1) % 50 == 0:
            log.info("  %d/%d classified", i + 1, len(tickers))
            time.sleep(1)  # be polite to the API
    df = pd.DataFrame(rows)
    os.makedirs(OUT_DIR, exist_ok=True)
    df.to_csv(SECTOR_MAP_CACHE, index=False)
    n_ok = (df["etf"] != "unclassified").sum()
    log.info("Sector map: %d/%d classified (%d unclassified, kept for audit)",
             n_ok, len(df), len(df) - n_ok)
    return df


def mask_to_membership(prices: pd.DataFrame, membership: pd.DataFrame) -> pd.DataFrame:
    """NaN-out each ticker's prices outside its S&P membership interval(s)."""
    masked = pd.DataFrame(index=prices.index, columns=prices.columns, dtype=float)
    for t in prices.columns:
        keep = pd.Series(False, index=prices.index)
        for _, row in membership[membership["ticker"] == t].iterrows():
            hi = row["end_date"] if pd.notna(row["end_date"]) else prices.index.max()
            keep |= (prices.index >= row["start_date"]) & (prices.index <= hi)
        masked[t] = prices[t].where(keep)
    return masked


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    membership = load_membership()

    log.info("Loading PIT price cache: %s", PIT_PRICES)
    prices = pd.read_csv(PIT_PRICES, index_col=0, parse_dates=True)
    prices = prices.loc[PRICE_START:PRICE_END]

    # Tickers that were members at some point in the price span AND have data.
    lo, hi = pd.Timestamp(PRICE_START), pd.Timestamp(PRICE_END)
    active = membership[(membership["start_date"] <= hi)
                        & (membership["end_date"].isna() | (membership["end_date"] >= lo))]
    tickers = sorted(set(active["ticker"]) & set(prices.columns))
    log.info("PIT members in %s..%s with price data: %d", PRICE_START, PRICE_END, len(tickers))

    sector_map = build_sector_map(tickers)

    log.info("Downloading SPDR ETF prices...")
    etf_px = yf.download([e.upper() for e in ETFS], start=PRICE_START, end=PRICE_END,
                         auto_adjust=True, progress=False)["Close"]

    report = []
    for etf in ETFS:
        sleeve_tickers = sector_map[(sector_map["etf"] == etf)
                                    & (sector_map["ticker"].isin(tickers))]["ticker"].tolist()
        etf_col = etf_px[etf.upper()].dropna() if etf.upper() in etf_px.columns else pd.Series(dtype=float)
        if not sleeve_tickers or etf_col.empty:
            report.append({"etf": etf, "n_tickers": len(sleeve_tickers),
                           "written": False, "reason": "no tickers or no ETF prices"})
            log.info("  %s: skipped (%d tickers, etf prices %s)",
                     etf, len(sleeve_tickers), "yes" if not etf_col.empty else "no")
            continue

        sleeve_px = mask_to_membership(prices[sleeve_tickers], membership)
        # Engine format: ETF column first, then constituents, all named "T Adj. Close".
        wide = pd.concat([etf_col.rename(etf.upper()), sleeve_px], axis=1)
        wide = wide.loc[etf_col.index.min(): etf_col.index.max()]
        wide.columns = [f"{c} Adj. Close" for c in wide.columns]
        wide.index.name = "Date"
        out = os.path.join(OUT_DIR, f"{etf}_pit.csv")
        wide.to_csv(out)
        cov = sleeve_px.notna().any(axis=1).mean()
        report.append({"etf": etf, "n_tickers": len(sleeve_tickers), "written": True,
                       "reason": ""})
        log.info("  %s: %d constituents -> %s", etf, len(sleeve_tickers), out)

    rep = pd.DataFrame(report)
    rep.to_csv(os.path.join(OUT_DIR, "pit_sleeve_report.csv"), index=False)
    n_uncls = (sector_map["etf"] == "unclassified").sum()
    log.info("Done. %d sleeves written; %d tickers unclassified (see sector_map.csv)",
             int(rep["written"].sum()), n_uncls)


if __name__ == "__main__":
    main()
