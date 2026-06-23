#!/usr/bin/env python3
"""
Filter the FNSPID raw dataset to the constituents of a sector ETF.

Reads the ticker list from the existing sector price CSV (same source used
by the backtest), then extracts matching headlines from the full FNSPID
raw Parquet and saves a sector-specific Parquet for compute_stress.py.

Requires: python scripts/download_fnspid.py to have been run first.

Usage:
    python scripts/cache_news.py --etf xlf
    python scripts/cache_news.py --etf xlk
    python scripts/cache_news.py --etf xlf --refresh   # re-filter even if cache exists
"""
import argparse
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(PROJECT_ROOT)
sys.path.insert(0, PROJECT_ROOT)

import pandas as pd
import yaml

from src.news_loader import FNSPIDLoader   # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--etf", required=True,
                   help="Sector ETF ticker (e.g. xlf, xlk, xle).")
    p.add_argument("--cache-dir", default="data/news",
                   help="Directory for cache files (default: data/news).")
    p.add_argument("--refresh", action="store_true",
                   help="Re-filter even if the sector cache already exists.")
    return p.parse_args()


def _load_tickers(cfg: dict, etf: str) -> list:
    """Extract ticker list from the sector price CSV."""
    prices_path = cfg["prices_file_path"].get(etf)
    if not prices_path:
        raise ValueError(f"ETF '{etf}' not found in config prices_file_path.")
    df = pd.read_csv(prices_path, nrows=0)  # headers only
    tickers = [
        col.replace(" Adj. Close", "").strip()
        for col in df.columns
        if "Adj. Close" in col
    ]
    return tickers


def main():
    args = parse_args()
    etf = args.etf.lower()

    with open("configs/optimise_trading_rules.yml") as fh:
        cfg = yaml.load(fh, Loader=yaml.SafeLoader)

    tickers = _load_tickers(cfg, etf)
    print(f"ETF      : {etf.upper()}")
    print(f"Tickers  : {len(tickers)} ({', '.join(tickers[:5])}{'...' if len(tickers) > 5 else ''})")

    loader = FNSPIDLoader(etf=etf, cache_dir=args.cache_dir)
    filtered = loader.filter_sector(tickers, force=args.refresh)

    print(f"\nDone. {len(filtered):,} headlines cached.")
    print(f"Output   : {loader.filtered_path}")
    print(f"Date range: {filtered['date'].min().date()} → {filtered['date'].max().date()}")

    top5 = filtered["ticker"].value_counts().head(5)
    print(f"\nTop 5 tickers by headline count:")
    for tkr, cnt in top5.items():
        print(f"  {tkr}: {cnt:,}")


if __name__ == "__main__":
    main()
