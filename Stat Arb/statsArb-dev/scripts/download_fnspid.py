#!/usr/bin/env python3
"""
One-time download of the FNSPID financial news dataset from HuggingFace.

FNSPID (Financial News and Stock Price Integration Dataset):
  - 15.7 million financial headlines (1999-2023)
  - 4,775 S&P 500 companies
  - Sources: Bloomberg, Reuters, Benzinga, NASDAQ
  - Free, no API key required
  - HuggingFace repo: Zihan1004/FNSPID

Downloads and converts to Parquet at data/news/fnspid_raw.parquet (~2-3 GB).
Subsequent scripts (cache_news.py, compute_stress.py) all read from this
local cache — no further internet access needed.

Usage:
    python scripts/download_fnspid.py
    python scripts/download_fnspid.py --output data/news/fnspid_raw.parquet
    python scripts/download_fnspid.py --refresh   # force re-download
"""
import argparse
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(PROJECT_ROOT)
sys.path.insert(0, PROJECT_ROOT)

from src.news_loader import FNSPIDLoader, FNSPID_RAW_DEFAULT   # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--output", default=FNSPID_RAW_DEFAULT,
        help=f"Path for the raw Parquet file (default: {FNSPID_RAW_DEFAULT}).",
    )
    p.add_argument(
        "--refresh", action="store_true",
        help="Force re-download even if the file already exists.",
    )
    return p.parse_args()


def _login_huggingface():
    """Log in using HF_TOKEN env var if set."""
    token = os.environ.get("HF_TOKEN")
    if token:
        try:
            from huggingface_hub import login
            login(token=token, add_to_git_credential=False)
            print("[HuggingFace] Logged in via HF_TOKEN env var.")
        except Exception as e:
            print(f"[HuggingFace] Login warning: {e}")
    else:
        print("[HuggingFace] No HF_TOKEN found — proceeding as anonymous "
              "(set HF_TOKEN env var if you get access errors).")


def main():
    args = parse_args()
    _login_huggingface()

    # Use a dummy ETF — loader.download_raw() doesn't use self.etf
    loader = FNSPIDLoader(etf="xlf", raw_path=args.output)
    loader.download_raw(force=args.refresh)

    if os.path.exists(args.output):
        size_gb = os.path.getsize(args.output) / 1e9
        import pandas as pd
        df = pd.read_parquet(args.output, columns=["date", "ticker"])
        print(f"\n=== FNSPID raw dataset ===")
        print(f"  File      : {args.output} ({size_gb:.2f} GB)")
        print(f"  Rows      : {len(df):,}")
        print(f"  Tickers   : {df['ticker'].nunique():,}")
        print(f"  Date range: {df['date'].min().date()} → {df['date'].max().date()}")
        print(f"\nNext steps:")
        print(f"  python scripts/cache_news.py --etf xlf")
        print(f"  python scripts/compute_stress.py --etf xlf --device cuda")


if __name__ == "__main__":
    main()
