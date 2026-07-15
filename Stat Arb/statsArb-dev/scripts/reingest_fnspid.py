#!/usr/bin/env python3
"""
Re-ingest the FNSPID dataset from the already-downloaded HuggingFace CSV cache.

Why: the original ingest matched the "Article" column (the scraped article BODY,
empty for most pre-2010 rows) instead of "Article_title" (the headline, populated
for every row). news_loader._TEXT_COLS is fixed to prefer Article_title; this
script re-parses the cached 23 GB CSV into data/news/fnspid_raw.parquet with the
corrected column mapping (plus url/publisher provenance). No re-download needed:
hf_hub_download hits the local cache.

Usage:
    python scripts/reingest_fnspid.py
    python scripts/reingest_fnspid.py --output data/news/fnspid_raw.parquet
"""
import argparse
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(PROJECT_ROOT)
sys.path.insert(0, PROJECT_ROOT)

from src.news_loader import FNSPIDLoader, FNSPID_RAW_DEFAULT, FNSPID_REPO  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--output", default=FNSPID_RAW_DEFAULT,
        help=f"Path for the raw Parquet file (default: {FNSPID_RAW_DEFAULT}).",
    )
    return p.parse_args()


def main():
    args = parse_args()
    from huggingface_hub import hf_hub_download

    # Same candidate order as FNSPIDLoader._download_via_hub. local_files_only
    # guarantees we only use the existing cache (fails loudly otherwise).
    csv_path = None
    for filename in [
        "Stock_news/nasdaq_exteral_data.csv",   # 23 GB — primary (repo typo intentional)
        "Stock_news/All_external.csv",           # 5.7 GB — fallback
    ]:
        try:
            csv_path = hf_hub_download(
                repo_id=FNSPID_REPO, filename=filename,
                repo_type="dataset", local_files_only=True,
            )
            break
        except Exception as exc:
            print(f"  {filename}: not in local cache ({exc})")

    if csv_path is None:
        sys.exit("No cached FNSPID CSV found. Run scripts/download_fnspid.py first.")

    print(f"[reingest_fnspid] Re-parsing cached CSV: {csv_path}")
    loader = FNSPIDLoader(etf="xlf", raw_path=args.output)
    loader._csv_to_parquet_chunked(csv_path)

    import pandas as pd
    df = pd.read_parquet(args.output, columns=["date", "article"])
    n_text = int((df["article"].astype(str).str.strip() != "").sum())
    size_gb = os.path.getsize(args.output) / 1e9
    print(f"\n=== FNSPID raw dataset (re-ingested with Article_title) ===")
    print(f"  File           : {args.output} ({size_gb:.2f} GB)")
    print(f"  Rows           : {len(df):,}")
    print(f"  Non-empty text : {n_text:,}")
    print(f"  Date range     : {df['date'].min().date()} -> {df['date'].max().date()}")
    print("\nNote: the XLF sector cache is now stale; re-run "
          "scripts/cache_news.py --etf xlf if you need it.")


if __name__ == "__main__":
    main()
