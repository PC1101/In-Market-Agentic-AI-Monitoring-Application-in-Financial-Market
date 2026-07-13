"""Download the FNSPID news CSV from HuggingFace and peek at its schema.

FNSPID (Dong et al. 2024): 15.7M time-aligned financial news records, 1999-2023.
Dataset repo: Zihan1004/FNSPID (the canonical repo from the paper's authors;
Zdong104/FNSPID_Financial_News_Dataset, named in some secondary write-ups, 401s /
does not resolve on HuggingFace as of 2026-07).
We download the news file once to data/fnspid/raw/, then Task 4 stream-filters it
to the backtest window (2003-2016) as a parquet store; the raw CSV can be deleted
afterwards if disk is tight.

Usage (from monitoring/):
    python news/download_fnspid.py --list          # show repo files + sizes, pick one
    python news/download_fnspid.py                 # download default news CSV
    python news/download_fnspid.py --peek          # print first rows of downloaded CSV
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

REPO_ID = "Zihan1004/FNSPID"  # canonical repo; Zdong104/FNSPID_Financial_News_Dataset 401s (private/gone)
DEFAULT_FILE = "Stock_news/nasdaq_exteral_data.csv"  # (sic — typo is upstream's)
RAW_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "fnspid" / "raw"


def list_files() -> None:
    from huggingface_hub import HfApi

    api = HfApi()
    info = api.repo_info(REPO_ID, repo_type="dataset", files_metadata=True)
    for f in sorted(info.siblings, key=lambda s: s.rfilename):
        size_gb = (f.size or 0) / 1e9
        print(f"{size_gb:8.2f} GB  {f.rfilename}")


def download(filename: str) -> Path:
    from huggingface_hub import hf_hub_download

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    path = hf_hub_download(
        repo_id=REPO_ID, filename=filename, repo_type="dataset",
        local_dir=RAW_DIR,
    )
    print(f"Downloaded to {path}")
    return Path(path)


def peek(filename: str, n: int = 3) -> None:
    path = RAW_DIR / filename
    head = pd.read_csv(path, nrows=n)
    print("Columns:", list(head.columns))
    print(head.to_string(max_colwidth=60))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true", help="list repo files and sizes")
    ap.add_argument("--peek", action="store_true", help="print head of downloaded CSV")
    ap.add_argument("--file", default=DEFAULT_FILE)
    args = ap.parse_args()
    if args.list:
        list_files()
    elif args.peek:
        peek(args.file)
    else:
        download(args.file)


if __name__ == "__main__":
    main()
