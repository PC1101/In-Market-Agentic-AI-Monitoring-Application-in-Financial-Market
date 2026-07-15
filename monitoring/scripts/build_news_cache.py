#!/usr/bin/env python3
"""Build the monitoring news caches from raw NYT archive JSON and FNSPID parquet.

Outputs (both consumed by ``monitoring/news/records.py``):

* ``monitoring/news/data/nyt_archive.parquet`` — one row per NYT article:
  ``pub_ts`` (exact publication timestamp), ``headline``, ``abstract``,
  ``section_name``, ``news_desk``, ``url``, ``source="nyt"``. All sections are
  kept — section filtering happens in the pipeline so it can be tuned without
  re-fetching.
* ``monitoring/news/data/fnspid_windows.parquet`` — FNSPID headlines sliced to
  the six evaluation windows (+30-day lead-ins), non-English titles dropped
  (pre-2009 FNSPID is a Russian general-news scrape — see WEEK3_STATUS.md):
  ``pub_ts`` (date at midnight — day granularity), ``headline``, ``ticker``,
  ``url``, ``publisher``, ``source="fnspid"``.

Source policy: NYT is the primary market-wide source for ALL windows (uniform
coverage avoids a source-mix confound across windows); FNSPID adds ticker-tagged
headlines from 2010 onward.

Usage (from ``monitoring/``):
    python scripts/build_news_cache.py
    python scripts/build_news_cache.py --fnspid-raw "path/to/fnspid_raw.parquet"
"""
from __future__ import annotations

import argparse
import gzip
import json
import re
import sys
from pathlib import Path

import pandas as pd

MONITORING_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MONITORING_ROOT))

from windows import ALL_WINDOWS  # noqa: E402

DATA_DIR = MONITORING_ROOT / "news" / "data"
RAW_DIR = DATA_DIR / "nyt_raw"
FNSPID_RAW_DEFAULT = (MONITORING_ROOT.parent / "Stat Arb" / "statsArb-dev"
                      / "data" / "news" / "fnspid_raw.parquet")
LEAD_IN = pd.Timedelta(days=31)

_CYRILLIC = re.compile(r"[\u0400-\u04FF]")


def build_nyt(out_path: Path) -> None:
    files = sorted(RAW_DIR.glob("*.json.gz"))
    if not files:
        print(f"[nyt] no raw months in {RAW_DIR} — run scripts/fetch_nyt_archive.py first")
        return
    frames = []
    for f in files:
        with gzip.open(f, "rb") as fh:
            docs = json.loads(fh.read())["response"]["docs"]
        rows = []
        for d in docs:
            headline = ((d.get("headline") or {}).get("main") or "").strip()
            if not headline:
                continue
            rows.append({
                "pub_ts": d.get("pub_date"),
                "headline": headline,
                "abstract": (d.get("abstract") or d.get("snippet") or "").strip() or None,
                "section_name": d.get("section_name"),
                "news_desk": d.get("news_desk"),
                "url": d.get("web_url"),
            })
        frames.append(pd.DataFrame(rows))
        print(f"  {f.name}: {len(rows):,} articles")
    df = pd.concat(frames, ignore_index=True)
    df["pub_ts"] = pd.to_datetime(df["pub_ts"], errors="coerce", utc=True).dt.tz_localize(None)
    df = df.dropna(subset=["pub_ts"]).sort_values("pub_ts").reset_index(drop=True)
    df["source"] = "nyt"
    df.to_parquet(out_path, index=False)
    print(f"[nyt] {len(df):,} articles ({df['pub_ts'].min().date()} -> "
          f"{df['pub_ts'].max().date()}) -> {out_path}")


def build_fnspid(raw_path: Path, out_path: Path) -> None:
    if not raw_path.exists():
        print(f"[fnspid] raw parquet not found at {raw_path} — skipping "
              "(only NYT will be available)")
        return
    import pyarrow.parquet as pq

    spans = [(w.start_ts - LEAD_IN, w.end_ts) for w in ALL_WINDOWS]
    pf = pq.ParquetFile(raw_path)
    cols = [c for c in ["date", "ticker", "article", "url", "publisher"]
            if c in pf.schema_arrow.names]
    kept = []
    for batch in pf.iter_batches(batch_size=1_000_000, columns=cols):
        df = batch.to_pandas()
        d = pd.to_datetime(df["date"], errors="coerce", utc=True).dt.tz_localize(None)
        mask = pd.Series(False, index=df.index)
        for lo, hi in spans:
            mask |= (d >= lo) & (d <= hi)
        if not bool(mask.any()):
            continue
        sub = df.loc[mask].copy()
        sub["pub_ts"] = d[mask]
        kept.append(sub)
    if not kept:
        print("[fnspid] no rows fell inside the evaluation windows")
        return
    df = pd.concat(kept, ignore_index=True)

    # Drop non-English titles (Cyrillic) and empty/short strings.
    head = df["article"].astype(str).str.strip()
    ok = (head.str.len() > 10) & ~head.map(lambda s: bool(_CYRILLIC.search(s)))
    df = df.loc[ok].copy()
    df["headline"] = head[ok]
    # FNSPID repeats the same wire headline across many ticker rows — keep one
    # per (calendar day, headline) so daily counts measure stories, not fan-out.
    df["_day"] = df["pub_ts"].dt.normalize()
    df = df.drop_duplicates(subset=["_day", "headline"]).drop(columns="_day")
    df["ticker"] = df.get("ticker", pd.Series(index=df.index, dtype=str))
    df["source"] = "fnspid"
    keep_cols = ["pub_ts", "headline", "ticker", "source"] + [
        c for c in ("url", "publisher") if c in df.columns
    ]
    df = df[keep_cols].sort_values("pub_ts").reset_index(drop=True)
    df.to_parquet(out_path, index=False)
    print(f"[fnspid] {len(df):,} usable headlines in windows "
          f"({df['pub_ts'].min().date()} -> {df['pub_ts'].max().date()}) -> {out_path}")
    per_win = {
        w.name: int(((df["pub_ts"] >= w.start_ts - LEAD_IN) & (df["pub_ts"] <= w.end_ts)).sum())
        for w in ALL_WINDOWS
    }
    print("  per window:", json.dumps(per_win))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--fnspid-raw", default=str(FNSPID_RAW_DEFAULT),
                    help="path to the (re-ingested) fnspid_raw.parquet")
    args = ap.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    build_nyt(DATA_DIR / "nyt_archive.parquet")
    build_fnspid(Path(args.fnspid_raw), DATA_DIR / "fnspid_windows.parquet")


if __name__ == "__main__":
    main()
