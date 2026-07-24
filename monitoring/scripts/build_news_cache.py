"""Build the monitoring news caches from raw NYT archive JSON and FNSPID parquet.

Outputs (both consumed by the news pipeline):

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

from windows import ALL_WINDOWS

DATA_DIR = MONITORING_ROOT / "news" / "data"
RAW_DIR = DATA_DIR / "nyt_raw"
FNSPID_RAW_DEFAULT = (
    MONITORING_ROOT.parent / "Stat Arb" / "statsArb-dev" / "data" / "news" / "fnspid_raw.parquet"
)
LEAD_IN = pd.Timedelta(days=31)

_CYRILLIC = re.compile(r"[\u0400-\u04FF]")


def build_nyt(out_path: Path) -> None:
    """Parse cached gzipped NYT Archive JSON into a single parquet."""
    files = sorted(RAW_DIR.glob("*.json.gz"))
    if not files:
        print(f"[nyt] no raw months in {RAW_DIR} — run scripts/fetch_nyt_archive.py first")
        return

    frames: list[pd.DataFrame] = []
    for f in files:
        with gzip.open(f, "rb") as fh:
            docs = json.loads(fh.read())
        rows = []
        for d in docs:
            headline = (d.get("headline") or {}).get("main", "").strip()
            rows.append({
                "pub_ts": d.get("pub_date", ""),
                "headline": headline,
                "abstract": (d.get("abstract") or "").strip(),
                "section_name": d.get("section_name", ""),
                "news_desk": d.get("news_desk", ""),
                "url": d.get("web_url", ""),
                "source": "nyt",
            })
        frames.append(pd.DataFrame(rows))

    df = pd.concat(frames, ignore_index=True)
    df["pub_ts"] = pd.to_datetime(df["pub_ts"], errors="coerce").dt.tz_localize(None)
    df = df.dropna(subset=["pub_ts"])
    df = df[df["headline"] != ""]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    print(f"[nyt] {len(df):,} articles -> {out_path}")


def build_fnspid(raw_path: Path, out_path: Path) -> None:
    """Slice FNSPID parquet to evaluation windows, drop non-English titles."""
    if not raw_path.exists():
        print(f"[fnspid] raw parquet not found at {raw_path} — skipping "
              f"(only NYT will be available)")
        return

    import pyarrow.parquet as parquet

    # Build (lo, hi) date spans per window with lead-in.
    spans = [
        (w.start_ts - LEAD_IN, w.end_ts)
        for w in ALL_WINDOWS
    ]
    schema_arrow = parquet.ParquetFile(raw_path).schema_arrow
    cols = [c for c in schema_arrow.names
            if c in {"date", "article", "Stock_symbol", "Url", "Publisher",
                     "Article_title", "ticker", "title", "url", "publisher"}]

    kept: list[pd.DataFrame] = []
    for batch in parquet.ParquetFile(raw_path).iter_batches(batch_size=1_000_000, columns=cols):
        df = batch.to_pandas()
        df["date"] = pd.to_datetime(df.get("date", df.get("Date")), errors="coerce")
        df["date"] = df["date"].dt.tz_localize(None)
        pub_ts = pd.Series(df["date"], name="pub_ts")

        mask = pd.Series(False, index=df.index)
        for lo, hi in spans:
            mask |= (pub_ts >= lo) & (pub_ts <= hi)

        sub = df.loc[mask].copy()
        if sub.empty:
            continue

        # Rename columns to canonical names.
        sub = sub.rename(columns={
            "Article_title": "headline", "article": "headline",
            "title": "headline",
            "Stock_symbol": "ticker", "ticker": "ticker",
            "Url": "url", "url": "url",
            "Publisher": "publisher", "publisher": "publisher",
        })
        for c in ("headline", "ticker", "url", "publisher"):
            if c not in sub.columns:
                sub[c] = ""

        # Drop Cyrillic-dominated titles (pre-2009 Lenta.ru scrape).
        head = sub["headline"].fillna("")
        ok = ~head.map(lambda s: bool(_CYRILLIC.search(s[:10])))
        sub = sub.loc[ok].copy()

        keep_cols = ["pub_ts", "headline", "ticker", "url", "publisher"]
        sub["pub_ts"] = pub_ts.loc[sub.index]
        sub["source"] = "fnspid"
        kept.append(sub[keep_cols + ["source"]])

    if not kept:
        print("[fnspid] no rows fell inside the evaluation windows")
        return

    per_win = pd.concat(kept, ignore_index=True)
    per_win = per_win.drop_duplicates(subset=["pub_ts", "headline"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    per_win.to_parquet(out_path, index=False)
    print(f"[fnspid] {len(per_win):,} articles -> {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--fnspid-raw", type=str, default=str(FNSPID_RAW_DEFAULT),
                    help="path to the (re-ingested) fnspid_raw.parquet")
    args = ap.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    build_nyt(DATA_DIR / "nyt_archive.parquet")
    build_fnspid(Path(args.fnspid_raw), DATA_DIR / "fnspid_windows.parquet")


if __name__ == "__main__":
    main()
