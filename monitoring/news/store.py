"""FNSPID parquet store: chunked build from the raw CSV, date-indexed queries.

The raw FNSPID news CSV is ~22 GB; we never load it whole. ``build_store`` streams
it in chunks, keeps only the backtest window and the columns the pipeline needs,
and writes one parquet file per calendar year. ``NewsStore.query`` then reads only
the year files a date range touches — "indexed by date, ready to query for any
date in the backtest window" (VRI Week 3).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

COLS = ["date", "ticker", "title", "summary", "publisher", "url"]

SCHEMA = pa.schema([
    ("date", pa.timestamp("ns")),
    ("ticker", pa.string()),
    ("title", pa.string()),
    ("summary", pa.string()),
    ("publisher", pa.string()),
    ("url", pa.string()),
])

#: raw-CSV column name -> canonical name (first match per canonical wins)
CANDIDATES = {
    "date": ["Date", "date", "Publish_date", "publish_date"],
    "title": ["Article_title", "Title", "title", "Headline"],
    "ticker": ["Stock_symbol", "Symbol", "ticker", "symbol"],
    "summary": ["Lsa_summary", "Textrank_summary", "Luhn_summary", "Summary", "summary"],
    "publisher": ["Publisher", "publisher"],
    "url": ["Url", "URL", "url"],
}


def detect_columns(csv_path) -> dict[str, str]:
    """Map raw header names to canonical names. Requires date + title."""
    header = list(pd.read_csv(csv_path, nrows=0).columns)
    mapping: dict[str, str] = {}
    for canon, options in CANDIDATES.items():
        for opt in options:
            if opt in header:
                mapping[opt] = canon
                break
    found = set(mapping.values())
    if "date" not in found or "title" not in found:
        raise ValueError(f"could not find date/title columns in {header}")
    return mapping


def build_store(csv_path, out_dir, start="2003-01-01", end="2016-12-31",
                chunksize: int = 200_000) -> dict[str, int]:
    """Stream-filter the raw CSV into per-year parquet files.

    Returns ingest stats: ``{"read": rows read, "kept": rows kept (date in range),
    "dropped_bad_date": rows dropped for unparseable dates}``.

    Accepted risk (one-time research ingest): rows whose fields are misaligned by
    malformed CSV almost always fail the date parse and are counted in
    ``dropped_bad_date``; ``on_bad_lines="skip"`` handles extra-field rows. A
    clean vs dirty run is distinguishable via the returned stats.
    """
    csv_path, out_dir = Path(csv_path), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)

    mapping = detect_columns(csv_path)
    writers: dict[int, pq.ParquetWriter] = {}
    read = kept = dropped_bad_date = 0
    try:
        for i, chunk in enumerate(
                pd.read_csv(csv_path, usecols=list(mapping), chunksize=chunksize,
                            dtype=str, on_bad_lines="skip"), start=1):
            read += len(chunk)
            chunk = chunk.rename(columns=mapping)
            for c in COLS:
                if c not in chunk.columns:
                    chunk[c] = ""
            dates = pd.to_datetime(chunk["date"], errors="coerce", utc=True)
            chunk["date"] = dates.dt.tz_localize(None)
            n_parseable = len(chunk)
            chunk = chunk.dropna(subset=["date"])
            dropped_bad_date += n_parseable - len(chunk)
            chunk = chunk[(chunk["date"] >= start_ts) & (chunk["date"] <= end_ts)]
            kept += len(chunk)
            if i % 25 == 0:
                print(f"  chunk {i}: {read:,} read, {kept:,} kept", flush=True)
            if chunk.empty:
                continue
            chunk = chunk[COLS]
            for c in COLS[1:]:
                chunk[c] = chunk[c].fillna("").astype(str)
            for year, group in chunk.groupby(chunk["date"].dt.year):
                if year not in writers:
                    writers[year] = pq.ParquetWriter(out_dir / f"year={year}.parquet", SCHEMA)
                writers[year].write_table(
                    pa.Table.from_pandas(group, schema=SCHEMA, preserve_index=False))
    finally:
        for w in writers.values():
            w.close()
    return {"read": read, "kept": kept, "dropped_bad_date": dropped_bad_date}


class NewsStore:
    """Read-side of the store: date-range (and optional ticker) queries."""

    def __init__(self, root):
        self.root = Path(root)

    def query(self, start, end, tickers: list[str] | None = None) -> pd.DataFrame:
        start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
        frames = []
        for year in range(start_ts.year, end_ts.year + 1):
            path = self.root / f"year={year}.parquet"
            if path.exists():
                frames.append(pd.read_parquet(path))
        if not frames:
            return pd.DataFrame(columns=COLS)
        df = pd.concat(frames, ignore_index=True)
        df = df[(df["date"] >= start_ts) & (df["date"] <= end_ts)]
        if tickers is not None:
            df = df[df["ticker"].isin(tickers)]
        return df.sort_values("date").reset_index(drop=True)[COLS]
