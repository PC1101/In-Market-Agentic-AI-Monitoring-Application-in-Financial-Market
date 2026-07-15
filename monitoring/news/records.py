"""Unified news record schema and loaders.

A news record is a plain JSON-serialisable dict:

    {
        "timestamp": "2007-08-03T14:05:00",   # ISO publication timestamp (str)
        "headline":  "...",                   # required, non-empty
        "abstract":  "..." | None,
        "source":    "nyt" | "fnspid" | ...,
        "tickers":   ["ABC", ...],            # possibly empty
        "section":   "Business" | None,
    }

The ``timestamp`` key matches what ``agentic.guardrails.filter_news_by_timestamp``
expects, so the existing information-parity helpers work on these records
unchanged. FNSPID timestamps are day-granularity (midnight); NYT carries exact
publication times.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd

# Documentation alias — records are plain dicts.
NewsRecord = dict


def _none_if_na(x):
    """Normalise missing values (None / NaN / pd.NA) to None."""
    if x is None or x is pd.NA:
        return None
    if isinstance(x, float) and math.isnan(x):
        return None
    return x


def _make_record(timestamp, headline, abstract=None, source="unknown",
                 tickers=None, section=None) -> dict:
    abstract = _none_if_na(abstract)
    section = _none_if_na(section)
    return {
        "timestamp": str(timestamp),
        "headline": str(headline).strip(),
        "abstract": (str(abstract).strip() or None) if abstract is not None else None,
        "source": source,
        "tickers": list(tickers) if tickers else [],
        "section": str(section) if section else None,
    }


def load_parquet_news(paths, start=None, end=None) -> list[dict]:
    """Load news records from one or more cache parquets built by
    ``scripts/build_news_cache.py`` (columns: pub_ts, headline, and optionally
    abstract, section_name, news_desk, ticker, source).

    Args:
        paths: iterable of parquet paths; missing files are skipped silently so
               a partial cache (e.g. NYT only) still works.
        start/end: optional inclusive date bounds applied on pub_ts.
    """
    records: list[dict] = []
    start = pd.Timestamp(start) if start is not None else None
    end = pd.Timestamp(end) if end is not None else None

    for path in paths:
        path = Path(path)
        if not path.exists():
            continue
        df = pd.read_parquet(path)
        ts = pd.to_datetime(df["pub_ts"], errors="coerce")
        keep = ts.notna()
        if start is not None:
            keep &= ts >= start
        if end is not None:
            keep &= ts <= end
        df = df.loc[keep]
        ts = ts[keep]

        get = lambda col: df[col] if col in df.columns else pd.Series(None, index=df.index)  # noqa: E731
        abstracts = get("abstract")
        sections = get("section_name")
        tickers = get("ticker")
        sources = df["source"] if "source" in df.columns else pd.Series(path.stem, index=df.index)

        for i in df.index:
            headline = str(df.at[i, "headline"]).strip()
            if not headline:
                continue
            tick = _none_if_na(tickers.at[i])
            tick = [str(tick).strip()] if tick is not None and str(tick).strip() else []
            records.append(_make_record(
                timestamp=ts.at[i].isoformat(),
                headline=headline,
                abstract=abstracts.at[i],
                source=str(sources.at[i]),
                tickers=tick,
                section=sections.at[i],
            ))
    records.sort(key=lambda r: r["timestamp"])
    return records


def load_jsonl_news(path) -> list[dict]:
    """Load news records from a JSONL file (one record dict per line).

    Used by the offline test fixtures; also handy for hand-curated news sets.
    Missing optional keys are defaulted; records without a headline are dropped.
    """
    records: list[dict] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        raw = json.loads(line)
        if not str(raw.get("headline", "")).strip():
            continue
        records.append(_make_record(
            timestamp=raw.get("timestamp", ""),
            headline=raw["headline"],
            abstract=raw.get("abstract"),
            source=raw.get("source", "fixture"),
            tickers=raw.get("tickers"),
            section=raw.get("section"),
        ))
    records.sort(key=lambda r: r["timestamp"])
    return records
