"""GDELT DOC 2.0 news ingestion — writes the same per-year parquet store as FNSPID.

GDELT (https://api.gdeltproject.org/api/v2/doc/doc) is free, no key, global,
point-in-time (every article carries a ``seendate``). This adapter maps GDELT
articles onto the exact FNSPID ``NewsStore`` schema
(``date, ticker, title, summary, publisher, url``) and writes ``year=YYYY.parquet``
files, so the existing ``NewsStore.query`` + filter/triage/FinBERT stack consumes
GDELT news unchanged. Per design §3 / §11.3 this is the FNSPID replacement for the
new markets (energy now, ASX later).

Coverage note: the DOC 2.0 API covers ~2017-present. Pre-2017 windows (e.g. the
2014-16 oil crash) need the GDELT GKG raw files instead — a documented future
extension. The 2020 / 2022 energy windows are within DOC 2.0 range.

Rate limit: GDELT throttles to roughly one request per 5s; ``build_gdelt_store``
sleeps ``throttle_s`` between calls and retries 429s with backoff.
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

from news.store import COLS

DOC_API = "https://api.gdeltproject.org/api/v2/doc/doc"
_UA = "in-market-monitor/research (GDELT DOC 2.0 ingest)"


def parse_seendate(s: str) -> pd.Timestamp:
    """GDELT ``seendate`` ('YYYYMMDDTHHMMSSZ') -> Timestamp, or NaT if unparseable."""
    return pd.to_datetime(s, format="%Y%m%dT%H%M%SZ", errors="coerce", utc=False)


def build_query(terms: list[str]) -> str:
    """GDELT query: OR the terms, quoting any multi-word term."""
    parts = [f'"{t}"' if " " in t else t for t in terms]
    return "(" + " OR ".join(parts) + ")"


def article_to_row(art: dict, ticker: str) -> dict:
    """Map one GDELT article dict to a NewsStore row (summary is empty for DOC API)."""
    return {
        "date": parse_seendate(art.get("seendate", "")),
        "ticker": ticker,
        "title": art.get("title", "") or "",
        "summary": "",  # DOC 2.0 artlist has no body/summary; title feeds FinBERT
        "publisher": art.get("domain", "") or "",
        "url": art.get("url", "") or "",
    }


def articles_to_frame(articles: list[dict], ticker: str,
                      english_only: bool = False) -> pd.DataFrame:
    """List of GDELT articles -> DataFrame in exact NewsStore column order.

    Drops rows whose ``seendate`` does not parse (same discipline as FNSPID's
    ``dropped_bad_date``). Optionally keeps only English-language articles.
    """
    if english_only:
        articles = [a for a in articles if (a.get("language") or "") == "English"]
    rows = [article_to_row(a, ticker) for a in articles]
    df = pd.DataFrame(rows, columns=list(COLS))
    df = df[df["date"].notna()].reset_index(drop=True)
    return df


def fetch(query: str, start, end, maxrecords: int = 250,
          retries: int = 6, backoff_s: float = 6.0) -> list[dict]:
    """Fetch articles for ``query`` in [start, end] from the DOC 2.0 API.

    Dates are formatted as GDELT's ``YYYYMMDDHHMMSS``. Retries 429/5xx with
    exponential backoff, honouring a ``Retry-After`` header when present. Returns
    the raw ``articles`` list (possibly empty).
    """
    params = {
        "query": query,
        "mode": "artlist",
        "format": "json",
        "maxrecords": str(maxrecords),
        "startdatetime": pd.Timestamp(start).strftime("%Y%m%d%H%M%S"),
        "enddatetime": pd.Timestamp(end).strftime("%Y%m%d%H%M%S"),
        "sort": "datedesc",
    }
    url = f"{DOC_API}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={"User-Agent": _UA})
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                body = resp.read().decode("utf-8", errors="replace")
            if not body.strip():
                return []
            return json.loads(body).get("articles", [])
        except (urllib.error.HTTPError, json.JSONDecodeError) as e:
            if attempt == retries - 1:
                raise
            # Prefer the server's Retry-After; else exponential backoff (6,12,24,…).
            wait = backoff_s * (2 ** attempt)
            if isinstance(e, urllib.error.HTTPError):
                ra = e.headers.get("Retry-After") if e.headers else None
                if ra and ra.isdigit():
                    wait = max(wait, float(ra))
            time.sleep(wait)
    return []


def build_gdelt_store(symbol_queries: dict[str, list[str]], start, end,
                      out_dir: str | Path, throttle_s: float = 5.0,
                      english_only: bool = True, maxrecords: int = 250) -> dict[str, int]:
    """Fetch news for each symbol and write per-year parquet files (NewsStore layout).

    Args:
        symbol_queries: {ticker: [search terms]} — one GDELT query per symbol.
        start, end: overall date range.
        out_dir: store root; files written as ``year=YYYY.parquet``.
        throttle_s: sleep between API calls (GDELT rate limit).
        english_only: keep only English articles.

    Returns per-year row counts written.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    frames: list[pd.DataFrame] = []
    for ticker, terms in symbol_queries.items():
        arts = fetch(build_query(terms), start, end, maxrecords=maxrecords)
        frames.append(articles_to_frame(arts, ticker, english_only=english_only))
        time.sleep(throttle_s)
    if not frames:
        return {}
    allrows = pd.concat(frames, ignore_index=True)
    allrows = allrows[(allrows["date"] >= pd.Timestamp(start)) &
                      (allrows["date"] <= pd.Timestamp(end))]
    counts: dict[str, int] = {}
    for year, grp in allrows.groupby(allrows["date"].dt.year):
        path = out_dir / f"year={int(year)}.parquet"
        # Append to any existing year file (idempotent-ish rebuilds overwrite).
        grp[list(COLS)].to_parquet(path, index=False)
        counts[str(int(year))] = len(grp)
    return counts
