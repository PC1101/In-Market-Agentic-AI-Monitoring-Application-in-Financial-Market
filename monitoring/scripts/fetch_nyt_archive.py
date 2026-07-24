"""Fetch NYT Archive API months covering the six evaluation windows.

Why NYT: FNSPID has almost no usable English headlines before mid-2010 (pre-2009
rows are a Russian general-news scrape with empty tickers — see WEEK3_STATUS.md),
so the early event windows (Aug-2007 quant meltdown, GFC-2008, momentum crash
2009) need a supplementary point-in-time news source. The NYT Archive API returns
every NYT article's metadata for a given month — headline, abstract, exact
``pub_date`` timestamp, section — which is point-in-time safe by construction.

Requires a free API key (https://developer.nytimes.com, enable "Archive API"):
    export NYT_API_KEY=...          # never committed

Rate limits: 500 req/day, 5 req/min — we sleep 12s between calls. Each month's
raw JSON is cached gzipped at ``monitoring/news/data/nyt_raw/{yyyy}-{mm}.json.gz``
and never re-fetched (idempotent; delete a file to re-fetch it).

Usage (from ``monitoring/``):
    python scripts/fetch_nyt_archive.py                 # event windows + calm samples (25 months)
    python scripts/fetch_nyt_archive.py --full-calm     # + full calm spans (Week-5 ready)
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

MONITORING_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MONITORING_ROOT))

from windows import EVENT_WINDOWS, CALM_WINDOWS

RAW_DIR = MONITORING_ROOT / "news" / "data" / "nyt_raw"
API_URL = "https://api.nytimes.com/svc/archive/v1/{year}/{month}.json?api-key={key}"
SLEEP_S = 12.5  # 5 req/min limit -> 12s gap is safe

# For --full-calm=False, just fetch 3-month samples from each calm window start.
CALM_SAMPLE_MONTHS = 3


def months_between(start: str, end: str) -> list[tuple[int, int]]:
    """Inclusive (year, month) list covering [start, end] ISO dates."""
    sy, sm = int(start[:4]), int(start[5:7])
    ey, em = int(end[:4]), int(end[5:7])
    out: list[tuple[int, int]] = []
    y, m = sy, sm
    while (y, m) <= (ey, em):
        out.append((y, m))
        m += 1
        if m == 13:
            m = 1
            y += 1
    return out


def month_before(year: int, month: int) -> tuple[int, int]:
    if month == 1:
        return year - 1, 12
    return year, month - 1


def required_months(full_calm: bool = False) -> list[tuple[int, int]]:
    """Window spans + one lead-in month (the aggregator needs a ~30-day baseline)."""
    months: set[tuple[int, int]] = set()
    for w in EVENT_WINDOWS:
        span = months_between(w.start, w.end)
        months.update(span)
        months.add(month_before(*span[0]))
    if full_calm:
        for w in CALM_WINDOWS:
            months.update(months_between(w.start, w.end))
    else:
        for w in CALM_WINDOWS:
            span = months_between(w.start, w.end)
            months.update(span[:CALM_SAMPLE_MONTHS])
            months.add(month_before(*span[0]))
    return sorted(months)


def fetch_month(year: int, month: int, key: str) -> None:
    out = RAW_DIR / f"{year:04d}-{month:02d}.json.gz"
    if out.exists():
        print(f"  {out.name}: cached, skip")
        return
    url = API_URL.format(year=year, month=month, key=key)
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    resp = urllib.request.urlopen(req, timeout=120)
    payload = json.loads(resp.read())
    docs = payload["response"]["docs"]
    with gzip.open(out, "wb") as fh:
        fh.write(json.dumps(docs).encode())
    print(f"  {out.name}: {len(docs):,} docs")
    time.sleep(SLEEP_S)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--full-calm", action="store_true",
                    help="fetch the full calm-window spans (2004-2006, 2013-2014) "
                         "instead of 3-month samples")
    args = ap.parse_args()

    key = os.environ.get("NYT_API_KEY")
    if not key:
        sys.exit("NYT_API_KEY env var not set. Get a free key at "
                 "https://developer.nytimes.com (enable the Archive API).")

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    months = required_months(full_calm=args.full_calm)
    todo = [(y, m) for y, m in months
            if not (RAW_DIR / f"{y:04d}-{m:02d}.json.gz").exists()]
    print(f"{len(months)} months required, {len(todo)} to fetch "
          f"(~{len(todo) * SLEEP_S / 60:.0f} min at rate limit)\n")

    missing: list[str] = []
    for y, m in todo:
        try:
            fetch_month(y, m, key)
        except Exception as e:
            print(f"  {y:04d}-{m:02d}: ERROR {e}")
            missing.append(f"{y:04d}-{m:02d}")

    cached = sorted(p.name for p in RAW_DIR.glob("*.json.gz"))
    print(f"\nDone. {len(cached)} months cached in {RAW_DIR}")
    if missing:
        print(f"FAILED: {missing}")


if __name__ == "__main__":
    main()
