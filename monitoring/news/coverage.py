"""Per-window news coverage report — the Week-3 data gate.

Measures the news the pipeline **actually consumes**: the NYT Archive +
Cyrillic-cleaned FNSPID caches built by ``scripts/build_news_cache.py`` (the
same ``NewsConfig`` sources the agents see), not the raw FNSPID store. The raw
store passes pre-2009 windows on *volume* while the content is the Russian
Lenta.ru scrape — 91k articles in calm_2004_2006 yielded exactly 1 English
risk hit — so a store-level gate is blind to unusable content (see
WEEK3_STATUS "Data findings").

Counts per window: total records, keyword-filter candidates/matches
(``filters.split_records``), and a per-source breakdown. An event window with
fewer risk-relevant articles than ``MIN_RISK_ARTICLES`` is flagged — a
supervisor-decision point, not something to paper over.

Usage (from monitoring/): python -m news.coverage
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import pandas as pd

from windows import ALL_WINDOWS

from .filters import split_records
from .records import load_parquet_news

#: an event window with fewer risk-relevant articles than this is flagged.
MIN_RISK_ARTICLES = 25

RESULTS = Path(__file__).resolve().parent.parent / "results"
DEFAULT_DATA_DIR = Path(__file__).resolve().parent / "data"
DEFAULT_SOURCES = ("nyt_archive.parquet", "fnspid_windows.parquet")


def coverage_report(source_paths, windows=ALL_WINDOWS) -> pd.DataFrame:
    """Coverage of the pipeline's cache parquets, window by window."""
    rows = []
    for w in windows:
        records = load_parquet_news(source_paths, start=w.start, end=w.end)
        candidates, matched = split_records(records)
        by_source = Counter(r["source"] for r in records)
        n_days = max((pd.Timestamp(w.end) - pd.Timestamp(w.start)).days, 1)
        rows.append({
            "window": w.name,
            "kind": w.kind,
            "start": w.start,
            "end": w.end,
            "n_articles": len(records),
            "n_candidates": len(candidates),
            "n_risk_articles": len(matched),
            "articles_per_day": round(len(records) / n_days, 2),
            "by_source": dict(sorted(by_source.items())),
            "adequate": bool(len(matched) >= MIN_RISK_ARTICLES) if w.kind == "event" else True,
        })
    return pd.DataFrame(rows)


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    paths = [DEFAULT_DATA_DIR / s for s in DEFAULT_SOURCES]
    missing = [p for p in paths if not p.exists()]
    if len(missing) == len(paths):
        raise SystemExit(f"no news caches in {DEFAULT_DATA_DIR} — run "
                         "scripts/fetch_nyt_archive.py + scripts/build_news_cache.py")
    rep = coverage_report(paths)
    rep.to_csv(RESULTS / "news_coverage.csv", index=False)
    print(rep.to_string(index=False))
    thin = rep[(rep["kind"] == "event") & (~rep["adequate"])]
    if not thin.empty:
        print("\n*** GATE: inadequate news coverage on event window(s): "
              f"{', '.join(thin['window'])} ***")
        print("Escalate to supervisor before Week-5 evaluation; document in WEEK3_STATUS.md.")


if __name__ == "__main__":
    main()
