"""Per-window FNSPID coverage report — the Week-3 data gate.

FNSPID's news density is known to be much higher post-2009. The Aug-2007 quant
meltdown is this project's headline gating event, so before building agents on
top of the news store we MEASURE how much news each evaluation window actually
has, write it to results/news_coverage.csv, and flag inadequate event windows.
An inadequate 2007 window is a supervisor-decision point, not something to
paper over (see WEEK3_STATUS).

Usage (from monitoring/): python -m news.coverage
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from news.filter import filter_news
from news.store import NewsStore
from windows import ALL_WINDOWS

#: an event window with fewer risk-relevant articles than this is flagged.
MIN_RISK_ARTICLES = 25

RESULTS = Path(__file__).resolve().parent.parent / "results"
STORE = Path(__file__).resolve().parent.parent.parent / "data" / "fnspid" / "store"


def coverage_report(store: NewsStore, windows=ALL_WINDOWS) -> pd.DataFrame:
    rows = []
    for w in windows:
        df = store.query(w.start, w.end)
        risky = filter_news(df)
        n_days = max((pd.Timestamp(w.end) - pd.Timestamp(w.start)).days, 1)
        rows.append({
            "window": w.name,
            "kind": w.kind,
            "start": w.start,
            "end": w.end,
            "n_articles": len(df),
            "n_risk_articles": len(risky),
            "articles_per_day": round(len(df) / n_days, 2),
            "adequate": bool(len(risky) >= MIN_RISK_ARTICLES) if w.kind == "event" else True,
        })
    return pd.DataFrame(rows)


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    rep = coverage_report(NewsStore(STORE))
    rep.to_csv(RESULTS / "news_coverage.csv", index=False)
    print(rep.to_string(index=False))
    thin = rep[(rep["kind"] == "event") & (~rep["adequate"])]
    if not thin.empty:
        print("\n*** GATE: inadequate news coverage on event window(s): "
              f"{', '.join(thin['window'])} ***")
        print("Escalate to supervisor before Week-5 evaluation; document in WEEK3_STATUS.md.")


if __name__ == "__main__":
    main()
