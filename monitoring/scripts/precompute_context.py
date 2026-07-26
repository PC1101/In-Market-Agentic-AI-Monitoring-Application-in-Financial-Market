"""Pre-compute news context (intensity_z + filtered articles) for a window.

Separates the heavy parquet I/O from the Ollama inference process so the
agentic pipeline starts instantly and the Ollama runner never goes stale.

Usage:
    cd monitoring && python scripts/precompute_context.py --window covid_2020
    # then: python run_agentic.py --window covid_2020 --strategy JT_MOM \
    #         --model ollama:qwen2.5:1.5b \
    #         --finbert-cache results/finbert_cache/covid_2020.json \
    #         --context-cache results/context_cache/covid_2020.json
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

_MONITORING = Path(__file__).resolve().parent.parent
if str(_MONITORING) not in sys.path:
    sys.path.insert(0, str(_MONITORING))

import pandas as pd

from news.store import NewsStore
from news.filter import filter_news
from news.aggregate import daily_signals
from pnl_loader import load_pnl, returns_series
from run_classical import STRATEGY_CURVES
from windows import get_window

ROOT = Path(__file__).resolve().parent.parent.parent
STORE_DIR = ROOT / "data" / "fnspid" / "store"
CACHE_DIR = Path(__file__).resolve().parent.parent / "results" / "context_cache"

BASELINE_PAD = 120   # same as run_agentic.py
NEWS_DAYS = 7        # same as run_agentic.py (COVERAGE_DAYS)
MAX_ARTICLES = 40    # ceiling; run_agentic --max-articles slices from this


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", required=True)
    ap.add_argument("--strategy", default="JT_MOM", choices=list(STRATEGY_CURVES))
    args = ap.parse_args()

    window = get_window(args.window)

    curve = next((p for p in STRATEGY_CURVES[args.strategy]
                  if Path(p).exists()), None)
    if curve is None:
        raise SystemExit(f"no {args.strategy} curve found")
    series = returns_series(load_pnl(curve))
    days = series.index[(series.index >= window.start_ts) & (series.index <= window.end_ts)]

    store = NewsStore(STORE_DIR)

    print(f"Pre-computing context for {args.window} ({len(days)} trading days)")
    t0 = time.time()

    # Baseline signals (one big query, same as run_agentic.py)
    print("  Loading news history for baseline signals...", flush=True)
    news_hist = store.query(window.start_ts - pd.Timedelta(days=BASELINE_PAD), window.end_ts)
    signals = daily_signals(news_hist, filter_news(news_hist))
    print(f"  Signals done ({len(news_hist)} articles, {len(signals)} days)")

    cache: dict[str, dict] = {}
    for i, day in enumerate(days):
        # intensity_z from baseline signals
        z = float(signals["intensity_z"].get(day.normalize(), float("nan"))) \
            if not signals.empty else float("nan")

        # Per-day filtered articles (same query + sort as run_agentic.py)
        day_news = store.query(day - pd.Timedelta(days=NEWS_DAYS), day)
        risky = filter_news(day_news).sort_values("n_risk_terms", ascending=False)
        articles = [{"date": str(r["date"].date()), "ticker": r["ticker"],
                     "title": r["title"], "summary": r["summary"]}
                    for _, r in risky.head(MAX_ARTICLES).iterrows()]

        cache[str(day.date())] = {
            "intensity_z": None if math.isnan(z) else round(z, 2),
            "articles": articles,
        }

        if (i + 1) % 10 == 0 or i == 0:
            print(f"  [{i+1}/{len(days)}] {day.date()} z={z:.2f} "
                  f"({len(articles)} articles)", flush=True)

    elapsed = time.time() - t0
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    out = CACHE_DIR / f"{args.window}.json"
    with open(out, "w") as f:
        json.dump(cache, f)
    print(f"Done in {elapsed:.1f}s -> {out} ({len(cache)} days)")


if __name__ == "__main__":
    main()
