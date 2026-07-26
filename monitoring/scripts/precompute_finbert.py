"""Pre-compute FinBERT stress scores for a window and save to JSON cache.

Runs FinBERT in a separate process so the main agentic pipeline never loads
torch/transformers, keeping it lightweight enough for reliable Ollama calls.

Usage:
    cd monitoring && python scripts/precompute_finbert.py --window covid_2020
    # then: python run_agentic.py --window covid_2020 --strategy JT_MOM \
    #         --model ollama:qwen2.5:1.5b --finbert-cache results/finbert_cache/covid_2020.json
"""
import argparse
import json
import sys
import time
from pathlib import Path

# Ensure monitoring/ is on sys.path when run as scripts/precompute_finbert.py
_MONITORING = Path(__file__).resolve().parent.parent
if str(_MONITORING) not in sys.path:
    sys.path.insert(0, str(_MONITORING))

import pandas as pd

# Imports are relative to monitoring/
from news.store import NewsStore
from news.filter import filter_news
from news.finbert import daily_max_stress
from pnl_loader import load_pnl, returns_series
from run_classical import STRATEGY_CURVES
from windows import get_window

ROOT = Path(__file__).resolve().parent.parent.parent
STORE_DIR = ROOT / "data" / "fnspid" / "store"
CACHE_DIR = Path(__file__).resolve().parent.parent / "results" / "finbert_cache"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", required=True)
    ap.add_argument("--strategy", default="JT_MOM", choices=list(STRATEGY_CURVES))
    ap.add_argument("--device", default=None,
                    help="torch device for FinBERT (default: auto — GPU if available)")
    args = ap.parse_args()

    window = get_window(args.window)

    # Get the trading days from any available curve for this strategy
    curve = next((p for p in STRATEGY_CURVES[args.strategy]
                  if Path(p).exists()), None)
    if curve is None:
        raise SystemExit(f"no {args.strategy} curve found")
    series = returns_series(load_pnl(curve))
    days = series.index[(series.index >= window.start_ts) & (series.index <= window.end_ts)]

    # Override FinBERT device if requested (default: let it auto-detect → GPU)
    if args.device:
        import news.finbert as fb
        fb._PIPELINE = None  # force reload
        fb._PIPELINE_FAILED = False
        # Monkey-patch the device; _load_pipeline will use it
        fb._DEVICE_OVERRIDE = args.device

    store = NewsStore(STORE_DIR)
    cache: dict[str, float] = {}

    print(f"Pre-computing FinBERT stress for {args.window} ({len(days)} trading days)")
    t0 = time.time()

    for i, day in enumerate(days):
        day_risk = filter_news(store.query(day - pd.Timedelta(days=1), day))
        if not day_risk.empty:
            titles = day_risk["title"].fillna("")
            summaries = day_risk["summary"].fillna("") if "summary" in day_risk.columns else titles
            headlines = (titles + " " + summaries).tolist()
        else:
            headlines = []
        stress = daily_max_stress(headlines)
        cache[str(day.date())] = round(stress, 6)
        if (i + 1) % 10 == 0 or i == 0:
            print(f"  [{i+1}/{len(days)}] {day.date()} stress={stress:.4f} "
                  f"({len(headlines)} headlines)", flush=True)

    elapsed = time.time() - t0
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    out = CACHE_DIR / f"{args.window}.json"
    with open(out, "w") as f:
        json.dump(cache, f, indent=2)
    print(f"Done in {elapsed:.1f}s -> {out} ({len(cache)} days)")


if __name__ == "__main__":
    main()
