"""End-to-end agentic monitoring on one evaluation window (VRI Week 3).

Daily loop over the window: build the day's causal context (telemetry, classical
alarms, as-of macro, filtered news), triage how much model to spend (skip /
cheap / thinking / classical-escalation), and on escalated days run the News
Context Agent then the Performance Supervisor (v2 prompt). Every step is
lookahead-guarded; every model call is logged to JSONL.

Usage (from monitoring/):
    python run_agentic.py --window quant_meltdown_2007 --strategy AL_PCA --model stub
    python run_agentic.py --window quant_meltdown_2007 --strategy AL_PCA \
        --model ollama:qwen2.5:3b
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd

from pnl_loader import load_pnl, returns_series
from windows import get_window
from detectors import PageHinkley, BOCPD, HMMDetector, DistributionalThreshold, aggregate_alarms
from run_classical import STRATEGY_CURVES
from agentic import as_of_context, RunLogger, run_supervisor
from agentic.guardrails import assert_no_lookahead, LookaheadError
from agentic.model import default_model
from agentic.news_agent import run_news_agent
from agentic.prompts import build_supervisor_prompt_v2, SUPERVISOR_PROMPT_VERSION_V2
from news.store import NewsStore
from news.filter import filter_news
from news.aggregate import daily_signals
from news.triage import decide

ROOT = Path(__file__).resolve().parent.parent
RESULTS = Path(__file__).resolve().parent / "results"
STORE_DIR = ROOT / "data" / "fnspid" / "store"
MACRO_DIR = ROOT / "data" / "macro"

RECENT_DAYS = 5      # detector/aggregate lookback for triage (matches the aggregation rule)
NEWS_DAYS = 5        # news shown to the agent: published in the last N days
BASELINE_PAD = 120   # calendar days of news history before the window for the z baseline


def _recent(alarms: list, day: pd.Timestamp, days: int = RECENT_DAYS) -> bool:
    lo = day - pd.Timedelta(days=days)
    return any(lo < pd.Timestamp(a) <= day for a in alarms)


def run_window(series: pd.Series, window, store: NewsStore, model,
               logger: RunLogger | None = None, macro_dir=MACRO_DIR,
               max_articles: int = 40) -> list[dict]:
    """Run the daily agentic loop over one window. Returns one record per trading day."""
    # Classical detectors run continuously over the full curve (warm-up history included).
    results = [d.detect(series) for d in
               (PageHinkley(), BOCPD(), HMMDetector(), DistributionalThreshold())]
    alarms_by_det = {r.detector: r.alarms for r in results}
    agg_alarms = aggregate_alarms(results, window_days=5, min_detectors=2)

    # News signals need trailing history for the causal z baseline.
    news_hist = store.query(window.start_ts - pd.Timedelta(days=BASELINE_PAD), window.end_ts)
    signals = daily_signals(news_hist, filter_news(news_hist))

    try:
        from macro.context import macro_context
    except ImportError:  # macro package optional at run time
        macro_context = None

    days = series.index[(series.index >= window.start_ts) & (series.index <= window.end_ts)]
    records: list[dict] = []
    for day in days:
        z = float(signals["intensity_z"].get(day.normalize(), float("nan"))) \
            if not signals.empty else float("nan")
        n_recent = sum(_recent(a, day) for a in alarms_by_det.values())
        dec = decide(z, n_recent, _recent(agg_alarms, day))

        rec = {"as_of": str(day.date()), "window": window.name,
               "triage_mode": dec.mode, "triage_reason": dec.reason,
               "intensity_z": None if pd.isna(z) else round(z, 2)}
        if logger is not None:
            logger.log({"agent": "triage", "prompt_version": "triage-v1", **rec})

        if dec.mode == "skip":
            records.append(rec)
            continue

        # --- News Context Agent ---
        day_news = store.query(day - pd.Timedelta(days=NEWS_DAYS), day)
        risky = filter_news(day_news).sort_values("n_risk_terms", ascending=False)
        articles = [{"date": str(r["date"].date()), "ticker": r["ticker"],
                     "title": r["title"], "summary": r["summary"]}
                    for _, r in risky.iterrows()]
        t0 = time.time()
        news = run_news_agent(str(day.date()), articles, model, logger=logger,
                              extra_label=f"{window.name}:{dec.mode}",
                              max_articles=max_articles)
        rec["news_latency_s"] = round(time.time() - t0, 1)

        # --- Performance Supervisor v2 ---
        ctx = as_of_context(series, day, alarms_by_det)
        if macro_context is not None and Path(macro_dir).exists():
            ctx["macro"] = macro_context(day, macro_dir)
        try:
            assert_no_lookahead({"news": news.to_dict()}, day)
            ctx["news"] = news.to_dict()
        except LookaheadError:
            # model narrative hallucinated a future date — drop the news block, fail closed
            rec["news_dropped_lookahead"] = True
        ctx["triage_mode"] = dec.mode

        t0 = time.time()
        assessment = run_supervisor(ctx, model, logger=logger,
                                    extra_label=f"{window.name}:{dec.mode}",
                                    prompt_builder=build_supervisor_prompt_v2,
                                    prompt_version=SUPERVISOR_PROMPT_VERSION_V2)
        rec["supervisor_latency_s"] = round(time.time() - t0, 1)
        rec["assessment"] = assessment.to_dict()
        records.append(rec)
    return records


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", default="quant_meltdown_2007")
    ap.add_argument("--strategy", default="AL_PCA", choices=list(STRATEGY_CURVES))
    ap.add_argument("--model", default=None, help="stub | ollama | ollama:<name> (default: MONITOR_MODEL env or stub)")
    args = ap.parse_args()

    window = get_window(args.window)
    curve = next((p for p in STRATEGY_CURVES[args.strategy]
                  if Path(p).exists()
                  and (s := returns_series(load_pnl(p))).index.min() <= window.start_ts
                  and s.index.max() >= window.end_ts - pd.Timedelta(days=7)), None)
    if curve is None:
        raise SystemExit(f"no {args.strategy} curve covers {window.name}")
    series = returns_series(load_pnl(curve))

    model = default_model(args.model)
    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / f"agentic_{args.window}_{args.strategy}.jsonl"
    if out.exists():
        out.unlink()
    logger = RunLogger(out)

    records = run_window(series, window, NewsStore(STORE_DIR), model, logger=logger)

    modes = pd.Series([r["triage_mode"] for r in records]).value_counts().to_dict()
    n_assessed = sum("assessment" in r for r in records)
    lat = [r["supervisor_latency_s"] for r in records if "supervisor_latency_s" in r]
    print(f"{window.name} × {args.strategy} × {model.name}")
    print(f"  trading days: {len(records)}  triage: {modes}")
    print(f"  supervisor assessments (schema-valid): {n_assessed}")
    if lat:
        print(f"  mean supervisor latency: {sum(lat)/len(lat):.1f}s")
    print(f"  log: {out}")


if __name__ == "__main__":
    main()
