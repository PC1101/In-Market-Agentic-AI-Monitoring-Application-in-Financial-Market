"""End-to-end agentic monitoring on one evaluation window (VRI Week 3).

Daily loop over the window: build the day's causal context (telemetry, classical
alarms, as-of macro, filtered news), triage how much model to spend (skip /
cheap / thinking / classical-escalation), and on escalated days run the News
Context Agent then the Performance Supervisor (v3 prompt). Every step is
lookahead-guarded; every model call is logged to JSONL.

Usage (from monitoring/):
    python run_agentic.py --window quant_meltdown_2007 --strategy AL_PCA --model stub
    python run_agentic.py --window quant_meltdown_2007 --strategy AL_PCA \
        --model ollama:qwen2.5:3b
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd

from pnl_loader import load_pnl, returns_series
from windows import get_window
from detectors import PageHinkley, BOCPD, HMMDetector, DistributionalThreshold, aggregate_alarms
from run_classical import STRATEGY_CURVES, _hmm_training_returns
from agentic import as_of_context, RunLogger, run_supervisor
from agentic.guardrails import assert_no_lookahead, LookaheadError, mask_dates_in_context, mask_dates_in_articles
from agentic.model import default_model
from agentic.news_agent import run_news_agent
from agentic.prompts import build_supervisor_prompt_v3, SUPERVISOR_PROMPT_VERSION_V3
from news.store import NewsStore
from news.filter import filter_news
from news.aggregate import daily_signals
from news.triage import decide, RECENT_DAYS, COVERAGE_DAYS
from news.finbert import daily_max_stress

ROOT = Path(__file__).resolve().parent.parent
RESULTS = Path(__file__).resolve().parent / "results"
STORE_DIR = ROOT / "data" / "fnspid" / "store"
MACRO_DIR = ROOT / "data" / "macro"

# RECENT_DAYS = 3 (preregistration §6.2) — imported from triage
NEWS_DAYS = COVERAGE_DAYS   # 7-day news context window (preregistration §6.2)
BASELINE_PAD = 120          # calendar days of news history before the window for the z baseline


def _recent(alarms: list, day: pd.Timestamp, days: int = RECENT_DAYS) -> bool:
    lo = day - pd.Timedelta(days=days)
    return any(lo < pd.Timestamp(a) <= day for a in alarms)


def run_window(series: pd.Series, window, store: NewsStore, model,
               logger: RunLogger | None = None, macro_dir=MACRO_DIR,
               max_articles: int = 40, strategy: str | None = None,
               condition: str = "A") -> list[dict]:
    """Run the daily agentic loop over one window. Returns one record per trading day."""
    # Classical detectors run continuously over the full curve (warm-up history included).
    # HMM is fit on pre-event training data when available (out-of-sample for event windows).
    hmm_train = _hmm_training_returns(STRATEGY_CURVES.get(strategy, [])) if strategy else None
    results = [d.detect(series) for d in
               (PageHinkley(), BOCPD(), HMMDetector(train_returns=hmm_train),
                DistributionalThreshold())]
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
        n_recent = sum(_recent(a, day, RECENT_DAYS) for a in alarms_by_det.values())

        # FinBERT stress score: max negative-sentiment probability across today's risk articles
        day_risk = filter_news(store.query(day - pd.Timedelta(days=1), day))
        if not day_risk.empty:
            titles = day_risk["title"].fillna("")
            summaries = day_risk["summary"].fillna("") if "summary" in day_risk.columns else titles
            headlines_today = (titles + " " + summaries).tolist()
        else:
            headlines_today = []
        stress = daily_max_stress(headlines_today)

        dec = decide(z, n_recent, _recent(agg_alarms, day), stress_score=stress)

        rec = {"as_of": str(day.date()), "window": window.name,
               "triage_mode": dec.mode, "triage_reason": dec.reason,
               "intensity_z": None if pd.isna(z) else round(z, 2),
               "finbert_stress": round(stress, 4) if stress > 0.0 else None}
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
        # Condition B: mask dates in articles before sending to news agent
        news_articles = mask_dates_in_articles(articles) if condition == "B" else articles
        as_of_for_news = "XXXX-XX-XX" if condition == "B" else str(day.date())
        t0 = time.time()
        try:
            news = run_news_agent(as_of_for_news, news_articles, model, logger=logger,
                                  extra_label=f"{window.name}:{dec.mode}",
                                  max_articles=max_articles)
        except (json.JSONDecodeError, ValueError) as exc:
            # Small models sometimes produce unparseable output on heavy-news days;
            # skip this day's news block rather than aborting the entire window.
            rec["news_error"] = str(exc)[:200]
            records.append(rec)
            continue
        rec["news_latency_s"] = round(time.time() - t0, 1)

        # --- Performance Supervisor v3 ---
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

        # Condition B: mask all dates in context before sending to supervisor
        if condition == "B":
            ctx = mask_dates_in_context(ctx)

        t0 = time.time()
        try:
            assessment = run_supervisor(ctx, model, logger=logger,
                                        extra_label=f"{window.name}:{dec.mode}",
                                        prompt_builder=build_supervisor_prompt_v3,
                                        prompt_version=SUPERVISOR_PROMPT_VERSION_V3)
        except (json.JSONDecodeError, ValueError) as exc:
            rec["supervisor_error"] = str(exc)[:200]
            records.append(rec)
            continue
        rec["supervisor_latency_s"] = round(time.time() - t0, 1)
        rec["assessment"] = assessment.to_dict()
        records.append(rec)
    return records


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", default="quant_meltdown_2007")
    ap.add_argument("--strategy", default="AL_PCA", choices=list(STRATEGY_CURVES))
    ap.add_argument("--model", default=None, help="stub | ollama | ollama:<name> (default: MONITOR_MODEL env or stub)")
    ap.add_argument("--condition", default="A", choices=["A", "B"],
                    help="A=standard, B=date-masked (leakage Condition B)")
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
    cond_suffix = f"_cond{args.condition}" if args.condition != "A" else ""
    out = RESULTS / f"agentic_{args.window}_{args.strategy}{cond_suffix}.jsonl"
    if out.exists():
        out.unlink()
    logger = RunLogger(out)

    records = run_window(series, window, NewsStore(STORE_DIR), model, logger=logger,
                         strategy=args.strategy, condition=args.condition)

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
