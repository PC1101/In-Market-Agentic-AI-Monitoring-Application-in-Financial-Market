"""Daily agentic monitoring loop over one evaluation window (Week-3/4 bridge).

For every trading day in the window: build the causal news block (T-1 cutoff),
triage how much model to spend (SKIP / CHEAP_MODEL / THINKING_MODEL /
CLASSICAL_ESCALATION), run the News Context Agent on escalated days, then run
the Performance Supervisor (v3 prompt, news-aware) on every non-SKIP day.
Every model call is lookahead-guarded and logged to JSONL.

This is the daily-cadence counterpart of ``run_classical.py --news`` (which
invokes the agents at event onsets only). It reuses that module's causal
plumbing — ``_news_for_date``, ``_load_window_records``, ``_frame_slice`` —
so both entry points share one triage/guardrail implementation.

Origin note: the daily-loop structure is merged from the week3-news-agents
branch (teammate's ``run_agentic.py``), re-based onto this branch's news
pipeline (NYT+FNSPID caches, FinBERT signal, TriageMode, supervisor-v3).

Usage (from monitoring/):
    python run_agentic.py --window quant_meltdown_2007 --strategy AL_PCA \
        --news-scorer fake                       # offline, stub model
    python run_agentic.py --window quant_meltdown_2007 --strategy AL_PCA \
        --model ollama:llama3.2:3b --news-device cuda
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
from agentic import as_of_context, RunLogger, run_supervisor, make_model
from news import NewsConfig, TriageMode, FakeScorer, FinBERTScorer
from run_classical import (STRATEGY_CURVES, _MemoScorer, _news_for_date,
                           _load_window_records, _frame_slice)

RESULTS = Path(__file__).resolve().parent / "results"


def run_window(series: pd.Series, window, news_cfg: NewsConfig,
               supervisor_model, news_model, logger: RunLogger | None = None,
               records: list[dict] | None = None,
               label: str | None = None) -> list[dict]:
    """Run the daily agentic loop over one window. Returns one record per trading day."""
    label = label or window.name

    # Classical detectors run continuously over the full curve (real warm-up).
    results = [d.detect(series) for d in
               (PageHinkley(), BOCPD(), HMMDetector(), DistributionalThreshold())]
    alarms_by_det = {r.detector: r.alarms for r in results}
    alarms_by_det["aggregate"] = aggregate_alarms(results, window_days=5, min_detectors=2)

    if records is None:
        records = _load_window_records(window, news_cfg)
    ts_keys = [r["timestamp"] for r in records]

    days = series.index[(series.index >= window.start_ts) & (series.index <= window.end_ts)]
    out: list[dict] = []
    for day in days:
        frame = _frame_slice(records, ts_keys, day, news_cfg.lookback_days)
        news, mode = _news_for_date(day, news_cfg, frame, alarms_by_det, series,
                                    news_model, logger, label)
        rec = {"as_of": str(day.date()), "window": window.name,
               "triage_mode": mode}
        if isinstance(news, dict) and "signal" in news:
            rec["signal"] = news["signal"]

        if mode == TriageMode.SKIP.value:
            # No LLM spend on quiet days — this is the triage deliverable.
            out.append(rec)
            continue

        ctx = as_of_context(series, day, alarms_by_det, news=news)
        t0 = time.time()
        assessment = run_supervisor(ctx, supervisor_model, logger=logger,
                                    extra_label=f"{label}:{mode}")
        rec["supervisor_latency_s"] = round(time.time() - t0, 1)
        rec["assessment"] = assessment.to_dict()
        out.append(rec)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Daily agentic monitoring on one window.")
    ap.add_argument("--window", default="quant_meltdown_2007")
    ap.add_argument("--strategy", default="AL_PCA", choices=list(STRATEGY_CURVES))
    ap.add_argument("--model", default="stub",
                    help="'stub' (default, offline) or 'ollama:<name>' "
                         "e.g. 'ollama:llama3.2:3b'")
    ap.add_argument("--news-data-dir", default=None,
                    help="directory holding the news parquet caches "
                         "(default: monitoring/news/data)")
    ap.add_argument("--news-scorer", default="finbert", choices=["finbert", "fake"])
    ap.add_argument("--news-device", default="cpu", choices=["cpu", "cuda"])
    args = ap.parse_args()

    window = get_window(args.window)
    curve = next((p for p in STRATEGY_CURVES[args.strategy]
                  if Path(p).exists()
                  and (s := returns_series(load_pnl(p))).index.min() <= window.start_ts
                  and s.index.max() >= window.end_ts - pd.Timedelta(days=7)), None)
    if curve is None:
        raise SystemExit(f"no {args.strategy} curve covers {window.name}")
    series = returns_series(load_pnl(curve))

    scorer = _MemoScorer(FinBERTScorer(device=args.news_device)
                         if args.news_scorer == "finbert" else FakeScorer())
    kwargs = {"scorer": scorer}
    if args.news_data_dir:
        kwargs["data_dir"] = Path(args.news_data_dir)
    news_cfg = NewsConfig(**kwargs)
    missing = [p for p in news_cfg.source_paths if not p.exists()]
    if len(missing) == len(news_cfg.source_paths):
        ap.error(f"no news caches found in {news_cfg.data_dir} — run "
                 f"scripts/fetch_nyt_archive.py + scripts/build_news_cache.py")

    supervisor_model = make_model(args.model)
    news_model = make_model(args.model, agent="news_context")

    RESULTS.mkdir(parents=True, exist_ok=True)
    log_path = RESULTS / f"agentic_{args.window}_{args.strategy}.jsonl"
    if log_path.exists():
        log_path.unlink()
    logger = RunLogger(log_path)
    label = f"{args.strategy}:{window.name}"

    days = run_window(series, window, news_cfg, supervisor_model, news_model,
                      logger=logger, label=label)

    days_path = RESULTS / f"agentic_{args.window}_{args.strategy}_days.jsonl"
    with open(days_path, "w", encoding="utf-8") as f:
        for rec in days:
            f.write(json.dumps(rec, default=str) + "\n")

    modes = pd.Series([r["triage_mode"] for r in days]).value_counts().to_dict()
    n_assessed = sum("assessment" in r for r in days)
    lat = [r["supervisor_latency_s"] for r in days if "supervisor_latency_s" in r]
    print(f"{window.name} x {args.strategy} x {supervisor_model.name}")
    print(f"  trading days: {len(days)}  triage: {modes}")
    print(f"  supervisor assessments (schema-valid): {n_assessed}")
    if lat:
        print(f"  mean supervisor latency: {sum(lat)/len(lat):.1f}s")
    print(f"  agent log: {log_path}")
    print(f"  daily records: {days_path}")


if __name__ == "__main__":
    main()
