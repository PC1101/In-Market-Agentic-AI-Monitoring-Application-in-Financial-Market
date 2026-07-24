"""Pretraining-leakage harness (preregistration §8).

Runs every test event window under three conditions:
  A — Standard frozen pipeline (delegates to run_agentic.run_window).
  B — Date-masked: mask_dates_in_context + mask_dates_in_articles before LLM.
  C — Synthetic: M=10 block-bootstrapped windows with injected crash templates.

Computes the preregistration leakage bound:
  evidence_skill_lower_bound  = performance_C
  memorisation_upper_bound    = performance_A - min(performance_B, performance_C)

If A ≫ C the write-up states that real-event performance is materially attributable
to pretraining memorisation. If A ≈ C the monitor shows evidence-driven skill.

Usage:
    python run_leakage.py --strategy AL_PCA --model stub
    python run_leakage.py --strategy JT_MOM --model ollama:qwen2.5:1.5b
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from pnl_loader import load_pnl, returns_series
from windows import (
    get_window, TEST_STRATEGY_WINDOWS, TEST_EVENT_WINDOWS,
    CALM_WINDOWS, EVENT_WINDOWS,
)
from detectors import PageHinkley, BOCPD, HMMDetector, DistributionalThreshold, aggregate_alarms
from run_classical import STRATEGY_CURVES, _hmm_training_returns
from metrics import evaluate_window, WindowMetrics
from agentic import as_of_context, RunLogger, run_supervisor
from agentic.guardrails import assert_no_lookahead, LookaheadError, mask_dates_in_context
from agentic.model import default_model
from agentic.news_agent import run_news_agent
from agentic.prompts import build_supervisor_prompt_v3, SUPERVISOR_PROMPT_VERSION_V3
from leakage.synthetic import generate_synthetic_windows, SyntheticWindow
from news.triage import decide, RECENT_DAYS, COVERAGE_DAYS
from news.finbert import daily_max_stress
from agentic.guardrails import mask_dates_in_articles

ROOT = Path(__file__).resolve().parent.parent
RESULTS = Path(__file__).resolve().parent / "results"
STORE_DIR = ROOT / "data" / "fnspid" / "store"

NEWS_DAYS = COVERAGE_DAYS  # 7 days


def _recent(alarms: list, day: pd.Timestamp, days: int = RECENT_DAYS) -> bool:
    lo = day - pd.Timedelta(days=days)
    return any(lo < pd.Timestamp(a) <= day for a in alarms)


def _run_daily_loop(series: pd.Series, window, articles_by_day: dict,
                    model, logger, condition: str = "A",
                    hmm_train=None) -> list[dict]:
    """Core daily loop shared across conditions A, B, and C.

    Args:
        series: daily return series covering the window.
        window: Window object (real or synthetic).
        articles_by_day: {date_str: [article_dicts]} for news agent.
        model: LocalModel implementation.
        logger: optional RunLogger.
        condition: "A" (standard), "B" (date-masked), or "C" (synthetic).
        hmm_train: pre-event training returns for HMM.
    """
    results = [d.detect(series) for d in (
        PageHinkley(), BOCPD(),
        HMMDetector(train_returns=hmm_train),
        DistributionalThreshold(),
    )]
    alarms_by_det = {r.detector: r.alarms for r in results}
    agg_alarms = aggregate_alarms(results, window_days=5, min_detectors=2)

    days = series.index[
        (series.index >= window.start_ts) & (series.index <= window.end_ts)
    ]
    records: list[dict] = []

    for day in days:
        day_str = str(day.date())
        # Triage: simplified (no live NewsStore for condition C / date-masked)
        n_recent = sum(_recent(a, day, RECENT_DAYS) for a in alarms_by_det.values())

        # Articles for this day (pre-fetched or synthetically generated)
        articles_today = articles_by_day.get(day_str, [])
        titles_today = [a.get("title", "") + " " + a.get("summary", "")
                        for a in articles_today]
        stress = daily_max_stress(titles_today) if titles_today else 0.0

        dec = decide(float("nan"), n_recent, _recent(agg_alarms, day),
                     stress_score=stress)

        rec = {
            "as_of": day_str,
            "window": window.name,
            "condition": condition,
            "triage_mode": dec.mode,
            "triage_reason": dec.reason,
            "finbert_stress": round(stress, 4) if stress > 0.0 else None,
        }

        if dec.mode == "skip":
            records.append(rec)
            continue

        # News agent
        if condition == "B":
            masked_articles = mask_dates_in_articles(articles_today)
        else:
            masked_articles = articles_today

        news = run_news_agent(
            "XXXX-XX-XX" if condition == "B" else day_str,
            masked_articles, model, logger=logger,
            extra_label=f"{window.name}:cond{condition}:{dec.mode}",
        )

        # Build context
        ctx = as_of_context(series, day, alarms_by_det)
        try:
            assert_no_lookahead({"news": news.to_dict()}, day)
            ctx["news"] = news.to_dict()
        except LookaheadError:
            rec["news_dropped_lookahead"] = True

        ctx["triage_mode"] = dec.mode

        # Apply date masking for condition B
        if condition == "B":
            ctx = mask_dates_in_context(ctx)

        t0 = time.time()
        try:
            assessment = run_supervisor(
                ctx, model, logger=logger,
                extra_label=f"{window.name}:cond{condition}",
                prompt_builder=build_supervisor_prompt_v3,
                prompt_version=SUPERVISOR_PROMPT_VERSION_V3,
            )
            rec["supervisor_latency_s"] = round(time.time() - t0, 1)
            rec["assessment"] = assessment.to_dict()
        except Exception as e:
            rec["error"] = str(e)

        records.append(rec)

    return records


def _articles_from_store(store, window, news_days: int) -> dict:
    """Pre-fetch news articles for the window, keyed by date string.

    Returns {} if the NewsStore is unavailable.
    """
    try:
        from news.store import NewsStore
        from news.filter import filter_news
        if store is None:
            return {}
        day_range = pd.bdate_range(window.start_ts, window.end_ts)
        articles_by_day: dict = {}
        for day in day_range:
            day_str = str(day.date())
            raw = store.query(day - pd.Timedelta(days=news_days), day)
            risky = filter_news(raw).sort_values("n_risk_terms", ascending=False)
            articles_by_day[day_str] = [
                {"date": str(r["date"].date()), "ticker": r["ticker"],
                 "title": r["title"], "summary": r["summary"]}
                for _, r in risky.iterrows()
            ]
        return articles_by_day
    except Exception:
        return {}


def run_condition_a(window_name: str, strategy: str, series: pd.Series,
                    model, store=None, logger=None, hmm_train=None) -> list[dict]:
    """Condition A: standard frozen pipeline."""
    window = get_window(window_name)
    articles_by_day = _articles_from_store(store, window, NEWS_DAYS)
    return _run_daily_loop(series, window, articles_by_day, model, logger,
                           condition="A", hmm_train=hmm_train)


def run_condition_b(window_name: str, strategy: str, series: pd.Series,
                    model, store=None, logger=None, hmm_train=None) -> list[dict]:
    """Condition B: date-masked pipeline.

    Identical to A but all ISO dates are replaced with XXXX-XX-XX before the
    LLM sees the context and articles.
    """
    window = get_window(window_name)
    articles_by_day = _articles_from_store(store, window, NEWS_DAYS)
    return _run_daily_loop(series, window, articles_by_day, model, logger,
                           condition="B", hmm_train=hmm_train)


def run_condition_c(synthetic_windows: list[SyntheticWindow],
                    model, logger=None) -> list[list[dict]]:
    """Condition C: synthetic event windows (M=10).

    Classical detectors run on synthetic returns; news is fabricated.
    No real dates, company names, or event identities are present.
    """
    all_records = []
    for sw in synthetic_windows:
        articles_by_day: dict = {}
        for art in sw.articles:
            d = art["date"]
            articles_by_day.setdefault(d, []).append(art)
        records = _run_daily_loop(
            sw.returns, sw.window, articles_by_day, model, logger,
            condition="C", hmm_train=None,  # no pre-event history for synthetic
        )
        all_records.append(records)
    return all_records


def _score_records(records: list[dict], window_name: str) -> dict:
    """Extract detection / latency from agentic JSONL-style records."""
    from agentic.alarm_extraction import extract_agentic_alarms, evaluate_agentic_window
    w = get_window(window_name)
    trading = pd.bdate_range(w.start_ts, w.end_ts)
    wm = evaluate_agentic_window(records, w, trading)
    return {
        "detected": wm.detected,
        "latency_days": wm.latency_days,
        "n_false_positives": wm.n_false_positives,
        "n_runtime_failures": wm.n_runtime_failures,
    }


def _score_synthetic(all_records: list[list[dict]],
                     synthetic_windows: list[SyntheticWindow]) -> dict:
    """Score Condition C: detection rate and mean latency across M windows."""
    from agentic.alarm_extraction import extract_agentic_alarms, evaluate_agentic_window
    detected_list = []
    latency_list = []
    for records, sw in zip(all_records, synthetic_windows):
        trading = pd.bdate_range(sw.window.start_ts, sw.window.end_ts)
        wm = evaluate_agentic_window(records, sw.window, trading)
        detected_list.append(bool(wm.detected))
        if wm.latency_days is not None:
            latency_list.append(wm.latency_days)
    n = len(detected_list)
    recall = sum(detected_list) / n if n else float("nan")
    mean_lat = float(np.mean(latency_list)) if latency_list else None
    return {
        "recall_mean": round(recall, 4),
        "recall_std": round(float(np.std(detected_list)), 4) if n > 1 else 0.0,
        "mean_latency": round(mean_lat, 1) if mean_lat is not None else None,
        "n_windows": n,
        "n_detected": sum(detected_list),
    }


def compute_leakage_bound(score_a: dict, score_b: dict, score_c: dict) -> dict:
    """Compute §8 leakage bounds from condition scores.

    score_a / score_b: {"detected": bool, "latency_days": int | None, ...}
    score_c: {"recall_mean": float, "mean_latency": float | None, ...}
    """
    # Recall proxy: 1 if detected else 0
    perf_a = 1.0 if score_a.get("detected") else 0.0
    perf_b = 1.0 if score_b.get("detected") else 0.0
    perf_c = score_c.get("recall_mean", 0.0)

    evidence_skill = perf_c
    memorisation = perf_a - min(perf_b, perf_c)
    memorisation = max(0.0, memorisation)  # clamp — can't be negative

    if perf_a > 0.5 and perf_c < 0.3:
        conclusion = "memorisation-dominated"
    elif perf_c >= perf_a * 0.6:
        conclusion = "evidence-driven"
    else:
        conclusion = "inconclusive"

    return {
        "evidence_skill_lower_bound": round(evidence_skill, 4),
        "memorisation_upper_bound": round(memorisation, 4),
        "perf_A": round(perf_a, 4),
        "perf_B": round(perf_b, 4),
        "perf_C_mean": round(perf_c, 4),
        "conclusion": conclusion,
    }


def _load_series_for_window(window_name: str, strategy: str) -> pd.Series | None:
    """Load the continuous PnL series that covers the given window."""
    window = get_window(window_name)
    for path in STRATEGY_CURVES.get(strategy, []):
        if not Path(path).exists():
            continue
        try:
            s = returns_series(load_pnl(path))
            if s.index.min() <= window.start_ts and s.index.max() >= window.end_ts:
                return s
        except Exception:
            continue
    return None


def _load_calm_returns_for_synthetic() -> pd.Series:
    """Load development calm returns for block-bootstrapping synthetic windows."""
    for path in STRATEGY_CURVES.get("JT_MOM", []) + STRATEGY_CURVES.get("AL_PCA", []):
        if Path(path).exists():
            try:
                s = returns_series(load_pnl(path))
                # Use a calm slice (2004-2006 or 2013-2014)
                calm_slice = s.loc["2004-01-01":"2006-12-31"]
                if len(calm_slice) > 100:
                    return calm_slice
            except Exception:
                continue
    # Fallback: tiny synthetic calm returns if no real data available
    rng = np.random.default_rng(0)
    dates = pd.bdate_range("2004-01-02", periods=500)
    return pd.Series(rng.normal(0, 0.008, size=500), index=dates, name="port_ret")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--strategy", default="AL_PCA", choices=list(STRATEGY_CURVES))
    ap.add_argument("--model", default=None,
                    help="stub | ollama | ollama:<name> (default: MONITOR_MODEL env or stub)")
    ap.add_argument("--M", type=int, default=10,
                    help="Number of synthetic windows for Condition C (default: 10)")
    ap.add_argument("--dry-run", action="store_true",
                    help="Validate imports and setup only; do not run the model")
    args = ap.parse_args()

    model = default_model(args.model)
    RESULTS.mkdir(parents=True, exist_ok=True)

    # Load optional NewsStore
    store = None
    try:
        from news.store import NewsStore
        if STORE_DIR.exists():
            store = NewsStore(STORE_DIR)
    except Exception:
        pass

    # Test windows for this strategy
    test_event_names = [
        n for n in TEST_STRATEGY_WINDOWS.get(args.strategy, [])
        if get_window(n).kind == "event"
    ]

    hmm_train = _hmm_training_returns(STRATEGY_CURVES.get(args.strategy, []))

    if args.dry_run:
        print(f"[dry-run] strategy={args.strategy}, model={model.name}")
        print(f"[dry-run] test event windows: {test_event_names}")
        print("[dry-run] imports OK")
        return

    # Generate synthetic windows once (shared across all event windows)
    calm_returns = _load_calm_returns_for_synthetic()
    dev_event_returns = {}
    for w in EVENT_WINDOWS:
        for path in STRATEGY_CURVES.get(args.strategy, []):
            if Path(path).exists():
                try:
                    s = returns_series(load_pnl(path))
                    if s.index.min() <= w.start_ts and s.index.max() >= w.end_ts:
                        dev_event_returns[w.name] = s
                        break
                except Exception:
                    pass

    synthetic_windows = generate_synthetic_windows(
        calm_returns=calm_returns,
        dev_event_windows=list(EVENT_WINDOWS),
        dev_event_returns=dev_event_returns,
        M=args.M,
    )
    print(f"Generated {len(synthetic_windows)} synthetic windows")

    out_data: dict = {"strategy": args.strategy, "model": model.name, "windows": {}}

    for window_name in test_event_names:
        print(f"\n=== {window_name} ===")
        series = _load_series_for_window(window_name, args.strategy)
        if series is None:
            print(f"  [skip] no curve covers {window_name}")
            continue

        logger_path = RESULTS / f"leakage_{window_name}_{args.strategy}.jsonl"
        logger = RunLogger(logger_path)

        # Condition A
        print("  Running Condition A (standard)...")
        recs_a = run_condition_a(window_name, args.strategy, series, model,
                                  store=store, logger=logger, hmm_train=hmm_train)
        score_a = _score_records(recs_a, window_name)
        print(f"  A: detected={score_a['detected']}, latency={score_a['latency_days']}")

        # Condition B
        print("  Running Condition B (date-masked)...")
        recs_b = run_condition_b(window_name, args.strategy, series, model,
                                  store=store, logger=logger, hmm_train=hmm_train)
        score_b = _score_records(recs_b, window_name)
        print(f"  B: detected={score_b['detected']}, latency={score_b['latency_days']}")

        # Condition C (shared synthetic windows)
        print(f"  Running Condition C ({args.M} synthetic windows)...")
        recs_c = run_condition_c(synthetic_windows, model, logger=logger)
        score_c = _score_synthetic(recs_c, synthetic_windows)
        print(f"  C: recall={score_c['recall_mean']:.2f} (±{score_c['recall_std']:.2f}), "
              f"mean_latency={score_c['mean_latency']}")

        # Leakage bound
        bound = compute_leakage_bound(score_a, score_b, score_c)
        print(f"  Leakage bound: evidence_skill≥{bound['evidence_skill_lower_bound']:.2f}, "
              f"memorisation≤{bound['memorisation_upper_bound']:.2f} "
              f"[{bound['conclusion']}]")

        out_data["windows"][window_name] = {
            "A": score_a,
            "B": score_b,
            "C": score_c,
            "leakage_bound": bound,
        }

    out_path = RESULTS / f"leakage_analysis_{args.strategy}.json"
    out_path.write_text(json.dumps(out_data, indent=2, default=str))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
