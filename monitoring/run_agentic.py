"""End-to-end agentic monitoring on one evaluation window.

Two-pass design (recommended for Ollama inference):
  Pass 1 — pre-compute heavy I/O in a separate process:
    python scripts/precompute_finbert.py --window <name>
    python scripts/precompute_context.py --window <name>
  Pass 2 — lightweight inference (no parquet/torch in-process):
    python run_agentic.py --window <name> --strategy AL_PCA \
        --model ollama:qwen2.5:1.5b \
        --finbert-cache results/finbert_cache/<name>.json \
        --context-cache results/context_cache/<name>.json

Daily loop: build the day's causal context (telemetry, classical alarms, as-of
macro, filtered news), triage how much model to spend (skip / cheap / thinking /
classical-escalation), and on escalated days run the News Context Agent then the
Performance Supervisor (v3 prompt). Every step is lookahead-guarded; every model
call is logged to JSONL. Day-level DeadRunnerError retry prevents single-day
Ollama stalls from aborting an entire window.

Usage (from monitoring/):
    python run_agentic.py --window quant_meltdown_2007 --strategy AL_PCA --model stub
    python run_agentic.py --window quant_meltdown_2007 --strategy AL_PCA \
        --model ollama:qwen2.5:3b
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import pandas as pd

from pnl_loader import load_pnl, returns_series
from windows import get_window
from detectors import PageHinkley, BOCPD, HMMDetector, DistributionalThreshold, aggregate_alarms
from run_classical import STRATEGY_CURVES, _hmm_training_returns, _select_curves
from agentic import as_of_context, RunLogger, run_supervisor
from agentic.guardrails import assert_no_lookahead, LookaheadError, mask_dates_in_context, mask_dates_in_articles
from agentic.model import default_model, DeadRunnerError
from agentic.news_agent import run_news_agent
from agentic.prompts import build_supervisor_prompt_v3, SUPERVISOR_PROMPT_VERSION_V3
from news.store import NewsStore
from news.filter import filter_news
from news.aggregate import daily_signals
from news.triage import decide, RECENT_DAYS, COVERAGE_DAYS
# FinBERT import is deferred: only loaded when no --finbert-cache is provided,
# keeping the process lightweight (~300 MB vs 1.5 GB) for reliable Ollama calls.
_daily_max_stress = None  # lazy; set in main() or on first use

def _git_sha() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
            cwd=str(Path(__file__).parent),
        ).stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def _log_run_meta(logger, model_name: str, curve: str, window: str,
                  strategy: str, condition: str) -> None:
    """Write the first JSONL line: run provenance (model, curve, git SHA)."""
    logger.log({"agent": "run_meta", "model": model_name, "curve": curve,
                "git_sha": _git_sha(), "window": window,
                "strategy": strategy, "condition": condition})


ROOT = Path(__file__).resolve().parent.parent
RESULTS = Path(__file__).resolve().parent / "results"
STORE_DIR = ROOT / "data" / "fnspid" / "store"
MACRO_DIR = ROOT / "data" / "macro"

# RECENT_DAYS = 3 (preregistration §6.2) — imported from triage
NEWS_DAYS = COVERAGE_DAYS   # 7-day news context window (preregistration §6.2)
BASELINE_PAD = 120          # calendar days of news history before the window for the z baseline


def _call_with_retry(fn, model, error_prefix: str = "dead_runner_retry"):
    """Call fn(); on DeadRunnerError, cooldown 15s + ping + one retry.

    Returns (result, None) on success or (None, error_string) on failure.
    The 15s cooldown lets the Ollama runner thermal-throttle or recover from
    a stale HTTP-200-done-false state before the retry attempt.
    """
    try:
        return fn(), None
    except DeadRunnerError:
        time.sleep(15)
        try:
            model.complete("ping", "Return {\"ok\": true}")
            return fn(), None
        except (DeadRunnerError, json.JSONDecodeError, ValueError) as exc:
            return None, f"{error_prefix}: {exc}"[:200]
    except (json.JSONDecodeError, ValueError) as exc:
        return None, str(exc)[:200]


def _completed_dates(path: Path) -> set[str]:
    """Scan a JSONL log and return dates that are fully processed.

    A day is "done" only if it has terminal evidence: a performance_supervisor
    line (assessment completed) or a skip-triage line. Bare triage lines from
    days aborted mid-call are NOT counted — they need reprocessing.
    """
    supervisor_dates: set[str] = set()
    skip_triage_dates: set[str] = set()
    if not path.exists():
        return set()
    with open(path) as fh:
        for line in fh:
            try:
                rec = json.loads(line)
                agent = rec.get("agent", "")
                d = rec.get("as_of") or rec.get("date")
                if not d or agent == "run_meta":
                    continue
                if agent == "performance_supervisor":
                    supervisor_dates.add(d)
                elif agent == "triage" and rec.get("triage_mode") == "skip":
                    skip_triage_dates.add(d)
            except json.JSONDecodeError:
                pass
    return supervisor_dates | skip_triage_dates


def _recent(alarms: list, day: pd.Timestamp, days: int = RECENT_DAYS) -> bool:
    lo = day - pd.Timedelta(days=days)
    return any(lo < pd.Timestamp(a) <= day for a in alarms)


def _resolve_window(profile, name: str):
    """Resolve a window by name against the market's profile window set.

    sp500 preserves its legacy resolution (dev + test windows via ``get_window``)
    so every S&P 500 window name that worked before still works. Other markets
    (e.g. energy) resolve from their own ``profile.windows`` and fail closed —
    naming the market — when the window is not part of that market.
    """
    for w in profile.windows:
        if w.name == name:
            return w
    if profile.key == "sp500":
        return get_window(name)  # legacy dev+test lookup, unchanged for sp500
    raise SystemExit(
        f"--window {name!r} not in market {profile.key!r}; "
        f"known windows: {[w.name for w in profile.windows]}"
    )


def run_window(series: pd.Series, window, store: NewsStore | None, model,
               logger: RunLogger | None = None, macro_dir=MACRO_DIR,
               max_articles: int = 40, strategy: str | None = None,
               condition: str = "A", skip_dates: set | None = None,
               finbert_cache: dict[str, float] | None = None,
               context_cache: dict | None = None,
               strategy_curves: dict | None = None) -> list[dict]:
    """Run the daily agentic loop over one window. Returns one record per trading day.

    skip_dates: set of date strings (YYYY-MM-DD) already processed — these days
    are omitted from the loop to support day-level resume after crashes.
    finbert_cache: if provided, read pre-computed stress scores instead of
    loading FinBERT (keeps process lightweight for reliable Ollama calls).
    context_cache: if provided, read pre-computed intensity_z and articles per day
    instead of querying the news store (eliminates heavy parquet I/O).
    """
    # Classical detectors run continuously over the full curve (warm-up history included).
    # HMM is fit on pre-event training data when available (out-of-sample for event windows).
    _curves = strategy_curves if strategy_curves is not None else STRATEGY_CURVES
    hmm_train = _hmm_training_returns(_curves.get(strategy, [])) if strategy else None
    results = [d.detect(series) for d in
               (PageHinkley(), BOCPD(), HMMDetector(train_returns=hmm_train),
                DistributionalThreshold())]
    alarms_by_det = {r.detector: r.alarms for r in results}
    agg_alarms = aggregate_alarms(results, window_days=5, min_detectors=2)

    if context_cache is None:
        # News signals need trailing history for the causal z baseline.
        news_hist = store.query(window.start_ts - pd.Timedelta(days=BASELINE_PAD), window.end_ts)
        signals = daily_signals(news_hist, filter_news(news_hist))
    else:
        signals = None  # read from cache per day

    try:
        from macro.context import macro_context
    except ImportError:  # macro package optional at run time
        macro_context = None

    # Warmup ping: after init the Ollama runner may have gone stale.
    if hasattr(model, 'host'):
        try:
            model.complete("ping", "Return {\"ok\": true}")
        except Exception:
            pass  # best effort; the day loop has its own retry

    days = series.index[(series.index >= window.start_ts) & (series.index <= window.end_ts)]
    records: list[dict] = []
    _skip = skip_dates or set()
    for day in days:
        if str(day.date()) in _skip:
            continue

        day_str = str(day.date())

        # --- intensity_z ---
        if context_cache is not None:
            day_ctx = context_cache.get(day_str, {})
            z = float(day_ctx.get("intensity_z") or float("nan"))
        else:
            z = float(signals["intensity_z"].get(day.normalize(), float("nan"))) \
                if not signals.empty else float("nan")

        n_recent = sum(_recent(a, day, RECENT_DAYS) for a in alarms_by_det.values())

        # FinBERT stress score: max negative-sentiment probability across today's risk articles
        if finbert_cache is not None:
            stress = finbert_cache.get(day_str, 0.0)
        else:
            global _daily_max_stress
            if _daily_max_stress is None:
                from news.finbert import daily_max_stress as _dms
                _daily_max_stress = _dms
            day_risk = filter_news(store.query(day - pd.Timedelta(days=1), day))
            if not day_risk.empty:
                titles = day_risk["title"].fillna("")
                summaries = day_risk["summary"].fillna("") if "summary" in day_risk.columns else titles
                headlines_today = (titles + " " + summaries).tolist()
            else:
                headlines_today = []
            stress = _daily_max_stress(headlines_today)

        dec = decide(z, n_recent, _recent(agg_alarms, day), stress_score=stress)

        rec = {"as_of": day_str, "window": window.name,
               "triage_mode": dec.mode, "triage_reason": dec.reason,
               "intensity_z": None if pd.isna(z) else round(z, 2),
               "finbert_stress": round(stress, 4) if stress > 0.0 else None}
        if logger is not None:
            logger.log({"agent": "triage", "prompt_version": "triage-v1", **rec})

        if dec.mode == "skip":
            records.append(rec)
            continue

        # --- News Context Agent ---
        if context_cache is not None:
            articles = day_ctx.get("articles", [])
        else:
            day_news = store.query(day - pd.Timedelta(days=NEWS_DAYS), day)
            risky = filter_news(day_news).sort_values("n_risk_terms", ascending=False)
            articles = [{"date": str(r["date"].date()), "ticker": r["ticker"],
                         "title": r["title"], "summary": r["summary"]}
                        for _, r in risky.iterrows()]
        # Condition B: mask dates in articles before sending to news agent
        news_articles = mask_dates_in_articles(articles) if condition == "B" else articles
        as_of_for_news = "XXXX-XX-XX" if condition == "B" else str(day.date())
        t0 = time.time()
        news, err = _call_with_retry(
            lambda: run_news_agent(as_of_for_news, news_articles, model, logger=logger,
                                   extra_label=f"{window.name}:{dec.mode}",
                                   max_articles=max_articles),
            model, error_prefix="dead_runner_retry")
        if err:
            rec["news_error"] = err
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
        assessment, err = _call_with_retry(
            lambda: run_supervisor(ctx, model, logger=logger,
                                   extra_label=f"{window.name}:{dec.mode}",
                                   prompt_builder=build_supervisor_prompt_v3,
                                   prompt_version=SUPERVISOR_PROMPT_VERSION_V3),
            model, error_prefix="dead_runner_retry")
        if err:
            rec["supervisor_error"] = err
            records.append(rec)
            continue
        rec["supervisor_latency_s"] = round(time.time() - t0, 1)
        rec["assessment"] = assessment.to_dict()
        records.append(rec)
    return records


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", default="quant_meltdown_2007")
    ap.add_argument("--market", default="sp500", help="market profile key (default: sp500)")
    ap.add_argument("--strategy", default="AL_PCA", choices=list(STRATEGY_CURVES))
    ap.add_argument("--model", default=None, help="stub | ollama | ollama:<name> (default: MONITOR_MODEL env or stub)")
    ap.add_argument("--condition", default="A", choices=["A", "B"],
                    help="A=standard, B=date-masked (leakage Condition B)")
    ap.add_argument("--max-articles", type=int, default=40,
                    help="max news articles per day (lower = smaller prompt, faster inference)")
    ap.add_argument("--finbert-cache", default=None,
                    help="path to pre-computed FinBERT stress JSON (avoids loading torch)")
    ap.add_argument("--context-cache", default=None,
                    help="path to pre-computed context JSON (avoids heavy parquet I/O)")
    args = ap.parse_args()

    # Market-aware routing: resolve curves and the window from the profile so
    # non-sp500 markets (e.g. energy) run their own curves + windows. sp500
    # yields exactly its legacy inputs, so its path is unchanged.
    from config.profiles import get_profile
    profile = get_profile(args.market)
    strategy_curves = _select_curves(profile)

    window = _resolve_window(profile, args.window)
    curve = next((p for p in strategy_curves[args.strategy]
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

    skip_dates = _completed_dates(out)
    if skip_dates:
        print(f"  Resuming: {len(skip_dates)} days done in {out.name}")
    if not skip_dates:
        # Fresh run — overwrite any existing file
        if out.exists():
            out.unlink()

    logger = RunLogger(out)
    if not skip_dates:
        _log_run_meta(logger, model.name, str(curve), args.window, args.strategy, args.condition)

    finbert_cache = None
    if args.finbert_cache:
        with open(args.finbert_cache) as f:
            finbert_cache = json.load(f)
        print(f"  FinBERT cache: {len(finbert_cache)} days from {args.finbert_cache}")

    context_cache = None
    if args.context_cache:
        with open(args.context_cache) as f:
            context_cache = json.load(f)
        print(f"  Context cache: {len(context_cache)} days from {args.context_cache}")

    store = None if context_cache else NewsStore(STORE_DIR)
    records = run_window(series, window, store, model, logger=logger,
                         strategy=args.strategy, condition=args.condition,
                         skip_dates=skip_dates, max_articles=args.max_articles,
                         finbert_cache=finbert_cache, context_cache=context_cache,
                         strategy_curves=strategy_curves)

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
