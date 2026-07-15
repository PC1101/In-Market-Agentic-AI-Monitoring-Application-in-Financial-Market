"""End-to-end classical monitoring across both strategies and all six windows.

For each strategy we run the four detectors *continuously* over each daily PnL curve, so
every window inherits genuine preceding history for detector warm-up. Alarms are then
scored window-by-window (detection latency, precision, recall on event windows;
false-positive rate on calm windows), for each detector and for the >=2-within-5-days
aggregate. A small agentic pass (offline stub) is also run at each event onset to exercise
the Week-2 agentic scaffold end-to-end and emit a JSONL agent log.

Data coverage note (Week-1 dependency): the AL PCA curves currently available cover five
of the six windows via ``baseline_2007_2015`` (2007-2014) plus ``calm_2004_2006``. The JT
momentum daily curve is produced by ``XSectional/backtest.py:run_backtest_daily``; if it
has not been generated yet this script reports JT as data-pending rather than fabricating.
"""

from __future__ import annotations

import argparse
import bisect
import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from pnl_loader import load_pnl, returns_series
from windows import ALL_WINDOWS, EVENT_WINDOWS
from detectors import PageHinkley, BOCPD, HMMDetector, DistributionalThreshold, aggregate_alarms
from metrics import evaluate_window, aggregate_metrics, WindowMetrics
from agentic import (as_of_context, LocalModel, RunLogger, run_supervisor,
                     run_news_agent, make_model)
from news import NewsConfig, build_news_block, TriageMode, FakeScorer, FinBERTScorer
from news.records import load_parquet_news

ROOT = Path(__file__).resolve().parent.parent
RESULTS = Path(__file__).resolve().parent / "results"

STAT_ARB = ROOT / "Stat Arb" / "statsArb-dev" / "results" / "full_universe"
JT_DAILY = ROOT / "XSectional" / "results" / "equity_curve_daily.csv"

def _prefer_pit(tag: str):
    """Use the point-in-time curve for a window tag if built, else the 2020-snapshot one."""
    pit = STAT_ARB / f"{tag}_pit" / "equity_curve.csv"
    return pit if pit.exists() else STAT_ARB / tag / "equity_curve.csv"


# Each strategy is a list of continuous daily PnL curves.
STRATEGY_CURVES = {
    "AL_PCA": [
        _prefer_pit("baseline_2007_2015"),
        _prefer_pit("calm_2004_2006"),
    ],
    "JT_MOM": [JT_DAILY],
}


def _new_detectors():
    return [PageHinkley(), BOCPD(), HMMDetector(), DistributionalThreshold()]


def _windows_in_curve(series: pd.Series, end_tol_days: int = 7):
    """Return the windows covered by a curve's date span.

    The curve must start on/before the window and extend to within ``end_tol_days`` of
    the window end (tolerating minor data-boundary gaps, e.g. a curve that ends one
    trading day short of a calm window's nominal close).
    """
    lo, hi = series.index.min(), series.index.max()
    tol = pd.Timedelta(days=end_tol_days)
    return [w for w in ALL_WINDOWS if w.start_ts >= lo and w.end_ts <= hi + tol]


def _n_trading_days(series: pd.Series, window) -> int:
    mask = (series.index >= window.start_ts) & (series.index <= window.end_ts)
    return int(mask.sum())


# ---------------------------------------------------------------------------
# Week-3 news path (only active with --news)
# ---------------------------------------------------------------------------

class _MemoScorer:
    """Cache ``daily_stress`` by headline tuple — the daily triage loop rescores
    the same day's matched headlines for every decision date whose lookback
    covers it, which would be prohibitively slow with CPU FinBERT."""

    def __init__(self, inner):
        self.inner = inner
        self._cache: dict[tuple, float] = {}

    def daily_stress(self, headlines: list[str]) -> float:
        key = tuple(headlines)
        if key not in self._cache:
            self._cache[key] = self.inner.daily_stress(list(headlines))
        return self._cache[key]


def _detector_state(alarms_by_det: dict, series: pd.Series, as_of,
                    days: int = 5) -> tuple[int, bool]:
    """Classical-detector state visible at ``as_of``: (# individual alarms in the
    last ``days`` trading days, whether the aggregate rule fired in that span)."""
    idx = series.index[series.index <= pd.Timestamp(as_of)]
    recent = set(idx[-days:])
    n_recent = 0
    aggregate = False
    for det, alarms in alarms_by_det.items():
        hits = sum(1 for a in alarms if pd.Timestamp(a).normalize() in recent)
        if det == "aggregate":
            aggregate = hits > 0
        else:
            n_recent += hits
    return n_recent, aggregate


def _log_news_escalation(agent_logger, as_of: str, reason: str, label: str):
    if agent_logger is not None:
        agent_logger.log_invocation(
            agent="news_context", prompt_version="triage", as_of=as_of,
            raw_output=None, assessment=None, error=reason,
            extra={"triage_mode": TriageMode.CLASSICAL_ESCALATION.value,
                   "label": label},
        )


def _news_for_date(as_of, news_cfg: NewsConfig, records: list[dict],
                   alarms_by_det: dict, series: pd.Series,
                   news_model: LocalModel | None, agent_logger, label: str,
                   run_agent: bool = True) -> tuple[dict, str]:
    """Build the causal news block for ``as_of``, triage it, and (for the
    CHEAP/THINKING modes, when ``run_agent``) run the News Context Agent.

    Returns (news context dict for ``as_of_context``, final triage-mode value).
    Any agent failure degrades to CLASSICAL_ESCALATION — the supervisor then
    runs on classical outputs alone, and the escalation is logged.
    """
    n_recent, aggregate = _detector_state(alarms_by_det, series, as_of)
    block = build_news_block(as_of, news_cfg, records=records,
                             n_recent_alarms=n_recent, aggregate_alarm=aggregate)
    mode = TriageMode(block["triage_mode"])

    if mode == TriageMode.SKIP:
        return {"status": "quiet", "triage_mode": mode.value,
                "signal": block["signal"]}, mode.value

    if mode == TriageMode.CLASSICAL_ESCALATION:
        reason = "no news coverage while detectors alarm"
        _log_news_escalation(agent_logger, block["as_of"], reason, label)
        return {"status": "unavailable", "triage_mode": mode.value,
                "reason": reason}, mode.value

    if not run_agent or news_model is None:
        # Triage-tally pass: report the mode without spending an LLM call.
        return {"status": "not_run", "triage_mode": mode.value,
                "signal": block["signal"]}, mode.value

    variant = "thinking" if mode == TriageMode.THINKING_MODEL else "standard"
    try:
        summary = run_news_agent(block, news_model, logger=agent_logger,
                                 prompt_variant=variant, extra_label=label)
    except Exception as e:  # SchemaError after repair, transport errors, ...
        reason = f"news agent failed: {e}"
        _log_news_escalation(agent_logger, block["as_of"], reason, label)
        return {"status": "unavailable",
                "triage_mode": TriageMode.CLASSICAL_ESCALATION.value,
                "reason": reason}, TriageMode.CLASSICAL_ESCALATION.value

    news = dict(summary.to_dict(), triage_mode=mode.value, signal=block["signal"])
    return news, mode.value


def _load_window_records(w, news_cfg: NewsConfig) -> list[dict]:
    """Load all news records a window's daily decisions could ever see.

    Returned list is sorted by ISO timestamp string (loader guarantee), which
    ``_frame_slice`` relies on.
    """
    start = w.start_ts - pd.Timedelta(days=news_cfg.lookback_days + 2)
    return load_parquet_news(news_cfg.source_paths, start=start, end=w.end_ts)


def _frame_slice(records: list[dict], ts_keys: list[str], as_of,
                 lookback_days: int) -> list[dict]:
    """Slice the sorted window records down to one decision date's signal frame.

    Bounds by ISO-string comparison: [frame_start 00:00, end of day T-1]. Dense
    windows hold millions of records — the per-day pipeline must not rescan them
    all. ``build_news_block`` re-applies its exact causal cut on the slice.
    """
    as_of = pd.Timestamp(as_of).normalize()
    lo_key = str((as_of - pd.Timedelta(days=lookback_days + 1)).date())
    hi_key = str(as_of.date())  # records on day T start with this prefix -> excluded
    lo = bisect.bisect_left(ts_keys, lo_key)
    hi = bisect.bisect_left(ts_keys, hi_key)
    return records[lo:hi]


def run_strategy(name: str, curve_paths, agent_logger: RunLogger | None,
                 agent_model: LocalModel | None = None,
                 news_cfg: NewsConfig | None = None,
                 news_model: LocalModel | None = None,
                 window_filter: set[str] | None = None):
    """Run all detectors on a strategy's curves and score every covered window.

    With ``news_cfg`` set (--news), the event-onset agentic pass additionally
    builds the causal news block, triages it, runs the News Context Agent when
    warranted, and feeds the result to the supervisor; a daily triage-mode tally
    is also built per covered window (no LLM calls on non-onset days).
    """
    # detector name -> list[WindowMetrics] across all windows
    per_detector: dict[str, list[WindowMetrics]] = {}
    covered: set[str] = set()
    triage_tally: dict[str, dict[str, int]] = {}  # window -> mode -> n days

    for path in curve_paths:
        if not Path(path).exists():
            continue
        series = returns_series(load_pnl(path))
        windows = _windows_in_curve(series)
        if window_filter is not None:
            windows = [w for w in windows if w.name in window_filter]
        if not windows:
            continue

        detectors = _new_detectors()
        results = [d.detect(series) for d in detectors]
        alarms_by_det = {r.detector: r.alarms for r in results}
        alarms_by_det["aggregate"] = aggregate_alarms(results, window_days=5, min_detectors=2)

        for w in windows:
            covered.add(w.name)
            n_td = _n_trading_days(series, w)
            for det_name, alarms in alarms_by_det.items():
                wm = evaluate_window(alarms, w, tolerance_days=21, n_trading_days=n_td)
                per_detector.setdefault(det_name, []).append(wm)

        # Agentic scaffold demonstration: assess at each event onset in this curve.
        if agent_logger is not None:
            model = agent_model if agent_model is not None else make_model("stub")
            for w in windows:
                if w.kind != "event":
                    continue
                onset = w.onset_ts
                if onset not in series.index:
                    # snap to the first trading day on/after onset
                    later = series.index[series.index >= onset]
                    if len(later) == 0:
                        continue
                    onset = later[0]
                label = f"{name}:{w.name}"
                news = None
                if news_cfg is not None:
                    w_records = _load_window_records(w, news_cfg)
                    ts_keys = [r["timestamp"] for r in w_records]
                    frame = _frame_slice(w_records, ts_keys, onset,
                                         news_cfg.lookback_days)
                    news, _ = _news_for_date(
                        onset, news_cfg, frame, alarms_by_det, series,
                        news_model, agent_logger, label)
                ctx = as_of_context(series, onset, alarms_by_det, news=news)
                run_supervisor(ctx, model, logger=agent_logger,
                               extra_label=label)

        # Daily triage-mode tally per window (news path only; no agent calls).
        if news_cfg is not None:
            for w in windows:
                w_records = _load_window_records(w, news_cfg)
                ts_keys = [r["timestamp"] for r in w_records]
                days = series.index[(series.index >= w.start_ts)
                                    & (series.index <= w.end_ts)]
                tally = triage_tally.setdefault(w.name, {})
                for day in days:
                    frame = _frame_slice(w_records, ts_keys, day,
                                         news_cfg.lookback_days)
                    _, mode = _news_for_date(
                        day, news_cfg, frame, alarms_by_det, series,
                        news_model=None, agent_logger=None,
                        label=f"{name}:{w.name}", run_agent=False)
                    tally[mode] = tally.get(mode, 0) + 1

    headline = {det: aggregate_metrics(wms) for det, wms in per_detector.items()}
    return headline, per_detector, covered, triage_tally


def _fmt(x, nd=3):
    return "n/a" if x is None or (isinstance(x, float) and pd.isna(x)) else f"{x:.{nd}f}"


def main():
    ap = argparse.ArgumentParser(description="Classical monitoring + agentic pass.")
    ap.add_argument("--model", default="stub",
                    help="agent model spec: 'stub' (default, deterministic/CI-safe) "
                         "or 'ollama:<name>' e.g. 'ollama:llama3.2:3b'")
    ap.add_argument("--news", action="store_true",
                    help="enable the Week-3 news path: triage + News Context Agent "
                         "feeding the supervisor (default off — baseline unchanged)")
    ap.add_argument("--news-data-dir", default=None,
                    help="directory holding the news parquet caches "
                         "(default: monitoring/news/data)")
    ap.add_argument("--news-scorer", default="finbert", choices=["finbert", "fake"],
                    help="sentiment scorer for the news-stress signal: 'finbert' "
                         "(default) or 'fake' (deterministic, offline)")
    ap.add_argument("--news-device", default="cpu", choices=["cpu", "cuda"],
                    help="device for the FinBERT scorer (default cpu; 'cuda' is "
                         "~10x faster but shares VRAM with the Ollama model at "
                         "onset dates)")
    ap.add_argument("--windows", default=None,
                    help="comma-separated window names to restrict scoring + news "
                         "runs to (e.g. 'quant_meltdown_2007')")
    args = ap.parse_args()
    agent_model = make_model(args.model)

    window_filter = None
    if args.windows:
        window_filter = {s.strip() for s in args.windows.split(",") if s.strip()}
        unknown = window_filter - {w.name for w in ALL_WINDOWS}
        if unknown:
            ap.error(f"unknown window(s): {sorted(unknown)}")

    news_cfg = news_model = None
    if args.news:
        scorer = _MemoScorer(FinBERTScorer(device=args.news_device)
                             if args.news_scorer == "finbert" else FakeScorer())
        kwargs = {"scorer": scorer}
        if args.news_data_dir:
            kwargs["data_dir"] = Path(args.news_data_dir)
        news_cfg = NewsConfig(**kwargs)
        missing = [p for p in news_cfg.source_paths if not p.exists()]
        if len(missing) == len(news_cfg.source_paths):
            ap.error(f"--news: no news caches found in {news_cfg.data_dir} — "
                     f"run scripts/fetch_nyt_archive.py + scripts/build_news_cache.py")
        news_model = make_model(args.model, agent="news_context")

    RESULTS.mkdir(parents=True, exist_ok=True)
    agent_logger = RunLogger(RESULTS / "agent_log.jsonl")
    # start each run with a fresh agent log
    if (RESULTS / "agent_log.jsonl").exists():
        (RESULTS / "agent_log.jsonl").unlink()

    summary = {}
    triage_report = {}
    detector_order = ["page_hinkley", "bocpd", "hmm", "distributional", "aggregate"]

    for strat, paths in STRATEGY_CURVES.items():
        missing = [str(p) for p in paths if not Path(p).exists()]
        if len(missing) == len(paths):
            print(f"\n### {strat}: NO curves found — data pending. Missing:")
            for m in missing:
                print(f"    {m}")
            summary[strat] = {"status": "data_pending", "missing": missing}
            continue

        headline, per_detector, covered, triage_tally = run_strategy(
            strat, paths, agent_logger, agent_model=agent_model,
            news_cfg=news_cfg, news_model=news_model, window_filter=window_filter)
        if triage_tally:
            triage_report[strat] = triage_tally

        print(f"\n### {strat} — windows covered: {sorted(covered)}")
        for p in paths:
            if Path(p).exists():
                print(f"    curve: {Path(p).relative_to(ROOT)}")
        header = f"{'detector':16} {'recall':>7} {'precision':>10} {'mean_lat':>9} {'FPR/day':>9}"
        print(header)
        print("-" * len(header))
        strat_out = {}
        for det in detector_order:
            if det not in headline:
                continue
            h = headline[det]
            print(f"{det:16} {_fmt(h.recall,2):>7} {_fmt(h.precision,2):>10} "
                  f"{_fmt(h.mean_latency_days,1):>9} {_fmt(h.false_positive_rate,4):>9}")
            strat_out[det] = {
                "recall": h.recall, "precision": h.precision,
                "mean_latency_days": h.mean_latency_days,
                "false_positive_rate": h.false_positive_rate,
                "n_events": h.n_events, "n_detected": h.n_detected,
                "per_window": [asdict(m) for m in h.per_window],
            }
        summary[strat] = {"status": "ok", "covered": sorted(covered), "detectors": strat_out}

    (RESULTS / "classical_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"\nWrote {RESULTS / 'classical_summary.json'}")
    print(f"Wrote {RESULTS / 'agent_log.jsonl'} ({len(agent_logger.read_all())} agent assessments)")

    if news_cfg is not None:
        (RESULTS / "news_triage_report.json").write_text(
            json.dumps(triage_report, indent=2))
        print(f"Wrote {RESULTS / 'news_triage_report.json'}")
        for strat, tallies in triage_report.items():
            print(f"\n### {strat} — daily news triage modes")
            for wname, tally in tallies.items():
                parts = ", ".join(f"{m}={n}" for m, n in sorted(tally.items()))
                print(f"    {wname}: {parts}")


if __name__ == "__main__":
    main()
