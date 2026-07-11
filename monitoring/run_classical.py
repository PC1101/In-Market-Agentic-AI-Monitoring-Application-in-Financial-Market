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

import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from pnl_loader import load_pnl, returns_series
from windows import ALL_WINDOWS, EVENT_WINDOWS
from detectors import PageHinkley, BOCPD, HMMDetector, DistributionalThreshold, aggregate_alarms
from metrics import evaluate_window, aggregate_metrics, WindowMetrics
from agentic import as_of_context, OfflineStubModel, RunLogger, run_supervisor

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


def run_strategy(name: str, curve_paths, agent_logger: RunLogger | None):
    """Run all detectors on a strategy's curves and score every covered window."""
    # detector name -> list[WindowMetrics] across all windows
    per_detector: dict[str, list[WindowMetrics]] = {}
    covered: set[str] = set()

    for path in curve_paths:
        if not Path(path).exists():
            continue
        series = returns_series(load_pnl(path))
        windows = _windows_in_curve(series)
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
                ctx = as_of_context(series, onset, alarms_by_det)
                run_supervisor(ctx, OfflineStubModel(), logger=agent_logger,
                               extra_label=f"{name}:{w.name}")

    headline = {det: aggregate_metrics(wms) for det, wms in per_detector.items()}
    return headline, per_detector, covered


def _fmt(x, nd=3):
    return "n/a" if x is None or (isinstance(x, float) and pd.isna(x)) else f"{x:.{nd}f}"


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    agent_logger = RunLogger(RESULTS / "agent_log.jsonl")
    # start each run with a fresh agent log
    if (RESULTS / "agent_log.jsonl").exists():
        (RESULTS / "agent_log.jsonl").unlink()

    summary = {}
    detector_order = ["page_hinkley", "bocpd", "hmm", "distributional", "aggregate"]

    for strat, paths in STRATEGY_CURVES.items():
        missing = [str(p) for p in paths if not Path(p).exists()]
        if len(missing) == len(paths):
            print(f"\n### {strat}: NO curves found — data pending. Missing:")
            for m in missing:
                print(f"    {m}")
            summary[strat] = {"status": "data_pending", "missing": missing}
            continue

        headline, per_detector, covered = run_strategy(strat, paths, agent_logger)

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


if __name__ == "__main__":
    main()
