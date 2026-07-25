"""
Classical arm on the confirmatory TEST windows (D5 step 2).

Frozen detectors (identical instantiation to run_classical.py), zero recalibration.
Detectors run continuously over each strategy curve (full warm-up history), and
alarms are scored only against the TEST windows eligible for that strategy
(TEST_STRATEGY_WINDOWS). Output: results/classical_test_summary.json.

This is an orchestration/analysis script: it does not modify the frozen system
(run_classical.py detectors, thresholds, scoring in metrics.py are reused as-is).

Usage:
    cd monitoring && python scripts/run_classical_test.py
"""
import json
import sys
from dataclasses import asdict
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pnl_loader import load_pnl, returns_series
from run_classical import STRATEGY_CURVES, _hmm_training_returns, _new_detectors
from detectors import aggregate_alarms
from metrics import evaluate_window, aggregate_metrics
from windows import TEST_WINDOWS, TEST_STRATEGY_WINDOWS

RESULTS = Path(__file__).resolve().parent.parent / "results"
TOLERANCE_DAYS = 21  # identical to dev scoring
END_TOL = pd.Timedelta(days=7)  # identical to run_classical._windows_in_curve


def main() -> None:
    summary: dict = {}
    detector_order = ["page_hinkley", "bocpd", "hmm", "distributional", "aggregate"]

    for strat, paths in STRATEGY_CURVES.items():
        eligible = list(TEST_STRATEGY_WINDOWS.get(strat, []))
        hmm_train = _hmm_training_returns(paths)
        per_detector: dict[str, list] = {}
        covered: set[str] = set()
        curves_used: list[str] = []

        for path in paths:
            p = Path(path)
            if not p.exists():
                continue
            series = returns_series(load_pnl(p))
            lo, hi = series.index.min(), series.index.max()
            windows = [
                w for w in TEST_WINDOWS
                if w.name in eligible and w.start_ts >= lo and w.end_ts <= hi + END_TOL
            ]
            if not windows:
                continue
            curves_used.append(str(p))

            detectors = _new_detectors(train_returns=hmm_train)
            results = [d.detect(series) for d in detectors]
            alarms_by_det = {r.detector: r.alarms for r in results}
            alarms_by_det["aggregate"] = aggregate_alarms(
                results, window_days=5, min_detectors=2
            )

            for w in windows:
                covered.add(w.name)
                mask = (series.index >= w.start_ts) & (series.index <= w.end_ts)
                n_td = int(mask.sum())
                for det_name, alarms in alarms_by_det.items():
                    wm = evaluate_window(
                        alarms, w, tolerance_days=TOLERANCE_DAYS, n_trading_days=n_td
                    )
                    per_detector.setdefault(det_name, []).append(wm)

        not_covered = sorted(set(eligible) - covered)
        headline = {det: aggregate_metrics(wms) for det, wms in per_detector.items()}

        print(f"\n### {strat} — test windows covered: {sorted(covered)}")
        if not_covered:
            print(f"    NOT covered (no curve): {not_covered}")
        for c in curves_used:
            print(f"    curve: {c}")
        header = f"{'detector':16} {'recall':>7} {'precision':>10} {'mean_lat':>9} {'FPR/day':>9}"
        print(header)
        print("-" * len(header))
        strat_out = {}
        for det in detector_order:
            if det not in headline:
                continue
            h = headline[det]

            def _f(x, nd=3):
                return "n/a" if x is None or (isinstance(x, float) and pd.isna(x)) else f"{x:.{nd}f}"

            print(f"{det:16} {_f(h.recall, 2):>7} {_f(h.precision, 2):>10} "
                  f"{_f(h.mean_latency_days, 1):>9} {_f(h.false_positive_rate, 4):>9}")
            strat_out[det] = {
                "recall": h.recall, "precision": h.precision,
                "mean_latency_days": h.mean_latency_days,
                "false_positive_rate": h.false_positive_rate,
                "n_events": h.n_events, "n_detected": h.n_detected,
                "per_window": [asdict(m) for m in h.per_window],
            }
        summary[strat] = {
            "status": "ok" if covered else "data_pending",
            "covered": sorted(covered),
            "not_covered": not_covered,
            "curves": curves_used,
            "detectors": strat_out,
        }

    out = RESULTS / "classical_test_summary.json"
    out.write_text(json.dumps(summary, indent=2, default=str))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
