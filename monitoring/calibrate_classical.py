"""Classical detector grid calibration (preregistration §6.1).

Searches {0.5×, 1×, 2×} of each detector's key sensitivity parameters on the
**development set only** (DEV_WINDOWS — never TEST_WINDOWS).

Objective: J = recall − 50 × FPR/day, tie-break by lower mean censored latency.

Outputs:
  results/calibration_grid.json — best config per detector per strategy, full grid,
                                   and shared (pooled) config for H3 analysis.

Usage:
    python calibrate_classical.py [--strategy AL_PCA|JT_MOM|both]
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from itertools import product

import pandas as pd

from pnl_loader import load_pnl, returns_series
from windows import DEV_WINDOWS, Window
from detectors import PageHinkley, BOCPD, HMMDetector, DistributionalThreshold
from metrics import evaluate_window, aggregate_metrics, WindowMetrics, HeadlineMetrics
from run_classical import STRATEGY_CURVES, _hmm_training_returns, _n_trading_days

RESULTS = Path(__file__).resolve().parent / "results"

# ---------------------------------------------------------------------------
# Parameter grids (§6.1: {0.5×, 1×, 2×} of current repo defaults)
# ---------------------------------------------------------------------------

_MULTIPLIERS = (0.5, 1.0, 2.0)

DETECTOR_GRIDS: dict[str, list[dict]] = {
    "page_hinkley": [
        {"delta": round(0.5 * d, 4), "lambda_": round(8.0 * l, 4)}
        for d, l in product(_MULTIPLIERS, _MULTIPLIERS)
    ],
    "bocpd": [
        {"hazard": round(1 / 250 * h, 6), "min_run": max(1, round(25 * r))}
        for h, r in product(_MULTIPLIERS, _MULTIPLIERS)
    ],
    "hmm": [
        {"threshold": round(0.5 * t, 4)}
        for t in _MULTIPLIERS
    ],
    "distributional": [
        {"z_threshold": round(3.0 * z, 4), "vol_threshold": round(2.0 * v, 4)}
        for z, v in product(_MULTIPLIERS, _MULTIPLIERS)
    ],
}

# Human-readable multiplier labels (for each detector, in grid order)
_MULTIPLIER_LABELS: dict[str, list[dict]] = {
    "page_hinkley": [
        {"delta": d, "lambda_": l}
        for d, l in product(_MULTIPLIERS, _MULTIPLIERS)
    ],
    "bocpd": [
        {"hazard": h, "min_run": r}
        for h, r in product(_MULTIPLIERS, _MULTIPLIERS)
    ],
    "hmm": [{"threshold": t} for t in _MULTIPLIERS],
    "distributional": [
        {"z_threshold": z, "vol_threshold": v}
        for z, v in product(_MULTIPLIERS, _MULTIPLIERS)
    ],
}


@dataclass
class GridResult:
    detector: str
    params: dict
    multipliers: dict
    strategy: str
    J: float              # recall − 50 × FPR/day
    recall: float
    fpr_per_day: float
    mean_latency: float | None
    per_window: list[dict] = field(default_factory=list)  # serialisable WindowMetrics


def _J(recall: float, fpr: float) -> float:
    """Objective function J = recall − 50 × FPR/day."""
    if pd.isna(recall) or pd.isna(fpr):
        return float("-inf")
    return recall - 50.0 * fpr


def _make_detector(det_name: str, params: dict, hmm_train=None):
    """Instantiate a detector by name with the given params."""
    if det_name == "page_hinkley":
        return PageHinkley(**params)
    if det_name == "bocpd":
        return BOCPD(**params)
    if det_name == "hmm":
        return HMMDetector(train_returns=hmm_train, **params)
    if det_name == "distributional":
        return DistributionalThreshold(**params)
    raise ValueError(f"unknown detector {det_name!r}")


def _windows_in_curve(series: pd.Series, end_tol_days: int = 7) -> list[Window]:
    lo, hi = series.index.min(), series.index.max()
    tol = pd.Timedelta(days=end_tol_days)
    return [w for w in DEV_WINDOWS if w.start_ts >= lo and w.end_ts <= hi + tol]


def evaluate_config(
    det_name: str,
    params: dict,
    multipliers: dict,
    curve_paths: list,
    hmm_train=None,
    strategy: str = "",
) -> GridResult:
    """Run one detector config across all curves and score all covered DEV_WINDOWS."""
    all_window_metrics: dict[str, WindowMetrics] = {}

    for path in curve_paths:
        if not Path(path).exists():
            continue
        series = returns_series(load_pnl(path))
        windows = _windows_in_curve(series)
        if not windows:
            continue
        det = _make_detector(det_name, params, hmm_train=hmm_train)
        result = det.detect(series)
        for w in windows:
            n_td = _n_trading_days(series, w)
            wm = evaluate_window(result.alarms, w, tolerance_days=21, n_trading_days=n_td)
            if w.name not in all_window_metrics:
                all_window_metrics[w.name] = wm

    wm_list = list(all_window_metrics.values())
    headline = aggregate_metrics(wm_list) if wm_list else HeadlineMetrics(
        n_events=0, n_detected=0, recall=float("nan"), precision=float("nan"),
        mean_latency_days=None, false_positive_rate=float("nan"),
        n_false_positives_calm=0, per_window=[],
    )

    j = _J(headline.recall, headline.false_positive_rate)
    return GridResult(
        detector=det_name,
        params=params,
        multipliers=multipliers,
        strategy=strategy,
        J=j,
        recall=headline.recall,
        fpr_per_day=headline.false_positive_rate,
        mean_latency=headline.mean_latency_days,
        per_window=[asdict(m) for m in wm_list],
    )


def calibrate_strategy(strategy: str) -> dict[str, GridResult]:
    """Run the full 30-config grid for one strategy on DEV_WINDOWS.

    Returns {detector_name: best_GridResult}.
    """
    curve_paths = STRATEGY_CURVES[strategy]
    hmm_train = _hmm_training_returns(curve_paths)
    best: dict[str, GridResult] = {}
    full_grid: dict[str, list[GridResult]] = {}

    for det_name, param_list in DETECTOR_GRIDS.items():
        mult_list = _MULTIPLIER_LABELS[det_name]
        results: list[GridResult] = []
        for params, mults in zip(param_list, mult_list):
            r = evaluate_config(det_name, params, mults, curve_paths,
                                hmm_train=hmm_train, strategy=strategy)
            results.append(r)

        full_grid[det_name] = results

        # Best = max J; tie-break by lower mean latency (lower is better).
        def _sort_key(r: GridResult):
            lat = r.mean_latency if r.mean_latency is not None else 21.0
            return (-r.J, lat)

        best[det_name] = min(results, key=_sort_key)  # min because negated J

    return best, full_grid


def _shared_config(al_best: dict, jt_best: dict) -> dict:
    """Pick the shared config with best average J across both strategies."""
    shared = {}
    for det_name in DETECTOR_GRIDS:
        al = al_best.get(det_name)
        jt = jt_best.get(det_name)
        if al is None and jt is None:
            continue
        if al is None:
            winner = jt
        elif jt is None:
            winner = al
        else:
            winner = al if al.J >= jt.J else jt
        shared[det_name] = {
            "params": winner.params,
            "multipliers": winner.multipliers,
            "J_AL_PCA": al.J if al else None,
            "J_JT_MOM": jt.J if jt else None,
            "J_avg": float(
                (al.J if al else 0) / (1 if al else 0.5) +
                (jt.J if jt else 0) / (1 if jt else 0.5)
            ) if al and jt else (al.J if al else jt.J),
        }
    return shared


def calibrate_all(strategies: list[str] | None = None) -> dict:
    """Run calibration for all requested strategies and build the output structure."""
    if strategies is None:
        strategies = list(STRATEGY_CURVES.keys())

    out: dict = {}
    best_by_strategy: dict[str, dict[str, GridResult]] = {}

    for strat in strategies:
        missing = [str(p) for p in STRATEGY_CURVES[strat] if not Path(p).exists()]
        if len(missing) == len(STRATEGY_CURVES[strat]):
            print(f"[{strat}] no curves found — skipping")
            continue
        print(f"[{strat}] running calibration grid...")
        best, full_grid = calibrate_strategy(strat)
        best_by_strategy[strat] = best

        out[strat] = {
            "best_per_detector": {
                det: {
                    "params": r.params,
                    "multipliers": r.multipliers,
                    "J": round(r.J, 6),
                    "recall": round(r.recall, 4) if not pd.isna(r.recall) else None,
                    "fpr_per_day": round(r.fpr_per_day, 6) if not pd.isna(r.fpr_per_day) else None,
                    "mean_latency": round(r.mean_latency, 2) if r.mean_latency is not None else None,
                }
                for det, r in best.items()
            },
            "full_grid": {
                det: [
                    {
                        "params": r.params,
                        "multipliers": r.multipliers,
                        "J": round(r.J, 6),
                        "recall": round(r.recall, 4) if not pd.isna(r.recall) else None,
                        "fpr_per_day": round(r.fpr_per_day, 6) if not pd.isna(r.fpr_per_day) else None,
                        "mean_latency": round(r.mean_latency, 2) if r.mean_latency is not None else None,
                    }
                    for r in results
                ]
                for det, results in full_grid.items()
            },
        }
        # Print summary
        print(f"  {'detector':20} {'J':>8} {'recall':>7} {'FPR/day':>10} {'latency':>9} {'params'}")
        for det, r in best.items():
            lat = f"{r.mean_latency:.1f}" if r.mean_latency else "n/a"
            rec = f"{r.recall:.2f}" if not pd.isna(r.recall) else "n/a"
            fpr = f"{r.fpr_per_day:.4f}" if not pd.isna(r.fpr_per_day) else "n/a"
            print(f"  {det:20} {r.J:8.4f} {rec:>7} {fpr:>10} {lat:>9}  {r.params}")

    if len(best_by_strategy) == 2:
        out["shared_config"] = _shared_config(
            best_by_strategy.get("AL_PCA", {}),
            best_by_strategy.get("JT_MOM", {}),
        )

    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--strategy", default="both",
                    choices=list(STRATEGY_CURVES) + ["both"],
                    help="strategy to calibrate (default: both)")
    args = ap.parse_args()

    strategies = list(STRATEGY_CURVES.keys()) if args.strategy == "both" else [args.strategy]
    RESULTS.mkdir(parents=True, exist_ok=True)
    out = calibrate_all(strategies)

    out_path = RESULTS / "calibration_grid.json"
    out_path.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
