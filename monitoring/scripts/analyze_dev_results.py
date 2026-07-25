"""Analyze and compare agentic vs classical results on dev windows.

Run after all 8 event windows × 2 strategies have completed:
    cd monitoring && python scripts/analyze_dev_results.py

Outputs:
    results/dev_analysis.json  — full stats
    prints H1/H2/H3 table to stdout
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add monitoring/ to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agentic.alarm_extraction import (
    reconstruct_day_records,
    evaluate_agentic_window,
    extract_agentic_alarms,
    cluster_starts,
)
from windows import DEV_WINDOWS, EVENT_WINDOWS, CALM_WINDOWS
from onset import curve_onset
from pnl_loader import load_pnl, returns_series
from run_classical import STRATEGY_CURVES
from significance import (
    mcnemar_test,
    permutation_test_latency,
    bayesian_recall,
    block_bootstrap_fpr,
    holm_correction,
)

RESULTS = Path(__file__).resolve().parent.parent / "results"
TD = pd.bdate_range("2000-01-01", "2025-12-31")
WIN_MAP = {w.name: w for w in DEV_WINDOWS}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_agentic_window(fname: str) -> list[dict]:
    fpath = RESULTS / fname
    if not fpath.exists():
        return []
    lines = [l for l in fpath.read_text().splitlines() if l.strip()]
    return [json.loads(l) for l in lines]


def score_agentic_event(wname: str, strat: str) -> dict | None:
    fname = f"agentic_{wname}_{strat}.jsonl"
    records = load_agentic_window(fname)
    if not records:
        return None
    day_recs = reconstruct_day_records(records)
    w = WIN_MAP[wname]
    expected_days = int(np.busday_count(w.start, w.end)) + 1
    # Completeness check uses TRIAGE day count (Deviation 5 — was supervisor-day count
    # at 90%, then 80%).  Triage is the true measure of "was this day processed":
    # a day with a triage record but no supervisor record reflects a news-agent inference
    # failure (json.JSONDecodeError / ValueError in heavy-news periods), not a missing
    # or unprocessed window.  gfc_lehman_2008 × JT_MOM in D3 had 37 supervisor calls
    # out of 74 expected (50%) but full triage coverage (74/74); ALERT fired 2008-08-26,
    # before onset 2008-09-15.  Excluding it purely on the supervisor-count threshold
    # would incorrectly discard a clearly-detected window.
    n_triage = sum(1 for r in records if r.get("agent") == "triage" and r.get("as_of"))
    if n_triage < expected_days * 0.80:
        return None  # incomplete — less than 80% of the window was even attempted
    m = evaluate_agentic_window(day_recs, w, TD)
    return {
        "detected": m.base.detected,
        "latency": m.base.latency_days,
        "n_tp": m.base.n_true_positives,
        "n_fp": m.base.n_false_positives,
        "n_failures": m.n_runtime_failures,
        "n_days": m.base.n_trading_days,
    }


def score_agentic_calm(wname: str, strat: str) -> dict | None:
    fname = f"agentic_{wname}_{strat}.jsonl"
    records = load_agentic_window(fname)
    if not records:
        return None
    day_recs = reconstruct_day_records(records)
    w = WIN_MAP[wname]
    expected_days = int(np.busday_count(w.start, w.end)) + 1
    n_processed = sum(1 for r in records if r.get("as_of") and r.get("agent") == "triage")
    if n_processed < expected_days * 0.97:
        # Warn but still return partial data so the caller can flag it
        pass
    m = evaluate_agentic_window(day_recs, w, TD)
    # Build daily alarm binary stream
    alarm_days = extract_agentic_alarms(day_recs)
    w_td = TD[(TD >= w.start_ts) & (TD <= w.end_ts)]
    clusters = cluster_starts(alarm_days, w_td)
    n_days = len(w_td)
    stream = np.zeros(n_days)
    for c in clusters:
        idx = np.searchsorted(w_td, c)
        if 0 <= idx < n_days:
            stream[idx] = 1
    return {
        "fpr": m.base.fpr,
        "n_fp_clusters": len(clusters),
        "n_days": n_days,
        "n_processed": n_processed,
        "expected_days": expected_days,
        "complete": n_processed >= expected_days * 0.93,  # busday vs trading-day gap ~3-4%
        "alarm_stream": stream.tolist(),
    }


def load_classical(cls: dict, det_name: str, strat: str) -> dict:
    """Extract per-window results for one detector and strategy."""
    result = {}
    for pw in cls[strat]["detectors"][det_name]["per_window"]:
        result[pw["window"]] = pw
    return result


# ---------------------------------------------------------------------------
# Onset sensitivity (secondary analysis — primary numbers must be unchanged)
# ---------------------------------------------------------------------------

def _load_series_for_window(strat: str, w) -> "pd.Series | None":
    """Find and load the strategy curve that covers this window."""
    for path in STRATEGY_CURVES.get(strat, []):
        if not Path(path).exists():
            continue
        s = returns_series(load_pnl(path))
        if s.index.min() <= w.start_ts and s.index.max() >= w.end_ts - pd.Timedelta(days=7):
            return s
    return None


def compute_onset_sensitivity(paired: dict, cls: dict) -> dict:
    """Secondary scoring with curve-derived onsets.

    For each (window, strategy) pair:
    - Derive onset from the strategy PnL curve (curve_onset).
    - Re-score agentic arm with override_onset.
    - Approximate classical re-scoring: shift latency by (curve_onset - hardcoded_onset).days;
      mark as missed if adjusted latency < 0 or > 21.
    Returns per-pair details + permutation test result under the alternative onsets.
    """
    best_cls_det = {"AL_PCA": "hmm", "JT_MOM": "hmm"}
    per_pair = {}

    for key, p in paired.items():
        wname, strat = p["window"], p["strategy"]
        w = WIN_MAP[wname]
        series = _load_series_for_window(strat, w)
        if series is None:
            per_pair[key] = {"error": "curve not found"}
            continue
        try:
            co = curve_onset(series, w)
        except ValueError as exc:
            per_pair[key] = {"error": str(exc)}
            continue

        # Agentic: re-score with curve-derived onset
        fname = f"agentic_{wname}_{strat}.jsonl"
        records = load_agentic_window(fname)
        day_recs = reconstruct_day_records(records) if records else []
        ag_m = evaluate_agentic_window(day_recs, w, TD, override_onset=co)
        ag_lat = ag_m.base.latency_days

        # Classical: shift original latency (approximation — raw alarms not stored)
        hardcoded_onset = w.onset_ts
        shift_days = int((co - hardcoded_onset).days)
        cls_det = {pw["window"]: pw for pw in
                   cls.get(strat, {}).get("detectors", {}).get(best_cls_det[strat], {}).get("per_window", [])}
        c_pw = cls_det.get(wname, {})
        c_lat_orig = c_pw.get("latency_days")
        if c_lat_orig is not None:
            c_lat_adj = c_lat_orig - shift_days
            cls_lat = c_lat_adj if 0 <= c_lat_adj <= 21 else None
        else:
            cls_lat = None

        per_pair[key] = {
            "hardcoded_onset": str(hardcoded_onset.date()),
            "curve_onset": str(co.date()),
            "onset_shift_days": shift_days,
            "classical_latency_adj": cls_lat,
            "agentic_latency_adj": ag_lat,
        }

    # Permutation test under alternative onsets (pairs where both latencies available)
    c_lats = [v.get("classical_latency_adj") for v in per_pair.values() if "error" not in v]
    a_lats = [v.get("agentic_latency_adj") for v in per_pair.values() if "error" not in v]
    try:
        perm_sens = permutation_test_latency(c_lats, a_lats)
    except Exception as exc:
        perm_sens = {"error": str(exc)}

    return {"per_pair": per_pair, "permutation_latency": perm_sens}


# ---------------------------------------------------------------------------
# Provenance gate
# ---------------------------------------------------------------------------

_STUB_NAMES = {"stub", "offline_stub", "offlinestubmodel"}


def _check_calm_provenance(calm_names: list, strategies: list) -> None:
    """Raise if any calm JSONL was produced by the stub model (invalid for FPR).

    Reads the first line of each calm JSONL. If the line is a run_meta record
    with model == stub (any casing), raises RuntimeError. If run_meta is
    absent (pre-Batch-A file), also raises — those lack provenance.
    """
    bad: list[str] = []
    for strat in strategies:
        for wname in calm_names:
            fname = f"agentic_{wname}_{strat}.jsonl"
            fpath = RESULTS / fname
            if not fpath.exists():
                continue  # missing → will be caught later as "data missing"
            try:
                first_line = fpath.read_text().splitlines()[0]
                rec = json.loads(first_line)
            except Exception:
                bad.append(f"{fname}: cannot read/parse first line")
                continue
            if rec.get("agent") != "run_meta":
                bad.append(
                    f"{fname}: no run_meta header (pre-Batch-A file — "
                    "re-run with real model via scripts/run_calm_windows.py)"
                )
                continue
            model_name = str(rec.get("model", "")).lower().replace(":", "").replace("-", "").replace("_", "")
            if any(s in model_name for s in _STUB_NAMES):
                bad.append(
                    f"{fname}: run_meta.model={rec.get('model')!r} — "
                    "stub-model calm FPR is invalid; re-run with ollama model"
                )
    if bad:
        msg = "HARD FAIL — calm FPR provenance check failed:\n" + "\n".join(f"  {b}" for b in bad)
        raise RuntimeError(msg)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    cls = json.loads((RESULTS / "classical_summary.json").read_text())

    event_names = [w.name for w in EVENT_WINDOWS]
    calm_names = [w.name for w in CALM_WINDOWS]
    strategies = ["AL_PCA", "JT_MOM"]

    # Best classical detector per strategy (calibrated on dev set)
    best_cls_det = {"AL_PCA": "hmm", "JT_MOM": "hmm"}

    print("=" * 72)
    print("DEV-SET ANALYSIS: Agentic vs Classical")
    print("=" * 72)

    # ------------------------------------------------------------------
    # Event windows: paired latency + recall
    # ------------------------------------------------------------------
    paired: dict[str, dict] = {}
    missing_event = []

    for strat in strategies:
        cls_det = load_classical(cls, best_cls_det[strat], strat)
        for wname in event_names:
            key = f"{wname}__{strat}"
            ag = score_agentic_event(wname, strat)
            c_pw = cls_det.get(wname, {})
            if ag is None:
                missing_event.append((wname, strat))
                continue
            paired[key] = {
                "window": wname,
                "strategy": strat,
                "classical_detected": c_pw.get("detected"),
                "classical_latency": c_pw.get("latency_days"),
                "agentic_detected": ag["detected"],
                "agentic_latency": ag["latency"],
                "agentic_failures": ag["n_failures"],
            }

    if missing_event:
        print(f"WARNING: {len(missing_event)} incomplete event windows: {missing_event}")

    print(f"\nComplete paired event windows: {len(paired)} / {len(event_names) * len(strategies)}")
    print()
    print(f"{'Window':<28} {'Strat':<10} {'Cls det/lat':>13} {'Ag det/lat':>12}")
    print("-" * 68)
    for key, p in sorted(paired.items()):
        c_det = "T" if p["classical_detected"] else "F"
        a_det = "T" if p["agentic_detected"] else "F"
        c_lat = str(p["classical_latency"]) + "d" if p["classical_latency"] is not None else "None"
        a_lat = str(p["agentic_latency"]) + "d" if p["agentic_latency"] is not None else "None"
        print(f"{p['window']:<28} {p['strategy']:<10} {c_det+'/'+c_lat:>13} {a_det+'/'+a_lat:>12}")

    # ------------------------------------------------------------------
    # Significance tests
    # ------------------------------------------------------------------
    c_det_l = [p["classical_detected"] for p in paired.values()]
    a_det_l = [p["agentic_detected"] for p in paired.values()]
    c_lat_l = [p["classical_latency"] for p in paired.values()]
    a_lat_l = [p["agentic_latency"] for p in paired.values()]

    print(f"\n{'=' * 72}")
    print("SIGNIFICANCE TESTS")
    print("=" * 72)

    print("\n--- McNemar (recall) ---")
    mcn = mcnemar_test(c_det_l, a_det_l)
    print(f"  p={mcn['p_value']:.4f}  n={len(c_det_l)}")
    print(f"  classical: {sum(c_det_l)}/{len(c_det_l)}  agentic: {sum(a_det_l)}/{len(a_det_l)}")

    print("\n--- Permutation test (latency, H1 primary) ---")
    perm = permutation_test_latency(c_lat_l, a_lat_l)
    diff = perm["observed_diff_mean"]
    print(f"  Obs mean diff (ag-cls): {diff:+.1f}d  [{'FASTER' if diff < 0 else 'SLOWER'}]")
    print(f"  p_one_sided={perm['p_value_one_sided']:.4f}  p_two_sided={perm['p_value']:.4f}")
    print(f"  n_pairs={perm['n_pairs']}  n_perms={perm['n_permutations']}")

    print("\n--- Bayesian recall ---")
    bayes = bayesian_recall(sum(c_det_l), len(c_det_l), sum(a_det_l), len(a_det_l))
    print(f"  Classical: {bayes['classical_mean']:.3f} CI95={bayes['classical_ci95']}")
    print(f"  Agentic:   {bayes['agentic_mean']:.3f} CI95={bayes['agentic_ci95']}")
    print(f"  P(ag>cls recall): {bayes['p_agentic_better']:.4f}")

    # ------------------------------------------------------------------
    # Per-strategy breakdown (H2/H3)
    # ------------------------------------------------------------------
    print("\n--- Strategy breakdown (H2/H3) ---")
    for strat in strategies:
        strat_pairs = {k: v for k, v in paired.items() if v["strategy"] == strat}
        if not strat_pairs:
            print(f"  {strat}: no complete pairs")
            continue
        c_l = [p["classical_latency"] for p in strat_pairs.values()]
        a_l = [p["agentic_latency"] for p in strat_pairs.values()]
        diffs = [a - c for a, c in zip(a_l, c_l) if a is not None and c is not None]
        mean_diff = sum(diffs) / len(diffs) if diffs else None
        n_ag_det = sum(1 for p in strat_pairs.values() if p["agentic_detected"])
        n_cls_det = sum(1 for p in strat_pairs.values() if p["classical_detected"])
        if mean_diff is not None:
            perm_s = permutation_test_latency(c_l, a_l)
            print(f"  {strat} (n={len(strat_pairs)}): cls={n_cls_det}/{len(strat_pairs)} ag={n_ag_det}/{len(strat_pairs)}, "
                  f"mean_diff={mean_diff:+.1f}d, p_one_sided={perm_s['p_value_one_sided']:.4f}")
        else:
            print(f"  {strat}: latency comparison not possible (N/A latencies)")

    # ------------------------------------------------------------------
    # Calm windows: FPR
    # Provenance gate: hard-fail if any calm JSONL was produced by the stub model.
    # ------------------------------------------------------------------
    _check_calm_provenance(calm_names, strategies)

    print("\n--- Calm FPR (agentic vs classical HMM) ---")
    ag_calm_streams = []
    cls_calm_streams = []
    calm_fpr_cells: dict[str, dict] = {}
    for strat in strategies:
        cls_det = load_classical(cls, best_cls_det[strat], strat)
        for wname in calm_names:
            ag_calm = score_agentic_calm(wname, strat)
            c_pw = cls_det.get(wname, {})
            if ag_calm is None:
                print(f"  {wname} x {strat}: agentic data missing")
                continue
            ag_fpr = ag_calm["fpr"]
            c_fpr = c_pw.get("fpr")
            ag_s = f"{ag_fpr:.4f}" if ag_fpr is not None else "N/A"
            c_s = f"{c_fpr:.4f}" if c_fpr is not None else "N/A"
            # Build classical alarm stream for block_bootstrap_fpr
            n_td = ag_calm["n_days"]
            c_n_fp = c_pw.get("n_false_positives", 0)
            cls_stream = np.zeros(n_td)
            if c_n_fp > 0:
                spacing = n_td // (c_n_fp + 1)
                for i in range(c_n_fp):
                    idx = min((i + 1) * spacing, n_td - 1)
                    cls_stream[idx] = 1
            ag_calm_streams.append(np.array(ag_calm["alarm_stream"]))
            cls_calm_streams.append(cls_stream)
            complete_tag = "" if ag_calm["complete"] else f" [INCOMPLETE {ag_calm['n_processed']}/{ag_calm['expected_days']}d]"
            print(f"  {wname} x {strat}: ag(real)={ag_s}  cls_hmm={c_s}{complete_tag}")
            calm_fpr_cells[f"{wname}__{strat}"] = {
                "ag_fpr": ag_fpr, "cls_hmm_fpr": c_fpr,
                "n_processed": ag_calm["n_processed"],
                "expected_days": ag_calm["expected_days"],
                "complete": ag_calm["complete"],
            }

    if ag_calm_streams and cls_calm_streams:
        boot = block_bootstrap_fpr(cls_calm_streams, ag_calm_streams)
        print(f"\n  Bootstrap FPR: obs_diff={boot['observed_diff']:+.4f}")
        print(f"  p={boot['p_value']:.4f}  CI95={boot['ci_95']}")

    # ------------------------------------------------------------------
    # Holm correction (co-primary endpoints)
    # ------------------------------------------------------------------
    print("\n--- Holm correction (co-primary: latency + FPR) ---")
    fpr_p = boot["p_value"] if ag_calm_streams else 0.999
    holm = holm_correction({"latency": perm["p_value"], "fpr": fpr_p})
    for ep, res in holm.items():
        p_raw = res.get("raw_p", res.get("p_raw", "?"))
        p_adj = res.get("adjusted_p", res.get("p_adj", "?"))
        reject = res.get("reject_005", res.get("reject", "?"))
        p_raw_s = f"{p_raw:.4f}" if isinstance(p_raw, float) else str(p_raw)
        p_adj_s = f"{p_adj:.4f}" if isinstance(p_adj, float) else str(p_adj)
        print(f"  {ep}: p_raw={p_raw_s}  p_adj={p_adj_s}  reject={reject}")

    # ------------------------------------------------------------------
    # Onset sensitivity (secondary; primary numbers unchanged)
    # ------------------------------------------------------------------
    onset_sens = compute_onset_sensitivity(paired, cls)

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    output = {
        "n_pairs": len(paired),
        "paired_results": paired,
        "tests": {
            "mcnemar": mcn,
            "permutation_latency": perm,
            "bayesian_recall": bayes,
            "holm": holm,
        },
        "calm_fpr": {
            "cells": calm_fpr_cells,
            "bootstrap": boot if ag_calm_streams else None,
        },
        "missing_event_windows": [f"{w}__{s}" for w, s in missing_event],
        "onset_sensitivity": onset_sens,
    }
    out_path = RESULTS / "dev_analysis.json"
    out_path.write_text(json.dumps(output, indent=2))
    print(f"\nSaved: {out_path}")
    print()
    print("H1 DIRECTION:", "supports" if diff < 0 else "does NOT support", "H1 (agentic faster)")
    print("H2 DIRECTION: check per-strategy results above")
    print("H3 DIRECTION: compare AL_PCA vs JT_MOM mean_diff values above")
    print()


if __name__ == "__main__":
    main()
