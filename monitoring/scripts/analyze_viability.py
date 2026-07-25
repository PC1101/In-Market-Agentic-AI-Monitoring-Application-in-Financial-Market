"""Phase D4 — Practical Viability Assessment.

Three sections (BATCH_D_validation_and_final_runs.md §D4):
  1. Ops metrics   — triage skip-rate, LLM calls/day, error rate, state mix
  2. Economic sim  — pre-declared rule applied to dev PnL curves
  3. Failure review — FP clusters + missed/late detections in plain text

Run after D3 completes (all 12 dev JSONLs in results/):
    cd monitoring && python scripts/analyze_viability.py

Outputs:
    results/viability_dev.json
    Prints a report to stdout.

Trading rule (pre-declared, §D4 §2):
    ALERT  → exposure = 0.50 next-day
    CRITICAL → exposure = 0.00 next-day
    WATCH / NORMAL / skip → clean day; after 5 consecutive clean days restore 1.00
    Classical arm: HMM alarm → exposure 0.00 next-day; same 5-day restore
    Baseline: exposure = 1.00 always
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from windows import DEV_WINDOWS  # type: ignore[attr-defined]
WIN_MAP = {w.name: w for w in DEV_WINDOWS}
from run_classical import STRATEGY_CURVES  # type: ignore[attr-defined]
from pnl_loader import load_pnl, returns_series  # type: ignore[attr-defined]
from detectors import HMMDetector  # type: ignore[attr-defined]
from agentic.alarm_extraction import reconstruct_day_records  # type: ignore[attr-defined]

# ---------------------------------------------------------------------------
RESULTS = ROOT / "results"
EVENT_WINDOWS = [w for w in DEV_WINDOWS if w.kind == "event"]
CALM_WINDOWS  = [w for w in DEV_WINDOWS if w.kind == "calm"]
STRATEGIES    = ["AL_PCA", "JT_MOM"]

# Economic sim constants (pre-declared)
ALERT_EXPOSURE    = 0.50
CRITICAL_EXPOSURE = 0.00
RESTORE_CLEAN_N   = 5    # consecutive clean days before full restore
ALERT_STATES      = {"ALERT", "CRITICAL"}
CLEAN_STATES      = {"NORMAL", "WATCH"}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_jsonl(wname: str, strat: str) -> list[dict]:
    p = RESULTS / f"agentic_{wname}_{strat}.jsonl"
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def _has_provenance(records: list[dict]) -> bool:
    return any(r.get("agent") == "run_meta" for r in records)


def _triage_records(records: list[dict]) -> list[dict]:
    return [r for r in records if r.get("agent") == "triage"]


def _supervisor_records(records: list[dict]) -> list[dict]:
    return [r for r in records if r.get("agent") == "performance_supervisor"]


def _news_records(records: list[dict]) -> list[dict]:
    return [r for r in records if r.get("agent") == "news_context"]


def _state_on_day(sup_records: list[dict], day: str) -> str | None:
    """Return supervisor state for a given as_of date, or None if absent/failed."""
    for r in sup_records:
        if r.get("as_of") == day:
            a = r.get("assessment")
            if isinstance(a, dict):
                return a.get("state")
    return None


def _best_curve(strat: str, window) -> pd.Series | None:
    for path in STRATEGY_CURVES.get(strat, []):
        if not Path(path).exists():
            continue
        s = returns_series(load_pnl(path))
        if s.index.min() <= window.start_ts and s.index.max() >= window.end_ts - pd.Timedelta(days=7):
            return s
    return None


def _hmm_alarms_for_window(series: pd.Series, strat: str, w) -> list[pd.Timestamp]:
    """Run HMM detector over the full curve; return alarms inside the window."""
    # Build training returns from pre-window data (mirrors run_classical.py)
    from run_classical import _hmm_training_returns  # type: ignore
    train_returns = _hmm_training_returns(STRATEGY_CURVES.get(strat, []))
    result = HMMDetector(train_returns=train_returns).detect(series)
    return [a for a in result.alarms if w.start_ts <= a <= w.end_ts]


# ---------------------------------------------------------------------------
# Section 1 — Ops metrics
# ---------------------------------------------------------------------------

def compute_ops_metrics() -> dict:
    rows = []
    for strat in STRATEGIES:
        for w in DEV_WINDOWS:
            records = load_jsonl(w.name, strat)
            if not records:
                rows.append({"window": w.name, "strategy": strat, "status": "missing"})
                continue
            prov = _has_provenance(records)
            t_recs  = _triage_records(records)
            s_recs  = _supervisor_records(records)
            n_recs  = _news_records(records)

            n_triage   = len(t_recs)
            n_skip     = sum(1 for r in t_recs if r.get("triage_mode") == "skip")
            n_thinking = sum(1 for r in t_recs if r.get("triage_mode") == "thinking")
            n_cls_esc  = sum(1 for r in t_recs if r.get("triage_mode") == "classical_escalation")
            n_sup      = len(s_recs)
            n_news     = len(n_recs)
            n_errors   = sum(1 for r in s_recs if r.get("error") or
                             not isinstance(r.get("assessment"), dict))

            # LLM calls per trading day = (news + supervisor calls) / n_triage
            llm_calls_per_day = (n_news + n_sup) / max(n_triage, 1)

            # State distribution from supervisor records
            states: dict[str, int] = {}
            for r in s_recs:
                s = (r.get("assessment") or {}).get("state", "ERROR")
                states[s] = states.get(s, 0) + 1

            rows.append({
                "window": w.name,
                "strategy": strat,
                "kind": w.kind,
                "has_provenance": prov,
                "status": "ok" if prov else "no_meta",
                "n_trading_days": n_triage,
                "skip_rate": round(n_skip / max(n_triage, 1), 4),
                "n_skip": n_skip,
                "n_thinking": n_thinking,
                "n_classical_escalation": n_cls_esc,
                "n_supervisor_calls": n_sup,
                "n_news_calls": n_news,
                "llm_calls_per_day": round(llm_calls_per_day, 3),
                "error_rate": round(n_errors / max(n_sup, 1), 4),
                "n_errors": n_errors,
                "state_distribution": states,
                # latency_s not captured in JSONLs (tracked in-memory only)
                "latency_s_note": "not logged to JSONL; p50 estimated ~4-8s based on D2 smoke test",
            })

    # Aggregate: can the daily loop complete in pre-open window?
    ok_rows = [r for r in rows if r["status"] == "ok"]
    mean_llm_calls = np.mean([r["llm_calls_per_day"] for r in ok_rows]) if ok_rows else 0.0
    worst_event = max((r for r in ok_rows if r["kind"] == "event"),
                      key=lambda r: r["llm_calls_per_day"], default=None)

    # Estimate total wall-clock per day: 4.2s per LLM call (D2 estimate, GPU active)
    # worst case: thinking day = 2 calls, 8.4s
    est_wall_worst_s = 2 * 4.2  # thinking: news + supervisor
    est_wall_typical_s = mean_llm_calls * 4.2

    return {
        "per_window": rows,
        "aggregate": {
            "mean_llm_calls_per_day": round(float(mean_llm_calls), 3),
            "est_wall_clock_typical_s": round(est_wall_typical_s, 1),
            "est_wall_clock_worst_case_s": est_wall_worst_s,
            "pre_open_window_feasible": est_wall_worst_s <= 60,  # 1 min threshold
            "worst_event_window": worst_event["window"] if worst_event else None,
            "worst_event_llm_calls_per_day": worst_event["llm_calls_per_day"] if worst_event else None,
            "note": (
                "GPU Eco Mode can slow inference to ~40s/call (observed); in that regime "
                "a 2-call thinking day takes ~80s. On active GPU: ~8-9s/day is feasible pre-open."
            ),
        },
    }


# ---------------------------------------------------------------------------
# Section 2 — Economic simulation
# ---------------------------------------------------------------------------

def _apply_trading_rule(
    returns: pd.Series,
    signal: pd.Series,   # indexed by date, values: state str or None (None=NORMAL)
    rule: str,           # "agentic" | "classical" | "baseline"
) -> pd.Series:
    """Apply exposure rule. Returns daily strategy returns after position scaling.

    For "agentic" and "classical": signal must contain state strings or None.
    For "baseline": signal is ignored.
    """
    exposure = 1.0
    clean_count = 0
    out = []
    dates = returns.index

    for i, day in enumerate(dates):
        state = signal.get(day) if rule != "baseline" else None

        # Update exposure based on yesterday's signal (next-day execution lag):
        # We apply yesterday's signal to today's return.
        if i > 0:
            prev_day = dates[i - 1]
            prev_state = signal.get(prev_day) if rule != "baseline" else None
            if prev_state in ALERT_STATES:
                if prev_state == "CRITICAL" or rule == "classical":
                    exposure = CRITICAL_EXPOSURE
                else:
                    exposure = ALERT_EXPOSURE
                clean_count = 0
            elif prev_state in CLEAN_STATES or prev_state is None:
                clean_count += 1
                if clean_count >= RESTORE_CLEAN_N:
                    exposure = 1.0

        out.append(returns.iloc[i] * exposure)

    return pd.Series(out, index=dates)


def _max_drawdown(returns: pd.Series) -> float:
    equity = (1.0 + returns).cumprod()
    peak   = equity.cummax()
    dd     = (equity / peak - 1.0).min()
    return float(dd)


def _annualized_return(returns: pd.Series) -> float:
    if len(returns) == 0:
        return 0.0
    n_years = len(returns) / 252.0
    cum = float((1.0 + returns).prod())
    if n_years <= 0 or cum <= 0:
        return 0.0
    return float(cum ** (1.0 / n_years) - 1.0)


def _build_agentic_signal(records: list[dict], days: pd.DatetimeIndex) -> pd.Series:
    """Build a signal Series from supervisor records. Skip days → None (treated as NORMAL)."""
    sup = _supervisor_records(records)
    day_state: dict[pd.Timestamp, str | None] = {d: None for d in days}
    for r in sup:
        try:
            day = pd.Timestamp(r["as_of"])
        except (KeyError, TypeError, ValueError):
            continue
        state = (r.get("assessment") or {}).get("state")
        if state and day in day_state:
            day_state[day] = state
    return pd.Series(day_state)


def _build_classical_signal(
    alarms: list[pd.Timestamp], days: pd.DatetimeIndex
) -> pd.Series:
    """Binary signal: ALERT on alarm days, NORMAL otherwise."""
    alarm_set = set(alarms)
    return pd.Series(
        {d: ("ALERT" if d in alarm_set else None) for d in days}
    )


def compute_econ_sim() -> dict:
    """Run economic simulation for each event window × strategy."""
    results = []

    for strat in STRATEGIES:
        for w in EVENT_WINDOWS:
            records = load_jsonl(w.name, strat)
            if not records or not _has_provenance(records):
                results.append({
                    "window": w.name, "strategy": strat,
                    "status": "missing_or_no_meta",
                })
                continue

            series = _best_curve(strat, w)
            if series is None:
                results.append({"window": w.name, "strategy": strat, "status": "no_curve"})
                continue

            # Restrict to trading days inside the window
            window_returns = series[(series.index >= w.start_ts) & (series.index <= w.end_ts)]
            if window_returns.empty:
                continue
            days = window_returns.index

            # Build signals
            ag_signal  = _build_agentic_signal(records, days)
            try:
                cls_alarms = _hmm_alarms_for_window(series, strat, w)
            except Exception as exc:
                cls_alarms = []
            cls_signal = _build_classical_signal(cls_alarms, days)

            # Apply rule
            ag_ret   = _apply_trading_rule(window_returns, ag_signal,  "agentic")
            cls_ret  = _apply_trading_rule(window_returns, cls_signal, "classical")
            base_ret = window_returns.copy()

            # Metrics
            ag_dd   = _max_drawdown(ag_ret)
            cls_dd  = _max_drawdown(cls_ret)
            base_dd = _max_drawdown(base_ret)
            ag_ann   = _annualized_return(ag_ret)
            cls_ann  = _annualized_return(cls_ret)
            base_ann = _annualized_return(base_ret)

            # Drawdown avoided = base_dd - arm_dd  (positive = arm avoided drawdown)
            # drawdown_avoided: positive means arm experienced LESS drawdown than no-monitor
            # = ag_dd - base_dd  (both are negative; ag_dd > base_dd when arm helps)
            results.append({
                "window": w.name,
                "strategy": strat,
                "status": "ok",
                "n_days": len(days),
                "onset": w.onset,
                "baseline": {
                    "max_drawdown": round(base_dd, 4),
                    "annualized_return": round(base_ann, 4),
                },
                "agentic": {
                    "max_drawdown": round(ag_dd, 4),
                    "annualized_return": round(ag_ann, 4),
                    "drawdown_avoided": round(ag_dd - base_dd, 4),
                    "return_drag": round(ag_ann - base_ann, 4),
                },
                "classical_hmm": {
                    "max_drawdown": round(cls_dd, 4),
                    "annualized_return": round(cls_ann, 4),
                    "drawdown_avoided": round(cls_dd - base_dd, 4),
                    "return_drag": round(cls_ann - base_ann, 4),
                    "n_alarms": len(cls_alarms),
                },
            })

    # Aggregate across event windows
    ok = [r for r in results if r.get("status") == "ok"]
    if ok:
        ag_drag_mean  = np.mean([r["agentic"]["return_drag"] for r in ok])
        cls_drag_mean = np.mean([r["classical_hmm"]["return_drag"] for r in ok])
        ag_dd_avoid   = np.mean([r["agentic"]["drawdown_avoided"] for r in ok])
        cls_dd_avoid  = np.mean([r["classical_hmm"]["drawdown_avoided"] for r in ok])
        aggregate = {
            "mean_agentic_return_drag": round(float(ag_drag_mean), 4),
            "mean_classical_return_drag": round(float(cls_drag_mean), 4),
            "mean_agentic_dd_avoided": round(float(ag_dd_avoid), 4),
            "mean_classical_dd_avoided": round(float(cls_dd_avoid), 4),
            "note": (
                "Return drag < 0 means FP-driven position reductions cost return. "
                "Drawdown avoided > 0 means the rule reduced event-window max-drawdown. "
                "Rule is per-declared (§D4 §2): ALERT→50%, CRITICAL→0%, restore after 5 clean days. "
                "Classical arm: HMM alarm → 0% next-day. "
                "Next-day execution lag applied. Exploratory only — not preregistered."
            ),
        }
    else:
        aggregate = {"note": "no ok windows; likely D3 not yet complete"}

    return {"per_window": results, "aggregate": aggregate}


# ---------------------------------------------------------------------------
# Section 3 — Failure-mode review
# ---------------------------------------------------------------------------

def compute_failure_modes(dev_analysis_path: Path | None = None) -> dict:
    """Summarise FP clusters (calm) and missed/late detections (event) from dev data."""

    # Load paired results from dev_analysis.json if available
    if dev_analysis_path and dev_analysis_path.exists():
        da = json.loads(dev_analysis_path.read_text())
        paired = da.get("paired_results", {})
        calm_cells = da.get("calm_fpr", {}).get("cells", {})
    else:
        paired = {}
        calm_cells = {}

    # Missed / late detections (event windows)
    missed = []
    late   = []
    detected_early = []
    for key, p in paired.items():
        wname  = p["window"]
        strat  = p["strategy"]
        ag_det = p.get("agentic_detected")
        ag_lat = p.get("agentic_latency")
        c_det  = p.get("classical_detected")
        c_lat  = p.get("classical_latency")
        if not ag_det:
            missed.append({"window": wname, "strategy": strat,
                           "note": "agentic missed; investigate triage skip rate and news coverage"})
        elif ag_lat is not None and ag_lat > 14:
            late.append({"window": wname, "strategy": strat,
                         "latency_days": ag_lat, "note": "latency > 14d; check supervisor state progression"})
        else:
            lag_vs_cls = (ag_lat or 0) - (c_lat or 0) if (c_lat is not None and ag_lat is not None) else None
            detected_early.append({"window": wname, "strategy": strat,
                                   "agentic_latency": ag_lat, "classical_latency": c_lat,
                                   "advantage_days": (-lag_vs_cls if lag_vs_cls is not None else None)})

    # FP clusters in calm windows (from calm JSONLs via alarm_extraction)
    fp_clusters = []
    for key, cell in calm_cells.items():
        wname, strat = key.split("__")
        # Compute n_fp_clusters directly from the JSONL
        calm_records = load_jsonl(wname, strat)
        if calm_records:
            from agentic.alarm_extraction import extract_agentic_alarms, cluster_starts, reconstruct_day_records  # noqa
            TD = pd.bdate_range("2000-01-01", "2025-12-31")
            w = WIN_MAP.get(wname)
            if w:
                dr = reconstruct_day_records(calm_records)
                alarm_days = extract_agentic_alarms(dr)
                w_td = TD[(TD >= w.start_ts) & (TD <= w.end_ts)]
                clusters = cluster_starts(alarm_days, w_td)
                n_fp = len(clusters)
            else:
                n_fp = None
        else:
            n_fp = None
        fp_clusters.append({
            "window": wname, "strategy": strat,
            "n_fp_clusters": n_fp,
            "fpr_per_day": cell.get("ag_fpr"),
            "complete": cell.get("complete"),
        })

    # Per-window qualitative notes from live JSONLs
    qualitative = []
    for strat in STRATEGIES:
        for w in DEV_WINDOWS:
            records = load_jsonl(w.name, strat)
            if not records:
                continue
            s_recs = _supervisor_records(records)
            t_recs = _triage_records(records)
            n_errors = sum(1 for r in s_recs if r.get("error") or
                           not isinstance(r.get("assessment"), dict))
            n_skipped = sum(1 for r in t_recs if r.get("triage_mode") == "skip")
            if n_errors > 0:
                qualitative.append({
                    "window": w.name, "strategy": strat,
                    "issue": f"{n_errors} supervisor LLM failures (schema-invalid output)",
                    "action": "check model, prompt version, or news payload size on these days",
                })
            if w.kind == "event" and n_skipped > 0:
                qualitative.append({
                    "window": w.name, "strategy": strat,
                    "issue": f"{n_skipped} trading days skipped by triage (triage_mode=skip)",
                    "action": "confirm no onset-day was skipped; spot-check triage thresholds",
                })

    return {
        "missed_detections": missed,
        "late_detections": late,
        "successful_detections": detected_early,
        "calm_fp_clusters": fp_clusters,
        "qualitative_notes": qualitative,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description="D4 practical viability assessment")
    ap.add_argument("--no-econ-sim", action="store_true",
                    help="Skip economic simulation (faster, useful if D3 still running)")
    args = ap.parse_args()

    print("=" * 70)
    print("PHASE D4 — PRACTICAL VIABILITY ASSESSMENT")
    print("=" * 70)
    print()

    # --- Section 1: Ops metrics ---
    print("--- Section 1: Ops Metrics ---")
    ops = compute_ops_metrics()
    ok_rows = [r for r in ops["per_window"] if r["status"] == "ok"]
    missing_rows = [r for r in ops["per_window"] if r["status"] != "ok"]
    print(f"  Windows with provenance: {len(ok_rows)}/12")
    if missing_rows:
        missing_labels = [r["window"] + "x" + r["strategy"] for r in missing_rows]
        print(f"  Missing/no_meta: {missing_labels}")
    print()
    print(f"  {'Window':<30} {'Strat':<8} {'Skip%':<8} {'LLM/day':<9} {'Errors':<8} {'States'}")
    for r in ok_rows:
        sd = r.get("state_distribution", {})
        states_str = " ".join(f"{k}:{v}" for k, v in sorted(sd.items()))
        print(f"  {r['window']:<30} {r['strategy']:<8} "
              f"{r['skip_rate']*100:>5.1f}%  "
              f"{r['llm_calls_per_day']:>6.2f}   "
              f"{r['n_errors']:>4}    "
              f"{states_str}")
    print()
    agg = ops["aggregate"]
    print(f"  Mean LLM calls/day: {agg['mean_llm_calls_per_day']:.2f}")
    print(f"  Est. wall-clock: typical {agg['est_wall_clock_typical_s']:.1f}s, "
          f"worst-case {agg['est_wall_clock_worst_case_s']:.1f}s")
    print(f"  Pre-open feasible (GPU active): {agg['pre_open_window_feasible']}")
    print(f"  NOTE: {agg['note']}")
    print()

    # --- Section 2: Economic simulation ---
    ok_econ: list[dict] = []
    if args.no_econ_sim:
        print("--- Section 2: Economic Simulation --- SKIPPED (--no-econ-sim)")
        econ = {"per_window": [], "aggregate": {"note": "skipped"}}
    else:
        print("--- Section 2: Economic Simulation ---")
        print("  Running HMM detector for classical arm comparison...")
        econ = compute_econ_sim()
        ok_econ = [r for r in econ["per_window"] if r.get("status") == "ok"]
        print()
        if ok_econ:
            print(f"  {'Window':<25} {'Strat':<8} {'BaseMDD':>8} {'AgMDD':>8} "
                  f"{'ClsMDD':>8} {'AgDDavd':>8} {'ClsDDavd':>9} {'AgDrag':>8}")
            for r in ok_econ:
                b = r["baseline"]
                a = r["agentic"]
                c = r["classical_hmm"]
                print(f"  {r['window']:<25} {r['strategy']:<8} "
                      f"{b['max_drawdown']:>7.1%} "
                      f"{a['max_drawdown']:>7.1%} "
                      f"{c['max_drawdown']:>7.1%} "
                      f"{a['drawdown_avoided']:>+7.1%} "
                      f"{c['drawdown_avoided']:>+8.1%} "
                      f"{a['return_drag']:>+7.1%}")
        agg_e = econ["aggregate"]
        if ok_econ:
            print()
            print(f"  Mean agentic return drag: {agg_e['mean_agentic_return_drag']:+.1%}")
            print(f"  Mean classical return drag: {agg_e['mean_classical_return_drag']:+.1%}")
            print(f"  Mean agentic drawdown avoided: {agg_e['mean_agentic_dd_avoided']:+.1%}")
            print(f"  Mean classical drawdown avoided: {agg_e['mean_classical_dd_avoided']:+.1%}")
        print()

    # --- Section 3: Failure-mode review ---
    print("--- Section 3: Failure-Mode Review ---")
    da_path = RESULTS / "dev_analysis.json"
    fm = compute_failure_modes(da_path)
    print(f"  Missed detections: {len(fm['missed_detections'])}")
    for m in fm["missed_detections"]:
        print(f"    [{m['window']} x {m['strategy']}] {m['note']}")
    print(f"  Late detections (>14d): {len(fm['late_detections'])}")
    for m in fm["late_detections"]:
        print(f"    [{m['window']} x {m['strategy']}] latency={m['latency_days']}d — {m['note']}")
    print(f"  Early/on-time detections: {len(fm['successful_detections'])}")
    for m in fm["successful_detections"]:
        adv = f"advantage={m['advantage_days']}d" if m.get("advantage_days") is not None else ""
        cls_s = f"{m['classical_latency']}d" if m["classical_latency"] is not None else "missed"
        print(f"    [{m['window']} x {m['strategy']}] ag={m['agentic_latency']}d cls={cls_s} {adv}")
    print()
    print(f"  FP clusters in calm windows:")
    for fp in fm["calm_fp_clusters"]:
        comp = "COMPLETE" if fp["complete"] else "INCOMPLETE"
        fpr_s = f"{fp['fpr_per_day']:.6f}/d" if fp["fpr_per_day"] is not None else "n/a"
        print(f"    [{fp['window']} x {fp['strategy']}] n_fp={fp['n_fp_clusters']} fpr={fpr_s} [{comp}]")
    if fm["qualitative_notes"]:
        print()
        print(f"  Qualitative notes ({len(fm['qualitative_notes'])} items):")
        for qn in fm["qualitative_notes"]:
            print(f"    [{qn['window']} x {qn['strategy']}] {qn['issue']}")
            print(f"      → {qn['action']}")
    print()

    # --- Save output ---
    out = {
        "ops_metrics": ops,
        "econ_sim": econ,
        "failure_modes": fm,
        "_meta": {
            "script": "scripts/analyze_viability.py",
            "phase": "D4",
            "label": "exploratory — not preregistered",
        },
    }
    out_path = RESULTS / "viability_dev.json"
    out_path.write_text(json.dumps(out, indent=2, default=str))
    print(f"Saved: {out_path}")
    print()
    # Verdict
    agg_e = econ.get("aggregate", {})
    ok_count = sum(1 for r in ops["per_window"] if r["status"] == "ok")
    print("--- Verdict ---")
    feasible = ops["aggregate"]["pre_open_window_feasible"]
    print(f"  Pre-open feasibility (GPU active): {'YES' if feasible else 'NO'}")
    print(f"  Complete dev JSONLs (with provenance): {ok_count}/12")
    if ok_econ if not args.no_econ_sim else False:
        net_dd = agg_e.get("mean_agentic_dd_avoided", 0)
        net_drag = agg_e.get("mean_agentic_return_drag", 0)
        print(f"  Mean drawdown avoided by agentic: {net_dd:+.1%}")
        print(f"  Mean return drag from FP: {net_drag:+.1%}")
    print()


if __name__ == "__main__":
    main()
