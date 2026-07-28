"""PnL impact comparison: unmonitored vs classical-monitored vs agentic-monitored.

Simulates three regimes per (window, strategy) pair:
  1. UNMONITORED — hold throughout the window, no action on alarms.
  2. CLASSICAL — go flat on the first HMM ALERT day, stay flat for COOLDOWN days.
  3. AGENTIC — go flat on the first supervisor ALERT/CRITICAL day, stay flat for COOLDOWN.

Metrics: terminal cumulative return, max drawdown, drawdown avoided (vs unmonitored).

Usage (from monitoring/):
    python scripts/pnl_comparison.py

Outputs:
    results/pnl_comparison.json  — full per-window stats
    results/figures/fig10_pnl_comparison.png — equity curve overlays
    results/figures/fig11_drawdown_avoided.png — drawdown avoided bar chart
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pnl_loader import load_pnl, returns_series
from windows import (
    DEV_WINDOWS, TEST_EVENT_WINDOWS, TEST_CALM_WINDOWS,
    EVENT_WINDOWS, CALM_WINDOWS, ALL_WINDOWS,
)
from detectors import PageHinkley, BOCPD, HMMDetector, DistributionalThreshold, aggregate_alarms
from run_classical import STRATEGY_CURVES, _hmm_training_returns

RESULTS = Path(__file__).resolve().parent.parent / "results"
FIGDIR = RESULTS / "figures"

COOLDOWN_DAYS = 5  # stay flat for N trading days after going flat
BEST_DETECTOR = "hmm"  # best classical comparator

_LABELS = {
    "quant_meltdown_2007": "Quant Meltdown '07",
    "gfc_lehman_2008": "GFC / Lehman '08",
    "momentum_crash_2009": "Momentum Crash '09",
    "downgrade_2011": "US Downgrade '11",
    "flash_crash_2010": "Flash Crash '10",
    "china_deval_2015": "China Deval '15",
    "volmageddon_2018": "Volmageddon '18",
    "covid_2020": "COVID-19 '20",
}

# Colors
C_UNMON = "#888888"
C_CLS = "#2176AE"
C_AG = "#E8553A"


# ---------------------------------------------------------------------------
# Core simulation
# ---------------------------------------------------------------------------

def simulate_pnl(returns: pd.Series, alarm_dates: list[pd.Timestamp],
                 cooldown: int = COOLDOWN_DAYS) -> pd.Series:
    """Simulate equity curve: go flat on first alarm, stay flat for cooldown days.

    Returns a cumulative return series (starts at 0, in pct).
    """
    position = np.ones(len(returns))  # 1 = invested, 0 = flat
    alarm_set = set(alarm_dates)
    flat_until = None

    for i, day in enumerate(returns.index):
        if flat_until is not None and day <= flat_until:
            position[i] = 0.0
        elif day in alarm_set:
            # Go flat starting this day (don't capture this day's return)
            position[i] = 0.0
            # Stay flat for cooldown trading days
            future = returns.index[returns.index > day]
            if len(future) >= cooldown:
                flat_until = future[cooldown - 1]
            elif len(future) > 0:
                flat_until = future[-1]

    realized = returns.values * position
    cumulative = pd.Series(np.cumsum(realized), index=returns.index)
    return cumulative


def max_drawdown(cum_returns: pd.Series) -> float:
    """Max drawdown from cumulative returns (additive, not multiplicative)."""
    running_max = cum_returns.cummax()
    dd = cum_returns - running_max
    return float(dd.min())


def get_classical_alarm_dates(series: pd.Series, window, strategy: str) -> list[pd.Timestamp]:
    """Re-run detectors to get actual HMM alarm dates for a window."""
    hmm_train = _hmm_training_returns(STRATEGY_CURVES.get(strategy, []))
    detectors = [PageHinkley(), BOCPD(), HMMDetector(train_returns=hmm_train),
                 DistributionalThreshold()]
    results = [d.detect(series) for d in detectors]
    # Find HMM results
    for r in results:
        if r.detector == BEST_DETECTOR:
            # Filter to window period
            alarms = [a for a in r.alarms
                      if window.start_ts <= pd.Timestamp(a) <= window.end_ts]
            return [pd.Timestamp(a) for a in alarms]
    return []


def get_agentic_alarm_dates(window_name: str, strategy: str) -> list[pd.Timestamp]:
    """Parse agentic JSONL for ALERT/CRITICAL dates."""
    fname = RESULTS / f"agentic_{window_name}_{strategy}.jsonl"
    if not fname.exists():
        return []
    alarms = []
    with open(fname) as f:
        for line in f:
            try:
                rec = json.loads(line)
                if rec.get("agent") != "performance_supervisor":
                    continue
                assessment = rec.get("assessment", {})
                state = assessment.get("state", "")
                if state in ("ALERT", "CRITICAL"):
                    alarms.append(pd.Timestamp(rec["as_of"]))
            except (json.JSONDecodeError, KeyError):
                pass
    return sorted(set(alarms))


def load_series_for_window(strategy: str, window) -> pd.Series | None:
    """Find the strategy curve covering this window."""
    for path in STRATEGY_CURVES.get(strategy, []):
        if not Path(path).exists():
            continue
        s = returns_series(load_pnl(path))
        if s.index.min() <= window.start_ts and s.index.max() >= window.end_ts - pd.Timedelta(days=7):
            return s
    return None


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def run_comparison() -> list[dict]:
    all_windows = list(EVENT_WINDOWS) + list(TEST_EVENT_WINDOWS)
    strategies = ["AL_PCA", "JT_MOM"]
    results = []

    for strat in strategies:
        for w in all_windows:
            series = load_series_for_window(strat, w)
            if series is None:
                continue

            # Slice to window
            mask = (series.index >= w.start_ts) & (series.index <= w.end_ts)
            w_returns = series[mask]
            if len(w_returns) < 10:
                continue

            # Get alarm dates
            cls_alarms = get_classical_alarm_dates(series, w, strat)
            ag_alarms = get_agentic_alarm_dates(w.name, strat)

            # Simulate three regimes
            cum_unmon = simulate_pnl(w_returns, [])
            cum_cls = simulate_pnl(w_returns, cls_alarms)
            cum_ag = simulate_pnl(w_returns, ag_alarms)

            # Metrics
            dd_unmon = max_drawdown(cum_unmon)
            dd_cls = max_drawdown(cum_cls)
            dd_ag = max_drawdown(cum_ag)

            result = {
                "window": w.name,
                "strategy": strat,
                "set": "dev" if w in EVENT_WINDOWS else "test",
                "n_days": len(w_returns),
                # Terminal cumulative return
                "terminal_unmon": round(float(cum_unmon.iloc[-1]) * 100, 2),
                "terminal_cls": round(float(cum_cls.iloc[-1]) * 100, 2),
                "terminal_ag": round(float(cum_ag.iloc[-1]) * 100, 2),
                # Max drawdown
                "maxdd_unmon": round(dd_unmon * 100, 2),
                "maxdd_cls": round(dd_cls * 100, 2),
                "maxdd_ag": round(dd_ag * 100, 2),
                # Drawdown avoided (positive = good)
                "dd_avoided_cls": round((dd_cls - dd_unmon) * 100, 2),
                "dd_avoided_ag": round((dd_ag - dd_unmon) * 100, 2),
                # Alarm counts
                "n_cls_alarms": len(cls_alarms),
                "n_ag_alarms": len(ag_alarms),
                # First alarm dates
                "first_cls_alarm": str(cls_alarms[0].date()) if cls_alarms else None,
                "first_ag_alarm": str(ag_alarms[0].date()) if ag_alarms else None,
                # Equity curves for plotting
                "_cum_unmon": cum_unmon,
                "_cum_cls": cum_cls,
                "_cum_ag": cum_ag,
            }
            results.append(result)
            label = _LABELS.get(w.name, w.name)
            print(f"  {label} × {strat}: "
                  f"unmon={result['terminal_unmon']:+.1f}% dd={result['maxdd_unmon']:+.1f}% | "
                  f"cls={result['terminal_cls']:+.1f}% dd={result['maxdd_cls']:+.1f}% | "
                  f"ag={result['terminal_ag']:+.1f}% dd={result['maxdd_ag']:+.1f}%")

    return results


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def fig10_equity_curves(results: list[dict]):
    """Overlay equity curves for each event window."""
    n = len(results)
    cols = 3
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 3.5 * rows))
    axes = axes.flatten() if n > 1 else [axes]

    for i, r in enumerate(results):
        ax = axes[i]
        cum_u = r["_cum_unmon"] * 100
        cum_c = r["_cum_cls"] * 100
        cum_a = r["_cum_ag"] * 100

        ax.plot(cum_u.index, cum_u.values, color=C_UNMON, linewidth=1.2, label="Unmonitored")
        ax.plot(cum_c.index, cum_c.values, color=C_CLS, linewidth=1.2, label="Classical (HMM)")
        ax.plot(cum_a.index, cum_a.values, color=C_AG, linewidth=1.2, label="Agentic")

        # Mark first alarm dates
        if r["first_cls_alarm"]:
            cls_dt = pd.Timestamp(r["first_cls_alarm"])
            if cls_dt in cum_c.index:
                ax.axvline(cls_dt, color=C_CLS, linewidth=0.8, linestyle="--", alpha=0.6)
        if r["first_ag_alarm"]:
            ag_dt = pd.Timestamp(r["first_ag_alarm"])
            if ag_dt in cum_a.index:
                ax.axvline(ag_dt, color=C_AG, linewidth=0.8, linestyle="--", alpha=0.6)

        ax.axhline(0, color="black", linewidth=0.3)
        label = _LABELS.get(r["window"], r["window"])
        tag = "[DEV]" if r["set"] == "dev" else "[TEST]"
        ax.set_title(f"{tag} {label} × {r['strategy']}", fontsize=9, fontweight="bold")
        ax.set_ylabel("Cumulative Return (%)", fontsize=8)
        ax.tick_params(axis="x", rotation=30, labelsize=7)
        ax.tick_params(axis="y", labelsize=8)
        if i == 0:
            ax.legend(fontsize=7, loc="lower left")

    # Hide empty subplots
    for j in range(n, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Equity Curves Under Three Monitoring Regimes",
                 fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig10_pnl_comparison.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  fig10_pnl_comparison.png")


def fig11_drawdown_avoided(results: list[dict]):
    """Bar chart: drawdown avoided by each monitoring arm vs unmonitored."""
    n = len(results)
    fig, ax = plt.subplots(figsize=(14, 6))
    y = np.arange(n)
    bar_h = 0.35

    for i, r in enumerate(results):
        # Classical bar
        ax.barh(i + bar_h / 2, r["dd_avoided_cls"], bar_h, color=C_CLS,
                edgecolor="white", linewidth=0.5)
        ax.text(r["dd_avoided_cls"] + 0.1, i + bar_h / 2,
                f"{r['dd_avoided_cls']:+.1f}%", va="center", fontsize=8, color=C_CLS)

        # Agentic bar
        ax.barh(i - bar_h / 2, r["dd_avoided_ag"], bar_h, color=C_AG,
                edgecolor="white", linewidth=0.5)
        ax.text(r["dd_avoided_ag"] + 0.1, i - bar_h / 2,
                f"{r['dd_avoided_ag']:+.1f}%", va="center", fontsize=8, color=C_AG)

    labels = []
    for r in results:
        label = _LABELS.get(r["window"], r["window"])
        tag = "[D]" if r["set"] == "dev" else "[T]"
        labels.append(f"{tag} {label}\n({r['strategy']})")

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Drawdown Avoided vs Unmonitored (percentage points)", fontsize=11)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.invert_yaxis()

    # Separator between dev and test
    n_dev = sum(1 for r in results if r["set"] == "dev")
    if 0 < n_dev < n:
        ax.axhline(n_dev - 0.5, color="gray", linewidth=1, linestyle="--", alpha=0.5)

    cls_patch = mpatches.Patch(color=C_CLS, label="Classical (HMM)")
    ag_patch = mpatches.Patch(color=C_AG, label="Agentic")
    ax.legend(handles=[cls_patch, ag_patch], loc="lower right", fontsize=9)

    ax.set_title("Drawdown Avoided by Monitoring (vs Unmonitored Hold)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig11_drawdown_avoided.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  fig11_drawdown_avoided.png")


def fig12_summary_table(results: list[dict]):
    """Summary table of terminal returns and max drawdowns."""
    fig, ax = plt.subplots(figsize=(14, max(4, len(results) * 0.5 + 1.5)))
    ax.axis("off")

    header = ["Window", "Strategy", "Set",
              "Unmon\nReturn", "Cls\nReturn", "Ag\nReturn",
              "Unmon\nMaxDD", "Cls\nMaxDD", "Ag\nMaxDD",
              "DD Avoided\n(Cls)", "DD Avoided\n(Ag)"]
    rows = []
    for r in results:
        label = _LABELS.get(r["window"], r["window"])
        rows.append([
            label, r["strategy"], r["set"].upper(),
            f"{r['terminal_unmon']:+.1f}%", f"{r['terminal_cls']:+.1f}%", f"{r['terminal_ag']:+.1f}%",
            f"{r['maxdd_unmon']:+.1f}%", f"{r['maxdd_cls']:+.1f}%", f"{r['maxdd_ag']:+.1f}%",
            f"{r['dd_avoided_cls']:+.1f}%", f"{r['dd_avoided_ag']:+.1f}%",
        ])

    table = ax.table(cellText=rows, colLabels=header, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.6)

    # Style header
    for j in range(len(header)):
        table[0, j].set_facecolor("#263238")
        table[0, j].set_text_props(color="white", fontweight="bold", fontsize=7)

    # Highlight best DD avoided per row
    for i in range(1, len(rows) + 1):
        table[i, 0].set_text_props(fontweight="bold")
        # Color the DD avoided columns
        for j in [9, 10]:
            val = float(rows[i - 1][j].replace("%", "").replace("+", ""))
            if val > 0:
                table[i, j].set_facecolor("#C8E6C9")  # green — avoided drawdown
            elif val < -0.5:
                table[i, j].set_facecolor("#FFCDD2")  # red — worse than unmonitored

    ax.set_title("PnL Impact Summary: Monitoring vs Unmonitored",
                 fontsize=13, fontweight="bold", pad=20)
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig12_pnl_summary_table.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  fig12_pnl_summary_table.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    FIGDIR.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("PNL IMPACT COMPARISON: Unmonitored vs Classical vs Agentic")
    print("=" * 72)

    results = run_comparison()

    # Save (without the pandas series)
    save_results = [{k: v for k, v in r.items() if not k.startswith("_")} for r in results]
    out = RESULTS / "pnl_comparison.json"
    out.write_text(json.dumps(save_results, indent=2))
    print(f"\nSaved: {out}")

    # Summary stats
    dd_cls = [r["dd_avoided_cls"] for r in results]
    dd_ag = [r["dd_avoided_ag"] for r in results]
    print(f"\nMean drawdown avoided (classical): {np.mean(dd_cls):+.2f} pp")
    print(f"Mean drawdown avoided (agentic):   {np.mean(dd_ag):+.2f} pp")
    print(f"Agentic advantage:                 {np.mean(dd_ag) - np.mean(dd_cls):+.2f} pp")

    # Figures
    print(f"\nGenerating figures...")
    fig10_equity_curves(results)
    fig11_drawdown_avoided(results)
    fig12_summary_table(results)
    print(f"\nDone — 3 PnL figures generated.")


if __name__ == "__main__":
    main()
