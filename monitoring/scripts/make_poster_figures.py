"""Generate all poster/paper figures from experiment artifacts.

Usage (from monitoring/):
    python scripts/make_poster_figures.py

Outputs PNGs to monitoring/results/figures/.
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

from agentic.alarm_extraction import reconstruct_day_records
from windows import (
    DEV_WINDOWS, TEST_EVENT_WINDOWS, TEST_CALM_WINDOWS,
    EVENT_WINDOWS, CALM_WINDOWS, TEST_STRATEGY_WINDOWS,
)

RESULTS = Path(__file__).resolve().parent.parent / "results"
FIGDIR = RESULTS / "figures"

# Consistent color palette
C_CLS = "#2176AE"    # classical — blue
C_AG  = "#E8553A"    # agentic — red-orange
C_DEV = "#555555"    # dev set marker
C_TEST = "#000000"   # test set marker
C_MISS = "#CCCCCC"   # missed detection

LATENCY_CAP = 21     # ±21d detection window

# Nice window labels
_LABELS = {
    "quant_meltdown_2007": "Quant Meltdown '07",
    "gfc_lehman_2008": "GFC / Lehman '08",
    "momentum_crash_2009": "Momentum Crash '09",
    "downgrade_2011": "US Downgrade '11",
    "flash_crash_2010": "Flash Crash '10",
    "china_deval_2015": "China Deval '15",
    "volmageddon_2018": "Volmageddon '18",
    "covid_2020": "COVID-19 '20",
    "calm_2004_2006": "Calm '04-'06",
    "calm_2013_2014": "Calm '13-'14",
    "calm_2012": "Calm '12",
    "calm_2017": "Calm '17",
}


def _label(wname: str) -> str:
    return _LABELS.get(wname, wname)


def _load_json(name: str) -> dict:
    return json.loads((RESULTS / name).read_text())


# -----------------------------------------------------------------------
# Figure 1: Paired latency bar chart
# -----------------------------------------------------------------------
def fig1_latency_bars():
    dev = _load_json("dev_analysis.json")
    test = _load_json("test_analysis.json")

    pairs = []
    for key, p in dev["paired_results"].items():
        pairs.append({**p, "set": "dev"})
    for key, p in test["paired_results"].items():
        pairs.append({**p, "set": "test"})

    # Sort: dev events first, then test events
    dev_order = ["quant_meltdown_2007", "gfc_lehman_2008", "momentum_crash_2009", "downgrade_2011"]
    test_order = ["flash_crash_2010", "china_deval_2015", "volmageddon_2018", "covid_2020"]

    def sort_key(p):
        w = p["window"]
        if w in dev_order:
            return (0, dev_order.index(w), p["strategy"])
        return (1, test_order.index(w) if w in test_order else 99, p["strategy"])

    pairs.sort(key=sort_key)

    n = len(pairs)
    fig, ax = plt.subplots(figsize=(14, 6))
    y = np.arange(n)
    bar_h = 0.35

    for i, p in enumerate(pairs):
        c_lat = p["classical_latency"]
        a_lat = p["agentic_latency"]
        c_miss = c_lat is None
        a_miss = a_lat is None

        # Classical bar
        val_c = LATENCY_CAP if c_miss else c_lat
        ax.barh(i + bar_h / 2, val_c, bar_h, color=C_CLS if not c_miss else C_MISS,
                edgecolor="white", linewidth=0.5,
                hatch="///" if c_miss else None)
        if not c_miss:
            ax.text(val_c + 0.3, i + bar_h / 2, f"{c_lat}d", va="center", fontsize=8, color=C_CLS)
        else:
            ax.text(val_c + 0.3, i + bar_h / 2, "miss", va="center", fontsize=8, color="#888")

        # Agentic bar
        val_a = LATENCY_CAP if a_miss else a_lat
        ax.barh(i - bar_h / 2, val_a, bar_h, color=C_AG if not a_miss else C_MISS,
                edgecolor="white", linewidth=0.5,
                hatch="///" if a_miss else None)
        if not a_miss:
            ax.text(val_a + 0.3, i - bar_h / 2, f"{a_lat}d", va="center", fontsize=8, color=C_AG)
        else:
            ax.text(val_a + 0.3, i - bar_h / 2, "miss", va="center", fontsize=8, color="#888")

    labels = [f"{_label(p['window'])}\n({p['strategy']})" for p in pairs]
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Detection Latency (trading days from onset)", fontsize=11)
    ax.set_xlim(-0.5, LATENCY_CAP + 3)
    ax.axvline(0, color="black", linewidth=0.5, linestyle="--", alpha=0.3)
    ax.invert_yaxis()

    # Separator between dev and test
    n_dev = sum(1 for p in pairs if p["set"] == "dev")
    if 0 < n_dev < n:
        ax.axhline(n_dev - 0.5, color="gray", linewidth=1, linestyle="--", alpha=0.5)
        ax.text(LATENCY_CAP + 2, n_dev / 2 - 0.5, "DEV", ha="center", va="center",
                fontsize=9, color="gray", rotation=90, fontweight="bold")
        ax.text(LATENCY_CAP + 2, n_dev + (n - n_dev) / 2 - 0.5, "TEST", ha="center", va="center",
                fontsize=9, color="gray", rotation=90, fontweight="bold")

    cls_patch = mpatches.Patch(color=C_CLS, label="Classical (HMM)")
    ag_patch = mpatches.Patch(color=C_AG, label="Agentic (qwen2.5:1.5b)")
    miss_patch = mpatches.Patch(facecolor=C_MISS, hatch="///", edgecolor="gray", label="Missed (>21d)")
    ax.legend(handles=[cls_patch, ag_patch, miss_patch], loc="lower right", fontsize=9)

    ax.set_title("Detection Latency: Agentic vs Classical (Best HMM Detector)", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig1_latency_bars.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  fig1_latency_bars.png")


# -----------------------------------------------------------------------
# Figure 2: FPR dot plot (calm windows)
# -----------------------------------------------------------------------
def fig2_fpr_dotplot():
    dev = _load_json("dev_analysis.json")
    test = _load_json("test_analysis.json")

    fig, ax = plt.subplots(figsize=(8, 5))
    yticks = []
    ypos = []
    i = 0

    for dataset, data, marker in [("DEV", dev, "o"), ("TEST", test, "s")]:
        cells = data["calm_fpr"]["cells"]
        for key in sorted(cells.keys()):
            cell = cells[key]
            wname, strat = key.split("__")
            label = f"{_label(wname)} × {strat}"
            ag_fpr = cell["ag_fpr"]
            cls_fpr = cell["cls_hmm_fpr"]

            ax.scatter(cls_fpr * 100, i, marker=marker, s=80, color=C_CLS, zorder=5, edgecolors="white", linewidth=0.5)
            ax.scatter(ag_fpr * 100, i, marker=marker, s=80, color=C_AG, zorder=5, edgecolors="white", linewidth=0.5)
            # Connect with a line
            ax.plot([cls_fpr * 100, ag_fpr * 100], [i, i], color="#aaa", linewidth=1, zorder=1)

            yticks.append(f"[{dataset}] {label}")
            ypos.append(i)
            i += 1
        # Separator
        if dataset == "DEV":
            ax.axhline(i - 0.5, color="gray", linewidth=0.8, linestyle="--", alpha=0.4)

    ax.set_yticks(ypos)
    ax.set_yticklabels(yticks, fontsize=9)
    ax.set_xlabel("False Positive Rate (%)", fontsize=11)
    ax.invert_yaxis()
    ax.set_xlim(-0.5, max(12, ax.get_xlim()[1] + 1))

    cls_patch = plt.Line2D([], [], marker="o", color=C_CLS, linestyle="", markersize=8, label="Classical HMM")
    ag_patch = plt.Line2D([], [], marker="o", color=C_AG, linestyle="", markersize=8, label="Agentic")
    ax.legend(handles=[cls_patch, ag_patch], loc="lower right", fontsize=9)

    ax.set_title("False Positive Rate: Calm Windows", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig2_fpr_dotplot.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  fig2_fpr_dotplot.png")


# -----------------------------------------------------------------------
# Figure 3: Recall comparison (contingency grid)
# -----------------------------------------------------------------------
def fig3_recall_grid():
    dev = _load_json("dev_analysis.json")
    test = _load_json("test_analysis.json")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    for ax, data, title in [(axes[0], dev, "Dev Set (n=8)"), (axes[1], test, "Test Set (n=5)")]:
        pairs = data["paired_results"]
        n = len(pairs)
        both = sum(1 for p in pairs.values() if p["classical_detected"] and p["agentic_detected"])
        ag_only = sum(1 for p in pairs.values() if not p["classical_detected"] and p["agentic_detected"])
        cls_only = sum(1 for p in pairs.values() if p["classical_detected"] and not p["agentic_detected"])
        neither = sum(1 for p in pairs.values() if not p["classical_detected"] and not p["agentic_detected"])

        matrix = np.array([[both, cls_only], [ag_only, neither]])
        colors = np.array([["#4CAF50", "#FFC107"], ["#2196F3", "#F44336"]])

        for r in range(2):
            for c in range(2):
                rect = plt.Rectangle((c, 1 - r), 1, 1, facecolor=colors[r][c], alpha=0.6, edgecolor="white", linewidth=2)
                ax.add_patch(rect)
                ax.text(c + 0.5, 1.5 - r, str(matrix[r][c]), ha="center", va="center",
                        fontsize=24, fontweight="bold", color="white")

        ax.set_xlim(0, 2)
        ax.set_ylim(0, 2)
        ax.set_xticks([0.5, 1.5])
        ax.set_xticklabels(["Classical\nDetected", "Classical\nMissed"], fontsize=10)
        ax.set_yticks([0.5, 1.5])
        ax.set_yticklabels(["Agentic\nMissed", "Agentic\nDetected"], fontsize=10)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_aspect("equal")

    fig.suptitle("Detection Recall: Contingency Tables", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig3_recall_grid.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  fig3_recall_grid.png")


# -----------------------------------------------------------------------
# Figure 4: Triage mode distribution by window
# -----------------------------------------------------------------------
def fig4_triage_distribution():
    all_windows = list(EVENT_WINDOWS) + list(CALM_WINDOWS) + list(TEST_EVENT_WINDOWS) + list(TEST_CALM_WINDOWS)
    win_map = {w.name: w for w in all_windows}

    # Collect triage modes from all JSONLs
    data_rows = []
    for fpath in sorted(RESULTS.glob("agentic_*.jsonl")):
        records = [json.loads(l) for l in fpath.read_text().splitlines() if l.strip()]
        triage_recs = [r for r in records if r.get("agent") == "triage"]
        if not triage_recs:
            continue
        wname = triage_recs[0].get("window", "")
        # Extract strategy from filename
        fname = fpath.stem  # agentic_quant_meltdown_2007_AL_PCA
        strat = fname.rsplit("_", 2)[-2] + "_" + fname.rsplit("_", 1)[-1]  # AL_PCA or JT_MOM
        modes = [r.get("triage_mode", "unknown") for r in triage_recs]
        from collections import Counter
        counts = Counter(modes)
        total = len(modes)
        data_rows.append({
            "label": f"{_label(wname)}\n({strat})",
            "window": wname,
            "strategy": strat,
            "skip": counts.get("skip", 0) / total,
            "thinking": counts.get("thinking", 0) / total,
            "classical_escalation": counts.get("classical_escalation", 0) / total,
            "cheap": counts.get("cheap", 0) / total,
            "total": total,
            "kind": win_map.get(wname, None),
        })

    # Sort: dev event, dev calm, test event, test calm
    dev_event = [w.name for w in EVENT_WINDOWS]
    dev_calm = [w.name for w in CALM_WINDOWS]
    test_event = [w.name for w in TEST_EVENT_WINDOWS]
    test_calm = [w.name for w in TEST_CALM_WINDOWS]

    def sort_key(r):
        w = r["window"]
        if w in dev_event: return (0, dev_event.index(w), r["strategy"])
        if w in dev_calm: return (1, dev_calm.index(w), r["strategy"])
        if w in test_event: return (2, test_event.index(w), r["strategy"])
        if w in test_calm: return (3, test_calm.index(w), r["strategy"])
        return (4, 0, r["strategy"])

    data_rows.sort(key=sort_key)

    n = len(data_rows)
    fig, ax = plt.subplots(figsize=(12, max(6, n * 0.4)))
    y = np.arange(n)

    colors = {"skip": "#81C784", "cheap": "#FFD54F", "thinking": "#FF8A65", "classical_escalation": "#E53935"}
    labels_done = set()
    left = np.zeros(n)

    for mode in ["skip", "cheap", "thinking", "classical_escalation"]:
        vals = [r.get(mode, 0) for r in data_rows]
        lbl = mode.replace("_", " ").title() if mode not in labels_done else None
        ax.barh(y, vals, left=left, height=0.7, color=colors[mode], label=lbl, edgecolor="white", linewidth=0.5)
        labels_done.add(mode)
        left += vals

    # Day counts on right
    for i, r in enumerate(data_rows):
        ax.text(1.02, i, f"n={r['total']}", va="center", fontsize=8, color="#666")

    ax.set_yticks(y)
    ax.set_yticklabels([r["label"] for r in data_rows], fontsize=8)
    ax.set_xlabel("Proportion of Trading Days", fontsize=11)
    ax.set_xlim(0, 1.12)
    ax.invert_yaxis()
    ax.legend(loc="lower right", fontsize=9)
    ax.set_title("Triage Mode Distribution by Window", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig4_triage_distribution.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  fig4_triage_distribution.png")


# -----------------------------------------------------------------------
# Figure 5: News-agent failure rate by window
# -----------------------------------------------------------------------
def fig5_failure_rates():
    data_rows = []
    for fpath in sorted(RESULTS.glob("agentic_*.jsonl")):
        records = [json.loads(l) for l in fpath.read_text().splitlines() if l.strip()]
        triage_recs = [r for r in records if r.get("agent") == "triage"]
        if not triage_recs:
            continue
        wname = triage_recs[0].get("window", "")
        fname = fpath.stem
        strat = fname.rsplit("_", 2)[-2] + "_" + fname.rsplit("_", 1)[-1]

        non_skip = [r for r in triage_recs if r.get("triage_mode") != "skip"]
        supervisor_recs = [r for r in records if r.get("agent") == "performance_supervisor"]
        n_non_skip = len(non_skip)
        n_supervisor = len(supervisor_recs)

        if n_non_skip == 0:
            continue

        fail_rate = 1 - (n_supervisor / n_non_skip) if n_non_skip > 0 else 0
        data_rows.append({
            "label": f"{_label(wname)} × {strat}",
            "window": wname,
            "strategy": strat,
            "n_non_skip": n_non_skip,
            "n_supervisor": n_supervisor,
            "fail_rate": fail_rate,
        })

    data_rows.sort(key=lambda r: -r["fail_rate"])

    n = len(data_rows)
    fig, ax = plt.subplots(figsize=(10, max(5, n * 0.35)))
    y = np.arange(n)
    rates = [r["fail_rate"] * 100 for r in data_rows]

    bars = ax.barh(y, rates, height=0.6, color=[C_AG if r > 20 else "#FFB74D" if r > 5 else "#81C784" for r in rates],
                   edgecolor="white", linewidth=0.5)

    for i, r in enumerate(data_rows):
        ax.text(rates[i] + 0.5, i, f"{r['n_supervisor']}/{r['n_non_skip']}",
                va="center", fontsize=8, color="#555")

    ax.set_yticks(y)
    ax.set_yticklabels([r["label"] for r in data_rows], fontsize=9)
    ax.set_xlabel("News-Agent Failure Rate (% of non-skip days)", fontsize=11)
    ax.set_xlim(0, max(rates) + 10)
    ax.invert_yaxis()
    ax.set_title("News-Agent Failure Rate by Window (Deviation 7)", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig5_failure_rates.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  fig5_failure_rates.png")


# -----------------------------------------------------------------------
# Figure 6: Leakage analysis (A/B/C conditions)
# -----------------------------------------------------------------------
def fig6_leakage():
    # Dev leakage (quant_meltdown_2007)
    dev_leak = {
        "A": {"detected": True, "latency": 0, "n_fp": 27},
        "B": {"detected": True, "latency": 1, "n_fp": 12},
        "C_recall": 0.3, "C_mean_latency": 1.3,
    }

    # Test leakage (flash_crash_2010)
    test_leak_path = RESULTS / "leakage_analysis_AL_PCA.json"
    if test_leak_path.exists():
        test_data = json.loads(test_leak_path.read_text())
        pw = test_data.get("per_window", {}).get("flash_crash_2010", {})
        test_leak = {
            "A": {"detected": pw.get("pass_A", {}).get("detected"), "latency": pw.get("pass_A", {}).get("latency")},
            "B": {"detected": pw.get("pass_B", {}).get("detected"), "latency": pw.get("pass_B", {}).get("latency")},
            "C_recall": pw.get("pass_C", {}).get("recall_mean"),
            "C_mean_latency": None,  # Not always available
        }
    else:
        test_leak = None

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    # Panel 1: Latency under A/B
    conditions = ["A (Standard)", "B (Date-masked)"]
    for ax_i, (leak, title, window_label) in enumerate([
        (dev_leak, "Dev: Quant Meltdown '07", "quant_meltdown_2007"),
        (test_leak, "Test: Flash Crash '10", "flash_crash_2010"),
    ]):
        ax = axes[ax_i]
        if leak is None:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes, fontsize=14)
            ax.set_title(title, fontsize=11, fontweight="bold")
            continue

        lats = [
            leak["A"]["latency"] if leak["A"]["detected"] else LATENCY_CAP,
            leak["B"]["latency"] if leak["B"]["detected"] else LATENCY_CAP,
        ]
        bar_colors = ["#4CAF50", "#FF9800"]
        bars = ax.bar(conditions, lats, color=bar_colors, edgecolor="white", width=0.5)

        for bar, lat, cond in zip(bars, lats, ["A", "B"]):
            detected = leak[cond]["detected"]
            txt = f"{lat}d" if detected else "miss"
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                    txt, ha="center", fontsize=12, fontweight="bold")

        # Add C recall annotation
        c_recall = leak.get("C_recall")
        if c_recall is not None:
            ax.text(0.5, 0.92, f"Condition C recall: {c_recall:.0%}",
                    ha="center", va="top", transform=ax.transAxes,
                    fontsize=10, style="italic", color="#666",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))

        ax.set_ylabel("Detection Latency (days)", fontsize=10)
        ax.set_ylim(0, max(lats) + 3)
        ax.set_title(title, fontsize=11, fontweight="bold")

    fig.suptitle("Leakage Analysis: A/B/C Conditions", fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig6_leakage.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  fig6_leakage.png")


# -----------------------------------------------------------------------
# Figure 7: Strategy breakdown (H2/H3) — latency diffs
# -----------------------------------------------------------------------
def fig7_strategy_breakdown():
    dev = _load_json("dev_analysis.json")
    test = _load_json("test_analysis.json")

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    for ax, data, title in [(axes[0], dev, "Dev Set"), (axes[1], test, "Test Set")]:
        for s_i, strat in enumerate(["AL_PCA", "JT_MOM"]):
            pairs = {k: v for k, v in data["paired_results"].items() if v["strategy"] == strat}
            if not pairs:
                continue
            diffs = []
            labels = []
            for key, p in sorted(pairs.items()):
                c_lat = p["classical_latency"]
                a_lat = p["agentic_latency"]
                # Treat misses as LATENCY_CAP for visual comparison
                c_val = c_lat if c_lat is not None else LATENCY_CAP
                a_val = a_lat if a_lat is not None else LATENCY_CAP
                diffs.append(a_val - c_val)
                labels.append(_label(p["window"]))

            x = np.arange(len(diffs))
            offset = -0.2 if s_i == 0 else 0.2
            width = 0.35
            color = "#1565C0" if strat == "AL_PCA" else "#E65100"
            bars = ax.bar(x + offset, diffs, width, label=strat, color=color, alpha=0.8, edgecolor="white")

        ax.axhline(0, color="black", linewidth=0.8, linestyle="-")
        ax.set_ylabel("Latency Difference (agentic − classical, days)", fontsize=10)

        # Use the labels from the last strategy that had pairs
        if labels:
            ax.set_xticks(np.arange(max(
                len([v for v in data["paired_results"].values() if v["strategy"] == "AL_PCA"]),
                len([v for v in data["paired_results"].values() if v["strategy"] == "JT_MOM"]),
            )))
            # Build combined unique window labels
            all_windows_in_set = []
            seen = set()
            for p in sorted(data["paired_results"].values(), key=lambda p: p["window"]):
                if p["window"] not in seen:
                    all_windows_in_set.append(_label(p["window"]))
                    seen.add(p["window"])
            ax.set_xticks(np.arange(len(all_windows_in_set)))
            ax.set_xticklabels(all_windows_in_set, fontsize=8, rotation=30, ha="right")

        ax.legend(fontsize=9)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.annotate("← Agentic faster", xy=(0.02, 0.02), xycoords="axes fraction",
                    fontsize=8, color="green", alpha=0.7)
        ax.annotate("Classical faster →", xy=(0.02, 0.95), xycoords="axes fraction",
                    fontsize=8, color="red", alpha=0.7)

    fig.suptitle("Strategy Breakdown: Latency Difference by Window (H2/H3)",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig7_strategy_breakdown.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  fig7_strategy_breakdown.png")


# -----------------------------------------------------------------------
# Figure 8: Summary statistics panel (poster headline)
# -----------------------------------------------------------------------
def fig8_summary_panel():
    dev = _load_json("dev_analysis.json")
    test = _load_json("test_analysis.json")

    fig, ax = plt.subplots(figsize=(12, 3.5))
    ax.axis("off")

    # Dev stats
    dev_lat = dev["tests"]["permutation_latency"]
    dev_holm = dev["tests"]["holm"]
    dev_recall_ag = sum(1 for p in dev["paired_results"].values() if p["agentic_detected"])
    dev_recall_cls = sum(1 for p in dev["paired_results"].values() if p["classical_detected"])

    # Test stats
    test_lat = test["tests"]["permutation_latency"]
    test_recall_ag = sum(1 for p in test["paired_results"].values() if p["agentic_detected"])
    test_recall_cls = sum(1 for p in test["paired_results"].values() if p["classical_detected"])

    rows = [
        ["", "Dev Set (n=8)", "Test Set (n=5)"],
        ["Mean Latency Advantage",
         f"{abs(dev_lat['observed_diff_mean']):.1f} days faster",
         f"{abs(test_lat['observed_diff_mean']):.1f} days faster"],
        ["Permutation p (one-sided)",
         f"p = {dev_lat['p_value_one_sided']:.4f} *",
         f"p = {test_lat['p_value_one_sided']:.4f}"],
        ["Holm-adjusted p (latency)",
         f"p = {dev_holm['latency']['adjusted_p']:.4f} {'REJECT' if dev_holm['latency']['reject_005'] else ''}",
         f"p = {test['tests']['holm']['latency']['adjusted_p']:.4f}"],
        ["Recall (agentic / classical)",
         f"{dev_recall_ag}/8 vs {dev_recall_cls}/8",
         f"{test_recall_ag}/5 vs {test_recall_cls}/5"],
        ["FPR Bootstrap p",
         f"p = {dev['calm_fpr']['bootstrap']['p_value']:.4f} (no advantage)",
         f"p = {test['calm_fpr']['bootstrap']['p_value']:.4f} (no advantage)"],
    ]

    table = ax.table(cellText=rows, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.8)

    # Style header row
    for j in range(3):
        table[0, j].set_facecolor("#263238")
        table[0, j].set_text_props(color="white", fontweight="bold")

    # Style data rows
    for i in range(1, len(rows)):
        table[i, 0].set_text_props(fontweight="bold", ha="right")
        table[i, 0].set_facecolor("#ECEFF1")
        for j in range(1, 3):
            table[i, j].set_facecolor("#FAFAFA")

    # Highlight significant result
    table[2, 1].set_facecolor("#C8E6C9")
    table[3, 1].set_facecolor("#C8E6C9")

    ax.set_title("Headline Results", fontsize=14, fontweight="bold", pad=20)
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig8_summary_panel.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  fig8_summary_panel.png")


# -----------------------------------------------------------------------
# Figure 9: Detection timeline heatmap
# -----------------------------------------------------------------------
def fig9_detection_timeline():
    dev = _load_json("dev_analysis.json")
    test = _load_json("test_analysis.json")

    all_pairs = []
    for data, dataset in [(dev, "dev"), (test, "test")]:
        for key, p in data["paired_results"].items():
            all_pairs.append({**p, "set": dataset})

    # Sort same as fig1
    dev_order = ["quant_meltdown_2007", "gfc_lehman_2008", "momentum_crash_2009", "downgrade_2011"]
    test_order = ["flash_crash_2010", "china_deval_2015", "volmageddon_2018", "covid_2020"]

    def sort_key(p):
        w = p["window"]
        if w in dev_order: return (0, dev_order.index(w), p["strategy"])
        if w in test_order: return (1, test_order.index(w), p["strategy"])
        return (2, 0, p["strategy"])

    all_pairs.sort(key=sort_key)
    n = len(all_pairs)

    fig, ax = plt.subplots(figsize=(10, max(5, n * 0.4)))

    for i, p in enumerate(all_pairs):
        # Onset at x=0
        ax.axvline(0, color="#999", linewidth=0.3, linestyle=":")

        # Classical detection
        c_lat = p["classical_latency"]
        if c_lat is not None:
            ax.scatter(c_lat, i + 0.12, marker="|", s=200, color=C_CLS, linewidth=2.5, zorder=5)
        else:
            ax.scatter(LATENCY_CAP, i + 0.12, marker="x", s=60, color=C_CLS, linewidth=1.5, zorder=5, alpha=0.5)

        # Agentic detection
        a_lat = p["agentic_latency"]
        if a_lat is not None:
            ax.scatter(a_lat, i - 0.12, marker="|", s=200, color=C_AG, linewidth=2.5, zorder=5)
        else:
            ax.scatter(LATENCY_CAP, i - 0.12, marker="x", s=60, color=C_AG, linewidth=1.5, zorder=5, alpha=0.5)

    labels = [f"{_label(p['window'])} ({p['strategy']})" for p in all_pairs]
    ax.set_yticks(range(n))
    ax.set_yticklabels(labels, fontsize=9)
    ax.axvline(0, color="black", linewidth=1.5, label="Onset")
    ax.set_xlabel("Days relative to onset", fontsize=11)
    ax.set_xlim(-3, LATENCY_CAP + 2)
    ax.invert_yaxis()

    # Legend
    cls_line = plt.Line2D([], [], marker="|", color=C_CLS, linestyle="", markersize=12,
                          markeredgewidth=2.5, label="Classical first ALERT")
    ag_line = plt.Line2D([], [], marker="|", color=C_AG, linestyle="", markersize=12,
                         markeredgewidth=2.5, label="Agentic first ALERT")
    miss_x = plt.Line2D([], [], marker="x", color="#999", linestyle="", markersize=8, label="Missed (>21d)")
    onset_line = plt.Line2D([], [], color="black", linewidth=1.5, label="Onset date")
    ax.legend(handles=[onset_line, cls_line, ag_line, miss_x], loc="lower right", fontsize=9)

    ax.set_title("Detection Timeline: First ALERT Relative to Onset", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig9_detection_timeline.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  fig9_detection_timeline.png")


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------
def main():
    FIGDIR.mkdir(parents=True, exist_ok=True)
    print(f"Generating figures in {FIGDIR}/")

    fig1_latency_bars()
    fig2_fpr_dotplot()
    fig3_recall_grid()
    fig4_triage_distribution()
    fig5_failure_rates()
    fig6_leakage()
    fig7_strategy_breakdown()
    fig8_summary_panel()
    fig9_detection_timeline()

    print(f"\nDone — {len(list(FIGDIR.glob('*.png')))} figures generated.")


if __name__ == "__main__":
    main()
