"""Generate gross vs net comparison tables and figures.

Reads live-backtest results from {STRAT}_gross and {STRAT}_net directories,
produces:
  - metrics_gross_net.json + table PNG per strategy
  - fig_c_gross_net_equity.png — equity curves (gross vs net, managed vs unmanaged)
  - fig_d_cost_drag.png — cumulative cost drag in $ and annualised bps
  - sensitivity_table.png — low/base/high comparison

Usage (from monitoring/):
    python scripts/gross_net_comparison.py
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

RESULTS = Path(__file__).resolve().parents[1] / "results"
LB = RESULTS / "live_backtest"

STRATEGIES = ["JT_MOM", "AL_PCA"]
SCENARIOS = ["low", "base", "high"]

# Palette
C_GROSS = "#2176AE"
C_NET = "#E8553A"
C_UNMANAGED_G = "#90CAF9"
C_UNMANAGED_N = "#FFAB91"
C_SPY = "#888888"

# Cost schedule eras for annotation
ERAS = [("2004", "2009"), ("2010", "2018"), ("2019", "2025")]
ERA_LABELS = ["2004–09", "2010–18", "2019–25"]


def load_metrics(dirname: str) -> dict:
    p = LB / dirname / "metrics.json"
    if not p.exists():
        return {}
    return json.loads(p.read_text())


def load_portfolio(dirname: str) -> pd.DataFrame:
    p = LB / dirname / "portfolio_daily.csv"
    if not p.exists():
        return pd.DataFrame()
    return pd.read_csv(p, index_col=0, parse_dates=True)


def fmt_pct(v): return f"{v:.2%}" if not np.isnan(v) else "—"
def fmt_dollar(v): return f"${v:,.0f}" if not np.isnan(v) else "—"
def fmt_ratio(v): return f"{v:.2f}" if not np.isnan(v) else "—"


def make_comparison_table(strat: str, outdir: Path) -> dict:
    """Build and save the gross vs net metrics comparison table."""
    gross_m = load_metrics(f"{strat}_gross")
    net_m = load_metrics(f"{strat}_net")

    if not gross_m or not net_m:
        logger.warning("Missing metrics for %s, skipping table", strat)
        return {}

    rows_data = {}
    for label_key, display in [
        (f"{strat} + Agent", "Managed"),
        (f"{strat} (unmanaged)", "Unmanaged"),
    ]:
        for variant, suffix in [("Gross", gross_m), ("Net", net_m)]:
            m = suffix.get(label_key, {})
            rows_data[f"{display} ({variant})"] = m

    spy = gross_m.get("SPY B&H", {})
    rows_data["SPY B&H"] = spy

    # Write JSON
    outdir.mkdir(parents=True, exist_ok=True)
    json_path = outdir / f"metrics_gross_net_{strat}.json"
    json_path.write_text(json.dumps(rows_data, indent=2))

    # PNG table
    metrics_keys = ["cagr", "ann_vol", "sharpe", "max_dd", "calmar",
                    "terminal_value", "total_return"]
    header = [""] + [k.replace("_", " ").title() for k in metrics_keys]
    table_rows = []
    for label, m in rows_data.items():
        table_rows.append([
            label,
            fmt_pct(m.get("cagr", float("nan"))),
            fmt_pct(m.get("ann_vol", float("nan"))),
            fmt_ratio(m.get("sharpe", float("nan"))),
            fmt_pct(m.get("max_dd", float("nan"))),
            fmt_ratio(m.get("calmar", float("nan"))),
            fmt_dollar(m.get("terminal_value", float("nan"))),
            fmt_pct(m.get("total_return", float("nan"))),
        ])

    fig, ax = plt.subplots(figsize=(16, max(3.5, len(table_rows) * 0.9 + 1.5)))
    ax.axis("off")
    table = ax.table(cellText=table_rows, colLabels=header, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1, 1.8)

    for j in range(len(header)):
        table[0, j].set_facecolor("#263238")
        table[0, j].set_text_props(color="white", fontweight="bold", fontsize=8)
    for i in range(1, len(table_rows) + 1):
        table[i, 0].set_text_props(fontweight="bold")
        # Shade net rows
        label = table_rows[i - 1][0]
        if "(Net)" in label:
            for j in range(len(header)):
                table[i, j].set_facecolor("#FFF3E0")

    ax.set_title(f"{strat}: Gross vs Net Performance Comparison",
                 fontsize=13, fontweight="bold", pad=20)
    fig.tight_layout()
    fig.savefig(outdir / f"metrics_gross_net_{strat}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info("  Table: %s", outdir / f"metrics_gross_net_{strat}.png")

    return rows_data


def fig_equity_curves(strat: str, outdir: Path):
    """Fig C: gross vs net equity curves, managed and unmanaged panels."""
    gross_managed = load_portfolio(f"{strat}_gross")
    net_managed = load_portfolio(f"{strat}_net")
    gross_unmanaged = load_portfolio(f"{strat}_gross")
    net_unmanaged = load_portfolio(f"{strat}_net")

    if gross_managed.empty or net_managed.empty:
        logger.warning("Missing portfolio data for %s", strat)
        return

    fig, axes = plt.subplots(1, 2, figsize=(18, 7), sharey=True)

    # Panel 1: Unmanaged (gross vs net)
    ax = axes[0]
    # Reconstruct unmanaged from metrics (we need the unmanaged portfolio values)
    # Load from the unmanaged sim embedded in the run
    gross_m = load_metrics(f"{strat}_gross")
    net_m = load_metrics(f"{strat}_net")

    # Plot managed curves
    ax.plot(gross_managed.index, gross_managed["value"],
            color=C_GROSS, linewidth=1.2, label="Unmanaged (gross)")

    # For unmanaged, reconstruct from strat_ret column at full exposure
    strat_ret = gross_managed["strat_ret"]
    unman_equity_gross = 1e6 * (1 + strat_ret).cumprod()
    ax.plot(gross_managed.index, unman_equity_gross,
            color=C_UNMANAGED_G, linewidth=1.0, label="Managed (gross)", alpha=0.8)

    ax.set_yscale("log")
    ax.set_ylabel("Portfolio Value ($)", fontsize=11)
    ax.set_title(f"{strat}: Managed (Gross vs Net)", fontsize=12, fontweight="bold")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel 2: Managed (gross vs net)
    ax = axes[1]
    ax.plot(gross_managed.index, gross_managed["value"],
            color=C_GROSS, linewidth=1.2, label="Managed+Agent (gross)")
    ax.plot(net_managed.index, net_managed["value"],
            color=C_NET, linewidth=1.2, label="Managed+Agent (net)", linestyle="--")

    ax.set_yscale("log")
    ax.set_title(f"{strat}: Agent Overlay (Gross vs Net)", fontsize=12, fontweight="bold")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.suptitle(f"Gross vs Net: {strat}", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    path = outdir / f"fig_c_gross_net_equity_{strat}.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info("  Fig C: %s", path)


def fig_cost_drag(strat: str, outdir: Path):
    """Fig D: cumulative $ cost drag and annualised bps by era."""
    gross_port = load_portfolio(f"{strat}_gross")
    net_port = load_portfolio(f"{strat}_net")

    if gross_port.empty or net_port.empty:
        return

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), gridspec_kw={"height_ratios": [3, 2]})

    # Panel 1: Cumulative $ drag (gross - net portfolio value)
    drag = gross_port["value"] - net_port["value"]
    ax = axes[0]
    ax.fill_between(drag.index, 0, drag.values, color=C_NET, alpha=0.3)
    ax.plot(drag.index, drag.values, color=C_NET, linewidth=1.0)
    ax.set_ylabel("Cumulative Cost Drag ($)", fontsize=11)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax.set_title(f"{strat}: Cumulative Transaction Cost Drag (Managed + Agent)",
                 fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3)

    # Panel 2: Bar chart of annualised drag by era
    ax = axes[1]
    era_drags = []
    for (start, end), label in zip(ERAS, ERA_LABELS):
        mask = (net_port.index >= start) & (net_port.index <= f"{end}-12-31")
        if mask.sum() == 0:
            era_drags.append(0)
            continue
        gross_era = gross_port.loc[mask, "value"]
        net_era = net_port.loc[mask, "value"]
        # Annualised drag as fraction of gross value
        daily_drag_frac = (gross_era - net_era).diff().fillna(0) / gross_era
        ann_drag_bps = daily_drag_frac.mean() * 252 * 1e4
        era_drags.append(ann_drag_bps)

    colors = ["#BBDEFB", "#90CAF9", "#64B5F6"]
    bars = ax.bar(ERA_LABELS, era_drags, color=colors, edgecolor="#1565C0", linewidth=0.8)
    for bar, val in zip(bars, era_drags):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{val:.0f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_ylabel("Annualised Drag (bps)", fontsize=11)
    ax.set_title("Cost Drag by Era", fontsize=11, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    path = outdir / f"fig_d_cost_drag_{strat}.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info("  Fig D: %s", path)


def sensitivity_table(strat: str, outdir: Path):
    """Table 2: low/base/high sensitivity for key metrics."""
    rows = []
    for scenario in SCENARIOS:
        suffix = f"_net_{scenario}" if scenario != "base" else "_net"
        m = load_metrics(f"{strat}{suffix}")
        managed_key = f"{strat} + Agent"
        unmanaged_key = f"{strat} (unmanaged)"
        mm = m.get(managed_key, {})
        um = m.get(unmanaged_key, {})
        rows.append([
            scenario.capitalize(),
            fmt_pct(um.get("cagr", float("nan"))),
            fmt_ratio(um.get("sharpe", float("nan"))),
            fmt_pct(um.get("max_dd", float("nan"))),
            fmt_pct(mm.get("cagr", float("nan"))),
            fmt_ratio(mm.get("sharpe", float("nan"))),
            fmt_pct(mm.get("max_dd", float("nan"))),
        ])

    header = ["Scenario",
              "Unman. CAGR", "Unman. Sharpe", "Unman. MaxDD",
              "Agent CAGR", "Agent Sharpe", "Agent MaxDD"]

    fig, ax = plt.subplots(figsize=(14, 3))
    ax.axis("off")
    table = ax.table(cellText=rows, colLabels=header, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.8)
    for j in range(len(header)):
        table[0, j].set_facecolor("#263238")
        table[0, j].set_text_props(color="white", fontweight="bold", fontsize=8.5)
    for i in range(1, len(rows) + 1):
        table[i, 0].set_text_props(fontweight="bold")

    ax.set_title(f"{strat}: Cost Sensitivity (Low / Base / High)",
                 fontsize=13, fontweight="bold", pad=20)
    fig.tight_layout()
    path = outdir / f"sensitivity_{strat}.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info("  Sensitivity: %s", path)


def combined_summary_table(outdir: Path):
    """Combined summary across both strategies: gross, net-base, SPY."""
    all_rows = []
    for strat in STRATEGIES:
        for variant, dirname in [("Gross", f"{strat}_gross"), ("Net (base)", f"{strat}_net")]:
            m = load_metrics(dirname)
            for role, key in [("+ Agent", f"{strat} + Agent"),
                              ("Unmanaged", f"{strat} (unmanaged)")]:
                mm = m.get(key, {})
                all_rows.append([
                    f"{strat} {role} ({variant})",
                    fmt_pct(mm.get("cagr", float("nan"))),
                    fmt_pct(mm.get("ann_vol", float("nan"))),
                    fmt_ratio(mm.get("sharpe", float("nan"))),
                    fmt_pct(mm.get("max_dd", float("nan"))),
                    fmt_dollar(mm.get("terminal_value", float("nan"))),
                ])

    # Add SPY (from JT gross — identical across runs)
    spy = load_metrics("JT_MOM_gross").get("SPY B&H", {})
    all_rows.append([
        "SPY B&H",
        fmt_pct(spy.get("cagr", float("nan"))),
        fmt_pct(spy.get("ann_vol", float("nan"))),
        fmt_ratio(spy.get("sharpe", float("nan"))),
        fmt_pct(spy.get("max_dd", float("nan"))),
        fmt_dollar(spy.get("terminal_value", float("nan"))),
    ])

    header = ["Strategy", "CAGR", "Ann Vol", "Sharpe", "Max DD", "Terminal ($1M)"]

    fig, ax = plt.subplots(figsize=(16, max(4, len(all_rows) * 0.7 + 2)))
    ax.axis("off")
    table = ax.table(cellText=all_rows, colLabels=header, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1, 1.7)
    for j in range(len(header)):
        table[0, j].set_facecolor("#263238")
        table[0, j].set_text_props(color="white", fontweight="bold", fontsize=8.5)
    for i in range(1, len(all_rows) + 1):
        table[i, 0].set_text_props(fontweight="bold")
        if "Net" in all_rows[i - 1][0]:
            for j in range(len(header)):
                table[i, j].set_facecolor("#FFF3E0")

    ax.set_title("Live Backtest: Gross vs Net Summary (Both Strategies)",
                 fontsize=14, fontweight="bold", pad=20)
    fig.tight_layout()
    path = outdir / "summary_gross_net.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info("Combined summary: %s", path)


def main():
    outdir = RESULTS / "figures"
    outdir.mkdir(parents=True, exist_ok=True)

    for strat in STRATEGIES:
        logger.info("=== %s ===", strat)
        make_comparison_table(strat, outdir)
        fig_equity_curves(strat, outdir)
        fig_cost_drag(strat, outdir)
        sensitivity_table(strat, outdir)

    combined_summary_table(outdir)
    logger.info("Done — all gross/net figures in %s", outdir)


if __name__ == "__main__":
    main()
