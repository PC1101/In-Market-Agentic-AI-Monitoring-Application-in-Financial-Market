"""Performance metrics and figure generation for live-trading backtest."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

# Palette (from pnl_comparison.py)
C_UNMANAGED = "#888888"
C_AGENT = "#E8553A"
C_BENCH = "#2176AE"


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(value: pd.Series, rf: float = 0.0) -> dict:
    """Compute annualised performance metrics from a daily portfolio-value series.

    Args:
        value: daily portfolio value (dollar amounts).
        rf: annual risk-free rate for Sharpe computation.

    Returns:
        dict with CAGR, ann_vol, sharpe, max_dd, calmar, terminal_value,
        total_return.
    """
    daily_ret = value.pct_change().dropna()
    n_days = len(daily_ret)
    if n_days < 2:
        return {k: float("nan") for k in
                ["cagr", "ann_vol", "sharpe", "max_dd", "calmar",
                 "terminal_value", "total_return"]}

    total_return = value.iloc[-1] / value.iloc[0] - 1
    n_years = n_days / 252
    cagr = (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else 0.0
    ann_vol = daily_ret.std() * np.sqrt(252)
    sharpe = (cagr - rf) / ann_vol if ann_vol > 0 else 0.0

    peak = value.cummax()
    dd = (value - peak) / peak
    max_dd = float(dd.min())
    calmar = cagr / abs(max_dd) if max_dd != 0 else float("inf")

    return {
        "cagr": round(cagr, 6),
        "ann_vol": round(ann_vol, 6),
        "sharpe": round(sharpe, 4),
        "max_dd": round(max_dd, 6),
        "calmar": round(calmar, 4),
        "terminal_value": round(float(value.iloc[-1]), 2),
        "total_return": round(total_return, 6),
    }


def write_metrics_table(results: dict[str, dict], outdir: Path) -> Path:
    """Write metrics.json and a matplotlib summary-table PNG.

    Args:
        results: mapping of label -> metrics dict (from compute_metrics).
        outdir: output directory.

    Returns:
        Path to metrics.json.
    """
    outdir.mkdir(parents=True, exist_ok=True)

    # JSON
    json_path = outdir / "metrics.json"
    json_path.write_text(json.dumps(results, indent=2))

    # Table figure
    labels = list(results.keys())
    metrics_keys = ["cagr", "ann_vol", "sharpe", "max_dd", "calmar",
                    "terminal_value", "total_return"]
    header = [""] + [k.replace("_", " ").title() for k in metrics_keys]
    rows = []
    for label in labels:
        m = results[label]
        rows.append([
            label,
            f"{m['cagr']:.2%}",
            f"{m['ann_vol']:.2%}",
            f"{m['sharpe']:.2f}",
            f"{m['max_dd']:.2%}",
            f"{m['calmar']:.2f}",
            f"${m['terminal_value']:,.0f}",
            f"{m['total_return']:.2%}",
        ])

    fig, ax = plt.subplots(figsize=(14, max(2.5, len(rows) * 0.8 + 1.5)))
    ax.axis("off")
    table = ax.table(cellText=rows, colLabels=header, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.8)

    # Style header
    for j in range(len(header)):
        table[0, j].set_facecolor("#263238")
        table[0, j].set_text_props(color="white", fontweight="bold", fontsize=8)
    for i in range(1, len(rows) + 1):
        table[i, 0].set_text_props(fontweight="bold")

    ax.set_title("Live Trading Backtest: Performance Summary",
                 fontsize=13, fontweight="bold", pad=20)
    fig.tight_layout()
    fig.savefig(outdir / "metrics_table.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    return json_path


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def _shade_windows(ax, windows, xlim_start, xlim_end):
    """Add shaded window spans and onset vlines to an axes."""
    for w in windows:
        s, e = w.start_ts, w.end_ts
        if e < xlim_start or s > xlim_end:
            continue
        color = "#FFCDD2" if w.kind == "event" else "#E0E0E0"
        ax.axvspan(s, e, alpha=0.25, color=color, zorder=0)
        if w.onset_ts is not None:
            ax.axvline(w.onset_ts, color="#B71C1C", linewidth=0.6,
                       linestyle="--", alpha=0.5, zorder=1)


def fig_portfolio_value(managed_value: pd.Series,
                        unmanaged_value: pd.Series,
                        benchmark_value: pd.Series | None,
                        windows: tuple,
                        strategy: str,
                        outdir: Path) -> Path:
    """Fig (a): three portfolio-value ($) curves on log-y with window shading."""
    fig, ax = plt.subplots(figsize=(16, 7))

    ax.plot(unmanaged_value.index, unmanaged_value.values,
            color=C_UNMANAGED, linewidth=1.0, label=f"{strategy} (unmanaged)", alpha=0.7)
    ax.plot(managed_value.index, managed_value.values,
            color=C_AGENT, linewidth=1.2, label=f"{strategy} + Agent")
    if benchmark_value is not None:
        ax.plot(benchmark_value.index, benchmark_value.values,
                color=C_BENCH, linewidth=1.0, label="S&P 500 B&H", alpha=0.7)

    _shade_windows(ax, windows, managed_value.index[0], managed_value.index[-1])

    ax.set_yscale("log")
    ax.set_ylabel("Portfolio Value ($)", fontsize=12)
    ax.set_xlabel("")
    ax.set_title(f"Live Trading Backtest: {strategy} + Agentic Risk Overlay ($1M Start)",
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=10, loc="upper left")
    ax.grid(True, alpha=0.3)

    # Format y-axis as dollars
    from matplotlib.ticker import FuncFormatter
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"${x:,.0f}"))

    fig.tight_layout()
    path = outdir / "fig_a_portfolio_value.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def fig_pnl_drawdown(sim_df: pd.DataFrame,
                     windows: tuple,
                     strategy: str,
                     outdir: Path) -> Path:
    """Fig (b): 3 stacked panels — cumulative PnL ($), drawdown (%), exposure."""
    fig, axes = plt.subplots(3, 1, figsize=(16, 10), sharex=True,
                             gridspec_kw={"height_ratios": [3, 2, 1.5]})

    idx = sim_df.index
    xlim_s, xlim_e = idx[0], idx[-1]

    # Panel 1: cumulative PnL
    cum_pnl = sim_df["value"] - sim_df["value"].iloc[0]
    axes[0].plot(idx, cum_pnl.values, color=C_AGENT, linewidth=1.0)
    axes[0].axhline(0, color="black", linewidth=0.3)
    axes[0].set_ylabel("Cumulative PnL ($)", fontsize=10)
    axes[0].set_title(f"{strategy} + Agentic Overlay: PnL, Drawdown & Exposure",
                      fontsize=13, fontweight="bold")
    from matplotlib.ticker import FuncFormatter
    axes[0].yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"${x:,.0f}"))
    _shade_windows(axes[0], windows, xlim_s, xlim_e)
    axes[0].grid(True, alpha=0.3)

    # Panel 2: drawdown
    axes[1].fill_between(idx, sim_df["drawdown"].values * 100, 0,
                         color=C_AGENT, alpha=0.4)
    axes[1].plot(idx, sim_df["drawdown"].values * 100, color=C_AGENT, linewidth=0.5)
    axes[1].set_ylabel("Drawdown (%)", fontsize=10)
    _shade_windows(axes[1], windows, xlim_s, xlim_e)
    axes[1].grid(True, alpha=0.3)

    # Panel 3: exposure
    axes[2].fill_between(idx, sim_df["exposure"].values * 100, 0,
                         color="#4CAF50", alpha=0.4, step="post")
    axes[2].step(idx, sim_df["exposure"].values * 100, color="#4CAF50",
                 linewidth=0.8, where="post")
    axes[2].set_ylabel("Exposure (%)", fontsize=10)
    axes[2].set_ylim(-5, 115)
    _shade_windows(axes[2], windows, xlim_s, xlim_e)
    axes[2].grid(True, alpha=0.3)

    fig.tight_layout()
    path = outdir / "fig_b_pnl_drawdown_exposure.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path
