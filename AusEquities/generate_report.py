"""
generate_report.py — produce monitoring-style figures for AU backtest results.

Generates the same fig_a / fig_b / metrics_table outputs as the US strategies
in monitoring/results/live_backtest/, adapted for the AU strategies.

  fig_a_portfolio_value.png  — strategy vs ASX 200 EW benchmark ($1M start)
  fig_b_pnl_drawdown_exposure.png — PnL / drawdown / exposure (3-panel)
  metrics_table.png          — full performance table
  metrics.json

Run from anywhere; the script locates itself correctly.
"""

import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import data_norgate as dn   # noqa: E402

# Add monitoring to path so we can reuse report.compute_metrics
MONITORING = HERE.parent / "monitoring" / "live_backtest"
sys.path.insert(0, str(MONITORING))
from report import compute_metrics, write_metrics_table  # noqa: E402

INITIAL_VALUE = 1_000_000  # $1M starting capital

C_STRATEGY = "#E8553A"   # orange-red (matches monitoring palette)
C_BENCH    = "#2176AE"   # blue


# ─────────────────────────────────────────────────────────────────────────────
# Benchmark: equal-weighted ASX 200 buy-and-hold
# ─────────────────────────────────────────────────────────────────────────────

def build_benchmark(prices: pd.DataFrame, membership: pd.DataFrame) -> pd.Series:
    """
    Equal-weighted daily return of all current ASX 200 members.
    Uses only tickers that are members on each day (PIT).
    Returns a portfolio-value series starting at $1M.
    """
    ret = prices.astype(float).ffill().pct_change()
    # mask to members only, then take equal-weighted mean
    masked = ret.where(membership.reindex(ret.index).fillna(False))
    bm_ret = masked.mean(axis=1)
    bm_value = (1 + bm_ret).cumprod() * INITIAL_VALUE
    bm_value.name = "ASX 200 EW"
    return bm_value


# ─────────────────────────────────────────────────────────────────────────────
# Figures
# ─────────────────────────────────────────────────────────────────────────────

def fig_a_portfolio_value(
    strategy_value: pd.Series,
    benchmark_value: pd.Series,
    strategy_label: str,
    outdir: Path,
) -> Path:
    """Portfolio value ($) on log-y scale vs ASX 200 EW benchmark."""
    fig, ax = plt.subplots(figsize=(14, 6))

    ax.plot(strategy_value.index, strategy_value.values,
            color=C_STRATEGY, linewidth=1.4, label=strategy_label)
    ax.plot(benchmark_value.index, benchmark_value.values,
            color=C_BENCH, linewidth=1.0, alpha=0.7, label="ASX 200 EW B&H")

    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax.set_ylabel("Portfolio Value ($)", fontsize=11)
    ax.set_title(
        f"Live Trading Backtest: {strategy_label}  ($1M Start, log scale)",
        fontsize=13, fontweight="bold",
    )
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    path = outdir / "fig_a_portfolio_value.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


def fig_b_pnl_drawdown_exposure(
    strategy_value: pd.Series,
    drawdown: pd.Series,
    exposure: pd.Series,
    strategy_label: str,
    outdir: Path,
) -> Path:
    """3-panel: cumulative PnL ($), drawdown (%), gross exposure (%)."""
    fig, axes = plt.subplots(
        3, 1, figsize=(14, 10), sharex=True,
        gridspec_kw={"height_ratios": [3, 2, 1.5]},
    )

    cum_pnl = strategy_value - strategy_value.iloc[0]
    axes[0].plot(cum_pnl.index, cum_pnl.values, color=C_STRATEGY, linewidth=1.0)
    axes[0].axhline(0, color="black", linewidth=0.4)
    axes[0].yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"${x:,.0f}")
    )
    axes[0].set_ylabel("Cumulative PnL ($)", fontsize=10)
    axes[0].set_title(
        f"{strategy_label}: PnL, Drawdown & Exposure",
        fontsize=13, fontweight="bold",
    )
    axes[0].grid(True, alpha=0.3)

    dd_pct = drawdown * 100
    axes[1].fill_between(dd_pct.index, dd_pct.values, 0,
                         color=C_STRATEGY, alpha=0.35)
    axes[1].plot(dd_pct.index, dd_pct.values, color=C_STRATEGY, linewidth=0.6)
    axes[1].set_ylabel("Drawdown (%)", fontsize=10)
    axes[1].grid(True, alpha=0.3)

    exp_pct = exposure * 100
    axes[2].fill_between(exp_pct.index, exp_pct.values, 0,
                         color="#4CAF50", alpha=0.35, step="post")
    axes[2].step(exp_pct.index, exp_pct.values, color="#4CAF50",
                 linewidth=0.8, where="post")
    axes[2].set_ylim(-2, max(exp_pct.max() * 1.1, 10))
    axes[2].set_ylabel("Gross Exposure (%)", fontsize=10)
    axes[2].grid(True, alpha=0.3)

    fig.tight_layout()
    path = outdir / "fig_b_pnl_drawdown_exposure.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


# ─────────────────────────────────────────────────────────────────────────────
# Per-strategy report builders
# ─────────────────────────────────────────────────────────────────────────────

def report_jt_momentum(prices: pd.DataFrame, membership: pd.DataFrame,
                        benchmark_value: pd.Series) -> None:
    outdir = HERE / "results" / "jt_momentum"
    outdir.mkdir(parents=True, exist_ok=True)

    curve = pd.read_csv(outdir / "equity_curve.csv", index_col=0, parse_dates=True)
    strategy_value = curve["equity"] * INITIAL_VALUE

    # Drawdown from equity curve
    drawdown = curve["drawdown"]

    # Exposure: fixed 40% gross (20% long + 20% short each month)
    # between the first and last active day, 0 outside
    exposure = pd.Series(0.0, index=curve.index)
    active = curve["port_ret"].abs() > 0
    if active.any():
        exposure.loc[active.idxmax():] = 0.40

    label = "ASX 200 — JT Momentum 12-1"
    print(f"\n[JT Momentum]")

    fig_a_portfolio_value(strategy_value, benchmark_value.reindex(strategy_value.index).ffill(),
                          label, outdir)
    fig_b_pnl_drawdown_exposure(strategy_value, drawdown, exposure, label, outdir)

    m_strat = compute_metrics(strategy_value)
    m_bench = compute_metrics(benchmark_value.reindex(strategy_value.index).ffill())
    write_metrics_table(
        {"JT Momentum 12-1": m_strat, "ASX 200 EW Benchmark": m_bench},
        outdir,
    )
    print(f"  Metrics: Sharpe {m_strat['sharpe']:.3f}  CAGR {m_strat['cagr']:.2%}  "
          f"MaxDD {m_strat['max_dd']:.2%}")


def report_al_statarb(prices: pd.DataFrame, membership: pd.DataFrame,
                       benchmark_value: pd.Series) -> None:
    outdir = HERE / "results" / "al_statarb"
    outdir.mkdir(parents=True, exist_ok=True)

    curve = pd.read_csv(outdir / "equity_curve.csv", index_col=0, parse_dates=True)
    strategy_value = curve["equity"] * INITIAL_VALUE
    drawdown = curve["drawdown"]

    # Exposure from trades: we can reconstruct from long_ret / short_ret being non-zero
    # Approximate: count active long + short days / total tickers
    # Simpler: load s_scores and compute fraction of tickers with active signal
    # Even simpler: derive gross exposure from non-zero daily returns proxy
    # Use |long_ret| + |short_ret| normalised — but easier to just mark exposure
    # whenever the strategy is active (non-zero cum_ret change).
    gross_exp = (curve["long_ret"].abs() + curve["short_ret"].abs())
    # Cap and smooth — this gives relative exposure not 0/1 fraction
    # Better: load position data if available, else use a simple flag
    s_scores_path = outdir / "s_scores.csv"
    if s_scores_path.exists():
        s = pd.read_csv(s_scores_path, index_col=0, parse_dates=True)
        n_active = ((s < -1.25) | (s > 1.25)).sum(axis=1)
        total_tickers = s.shape[1]
        raw_exposure = (n_active / total_tickers).reindex(curve.index).ffill().fillna(0)
        # gross exposure = signal fraction × 2 (long + short)
        exposure = (raw_exposure * 2).clip(upper=1.0)
    else:
        active = curve["cum_ret"].abs() > 0
        exposure = pd.Series(0.0, index=curve.index)
        if active.any():
            exposure.loc[active.idxmax():] = 0.30

    label = "ASX 200 — AL PCA Stat Arb"
    print(f"\n[AL PCA Stat Arb]")

    fig_a_portfolio_value(strategy_value, benchmark_value.reindex(strategy_value.index).ffill(),
                          label, outdir)
    fig_b_pnl_drawdown_exposure(strategy_value, drawdown, exposure, label, outdir)

    m_strat = compute_metrics(strategy_value)
    m_bench = compute_metrics(benchmark_value.reindex(strategy_value.index).ffill())
    write_metrics_table(
        {"AL PCA Stat Arb": m_strat, "ASX 200 EW Benchmark": m_bench},
        outdir,
    )
    print(f"  Metrics: Sharpe {m_strat['sharpe']:.3f}  CAGR {m_strat['cagr']:.2%}  "
          f"MaxDD {m_strat['max_dd']:.2%}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    print("Loading prices and membership...")
    prices = dn.load_prices()
    membership = dn.load_membership(prices)

    print("Building ASX 200 EW benchmark...")
    benchmark_value = build_benchmark(prices, membership)

    report_jt_momentum(prices, membership, benchmark_value)
    report_al_statarb(prices, membership, benchmark_value)

    print("\nDone.")


if __name__ == "__main__":
    main()
