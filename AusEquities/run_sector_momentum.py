"""
run_sector_momentum.py — JT Momentum 12-1 run independently within each GICS sector.

Buys the top 20% and shorts the bottom 20% of momentum winners/losers within
each sector only, rebalancing monthly.  Each sector starts at $1M independently.
Compared against the official S&P/ASX 200 sector total-return benchmark.

Output: AusEquities/results/sector_momentum/sector_overlay.png
"""

import sys
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.cm as cm

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

import data_norgate as dn
import run_jt_momentum as jt
from run_sector_statarb import build_sector_map, load_sector_benchmark, INITIAL_VALUE

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

OUT_DIR = HERE / "results" / "sector_momentum"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Lower minimum leg size for small sectors (e.g. 2 stocks each side for Energy/IT)
MIN_STOCKS_LEG_SECTOR = 2


def run_sector_jt(sector, tickers, prices, membership):
    """
    Run JT 12-1 momentum on sector stocks only.
    Returns (gross_eq, net_eq, cagr_net) or (None, None, None) if skipped.
    """
    tickers = [t for t in tickers if t in prices.columns]
    if len(tickers) < MIN_STOCKS_LEG_SECTOR * 2:
        log.info("  %s: only %d tickers — skipping", sector, len(tickers))
        return None, None, None, 0

    sec_prices = prices[tickers]
    sec_mem    = membership.reindex(columns=tickers).fillna(False)

    # Compute 12-1 momentum scores within sector
    scores  = jt.compute_momentum_scores(sec_prices, sec_mem)

    # Build weights — temporarily lower the global min-leg threshold
    orig_min = jt.MIN_STOCKS_LEG
    jt.MIN_STOCKS_LEG = MIN_STOCKS_LEG_SECTOR
    weights = jt.build_weights(scores)
    jt.MIN_STOCKS_LEG = orig_min

    if weights.isna().all().all():
        log.info("  %s: no valid weights", sector)
        return None, None, None, 0

    # Run gross and net backtest
    daily_ret = sec_prices.pct_change(fill_method=None)
    # ffill() on weights first: skipped months (all-NaN rows) inherit previous
    # month's holdings so the portfolio doesn't suddenly go to zero exposure.
    w  = weights.astype(float).ffill().reindex(columns=daily_ret.columns)
    w_daily = w.reindex(daily_ret.index, method="ffill").shift(1)
    w0 = w_daily.fillna(0)
    delta_w  = w0.diff().abs()
    gross_ret = (w0 * daily_ret.fillna(0)).sum(axis=1)

    active = w0.abs().sum(axis=1) > 0
    if not active.any():
        return None, None, None, 0
    start = active.idxmax()

    # Gross track
    gross_port = []
    V_g = INITIAL_VALUE
    for i in range(len(daily_ret)):
        r = gross_ret.iloc[i]
        gross_port.append(r)
        V_g = max(V_g * (1 + r), 1.0)

    # Net track (tiered TC via jt._tiered_tc_frac)
    net_port  = []
    tc_paid   = []          # track each day's TC for cumulative reporting
    V_n = INITIAL_VALUE
    for i in range(len(daily_ret)):
        tc = jt._tiered_tc_frac(delta_w.iloc[i], V_n)
        r  = gross_ret.iloc[i] - tc
        net_port.append(r)
        tc_paid.append(tc * V_n)   # in dollar terms at current portfolio value
        V_n = max(V_n * (1 + r), 1.0)

    idx      = daily_ret.index
    gross_eq = (1 + pd.Series(gross_port, index=idx)).cumprod().loc[start:] * INITIAL_VALUE
    net_eq   = (1 + pd.Series(net_port,   index=idx)).cumprod().loc[start:] * INITIAL_VALUE

    # Rebase both by the GROSS starting value so initial entry TC is preserved in
    # the drag.  Dividing each series by its own start would cancel day-1 TC on net.
    scale    = gross_eq.iloc[0]
    gross_eq = gross_eq / scale * INITIAL_VALUE
    net_eq   = net_eq   / scale * INITIAL_VALUE

    total_tc_paid = sum(tc_paid)   # cumulative TC extracted from portfolio ($)

    n_periods = len(net_eq)
    cagr_net  = (net_eq.iloc[-1] / INITIAL_VALUE) ** (252 / n_periods) - 1

    log.info("  %s: %d stocks  gross CAGR %+.1f%%  net CAGR %+.1f%%",
             sector, len(tickers),
             ((gross_eq.iloc[-1] / INITIAL_VALUE) ** (252 / n_periods) - 1) * 100,
             cagr_net * 100)

    return gross_eq, net_eq, cagr_net, total_tc_paid


def _sharpe(eq: pd.Series) -> float:
    r = eq.pct_change().dropna()
    return float(np.sqrt(252) * r.mean() / r.std()) if r.std() > 0 else np.nan

def _maxdd(eq: pd.Series) -> float:
    return float((eq / eq.cummax() - 1).min())


def main():
    log.info("=== JT Momentum — Sector Analysis ===")

    log.info("Loading ASX 200 data...")
    prices     = dn.load_prices()
    membership = dn.load_membership(prices)

    log.info("Building GICS sector map...")
    sector_map = build_sector_map(list(prices.columns))

    # ASX 200 EW benchmark
    ret_all = prices.astype(float).ffill().pct_change()
    masked  = ret_all.where(membership.reindex(ret_all.index).fillna(False))
    bm_ret  = masked.mean(axis=1)
    bm_eq   = (1 + bm_ret).cumprod() * INITIAL_VALUE

    sectors_sorted = sorted(sector_map.keys())
    colours = cm.tab10(np.linspace(0, 1, len(sectors_sorted)))

    results = []
    for sector in sectors_sorted:
        log.info("[%s]", sector)
        gross_eq, net_eq, cagr, total_tc = run_sector_jt(
            sector, sector_map[sector], prices, membership
        )
        if net_eq is not None:
            results.append((sector, gross_eq, net_eq, cagr, total_tc))

    # Sort by CAGR descending for legend ordering
    results.sort(key=lambda x: x[3], reverse=True)   # sort by CAGR

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(16, 8))

    for i, (sector, gross_eq, net_eq, cagr, total_tc) in enumerate(results):
        colour = colours[i % len(colours)]
        ax.plot(net_eq.index, net_eq.values,
                color=colour, linewidth=1.3,
                label=f"{sector}  ({cagr:+.0%} CAGR)")

    # Benchmark
    if results:
        bm_start   = results[0][2].index[0]
        bm_aligned = bm_eq.reindex(
            pd.date_range(bm_start, bm_eq.index[-1], freq="B")
        ).ffill()
        bm_cagr = (bm_aligned.iloc[-1] / INITIAL_VALUE) ** (252 / len(bm_aligned)) - 1
        ax.plot(bm_aligned.index, bm_aligned.values,
                color="black", linewidth=2.0, linestyle="--",
                label=f"ASX 200 EW B&H  ({bm_cagr:+.0%} CAGR)", zorder=10)

    ax.axhline(INITIAL_VALUE, color="grey", linewidth=0.5, linestyle=":")
    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"${x/1e3:.0f}k")
    )
    ax.set_ylabel("Portfolio Value (each starts at $1M)", fontsize=12)
    ax.set_title(
        "JT Momentum 12-1 — Net Performance by GICS Sector  "
        "(ASX 200 C&P, each sector independent $1M, after all TC)",
        fontsize=13, fontweight="bold",
    )
    ax.legend(fontsize=9, loc="upper left", framealpha=0.85)
    ax.grid(True, alpha=0.25)

    path = OUT_DIR / "sector_overlay.png"
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved: %s", path)

    # ── Summary grid ──────────────────────────────────────────────────────────
    ncols = 3
    nrows = (len(results) + ncols - 1) // ncols
    fig2, axes = plt.subplots(nrows, ncols, figsize=(18, nrows * 4),
                              constrained_layout=True)
    axes_flat = axes.flatten()
    fmt = mticker.FuncFormatter(lambda x, _: f"${x/1e3:.0f}k")

    for ax, (sector, gross_eq, net_eq, cagr, total_tc) in zip(axes_flat, results):
        n = len(sector_map[sector])
        bench_sec = load_sector_benchmark(sector, net_eq.index)

        gross_cagr = (gross_eq.iloc[-1] / INITIAL_VALUE) ** (252 / len(gross_eq)) - 1

        ax.plot(gross_eq.index, gross_eq.values, color="#2176AE",
                linewidth=1.0, alpha=0.7, label=f"Gross {gross_cagr:+.0%}")
        ax.plot(net_eq.index, net_eq.values, color="#E8553A",
                linewidth=1.3, label=f"Net {cagr:+.0%}")
        if not bench_sec.empty:
            b = bench_sec.reindex(net_eq.index).ffill()
            b_cagr = (b.iloc[-1] / INITIAL_VALUE) ** (252 / len(b)) - 1
            ax.plot(b.index, b.values, color="#4CAF50", linewidth=1.0,
                    alpha=0.85, label=f"Sector TR {b_cagr:+.0%}")
        ax.axhline(INITIAL_VALUE, color="black", linewidth=0.4, linestyle="--")
        ax.yaxis.set_major_formatter(fmt)
        ax.set_title(
            f"{sector}\n"
            f"Net Sharpe {_sharpe(net_eq):.2f}  MaxDD {_maxdd(net_eq):.0%}  "
            f"({n} stks)",
            fontsize=8.5, fontweight="bold",
        )
        ax.legend(fontsize=7, loc="upper left")
        ax.grid(True, alpha=0.25)
        ax.tick_params(labelsize=7)

    for ax in axes_flat[len(results):]:
        ax.set_visible(False)

    fig2.suptitle(
        "JT Momentum 12-1 — Per Sector  (ASX 200 C&P, $1M Start, net of TC)",
        fontsize=13, fontweight="bold",
    )
    grid_path = OUT_DIR / "summary_grid.png"
    fig2.savefig(grid_path, dpi=180, bbox_inches="tight")
    plt.close(fig2)
    log.info("Saved: %s", grid_path)

    # ── Summary table ─────────────────────────────────────────────────────────
    print("\n=== JT Momentum — Sector Results ===")
    print(f"{'Sector':<30} {'N Stocks':>8} {'Net CAGR':>10} {'TC Paid $':>12} {'Terminal Drag $':>16}")
    print("-" * 75)
    for sector, gross_eq, net_eq, cagr, total_tc in results:
        n    = len(sector_map[sector])
        drag = gross_eq.iloc[-1] - net_eq.iloc[-1]
        print(f"{sector:<30} {n:>8} {cagr:>+10.1%} ${total_tc:>10,.0f} ${drag:>14,.0f}")

    print(f"\nOutputs in: {OUT_DIR}")


if __name__ == "__main__":
    main()
