"""
plot_sector_comparison.py — All sectors on one chart, each starting at $1M.

Re-runs sector backtests (fast, no s-score recomputation) and overlays all
net equity curves + the ASX 200 EW benchmark on a single axes.
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
import run_al_statarb as al
from run_sector_statarb import (
    build_sector_map, compute_sector_pca, load_sector_benchmark, INITIAL_VALUE
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

OUT_DIR = HERE / "results" / "sector_statarb"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MIN_STOCKS = 8


def run_sector_net(sector, tickers, prices, membership):
    tickers = [t for t in tickers if t in prices.columns]
    if membership.reindex(columns=tickers).any(axis=0).sum() < MIN_STOCKS:
        return None, None

    pca_factors = compute_sector_pca(prices, membership, tickers)
    if pca_factors.empty or len(pca_factors) < 10:
        return None, None

    s_scores = al.compute_s_scores(
        prices[tickers], membership.reindex(columns=tickers).fillna(False),
        pca_factors, n_window=60
    )
    if s_scores.empty:
        return None, None

    ret = prices[tickers].astype(float).ffill().pct_change()
    s_aligned = s_scores.reindex(ret.index).ffill()
    common = [t for t in s_scores.columns if t in ret.columns]
    if not common:
        return None, None

    ret_c = ret[common]
    s_c   = s_aligned[common]
    pos_long  = al.build_position_df(s_c < al.S_BO,  s_c > al.S_BC,  ret_c)
    pos_short = al.build_position_df(s_c > al.S_SO,  s_c < al.S_SC,  ret_c)

    _, long_net  = al.portfolio_returns(ret_c, pos_long,  "long")
    _, short_net = al.portfolio_returns(ret_c, pos_short, "short")

    net = (long_net + short_net).fillna(0)
    active = (pos_long.abs().sum(axis=1) + pos_short.abs().sum(axis=1)) > 0
    if not active.any():
        return None, None

    start  = active.idxmax()
    net_eq = (1 + net).cumprod().loc[start:] * INITIAL_VALUE

    # Rebase to exactly $1M at start (cumprod before start is all-zero returns = 1.0 already)
    net_eq = net_eq / net_eq.iloc[0] * INITIAL_VALUE

    final_cagr = ((net_eq.iloc[-1] / INITIAL_VALUE) ** (252 / len(net_eq)) - 1)
    return net_eq, final_cagr


def main():
    log.info("Loading ASX 200 data...")
    prices     = dn.load_prices()
    membership = dn.load_membership(prices)

    log.info("Building GICS sector map...")
    sector_map = build_sector_map(list(prices.columns))

    # ASX 200 EW benchmark
    log.info("Building ASX 200 EW benchmark...")
    ret_all = prices.astype(float).ffill().pct_change()
    masked  = ret_all.where(membership.reindex(ret_all.index).fillna(False))
    bm_ret  = masked.mean(axis=1)
    bm_eq   = (1 + bm_ret).cumprod() * INITIAL_VALUE
    bm_eq.name = "ASX 200 EW B&H"

    # Colour palette — one distinct colour per sector
    sectors_sorted = sorted(sector_map.keys())
    colours = cm.tab10(np.linspace(0, 1, len(sectors_sorted)))

    fig, ax = plt.subplots(figsize=(16, 8))

    sector_results = []

    for colour, sector in zip(colours, sectors_sorted):
        log.info("[%s]", sector)
        net_eq, cagr = run_sector_net(
            sector, sector_map[sector], prices, membership
        )
        if net_eq is None:
            log.info("  skipped")
            continue
        sector_results.append((sector, net_eq, cagr))

    # Sort by final value for legend ordering
    sector_results.sort(key=lambda x: x[2], reverse=True)

    for i, (sector, net_eq, cagr) in enumerate(sector_results):
        colour = colours[i % len(colours)]
        label  = f"{sector}  ({cagr:+.0%} CAGR)"
        ax.plot(net_eq.index, net_eq.values,
                color=colour, linewidth=1.3, label=label)

    # Benchmark — bold black
    bm_start = sector_results[0][1].index[0] if sector_results else bm_eq.index[0]
    bm_aligned = bm_eq.reindex(
        pd.date_range(bm_start, bm_eq.index[-1], freq="B")
    ).ffill()
    bm_cagr = ((bm_aligned.iloc[-1] / INITIAL_VALUE) ** (252 / len(bm_aligned)) - 1)
    ax.plot(bm_aligned.index, bm_aligned.values,
            color="black", linewidth=2.0, linestyle="--",
            label=f"ASX 200 EW B&H  ({bm_cagr:+.0%} CAGR)", zorder=10)

    ax.axhline(INITIAL_VALUE, color="grey", linewidth=0.5, linestyle=":")
    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"${x/1e3:.0f}k")
    )
    ax.set_ylabel("Portfolio Value (each starts at $1M)", fontsize=12)
    ax.set_title(
        "AL PCA Stat Arb — Net Performance by GICS Sector  "
        "(ASX 200 C&P, each sector independent $1M, after all TC + borrow)",
        fontsize=13, fontweight="bold",
    )
    ax.legend(fontsize=9, loc="upper left", framealpha=0.85)
    ax.grid(True, alpha=0.25)

    path = OUT_DIR / "sector_overlay.png"
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved: %s", path)
    print(f"\nOutputs in: {OUT_DIR}")


if __name__ == "__main__":
    main()
