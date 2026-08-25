"""
run_sector_statarb_longonly.py — AL PCA Stat Arb LONG-ONLY per GICS sector (S&P 500).

Long positions only (s-score < -1.25, close > -0.50). No short leg, no borrow cost.
Each sector starts at $1M independently.

Output: USEquities/results/sector_statarb_longonly/
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

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

import data_norgate as dn
import run_al_statarb as al
from run_sector_statarb import (
    build_sector_map, compute_sector_pca, load_sector_benchmark,
    INITIAL_VALUE, MIN_STOCKS, N_WINDOW,
    C_GROSS, C_NET, C_BENCH,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

OUT_DIR = HERE / "results" / "sector_statarb_longonly"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def run_sector_long(sector, tickers, prices, membership):
    tickers = [t for t in tickers if t in prices.columns]
    n_ever_active = membership.reindex(columns=tickers).any(axis=0).sum()
    if n_ever_active < MIN_STOCKS:
        log.info("  %s: only %d active tickers — skipping", sector, n_ever_active)
        return None

    log.info("  %s: %d tickers, computing PCA...", sector, len(tickers))
    pca_factors = compute_sector_pca(prices, membership, tickers)
    if pca_factors.empty or len(pca_factors) < 10:
        log.warning("  %s: insufficient PCA data — skipping", sector)
        return None

    log.info("  %s: fitting OU s-scores...", sector)
    sector_prices = prices[tickers]
    sector_mem    = membership.reindex(columns=tickers).fillna(False)

    s_scores = al.compute_s_scores(sector_prices, sector_mem, pca_factors, n_window=N_WINDOW)
    if s_scores.empty:
        return None

    ret = sector_prices.astype(float).ffill().pct_change()
    s_aligned = s_scores.reindex(ret.index).ffill()
    common = [t for t in s_scores.columns if t in ret.columns]
    if not common:
        return None

    ret_c = ret[common]
    s_c   = s_aligned[common]

    pos_long = al.build_position_df(s_c < al.S_BO, s_c > al.S_BC, ret_c)

    long_gross, long_net = al.portfolio_returns(ret_c, pos_long, "long")

    gross = long_gross.fillna(0)
    net   = long_net.fillna(0)

    active = pos_long.abs().sum(axis=1) > 0
    if not active.any():
        return None
    start = active.idxmax()

    gross_eq = (1 + gross).cumprod().loc[start:] * INITIAL_VALUE
    net_eq   = (1 + net).cumprod().loc[start:]   * INITIAL_VALUE
    bench_eq = load_sector_benchmark(sector, gross_eq.index)

    trades = al.extract_trades(pos_long, ret_c, "long")

    def _metrics(eq):
        r = eq.pct_change().dropna()
        n = len(r)
        total  = eq.iloc[-1] / eq.iloc[0] - 1
        cagr   = (1 + total) ** (252 / n) - 1 if n > 0 else np.nan
        vol    = r.std() * np.sqrt(252)
        sharpe = cagr / vol if vol > 0 else np.nan
        dd     = (eq / eq.cummax() - 1).min()
        return {"cagr": cagr, "vol": vol, "sharpe": sharpe, "max_dd": dd,
                "terminal": float(eq.iloc[-1])}

    log.info("  %s: gross CAGR %+.1f%%  net CAGR %+.1f%%  trades %d",
             sector,
             _metrics(gross_eq)["cagr"] * 100,
             _metrics(net_eq)["cagr"] * 100,
             len(trades))

    return {
        "sector":    sector,
        "n_tickers": len(tickers),
        "n_trades":  len(trades),
        "gross_eq":  gross_eq,
        "net_eq":    net_eq,
        "bench_eq":  bench_eq,
        "m_gross":   _metrics(gross_eq),
        "m_net":     _metrics(net_eq),
        "m_bench":   _metrics(bench_eq) if not bench_eq.empty else {},
    }


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    log.info("=== AL PCA Stat Arb LONG-ONLY — Sector Analysis (S&P 500) ===")

    prices     = dn.load_prices()
    membership = dn.load_membership(prices)
    log.info("  %d days × %d tickers", *prices.shape)

    sector_map = build_sector_map(list(prices.columns))

    results = []
    for sector in sorted(sector_map.keys()):
        log.info("\n[%s]", sector)
        res = run_sector_long(sector, sector_map[sector], prices, membership)
        if res is not None:
            results.append(res)

    if not results:
        log.error("No sectors produced results.")
        return

    results.sort(key=lambda r: r["m_net"]["sharpe"], reverse=True)

    # ── Summary grid ──────────────────────────────────────────────────────────
    n = len(results)
    ncols = 3
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(18, nrows * 4), constrained_layout=True)
    axes_flat = axes.flatten() if n > 1 else [axes]
    fmt = mticker.FuncFormatter(lambda x, _: f"${x/1e3:.0f}k")

    for ax, res in zip(axes_flat, results):
        mg, mn, mb = res["m_gross"], res["m_net"], res["m_bench"]
        ax.plot(res["gross_eq"].index, res["gross_eq"].values, color=C_GROSS,
                linewidth=1.0, alpha=0.7, label=f"Gross {mg['cagr']:+.0%}")
        ax.plot(res["net_eq"].index, res["net_eq"].values, color=C_NET,
                linewidth=1.3, label=f"Net {mn['cagr']:+.0%}")
        if not res["bench_eq"].empty:
            b = res["bench_eq"].reindex(res["net_eq"].index).ffill()
            ax.plot(b.index, b.values, color=C_BENCH, linewidth=1.0, alpha=0.8,
                    label=f"SPDR ETF {mb.get('cagr', 0):+.0%}")
        ax.axhline(INITIAL_VALUE, color="black", linewidth=0.4, linestyle="--")
        ax.yaxis.set_major_formatter(fmt)
        ax.set_title(
            f"{res['sector']}\n"
            f"Net Sharpe {mn['sharpe']:.2f}  MaxDD {mn['max_dd']:.0%}  "
            f"({res['n_tickers']} stks, {res['n_trades']} trades)",
            fontsize=8.5, fontweight="bold",
        )
        ax.legend(fontsize=7, loc="upper left")
        ax.grid(True, alpha=0.25)
        ax.tick_params(labelsize=7)

    for ax in axes_flat[n:]:
        ax.set_visible(False)
    fig.suptitle("AL PCA Stat Arb LONG-ONLY — Per Sector (S&P 500 C&P, $1M Start, net of TC)",
                 fontsize=13, fontweight="bold")
    fig.savefig(OUT_DIR / "summary_grid.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved: %s", OUT_DIR / "summary_grid.png")

    # ── Metrics CSV ───────────────────────────────────────────────────────────
    rows = []
    for res in results:
        mg, mn, mb = res["m_gross"], res["m_net"], res["m_bench"]
        rows.append({
            "Sector":         res["sector"],
            "N Stocks":       res["n_tickers"],
            "N Trades":       res["n_trades"],
            "Gross CAGR":     f"{mg['cagr']:+.1%}",
            "Net CAGR":       f"{mn['cagr']:+.1%}",
            "SPDR ETF CAGR":  f"{mb.get('cagr', float('nan')):+.1%}",
            "Net Sharpe":     f"{mn['sharpe']:.2f}",
            "Net MaxDD":      f"{mn['max_dd']:.1%}",
            "TC Drag $":      f"${(mg['terminal'] - mn['terminal']):,.0f}",
        })
    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "metrics_summary.csv", index=False)
    print("\n" + df.to_string(index=False))
    print(f"\nOutputs in: {OUT_DIR}")


if __name__ == "__main__":
    main()
