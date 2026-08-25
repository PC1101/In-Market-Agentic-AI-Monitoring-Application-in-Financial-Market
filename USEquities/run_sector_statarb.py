"""
run_sector_statarb.py — AL PCA Stat Arb run independently within each GICS sector (S&P 500).

Outputs: USEquities/results/sector_statarb/
  summary_grid.png      — grid, one panel per sector
  metrics_summary.csv   — CAGR / Sharpe / MaxDD for all sectors
"""

import json
import os
import sys
import logging
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import norgatedata as nd
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(it, **kw): return it

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

import data_norgate as dn
import run_al_statarb as al

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

START = "2024-08-01"
END   = "2026-08-24"
N_WINDOW      = 60
INITIAL_VALUE = 1_000_000
MIN_STOCKS    = 8

OUT_DIR = HERE / "results" / "sector_statarb"

# SPDR sector ETF benchmarks (Total Return, Norgate)
SECTOR_INDEX = {
    "Consumer Discretionary": "XLY",
    "Energy":                 "XLE",
    "Financials":             "XLF",
    "Health Care":            "XLV",
    "Information Technology": "XLK",
    "Materials":              "XLB",
    "Industrials":            "XLI",
    "Real Estate":            "XLRE",
    "Consumer Staples":       "XLP",
    "Communication Services": "XLC",
    "Utilities":              "XLU",
}

C_GROSS = "#2176AE"
C_NET   = "#E8553A"
C_BENCH = "#4CAF50"


# ─────────────────────────────────────────────────────────────────────────────
# GICS sector map
# ─────────────────────────────────────────────────────────────────────────────

def build_sector_map(symbols):
    """Return {sector_name: [list of tickers]} using current GICS classification."""
    by_sector = defaultdict(list)
    for sym in symbols:
        try:
            sector = nd.classification_at_level(sym, "GICS", "Name", 1)
            if sector:
                by_sector[sector].append(sym)
        except Exception:
            pass
    return dict(by_sector)


# ─────────────────────────────────────────────────────────────────────────────
# Sector-specific PCA factors
# ─────────────────────────────────────────────────────────────────────────────

def compute_sector_pca(prices, membership, sector_tickers, n_window=N_WINDOW):
    tickers = [t for t in sector_tickers if t in prices.columns]
    if not tickers:
        return pd.DataFrame()

    ret = prices[tickers].astype(float).ffill().pct_change()
    mem = membership.reindex(columns=tickers).fillna(False)

    avg_active = int(mem.mean(axis=0).gt(0).sum())
    n_pca = max(2, min(avg_active - 2, 15))

    n_days = len(ret)
    results = []
    n_skipped = 0

    for i in range(n_window, n_days):
        dt = ret.index[i]
        active_tickers = [t for t in tickers
                          if dt in mem.index and mem.loc[dt, t]]
        window = ret.iloc[i - n_window: i][active_tickers]
        window = window.replace(0, np.nan).dropna(axis=1, how="any")
        vol = window.std()
        window = window.loc[:, vol > 1e-10]

        k = min(n_pca, window.shape[1] - 1)
        if k < 2:
            results.append([dt] + [np.nan] * n_pca)
            continue

        rho = window.corr().dropna(axis=0, how="any").dropna(axis=1, how="any")
        sig_bar = window.std()[rho.columns]
        k = min(k, rho.shape[0] - 1)
        if k < 2:
            results.append([dt] + [np.nan] * n_pca)
            continue

        try:
            v = PCA(n_components=k).fit(rho).components_
            v = v / np.sum(np.abs(v))
            ret_dt = ret.iloc[i][rho.columns].fillna(0).values
            f = (v / sig_bar.values[np.newaxis, :]).dot(ret_dt)
            row = [dt] + list(f) + [np.nan] * (n_pca - k)
            results.append(row)
        except np.linalg.LinAlgError:
            n_skipped += 1
            results.append([dt] + [np.nan] * n_pca)

    cols = ["Date"] + [f"pca_{j}" for j in range(n_pca)]
    if not results:
        return pd.DataFrame(columns=cols[1:])
    pca_df = (pd.DataFrame(results, columns=cols)
              .set_index("Date").dropna(how="all"))
    return pca_df


# ─────────────────────────────────────────────────────────────────────────────
# Sector benchmark: SPDR ETF total-return price series
# ─────────────────────────────────────────────────────────────────────────────

def load_sector_benchmark(sector, idx):
    """Returns portfolio-value series ($1M start) for the SPDR sector ETF."""
    sym = SECTOR_INDEX.get(sector)
    if not sym:
        return pd.Series(dtype=float)
    try:
        rec = nd.price_timeseries(
            sym,
            stock_price_adjustment_setting=nd.StockPriceAdjustmentType.TOTALRETURN,
            start_date=START,
            end_date=END,
        )
        df = pd.DataFrame(rec)[["Date", "Close"]].copy()
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.set_index("Date")["Close"]
        df = df.reindex(idx).ffill()
        bench = df / df.iloc[0] * INITIAL_VALUE
        bench.name = f"{sector} ETF (TR)"
        return bench
    except Exception as e:
        log.warning("  Could not load benchmark for %s (%s): %s", sector, sym, e)
        return pd.Series(dtype=float)


# ─────────────────────────────────────────────────────────────────────────────
# Per-sector backtest
# ─────────────────────────────────────────────────────────────────────────────

def run_sector(sector, tickers, prices, membership):
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

    s_scores = al.compute_s_scores(sector_prices, sector_mem, pca_factors,
                                    n_window=N_WINDOW)
    if s_scores.empty:
        return None

    ret = sector_prices.astype(float).ffill().pct_change()
    s_aligned = s_scores.reindex(ret.index).ffill()
    common = [t for t in s_scores.columns if t in ret.columns]
    if not common:
        return None

    ret_c = ret[common]
    s_c   = s_aligned[common]

    pos_long  = al.build_position_df(s_c < al.S_BO, s_c > al.S_BC, ret_c)
    pos_short = al.build_position_df(s_c > al.S_SO, s_c < al.S_SC, ret_c)

    long_gross, long_net   = al.portfolio_returns(ret_c, pos_long,  "long")
    short_gross, short_net = al.portfolio_returns(ret_c, pos_short, "short")

    gross = (long_gross + short_gross).fillna(0)
    net   = (long_net   + short_net).fillna(0)

    active = (pos_long.abs().sum(axis=1) + pos_short.abs().sum(axis=1)) > 0
    if not active.any():
        return None
    start = active.idxmax()

    gross_eq = (1 + gross).cumprod().loc[start:] * INITIAL_VALUE
    net_eq   = (1 + net).cumprod().loc[start:]   * INITIAL_VALUE
    bench_eq = load_sector_benchmark(sector, gross_eq.index)

    trades = pd.concat([
        al.extract_trades(pos_long,  ret_c, "long"),
        al.extract_trades(pos_short, ret_c, "short"),
    ])

    def _metrics(eq):
        r = eq.pct_change().dropna()
        n = len(r)
        total = eq.iloc[-1] / eq.iloc[0] - 1
        cagr  = (1 + total) ** (252 / n) - 1 if n > 0 else np.nan
        vol   = r.std() * np.sqrt(252)
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


# ─────────────────────────────────────────────────────────────────────────────
# Summary plot & metrics
# ─────────────────────────────────────────────────────────────────────────────

def plot_summary(results):
    n = len(results)
    ncols = 3
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(18, nrows * 4),
                             constrained_layout=True)
    axes_flat = axes.flatten() if n > 1 else [axes]
    fmt = mticker.FuncFormatter(lambda x, _: f"${x/1e3:.0f}k")

    for ax, res in zip(axes_flat, results):
        gross_eq = res["gross_eq"]
        net_eq   = res["net_eq"]
        bench_eq = res["bench_eq"]
        mg, mn, mb = res["m_gross"], res["m_net"], res["m_bench"]

        ax.plot(gross_eq.index, gross_eq.values, color=C_GROSS,
                linewidth=1.0, alpha=0.7, label=f"Gross {mg['cagr']:+.0%}")
        ax.plot(net_eq.index, net_eq.values, color=C_NET,
                linewidth=1.3, label=f"Net {mn['cagr']:+.0%}")
        if not bench_eq.empty:
            bench_aligned = bench_eq.reindex(net_eq.index).ffill()
            ax.plot(bench_aligned.index, bench_aligned.values, color=C_BENCH,
                    linewidth=1.0, alpha=0.8,
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

    fig.suptitle(
        "AL PCA Stat Arb — Per Sector  (S&P 500 C&P, $1M Start, net of TC + borrow)",
        fontsize=13, fontweight="bold",
    )
    path = OUT_DIR / "summary_grid.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved: %s", path)


def save_metrics(results):
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
    path = OUT_DIR / "metrics_summary.csv"
    df.to_csv(path, index=False)
    log.info("Saved: %s", path)
    print("\n" + df.to_string(index=False))


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    log.info("=== AL PCA Stat Arb — S&P 500 Sector Analysis ===")

    log.info("Loading S&P 500 data from cache...")
    prices     = dn.load_prices()
    membership = dn.load_membership(prices)
    log.info("  %d days × %d tickers", *prices.shape)

    log.info("Building GICS sector map...")
    sector_map = build_sector_map(list(prices.columns))
    for sec, syms in sorted(sector_map.items(), key=lambda x: -len(x[1])):
        log.info("  %-30s %d stocks", sec, len(syms))

    results = []
    for sector in sorted(sector_map.keys()):
        log.info("\n[%s]", sector)
        res = run_sector(sector, sector_map[sector], prices, membership)
        if res is not None:
            results.append(res)

    if not results:
        log.error("No sectors produced results.")
        return

    results.sort(key=lambda r: r["m_net"]["sharpe"], reverse=True)
    plot_summary(results)
    save_metrics(results)
    print(f"\nOutputs in: {OUT_DIR}")


if __name__ == "__main__":
    main()
