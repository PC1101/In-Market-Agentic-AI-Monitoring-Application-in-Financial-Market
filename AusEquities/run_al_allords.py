"""
run_al_allords.py — AL PCA Stat Arb on the All Ordinaries (~600 stocks) universe.

Same strategy as run_al_statarb.py but trades the entire All Ordinaries C&P
universe instead of only ASX 200 members.  Compared against an ASX 200 EW
buy-and-hold benchmark.

Separate cache files are used so ASX 200 results are not touched:
  data/allords_prices.csv
  data/allords_membership.csv
  data/allords_pca_factors.csv
  results/al_allords/s_scores.csv

Outputs (AusEquities/results/al_allords/):
  equity_curve.csv
  fig_comparison.png  — strategy (gross + net) vs ASX 200 EW B&H
  metrics.json
"""

import json
import os
import sys
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import norgatedata as nd
from sklearn.decomposition import PCA
import statsmodels.api as sm
from statsmodels.tsa.tsatools import add_trend
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

# Reuse AL strategy logic (signal, positions, backtest)
import run_al_statarb as al
# Reuse ASX 200 data loading for the benchmark
import data_norgate as dn_200

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

# ── universe ───────────────────────────────────────────────────────────────────
WATCHLIST  = "All Ordinaries Current & Past"
INDEX_NAME = "All Ordinaries"
START      = "2024-08-01"
END        = "2026-08-24"

# ── strategy (same as ASX 200 run) ────────────────────────────────────────────
N_WINDOW   = 60
N_PCA      = 15
INITIAL_VALUE = 1_000_000

# ── caches ────────────────────────────────────────────────────────────────────
DATA_DIR        = HERE / "data"
PRICES_CACHE    = DATA_DIR / "allords_prices.csv"
MEMBER_CACHE    = DATA_DIR / "allords_membership.csv"
PCA_CACHE       = DATA_DIR / "allords_pca_factors.csv"
OUT_DIR         = HERE / "results" / "al_allords"

C_GROSS = "#2176AE"
C_NET   = "#E8553A"
C_BENCH = "#4CAF50"


# ─────────────────────────────────────────────────────────────────────────────
# Data loading (All Ordinaries)
# ─────────────────────────────────────────────────────────────────────────────

def load_prices() -> pd.DataFrame:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if PRICES_CACHE.exists():
        log.info("Prices: loading from cache")
        return pd.read_csv(PRICES_CACHE, index_col=0, parse_dates=True)

    symbols = nd.watchlist_symbols(WATCHLIST)
    log.info("Prices: downloading %d All Ordinaries C&P series...", len(symbols))
    frames = {}
    failed = []
    for sym in symbols:
        try:
            rec = nd.price_timeseries(
                sym,
                stock_price_adjustment_setting=nd.StockPriceAdjustmentType.TOTALRETURN,
                start_date=START, end_date=END,
            )
            df = pd.DataFrame(rec)[["Date", "Close"]].copy()
            df["Date"] = pd.to_datetime(df["Date"])
            df = df.set_index("Date")
            if not df.empty:
                frames[sym] = df["Close"]
        except Exception as exc:
            log.debug("  skip %s: %s", sym, exc)
            failed.append(sym)

    if failed:
        log.warning("  %d symbols had no data", len(failed))

    prices = pd.DataFrame(frames).sort_index()
    prices.index = pd.to_datetime(prices.index)
    prices.to_csv(PRICES_CACHE)
    log.info("Saved: %d days × %d tickers", *prices.shape)
    return prices


def load_membership(prices: pd.DataFrame) -> pd.DataFrame:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if MEMBER_CACHE.exists():
        log.info("Membership: loading from cache")
        return pd.read_csv(MEMBER_CACHE, index_col=0, parse_dates=True).astype(bool)

    symbols = list(prices.columns)
    log.info("Membership: building PIT mask for %d symbols...", len(symbols))
    mask = pd.DataFrame(False, index=prices.index, columns=symbols, dtype=bool)

    for sym in symbols:
        try:
            rec = nd.index_constituent_timeseries(
                sym, INDEX_NAME,
                padding_setting=nd.PaddingType.ALLCALENDARDAYS,
                start_date=START, end_date=END,
            )
            df = pd.DataFrame(rec).copy()
            df["Date"] = pd.to_datetime(df["Date"])
            df = df.set_index("Date")
            series = (
                df["Index Constituent"]
                .reindex(prices.index, method="ffill")
                .fillna(0).astype(bool)
            )
            mask[sym] = series
        except Exception as exc:
            log.debug("  membership skip %s: %s", sym, exc)

    mask.to_csv(MEMBER_CACHE)
    log.info("Saved membership: %d days × %d tickers", *mask.shape)
    return mask


def compute_pca_factors(prices: pd.DataFrame, membership: pd.DataFrame) -> pd.DataFrame:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if PCA_CACHE.exists():
        log.info("PCA factors: loading from cache")
        return pd.read_csv(PCA_CACHE, index_col=0, parse_dates=True)

    log.info("PCA factors: computing (%d components, %d-day window)...", N_PCA, N_WINDOW)
    ret = prices.astype(float).ffill().pct_change()
    n_days = len(ret)
    results = []
    n_skipped = 0

    for i in range(N_WINDOW, n_days):
        dt = ret.index[i]
        if dt not in membership.index:
            results.append([dt] + [np.nan] * N_PCA)
            continue
        active_tickers = membership.loc[dt][membership.loc[dt]].index.tolist()
        window = ret.iloc[i - N_WINDOW: i][active_tickers]
        window = window.replace(0, np.nan).dropna(axis=1, how="any")
        vol = window.std()
        window = window.loc[:, vol > 1e-10]

        if window.shape[1] < N_PCA + 1:
            results.append([dt] + [np.nan] * N_PCA)
            continue

        rho = window.corr().dropna(axis=0, how="any").dropna(axis=1, how="any")
        sig_bar = window.std()[rho.columns]
        if rho.shape[0] < N_PCA + 1:
            results.append([dt] + [np.nan] * N_PCA)
            continue

        try:
            v = PCA(n_components=N_PCA).fit(rho).components_
            v = v / np.sum(np.abs(v))
            ret_dt = ret.iloc[i][rho.columns].fillna(0).values
            f = (v / sig_bar.values[np.newaxis, :]).dot(ret_dt)
            results.append([dt, *f])
        except np.linalg.LinAlgError:
            n_skipped += 1
            results.append([dt] + [np.nan] * N_PCA)

        if (i - N_WINDOW) % 100 == 0:
            log.info("  PCA: day %d / %d", i - N_WINDOW, n_days - N_WINDOW)

    cols = ["Date"] + [f"pca_{k}" for k in range(N_PCA)]
    pca_df = (pd.DataFrame(results, columns=cols)
              .set_index("Date").dropna(how="all"))
    pca_df.to_csv(PCA_CACHE)
    log.info("Saved PCA: %d rows (skipped %d SVD failures)", len(pca_df), n_skipped)
    return pca_df


# ─────────────────────────────────────────────────────────────────────────────
# ASX 200 EW benchmark
# ─────────────────────────────────────────────────────────────────────────────

def build_asx200_benchmark() -> pd.Series:
    """Equal-weighted daily return of ASX 200 PIT members, $1M start."""
    prices_200   = dn_200.load_prices()
    member_200   = dn_200.load_membership(prices_200)
    ret = prices_200.astype(float).ffill().pct_change()
    masked = ret.where(member_200.reindex(ret.index).fillna(False))
    bm_ret = masked.mean(axis=1)
    bm_value = (1 + bm_ret).cumprod() * INITIAL_VALUE
    bm_value.name = "ASX 200 EW B&H"
    return bm_value


# ─────────────────────────────────────────────────────────────────────────────
# Gross / net backtest (reuses al.portfolio_returns)
# ─────────────────────────────────────────────────────────────────────────────

def run_gross_net(ret_c, pos_long, pos_short):
    long_gross,  long_net  = al.portfolio_returns(ret_c, pos_long,  "long")
    short_gross, short_net = al.portfolio_returns(ret_c, pos_short, "short")

    gross = (long_gross + short_gross).fillna(0)
    net   = (long_net   + short_net).fillna(0)

    gross_eq = (1 + gross).cumprod() * INITIAL_VALUE
    net_eq   = (1 + net).cumprod()   * INITIAL_VALUE

    active = (pos_long.abs().sum(axis=1) + pos_short.abs().sum(axis=1)) > 0
    start  = active.idxmax() if active.any() else gross_eq.index[0]
    return gross_eq.loc[start:], net_eq.loc[start:]


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

def metrics(value: pd.Series) -> dict:
    r = value.pct_change().dropna()
    n = len(r)
    total = value.iloc[-1] / value.iloc[0] - 1
    cagr  = (1 + total) ** (252 / n) - 1 if n > 0 else np.nan
    vol   = r.std() * np.sqrt(252)
    sharpe = cagr / vol if vol > 0 else np.nan
    dd = (value / value.cummax() - 1).min()
    return {"cagr": cagr, "vol": vol, "sharpe": sharpe,
            "max_dd": dd, "total_return": total,
            "terminal_value": float(value.iloc[-1])}


# ─────────────────────────────────────────────────────────────────────────────
# Plot
# ─────────────────────────────────────────────────────────────────────────────

def plot_comparison(gross_eq, net_eq, bench_eq, out_path):
    bench_aligned = bench_eq.reindex(net_eq.index).ffill()

    fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True,
                             gridspec_kw={"height_ratios": [3, 1.5]})

    # ── Panel 1: equity curves ─────────────────────────────────────────────
    axes[0].plot(gross_eq.index, gross_eq.values,
                 color=C_GROSS, linewidth=1.2, label="All Ords AL Stat Arb — Gross", alpha=0.8)
    axes[0].plot(net_eq.index, net_eq.values,
                 color=C_NET, linewidth=1.4, label="All Ords AL Stat Arb — Net (after TC)")
    axes[0].plot(bench_aligned.index, bench_aligned.values,
                 color=C_BENCH, linewidth=1.2, label="ASX 200 EW Buy & Hold", alpha=0.85)
    axes[0].axhline(INITIAL_VALUE, color="black", linewidth=0.4, linestyle="--")

    fmt = mticker.FuncFormatter(lambda x, _: f"${x:,.0f}")
    axes[0].yaxis.set_major_formatter(fmt)
    axes[0].set_ylabel("Portfolio Value ($)", fontsize=11)
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)

    m_gross = metrics(gross_eq)
    m_net   = metrics(net_eq)
    m_bench = metrics(bench_aligned.dropna())

    axes[0].set_title(
        f"All Ordinaries AL PCA Stat Arb vs ASX 200 EW Buy & Hold  ($1M Start)\n"
        f"Gross: CAGR {m_gross['cagr']:+.1%} Sharpe {m_gross['sharpe']:.2f}   "
        f"Net: CAGR {m_net['cagr']:+.1%} Sharpe {m_net['sharpe']:.2f}   "
        f"ASX 200 B&H: CAGR {m_bench['cagr']:+.1%} Sharpe {m_bench['sharpe']:.2f}",
        fontsize=11, fontweight="bold",
    )

    # ── Panel 2: drawdown ──────────────────────────────────────────────────
    def dd_series(eq):
        return (eq / eq.cummax() - 1) * 100

    axes[1].fill_between(net_eq.index, dd_series(net_eq).values, 0,
                         color=C_NET, alpha=0.3, label="Net DD")
    axes[1].plot(net_eq.index, dd_series(net_eq).values,
                 color=C_NET, linewidth=0.7)
    axes[1].plot(bench_aligned.index, dd_series(bench_aligned).values,
                 color=C_BENCH, linewidth=0.9, label="ASX 200 DD")
    axes[1].set_ylabel("Drawdown (%)", fontsize=10)
    axes[1].legend(fontsize=9, loc="lower left")
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved: %s", out_path)

    return m_gross, m_net, m_bench


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    log.info("=== AL PCA Stat Arb — All Ordinaries C&P ===")

    log.info("Step 1/6  Loading All Ordinaries prices...")
    prices = load_prices()
    log.info("  %d trading days × %d tickers", *prices.shape)

    log.info("Step 2/6  Loading PIT All Ordinaries membership...")
    membership = load_membership(prices)

    log.info("Step 3/6  Computing PCA factors...")
    pca_factors = compute_pca_factors(prices, membership)
    log.info("  %d rows × %d components", *pca_factors.shape)

    s_path = OUT_DIR / "s_scores.csv"
    if s_path.exists():
        log.info("Step 4/6  Loading cached s-scores...")
        s_scores = pd.read_csv(s_path, index_col=0, parse_dates=True)
    else:
        log.info("Step 4/6  Fitting OU s-scores (~8-10 min for 600+ stocks)...")
        s_scores = al.compute_s_scores(prices, membership, pca_factors,
                                       n_window=N_WINDOW)
        s_scores.to_csv(s_path)
        log.info("  Saved: %s", s_path)

    log.info("Step 5/6  Building positions and running backtest...")
    ret = prices.astype(float).ffill().pct_change()
    s_aligned = s_scores.reindex(ret.index).ffill()

    common = [t for t in s_scores.columns if t in ret.columns]
    ret_c = ret[common]
    s_c   = s_aligned[common]

    pos_long  = al.build_position_df(s_c < al.S_BO,  s_c > al.S_BC,  ret_c)
    pos_short = al.build_position_df(s_c > al.S_SO,  s_c < al.S_SC,  ret_c)

    gross_eq, net_eq = run_gross_net(ret_c, pos_long, pos_short)

    curve = al.build_equity_curve(
        *[s for s in [
            al.portfolio_returns(ret_c, pos_long,  "long")[1],
            al.portfolio_returns(ret_c, pos_short, "short")[1],
        ]]
    )
    curve.to_csv(OUT_DIR / "equity_curve.csv")

    log.info("Step 6/6  Building ASX 200 EW benchmark and plotting...")
    bench_eq = build_asx200_benchmark()

    m_gross, m_net, m_bench = plot_comparison(
        gross_eq, net_eq, bench_eq,
        OUT_DIR / "fig_comparison.png",
    )

    results = {
        "All Ords AL Stat Arb (gross)": {k: round(float(v), 6) for k, v in m_gross.items()},
        "All Ords AL Stat Arb (net)":   {k: round(float(v), 6) for k, v in m_net.items()},
        "ASX 200 EW Buy & Hold":        {k: round(float(v), 6) for k, v in m_bench.items()},
    }
    (OUT_DIR / "metrics.json").write_text(json.dumps(results, indent=2))

    print("\n=== Results ===")
    for label, m in results.items():
        print(f"\n  {label}")
        print(f"    CAGR       : {m['cagr']:+.2%}")
        print(f"    Sharpe     : {m['sharpe']:.3f}")
        print(f"    Ann Vol    : {m['vol']:.2%}")
        print(f"    Max DD     : {m['max_dd']:.2%}")
        print(f"    Terminal $ : ${m['terminal_value']:,.0f}")

    print(f"\nOutputs in: {OUT_DIR}")


if __name__ == "__main__":
    main()
