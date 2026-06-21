"""
Multi-strategy stat-arb comparison across 4 market regime windows.

Runs 5 strategy variants, benchmarks against S&P 500 (SPY), and produces
a 2x2 equity-curve overlay chart plus a full metrics table.

Strategies
----------
  1. L/S ETF-hedge   — baseline Avellaneda-Lee, sector ETF factor
  2. Long-only ETF   — long leg only, ETF-hedged residual
  3. Short-only ETF  — short leg only, ETF-hedged residual
  4. L/S PCA         — long-short with 15-factor PCA defactoring
  5. Pairs L/S       — cointegrated pairs, log-price spread OU signal

Windows
-------
  GFC        2007-01-03 to 2009-03-09
  Recovery   2009-03-10 to 2019-01-02
  COVID      2019-01-03 to 2020-11-20
  Full       2007-01-03 to 2020-11-20

Outputs (results/strategy_comparison/)
-------
  strategy_comparison.png   2x2 chart, all strategies overlaid per period
  strategy_metrics.csv      one row per (period x strategy)

Usage
-----
    python run_strategy_comparison.py
    python run_strategy_comparison.py --show      # display chart interactively
"""

import argparse
import os
import sys
import warnings
warnings.filterwarnings("ignore")

import matplotlib
if "--show" not in sys.argv:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(PROJECT_ROOT)
sys.path.insert(0, PROJECT_ROOT)

from src import backtest, bt_tools          # noqa: E402
from src.pairs_backtest import PairsBt      # noqa: E402

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PERIODS = [
    ("GFC\n(2007-2009)",          "2007-01-03", "2009-03-09"),
    ("Recovery\n(2009-2019)",     "2009-03-10", "2019-01-02"),
    ("COVID\n(2019-2020)",        "2019-01-03", "2020-11-20"),
    ("Full Period\n(2007-2020)",  "2007-01-03", "2020-11-20"),
]

# (display_name, model_type, defactoring, long_only, short_only, color, linestyle)
STRATEGIES = [
    ("L/S ETF-hedge",  "bt",    "etf", False, False, "#1f77b4", "-",  1.6),
    ("Long-only ETF",  "bt",    "etf", True,  False, "#2ca02c", "-",  1.4),
    ("Short-only ETF", "bt",    "etf", False, True,  "#9467bd", "-",  1.4),
    ("L/S PCA",        "bt",    "pca", False, False, "#ff7f0e", "--", 1.4),
    ("Pairs L/S",      "pairs", None,  False, False, "#d62728", "-",  1.4),
]

ETF       = "xlf"
N_WINDOW  = 60
KAPPA_MIN = 252.0 / 30.0   # ~8.4 /yr

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_cfgs():
    with open("configs/optimise_trading_rules.yml") as fh:
        base = yaml.load(fh, Loader=yaml.SafeLoader)
    with open("configs/pairs_trading.yml") as fh:
        pairs = yaml.load(fh, Loader=yaml.SafeLoader)["pairs"]
    return base, pairs


def load_spy():
    try:
        import yfinance as yf
        raw = yf.download("SPY", start="2006-01-01", end="2020-11-21",
                          auto_adjust=True, progress=False)
        spy = raw[("Close", "SPY")] if isinstance(raw.columns, pd.MultiIndex) else raw["Close"]
        spy = spy.squeeze()
        spy.index = pd.to_datetime(spy.index).tz_localize(None)
        spy.name = "SPY"
        print("  SPY downloaded via yfinance.")
        return spy
    except Exception as e:
        print(f"  SPY unavailable ({e}).")
        return None


def rebase(series: pd.Series) -> pd.Series:
    s = series.dropna()
    return s / s.iloc[0] if len(s) > 0 else s


def ann_return(endpnl: float, n_days: int) -> float:
    n_years = n_days / 252
    return float(endpnl ** (1 / n_years) - 1) if n_years > 0 else np.nan

# ---------------------------------------------------------------------------
# Core: run all strategies for one period
# ---------------------------------------------------------------------------

def run_period(st_dt: str, ed_dt: str, cfg: dict, pairs_cfg: dict) -> dict:
    """
    Returns dict: {strategy_name -> (port_ret_df, sharpe, maxdd, endpnl)}

    Efficiency: ETF variants (L/S, Long-only, Short-only) share one model fit.
    PCA and Pairs each get their own fit. Total = 3 fits per period.
    Must .copy() port_ret after each .run() since re-calling overwrites it.
    """
    prices_path  = cfg["prices_file_path"][ETF]
    base_cost    = tuple(cfg.get("transaction_cost", (0.0005, 0.0005)))
    pairs_cost   = tuple(pairs_cfg.get("transaction_cost", [0.0010, 0.0010]))
    sl           = cfg.get("sl", -0.10)
    weighting    = cfg.get("weighting_scheme", "equal_weighted")
    results      = {}

    # ---- ETF-hedge model: fit once, run 3 times -------------------------
    print("    [ETF] Fitting OU process...", flush=True)
    etf_model = backtest.bt(
        prices_file_path=prices_path, etf_name=ETF,
        st_dt=st_dt, ed_dt=ed_dt, n_window=N_WINDOW,
        defactoring="etf", performance_only=True,
        kappa_min=KAPPA_MIN, progress=False,
    )
    for name, _, _, long_only, short_only, *_ in STRATEGIES:
        if name not in ("L/S ETF-hedge", "Long-only ETF", "Short-only ETF"):
            continue
        s, m, e = etf_model.run(weighting_scheme=weighting, sl=sl,
                                 long_only=long_only, short_only=short_only,
                                 transaction_cost=base_cost)
        results[name] = (etf_model.port_ret.copy(), s, m, e)
        print(f"    [{name}] Sharpe {s:+.3f}  MaxDD {m:.2%}  EndPnL {e:.3f}x",
              flush=True)

    # ---- PCA model -------------------------------------------------------
    print("    [PCA] Fitting OU process...", flush=True)
    pca_model = backtest.bt(
        prices_file_path=prices_path, etf_name=ETF,
        st_dt=st_dt, ed_dt=ed_dt, n_window=N_WINDOW,
        defactoring="pca", performance_only=True,
        kappa_min=KAPPA_MIN, progress=False,
    )
    s, m, e = pca_model.run(weighting_scheme=weighting, sl=sl,
                             long_only=False, short_only=False,
                             transaction_cost=base_cost)
    results["L/S PCA"] = (pca_model.port_ret.copy(), s, m, e)
    print(f"    [L/S PCA] Sharpe {s:+.3f}  MaxDD {m:.2%}  EndPnL {e:.3f}x",
          flush=True)

    # ---- Pairs model -----------------------------------------------------
    print("    [Pairs] Selecting pairs and fitting OU...", flush=True)
    pairs_model = PairsBt(
        prices_file_path=prices_path, etf_name=ETF,
        st_dt=st_dt, ed_dt=ed_dt, n_window=N_WINDOW,
        p_value_cutoff=pairs_cfg.get("p_value_cutoff", 0.05),
        max_pairs=pairs_cfg.get("max_pairs", 50),
        reselect_freq=pairs_cfg.get("reselect_freq", 20),
        kappa_min=KAPPA_MIN, performance_only=True, progress=False,
    )
    s, m, e = pairs_model.run(
        weighting_scheme=pairs_cfg.get("weighting_scheme", "equal_weighted"),
        sl=pairs_cfg.get("sl", -0.10),
        long_only=False, short_only=False,
        transaction_cost=pairs_cost,
    )
    results["Pairs L/S"] = (pairs_model.port_ret.copy(), s, m, e)
    print(f"    [Pairs L/S] Sharpe {s:+.3f}  MaxDD {m:.2%}  EndPnL {e:.3f}x",
          flush=True)

    return results

# ---------------------------------------------------------------------------
# Chart
# ---------------------------------------------------------------------------

def plot_all(all_results: dict, spy_prices, show: bool = False):
    """2x2 grid — one panel per period, all strategies + SPY overlaid."""
    style = {s[0]: (s[5], s[6], s[7]) for s in STRATEGIES}   # name->(color,ls,lw)

    fig, axs = plt.subplots(2, 2, figsize=(16, 11))
    axs = axs.flatten()

    for i, (period_label, st_dt, ed_dt) in enumerate(PERIODS):
        ax = axs[i]
        period_res = all_results[period_label]

        # --- Strategy curves ---
        for name, *_ in STRATEGIES:
            if name not in period_res:
                continue
            port_ret, sharpe, maxdd, endpnl = period_res[name]
            color, ls, lw = style[name]
            cum = rebase(port_ret["cum_pnl"].dropna())
            ax.plot(cum.index, cum.values, color=color, linestyle=ls,
                    linewidth=lw,
                    label=f"{name}  Sh={sharpe:+.2f}  End={endpnl:.2f}x")

        # --- SPY benchmark ---
        if spy_prices is not None:
            sl_spy = spy_prices.loc[st_dt:ed_dt]
            if len(sl_spy) > 0:
                rb = rebase(sl_spy)
                ax.plot(rb.index, rb.values, color="#7f7f7f", linestyle=":",
                        linewidth=1.1, label=f"SPY B&H  End={float(rb.iloc[-1]):.2f}x")

        ax.axhline(1.0, color="black", linewidth=0.5, linestyle=":")
        ax.set_title(period_label.replace("\n", " "), fontsize=11,
                     fontweight="bold")
        ax.set_ylabel("Equity (base = 1.0)")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.xaxis.set_major_locator(mdates.YearLocator())
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
        ax.legend(loc="upper left", fontsize=7, framealpha=0.85)

    fig.suptitle(
        "XLF Stat-arb — Strategy Variants Across Market Regimes vs S&P 500",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout()

    out_dir = os.path.join("results", "strategy_comparison")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "strategy_comparison.png")
    fig.savefig(path, dpi=130, bbox_inches="tight")
    print(f"\nChart saved: {path}")
    if show:
        plt.show()
    else:
        plt.close(fig)

# ---------------------------------------------------------------------------
# Metrics table
# ---------------------------------------------------------------------------

def build_metrics(all_results: dict) -> pd.DataFrame:
    rows = []
    for period_label, st_dt, ed_dt in PERIODS:
        clean = period_label.replace("\n", " ")
        for name, *_ in STRATEGIES:
            if name not in all_results[period_label]:
                continue
            port_ret, sharpe, maxdd, endpnl = all_results[period_label][name]
            cum_ret = port_ret["cum_ret"].dropna()
            rows.append({
                "period":      clean,
                "strategy":    name,
                "sharpe":      round(sharpe, 3),
                "max_dd":      round(maxdd, 4),
                "end_pnl":     round(endpnl, 3),
                "ann_return":  round(ann_return(endpnl, len(cum_ret)), 4),
                "ann_vol":     round(float(cum_ret.std() * np.sqrt(252)), 4),
                "win_rate":    round(float((cum_ret > 0).mean()), 3),
            })
    return pd.DataFrame(rows)


def print_metrics(df: pd.DataFrame):
    w = 100
    print("\n" + "=" * w)
    print(f"{'Period':<26} {'Strategy':<20} {'Sharpe':>8} {'MaxDD':>8} "
          f"{'EndPnL':>8} {'AnnRet':>8} {'AnnVol':>8} {'WinRate':>8}")
    print("-" * w)
    cur_period = None
    for _, r in df.iterrows():
        if r["period"] != cur_period:
            if cur_period is not None:
                print()
            cur_period = r["period"]
        print(f"{r['period']:<26} {r['strategy']:<20} "
              f"{r['sharpe']:>8.3f} {r['max_dd']:>7.1%} "
              f"{r['end_pnl']:>7.3f}x {r['ann_return']:>7.2%} "
              f"{r['ann_vol']:>7.2%} {r['win_rate']:>7.1%}")
    print("=" * w)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--show", action="store_true",
                        help="Display chart interactively after saving.")
    args = parser.parse_args()

    cfg, pairs_cfg = load_cfgs()

    print("Loading SPY benchmark data...")
    spy_prices = load_spy()

    all_results = {}
    for i, (label, st_dt, ed_dt) in enumerate(PERIODS, 1):
        clean = label.replace("\n", " ")
        print(f"\n[{i}/4] {clean}  ({st_dt} to {ed_dt})")
        all_results[label] = run_period(st_dt, ed_dt, cfg, pairs_cfg)

    print("\nGenerating chart...")
    plot_all(all_results, spy_prices, show=args.show)

    metrics = build_metrics(all_results)
    out_dir = os.path.join("results", "strategy_comparison")
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "strategy_metrics.csv")
    metrics.to_csv(csv_path, index=False)
    print(f"Metrics saved: {csv_path}")

    print_metrics(metrics)


if __name__ == "__main__":
    main()
