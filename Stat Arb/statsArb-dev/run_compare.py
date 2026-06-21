"""
Side-by-side comparison of the Avellaneda-Lee ETF-hedge baseline vs the pairs
trading strategy on the same sector and date range.

Outputs (saved under results/comparison/<etf>/):
    comparison_metrics.csv  - headline stats for each strategy + blend
    comparison_plot.png     - overlaid equity curves, drawdowns, rolling Sharpe

Examples:

    python run_compare.py                          # XLF, default params
    python run_compare.py --etf xlk
    python run_compare.py --etf xlf --start 2015-01-02 --end 2020-11-19
    python run_compare.py --show                   # display chart interactively
"""

import argparse
import os
import sys

import matplotlib
if not any(flag in sys.argv for flag in ("--show", "--plot")):
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(PROJECT_ROOT)
sys.path.insert(0, PROJECT_ROOT)

from src import backtest, bt_tools          # noqa: E402
from src.pairs_backtest import PairsBt      # noqa: E402

DEFAULT_WINDOW = ("2007-01-03", "2015-01-02")
DEFAULT_KAPPA_MIN = 252.0 / 30.0


def load_cfgs():
    with open("configs/optimise_trading_rules.yml") as fh:
        base_cfg = yaml.load(fh, Loader=yaml.SafeLoader)
    with open("configs/pairs_trading.yml") as fh:
        pairs_cfg = yaml.load(fh, Loader=yaml.SafeLoader)["pairs"]
    return base_cfg, pairs_cfg


def parse_args(base_cfg, pairs_cfg):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--etf", default="xlf",
                        choices=sorted(base_cfg["prices_file_path"].keys()),
                        help="Sector ETF to run both strategies on (default: xlf).")
    parser.add_argument("--start", default=None,
                        help="Backtest start date YYYY-MM-DD.")
    parser.add_argument("--end", default=None,
                        help="Backtest end date YYYY-MM-DD.")
    parser.add_argument("--defactoring", default="etf", choices=["etf", "pca"],
                        help="Defactoring method for the baseline (default: etf).")
    parser.add_argument("--n-window", type=int, default=60,
                        help="Rolling estimation window (trading days).")
    parser.add_argument("--kappa-min", type=float, default=DEFAULT_KAPPA_MIN,
                        help=f"Min OU speed for both strategies (default: {DEFAULT_KAPPA_MIN:.2f}). "
                             "Use 0 to disable.")
    parser.add_argument("--max-pairs", type=int,
                        default=pairs_cfg.get("max_pairs", 50),
                        help="Max pairs per reselection (pairs strategy).")
    parser.add_argument("--p-value", type=float,
                        default=pairs_cfg.get("p_value_cutoff", 0.05),
                        help="Engle-Granger p-value cutoff (pairs strategy).")
    parser.add_argument("--reselect-freq", type=int,
                        default=pairs_cfg.get("reselect_freq", 20),
                        help="Days between pair reselections (default: 20).")
    parser.add_argument("--show", "--plot", action="store_true", dest="show",
                        help="Display the comparison chart interactively.")
    return parser.parse_args()


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _sharpe(ret_series: pd.Series) -> float:
    s = ret_series.dropna()
    return np.sqrt(252) * s.mean() / s.std() if s.std() > 0 else np.nan


def _maxdd(cum_pnl: pd.Series) -> float:
    return float((cum_pnl / cum_pnl.cummax() - 1).min())


def compute_rolling_sharpe(ret_series: pd.Series, window: int = 126) -> pd.Series:
    """126-day (~6m) rolling annualised Sharpe."""
    r = ret_series.dropna()
    roll_mean = r.rolling(window, min_periods=window // 2).mean()
    roll_std  = r.rolling(window, min_periods=window // 2).std()
    return np.sqrt(252) * roll_mean / roll_std


def print_comparison_table(results: dict):
    """Print a formatted comparison table."""
    header = f"{'Metric':<22} {'Baseline':>12} {'Pairs':>12} {'50/50 Blend':>14}"
    print("\n" + "=" * len(header))
    print("STRATEGY COMPARISON")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    rows = [
        ("Sharpe ratio",    "sharpe",   "{:>12.3f}"),
        ("Max drawdown",    "maxdd",    "{:>12.2%}"),
        ("End PnL (x)",     "endpnl",   "{:>12.3f}"),
        ("Total return",    "tot_ret",  "{:>12.2%}"),
        ("Ann. return",     "ann_ret",  "{:>12.2%}"),
        ("Daily vol (ann)", "ann_vol",  "{:>12.2%}"),
        ("Win rate",        "win_rate", "{:>12.1%}"),
        ("Corr w/ baseline","corr",     "{:>12.3f}"),
    ]
    for label, key, fmt in rows:
        vals = []
        for strat in ("baseline", "pairs", "blend"):
            v = results[strat].get(key, np.nan)
            if isinstance(v, float) and not np.isnan(v):
                vals.append(fmt.format(v))
            else:
                vals.append(f"{'—':>12}")
        print(f"{label:<22} {vals[0]} {vals[1]} {vals[2]}")
    print("=" * len(header))


def plot_comparison(baseline_port, pairs_port, blend_ret, etf, save_path, show=False):
    """3-panel comparison plot: equity curves, drawdowns, rolling Sharpe."""
    fig, axs = plt.subplots(3, 1, figsize=(13, 12), sharex=True)

    # Panel 1 — Equity curves
    baseline_port["cum_pnl"].plot(ax=axs[0], label="Baseline (ETF-hedge)", color="steelblue")
    pairs_port["cum_pnl"].plot(ax=axs[0], label="Pairs trading", color="darkorange")
    (blend_ret + 1).cumprod().plot(ax=axs[0], label="50/50 blend", color="green",
                                   linestyle="--", linewidth=1.2)
    axs[0].axhline(1.0, color="k", linewidth=0.6, linestyle=":")
    axs[0].set_title(f"{etf.upper()} — Cumulative PnL (equity curve)")
    axs[0].legend(loc="upper left")
    axs[0].set_ylabel("Equity (base = 1.0)")

    # Panel 2 — Drawdowns
    baseline_port["max_dd"].plot(ax=axs[1], label="Baseline", color="steelblue")
    pairs_port["max_dd"].plot(ax=axs[1], label="Pairs", color="darkorange")
    axs[1].set_title("Rolling 6m max drawdown")
    axs[1].legend(loc="lower left")
    axs[1].set_ylabel("Drawdown")

    # Panel 3 — Rolling Sharpe
    baseline_rs = compute_rolling_sharpe(baseline_port["cum_ret"])
    pairs_rs    = compute_rolling_sharpe(pairs_port["cum_ret"])
    blend_rs    = compute_rolling_sharpe(blend_ret)
    baseline_rs.plot(ax=axs[2], label="Baseline", color="steelblue")
    pairs_rs.plot(ax=axs[2], label="Pairs", color="darkorange")
    blend_rs.plot(ax=axs[2], label="50/50 blend", color="green", linestyle="--")
    axs[2].axhline(0, color="k", linewidth=0.6, linestyle=":")
    axs[2].set_title("Rolling 126-day Sharpe ratio")
    axs[2].legend(loc="upper left")
    axs[2].set_ylabel("Sharpe")

    fig.tight_layout()
    fig.savefig(save_path, dpi=110)
    if show:
        plt.show()
    else:
        plt.close(fig)


def strategy_stats(port_ret: pd.DataFrame, label: str, baseline_ret: pd.Series = None) -> dict:
    ret = port_ret["cum_ret"].dropna()
    cum = port_ret["cum_pnl"].dropna()
    n_years = len(ret) / 252
    win_rate = (ret > 0).mean()
    corr = ret.corr(baseline_ret) if baseline_ret is not None else np.nan

    return {
        "sharpe":   _sharpe(ret),
        "maxdd":    _maxdd(cum),
        "endpnl":   float(cum.iloc[-1]) if len(cum) else np.nan,
        "tot_ret":  float(cum.iloc[-1] - 1) if len(cum) else np.nan,
        "ann_ret":  float((cum.iloc[-1] ** (1 / n_years)) - 1) if n_years > 0 else np.nan,
        "ann_vol":  float(ret.std() * np.sqrt(252)),
        "win_rate": float(win_rate),
        "corr":     float(corr),
    }


def main():
    base_cfg, pairs_cfg = load_cfgs()
    args = parse_args(base_cfg, pairs_cfg)

    cfg_start, cfg_end = base_cfg.get("bt_dt", {}).get(args.etf, DEFAULT_WINDOW)
    st_dt = args.start or cfg_start
    ed_dt = args.end   or cfg_end

    kappa_min = args.kappa_min if args.kappa_min > 0 else None
    prices_path = base_cfg["prices_file_path"][args.etf]
    baseline_cost = tuple(base_cfg.get("transaction_cost", (0.0005, 0.0005)))
    pairs_cost = tuple(pairs_cfg.get("transaction_cost", [0.0010, 0.0010]))

    # ---- Run baseline ----
    print(f"\n[1/2] Running baseline (ETF-hedge, defactoring={args.defactoring})...")
    baseline_model = backtest.bt(
        prices_file_path=prices_path,
        etf_name=args.etf,
        st_dt=st_dt,
        ed_dt=ed_dt,
        n_window=args.n_window,
        defactoring=args.defactoring,
        performance_only=True,
        kappa_min=kappa_min,
        progress=True,
    )
    b_sharpe, b_maxdd, b_endpnl = baseline_model.run(
        weighting_scheme=base_cfg.get("weighting_scheme", "equal_weighted"),
        sl=base_cfg.get("sl", -0.10),
        long_only=base_cfg.get("long_only", False),
        transaction_cost=baseline_cost,
    )
    print(f"   Baseline  ->  Sharpe {b_sharpe:.3f}  MaxDD {b_maxdd:.2%}  EndPnL {b_endpnl:.3f}")

    # ---- Run pairs ----
    print(f"\n[2/2] Running pairs trading (p<{args.p_value}, max_pairs={args.max_pairs})...")
    pairs_model = PairsBt(
        prices_file_path=prices_path,
        etf_name=args.etf,
        st_dt=st_dt,
        ed_dt=ed_dt,
        n_window=args.n_window,
        p_value_cutoff=args.p_value,
        max_pairs=args.max_pairs,
        reselect_freq=args.reselect_freq,
        kappa_min=kappa_min,
        performance_only=True,
        progress=True,
    )
    p_sharpe, p_maxdd, p_endpnl = pairs_model.run(
        weighting_scheme=pairs_cfg.get("weighting_scheme", "equal_weighted"),
        sl=pairs_cfg.get("sl", -0.10),
        long_only=pairs_cfg.get("long_only", False),
        transaction_cost=pairs_cost,
    )
    print(f"   Pairs     ->  Sharpe {p_sharpe:.3f}  MaxDD {p_maxdd:.2%}  EndPnL {p_endpnl:.3f}")

    # ---- Combine on shared index ----
    baseline_ret = baseline_model.port_ret["cum_ret"].rename("baseline")
    pairs_ret    = pairs_model.port_ret["cum_ret"].rename("pairs")
    shared_idx   = baseline_ret.index.intersection(pairs_ret.index)
    baseline_ret = baseline_ret.reindex(shared_idx).fillna(0)
    pairs_ret    = pairs_ret.reindex(shared_idx).fillna(0)

    # 50/50 blend
    blend_ret = 0.5 * baseline_ret + 0.5 * pairs_ret
    blend_cum = (blend_ret + 1).cumprod()
    blend_dd  = blend_cum.rolling(126, min_periods=1).apply(
        lambda x: np.min(x / np.maximum.accumulate(x)) - 1)
    blend_port = pd.DataFrame({
        "cum_ret": blend_ret, "cum_pnl": blend_cum, "max_dd": blend_dd})

    # ---- Stats ----
    results = {
        "baseline": strategy_stats(baseline_model.port_ret, "baseline"),
        "pairs":    strategy_stats(pairs_model.port_ret,    "pairs",
                                   baseline_ret=baseline_ret),
        "blend":    strategy_stats(blend_port, "blend", baseline_ret=baseline_ret),
    }
    # Correlation of pairs vs baseline
    results["pairs"]["corr"]  = float(pairs_ret.corr(baseline_ret))
    results["blend"]["corr"]  = float(blend_ret.corr(baseline_ret))

    print_comparison_table(results)

    # ---- Save outputs ----
    out_dir = os.path.join("results", "comparison", args.etf)
    os.makedirs(out_dir, exist_ok=True)

    metrics_rows = []
    for strat, stats in results.items():
        row = {"strategy": strat}
        row.update(stats)
        metrics_rows.append(row)
    pd.DataFrame(metrics_rows).to_csv(
        os.path.join(out_dir, "comparison_metrics.csv"), index=False)

    plot_comparison(
        baseline_model.port_ret,
        pairs_model.port_ret,
        blend_ret,
        etf=args.etf,
        save_path=os.path.join(out_dir, "comparison_plot.png"),
        show=args.show,
    )

    print(f"\nSaved comparison_metrics.csv and comparison_plot.png to: {out_dir}")


if __name__ == "__main__":
    main()
