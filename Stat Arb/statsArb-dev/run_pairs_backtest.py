"""
Run a single-sector pairs trading backtest end to end and report the full
process: trade blotter, equity curve, and summary performance statistics.

Mirrors the structure of run_backtest.py but uses src/pairs_backtest.PairsBt
instead of src/backtest.bt.

Outputs (saved under results/pairs_trading/<etf>/):
    trades.csv           - every pair trade: pair_id, side, entry, exit, days, return
    equity_curve.csv     - daily long/short/combined PnL and rolling drawdown
    performance.png      - equity curve, drawdown, leg PnL, trade-return histogram
    pairs_selected.csv   - which pairs had a signal on each day (non-NaN s-scores)

Examples (run from the project root):

    python run_pairs_backtest.py                        # XLF, default params
    python run_pairs_backtest.py --etf xlk
    python run_pairs_backtest.py --etf xlf --start 2015-01-02 --end 2020-11-19
    python run_pairs_backtest.py --max-pairs 30 --p-value 0.01
    python run_pairs_backtest.py --kappa-min 0          # disable kappa filter
    python run_pairs_backtest.py --cost 0               # frictionless run
    python run_pairs_backtest.py --show                 # display chart interactively
"""
import argparse
import os
import sys

import matplotlib
if not any(flag in sys.argv for flag in ("--show", "--plot")):
    matplotlib.use("Agg")
import pandas as pd
import yaml

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(PROJECT_ROOT)
sys.path.insert(0, PROJECT_ROOT)

from src.pairs_backtest import PairsBt   # noqa: E402
from src import bt_tools                 # noqa: E402

DEFAULT_WINDOW = ("2007-01-03", "2015-01-02")
DEFAULT_KAPPA_MIN = 252.0 / 30.0        # ~8.4 / yr  (same as baseline)


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
                        help="Sector ETF universe to trade (default: xlf).")
    parser.add_argument("--start", default=None,
                        help="Backtest start date YYYY-MM-DD.")
    parser.add_argument("--end", default=None,
                        help="Backtest end date YYYY-MM-DD.")
    parser.add_argument("--n-window", type=int,
                        default=pairs_cfg.get("n_window", 60),
                        help="Rolling estimation window in trading days.")
    parser.add_argument("--p-value", type=float,
                        default=pairs_cfg.get("p_value_cutoff", 0.05),
                        help="Engle-Granger p-value cutoff for pair selection.")
    parser.add_argument("--max-pairs", type=int,
                        default=pairs_cfg.get("max_pairs", 50),
                        help="Max cointegrated pairs to trade per sector.")
    parser.add_argument("--reselect-freq", type=int,
                        default=pairs_cfg.get("reselect_freq", 20),
                        help="Days between pair reselections (default: 20 ~ monthly).")
    parser.add_argument("--kappa-min", type=float, default=DEFAULT_KAPPA_MIN,
                        help=f"Min OU mean-reversion speed (default: {DEFAULT_KAPPA_MIN:.2f}). "
                             "Use 0 to disable.")
    parser.add_argument("--cost", type=float, default=None,
                        help="Per-side transaction cost as a return fraction "
                             "(e.g. 0.001 = 10 bps). Default: value from pairs_trading.yml.")
    parser.add_argument("--show-trades", type=int, default=20,
                        help="Number of trades to preview in the console.")
    parser.add_argument("--show", "--plot", action="store_true", dest="show",
                        help="Display the performance chart interactively.")
    return parser.parse_args()


def build_blotter(model):
    """Extract long and short trade blotters from stored position DataFrames."""
    trades = pd.concat([
        bt_tools.extract_trades(model.pos_long_df,  model.pair_ret_df, "long"),
        bt_tools.extract_trades(model.pos_short_df, model.pair_ret_df, "short"),
    ]).sort_values("entry").reset_index(drop=True)
    return trades


def print_trade_stats(trades):
    if trades.empty:
        print("\nNo trades generated — try lowering --kappa-min or --p-value.")
        return
    wins = trades["trade_ret"] > 0
    print("\n=== Trade statistics ===")
    print(f"Total trades        : {len(trades)}  "
          f"(long {(trades['side']=='long').sum()}, "
          f"short {(trades['side']=='short').sum()})")
    print(f"Win rate            : {wins.mean():.1%}")
    print(f"Avg trade return    : {trades['trade_ret'].mean():+.2%}")
    print(f"Median trade return : {trades['trade_ret'].median():+.2%}")
    print(f"Best / worst trade  : {trades['trade_ret'].max():+.2%} "
          f"/ {trades['trade_ret'].min():+.2%}")
    print(f"Avg holding (days)  : {trades['days'].mean():.1f}")


def print_blotter_preview(trades, n_show):
    if trades.empty or n_show <= 0:
        return
    view = trades.copy()
    view["entry"] = view["entry"].dt.date
    view["exit"]  = view["exit"].dt.date
    view["trade_ret"] = (view["trade_ret"] * 100).round(2)
    view = view.rename(columns={"trade_ret": "ret_%"})
    print(f"\n=== Trade blotter (first {min(n_show, len(view))} of {len(view)}) ===")
    with pd.option_context("display.max_rows", None, "display.width", 140):
        print(view.head(n_show).to_string(index=False))


def print_running_pnl(port_ret):
    cum = port_ret["cum_pnl"]
    yearly = cum.groupby(cum.index.year).last()
    annual_ret = yearly.pct_change()
    annual_ret.iloc[0] = yearly.iloc[0] - 1.0
    print("\n=== Running PnL (year-end equity, base = 1.000) ===")
    for yr in yearly.index:
        print(f"  {yr}:  equity {yearly[yr]:.3f}   ({annual_ret[yr]:+.2%})")


def save_outputs(model, trades, etf, sharpe, maxdd, endpnl, show=False):
    out_dir = os.path.join("results", "pairs_trading", etf)
    os.makedirs(out_dir, exist_ok=True)

    trades.to_csv(os.path.join(out_dir, "trades.csv"), index=False)

    eq_cols = ["long_ret", "short_ret", "cum_short_ret", "cum_ret",
               "long_pnl", "short_pnl", "cum_pnl", "max_dd"]
    model.port_ret[eq_cols].to_csv(os.path.join(out_dir, "equity_curve.csv"))

    # Save which pairs had active signals on each day
    active = model.s_scores_df.notna().sum(axis=1).rename("active_pairs")
    active.to_csv(os.path.join(out_dir, "pairs_selected.csv"))

    bt_tools.plot_backtest_performance(
        model.port_ret,
        etf_name=None,
        trades=trades,
        sharpe=sharpe,
        maxdd=maxdd,
        endpnl=endpnl,
        save_path=os.path.join(out_dir, "performance.png"),
        show=show,
        title=f"{etf.upper()} pairs trading performance",
    )
    return out_dir


def main():
    base_cfg, pairs_cfg = load_cfgs()
    args = parse_args(base_cfg, pairs_cfg)

    cfg_start, cfg_end = base_cfg.get("bt_dt", {}).get(args.etf, DEFAULT_WINDOW)
    st_dt = args.start or cfg_start
    ed_dt = args.end   or cfg_end

    kappa_min = args.kappa_min if args.kappa_min > 0 else None

    cfg_cost = tuple(pairs_cfg.get("transaction_cost", [0.0010, 0.0010]))
    cost = (args.cost, args.cost) if args.cost is not None else cfg_cost

    prices_file_path = base_cfg["prices_file_path"][args.etf]

    print(f"ETF={args.etf.upper()}  strategy=pairs  window={st_dt}..{ed_dt}")
    print(f"n_window={args.n_window}  p_value<{args.p_value}  "
          f"max_pairs={args.max_pairs}  reselect_freq={args.reselect_freq}")
    print(f"kappa_min={kappa_min if kappa_min else 'off'}  "
          f"cost={cost[0]} ({cost[0]*1e4:g} bps/side)")
    print(f"prices: {prices_file_path}")
    print("Selecting pairs and fitting OU process (this may take a few minutes)...")

    model = PairsBt(
        prices_file_path=prices_file_path,
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

    sharpe, maxdd, endpnl = model.run(
        weighting_scheme=pairs_cfg.get("weighting_scheme", "equal_weighted"),
        sl=pairs_cfg.get("sl", -0.10),
        long_only=pairs_cfg.get("long_only", False),
        transaction_cost=cost,
    )

    trades = build_blotter(model)

    print("\n=== Portfolio performance ===")
    print(f"Sharpe ratio   : {sharpe:.3f}")
    print(f"Max drawdown   : {maxdd:.3%}")
    print(f"End PnL (x)    : {endpnl:.3f}  ({endpnl - 1:+.2%} total return)")
    print(f"Active pairs   : {model.s_scores_df.notna().sum(axis=1).mean():.1f} avg/day "
          f"(out of {len(model.pairs_info)} unique pairs selected)")

    print_trade_stats(trades)
    print_running_pnl(model.port_ret)
    print_blotter_preview(trades, args.show_trades)

    out_dir = save_outputs(model, trades, args.etf, sharpe, maxdd, endpnl, show=args.show)
    print(f"\nSaved trades.csv, equity_curve.csv, performance.png, "
          f"pairs_selected.csv to: {out_dir}")


if __name__ == "__main__":
    main()
