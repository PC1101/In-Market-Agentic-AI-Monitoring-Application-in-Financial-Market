"""
Run a single Avellaneda-Lee statistical-arbitrage backtest end to end and report
the full process: the trade blotter, the running equity curve, and summary stats.

This is a thin, self-contained entry point around src/backtest.py that does NOT
require ray (unlike mains/optimise_trading_rules.py). It reads the same
configs/optimise_trading_rules.yml so ETF -> price-file and date-range mappings
stay in one place.

Outputs (saved under results/run_backtest/<etf>_<defactoring>/):
    trades.csv         - every trade: ticker, side, entry, exit, days, return
    equity_curve.csv   - daily long/short/combined PnL and rolling drawdown
    performance.png     - equity curve, drawdown, leg PnL, trade-return histogram

Examples (run from anywhere; the script cd's to its own directory):

    python run_backtest.py                                 # XLF, PCA, kappa-filtered
    python run_backtest.py --etf xlk --defactoring etf
    python run_backtest.py --etf xlf --start 2019-01-02 --end 2020-11-19
    python run_backtest.py --kappa-min 0                   # disable the kappa filter
    python run_backtest.py --cost 0                        # frictionless (no tx cost)
    python run_backtest.py --show                          # also display the chart
"""
import argparse
import os
import sys

import matplotlib
# Pick the backend BEFORE importing pyplot: only use the non-blocking Agg backend
# when the user is NOT asking to display the chart (--show / --plot).
if not any(flag in sys.argv for flag in ("--show", "--plot")):
    matplotlib.use("Agg")
import pandas as pd              # noqa: E402
import yaml                      # noqa: E402

# All paths inside the project are relative to the project root, so make sure we
# are there regardless of where the script was launched from.
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(PROJECT_ROOT)
sys.path.insert(0, PROJECT_ROOT)

from src import backtest   # noqa: E402  (import after chdir/path setup)
from src import bt_tools   # noqa: E402

# Per-ETF backtest windows that differ from the default (newer sector ETFs that
# only started trading later). Mirrors mains/optimise_trading_rules.py.
DEFAULT_WINDOW = ("2007-01-03", "2015-01-02")

# Default mean-reversion-speed floor. Avellaneda-Lee trade only names whose OU
# reversion time 1/kappa is well below the estimation window. With a 60-day
# window, requiring 1/kappa < ~30 trading days gives kappa > 252/30 ~= 8.4 /yr.
DEFAULT_KAPPA_MIN = 252.0 / 30.0


def load_cfg():
    with open("configs/optimise_trading_rules.yml", "r") as fh:
        return yaml.load(fh, Loader=yaml.SafeLoader)


def parse_args(cfg):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--etf", default="xlf",
                        choices=sorted(cfg["prices_file_path"].keys()),
                        help="Sector ETF universe to trade (default: xlf).")
    parser.add_argument("--defactoring", default="pca", choices=["pca", "etf"],
                        help="Risk factor to strip from stock returns: the sector "
                             "ETF ('etf') or 15 PCA eigen-portfolios ('pca').")
    parser.add_argument("--start", default=None, help="Backtest start date YYYY-MM-DD.")
    parser.add_argument("--end", default=None, help="Backtest end date YYYY-MM-DD.")
    parser.add_argument("--n-window", type=int, default=60,
                        help="Rolling estimation window in trading days (default: 60).")
    parser.add_argument("--kappa-min", type=float, default=DEFAULT_KAPPA_MIN,
                        help=f"Min OU mean-reversion speed to trade a name "
                             f"(default: {DEFAULT_KAPPA_MIN:.2f}). Use 0 to disable.")
    parser.add_argument("--cost", type=float, default=None,
                        help="Per-side transaction cost as a return fraction "
                             "(e.g. 0.0005 = 5 bps). Use 0 for a frictionless run. "
                             "Default: value from the config.")
    parser.add_argument("--show-trades", type=int, default=20,
                        help="How many trades to print to the console (default: 20).")
    parser.add_argument("--show", "--plot", action="store_true", dest="show",
                        help="Display the performance chart interactively "
                             "(it is always saved to performance.png regardless).")
    return parser.parse_args()


def build_blotter(model):
    """Reconstruct the long and short trade blotter from the stored positions."""
    trades = pd.concat([
        bt_tools.extract_trades(model.pos_long_df, model.ret_df, "long"),
        bt_tools.extract_trades(model.pos_short_df, model.ret_df, "short"),
    ]).sort_values("entry").reset_index(drop=True)
    return trades


def print_trade_stats(trades):
    if trades.empty:
        print("\nNo trades were generated (try lowering --kappa-min).")
        return
    n = len(trades)
    wins = trades["trade_ret"] > 0
    print("\n=== Trade statistics ===")
    print(f"Total trades        : {n}  "
          f"(long {(trades['side'] == 'long').sum()}, "
          f"short {(trades['side'] == 'short').sum()})")
    print(f"Win rate            : {wins.mean():.1%}")
    print(f"Avg trade return    : {trades['trade_ret'].mean():+.2%}")
    print(f"Median trade return : {trades['trade_ret'].median():+.2%}")
    print(f"Best / worst trade  : {trades['trade_ret'].max():+.2%} / {trades['trade_ret'].min():+.2%}")
    print(f"Avg holding (days)  : {trades['days'].mean():.1f}")


def print_blotter_preview(trades, n_show):
    if trades.empty or n_show <= 0:
        return
    view = trades.copy()
    view["entry"] = view["entry"].dt.date
    view["exit"] = view["exit"].dt.date
    view["trade_ret"] = (view["trade_ret"] * 100).round(2)
    view = view.rename(columns={"trade_ret": "ret_%"})
    print(f"\n=== Trade blotter (first {min(n_show, len(view))} of {len(view)}) ===")
    with pd.option_context("display.max_rows", None, "display.width", 120):
        print(view.head(n_show).to_string(index=False))


def print_running_pnl(port_ret):
    """Year-end snapshots of the running equity curve (full daily series is in CSV)."""
    cum = port_ret["cum_pnl"]
    yearly = cum.groupby(cum.index.year).last()
    annual_ret = yearly.pct_change()
    annual_ret.iloc[0] = yearly.iloc[0] - 1.0  # first year from the 1.0 base
    print("\n=== Running PnL (year-end equity, base = 1.000) ===")
    for yr in yearly.index:
        print(f"  {yr}:  equity {yearly[yr]:.3f}   ({annual_ret[yr]:+.2%})")


def save_outputs(model, trades, etf, defactoring, sharpe, maxdd, endpnl, show=False):
    out_dir = os.path.join("results", "run_backtest", f"{etf}_{defactoring}")
    os.makedirs(out_dir, exist_ok=True)

    trades.to_csv(os.path.join(out_dir, "trades.csv"), index=False)

    eq_cols = ["long_ret", "short_ret", "cum_short_ret", "cum_ret",
               "long_pnl", "short_pnl", "cum_pnl", "max_dd"]
    model.port_ret[eq_cols].to_csv(os.path.join(out_dir, "equity_curve.csv"))

    bt_tools.plot_backtest_performance(
        model.port_ret, etf_name=model.etf_name, trades=trades,
        sharpe=sharpe, maxdd=maxdd, endpnl=endpnl,
        save_path=os.path.join(out_dir, "performance.png"), show=show,
        title=f"{etf.upper()} {defactoring} stat-arb performance",
    )
    return out_dir


def main():
    cfg = load_cfg()
    args = parse_args(cfg)

    cfg_start, cfg_end = cfg.get("bt_dt", {}).get(args.etf, DEFAULT_WINDOW)
    st_dt = args.start or cfg_start
    ed_dt = args.end or cfg_end
    kappa_min = args.kappa_min if args.kappa_min > 0 else None
    prices_file_path = cfg["prices_file_path"][args.etf]

    # Per-side transaction cost: CLI override, else config, applied on entry/exit days.
    cfg_cost = tuple(cfg.get("transaction_cost", (0.0005, 0.0005)))
    cost = (args.cost, args.cost) if args.cost is not None else cfg_cost

    print(f"ETF={args.etf.upper()}  defactoring={args.defactoring}  "
          f"window={st_dt}..{ed_dt}  n_window={args.n_window}  "
          f"kappa_min={kappa_min if kappa_min is not None else 'off'}  "
          f"cost={cost[0]} ({'frictionless' if cost[0] == 0 else f'{cost[0] * 1e4:g} bps/side'})")
    print(f"prices: {prices_file_path}")
    print("Fitting OU process / generating daily s-scores (this takes a moment)...")

    model = backtest.bt(
        prices_file_path=prices_file_path,
        etf_name=args.etf,
        st_dt=st_dt,
        ed_dt=ed_dt,
        n_window=args.n_window,
        defactoring=args.defactoring,
        performance_only=True,   # we render our own (non-blocking) charts below
        kappa_min=kappa_min,
        progress=True,           # show a tqdm bar over the OU fitting loop
    )

    sharpe, maxdd, endpnl = model.run(
        weighting_scheme=cfg.get("weighting_scheme", "equal_weighted"),
        sl=cfg.get("sl", -0.10),
        long_only=cfg.get("long_only", False),
        transaction_cost=cost,
    )

    trades = build_blotter(model)

    print("\n=== Portfolio performance ===")
    print(f"Sharpe ratio   : {sharpe:.3f}")
    print(f"Max drawdown   : {maxdd:.3%}")
    print(f"End PnL (x)    : {endpnl:.3f}  ({endpnl - 1:+.2%} total return)")

    print_trade_stats(trades)
    print_running_pnl(model.port_ret)
    print_blotter_preview(trades, args.show_trades)

    out_dir = save_outputs(model, trades, args.etf, args.defactoring,
                           sharpe, maxdd, endpnl, show=args.show)
    print(f"\nSaved trades.csv, equity_curve.csv and performance.png to: {out_dir}")


if __name__ == "__main__":
    main()
