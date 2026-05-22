"""
Full S&P 500 (sector-ETF) Avellaneda-Lee backtest with SECTOR-ETF HEDGING.

The bt class trades one sector ETF's constituents at a time. This runner executes
each of the 11 SPDR sector sleeves (defactoring='etf' => each stock hedged by its
own sector ETF, i.e. "sector-ETF hedging"), then aggregates the sleeves into one
portfolio. The union of the sector ETFs' holdings ~= the S&P 500 (506 names).

Aggregation: each sector sleeve is a self-contained dollar-neutral long/short book
producing a daily combined return; the portfolio return is the EQUAL-WEIGHTED mean
of the active sleeves on each day (sectors with no data yet are skipped, not zero-
filled). This treats every sector as an equally-capitalized sleeve.

IMPORTANT DATA CAVEAT: the price CSVs are a 2020 holdings snapshot, NOT point-in-
time membership. Backtests before 2020 are therefore survivorship-biased (delisted
/ removed names are absent). Treat results as an upper bound on realised edge.

Examples:
    python run_full_universe.py --start 2007-01-03 --end 2015-01-02
    python run_full_universe.py --start 2013-01-02 --end 2014-12-31 --sectors xlf,xlk,xli
    python run_full_universe.py --defactoring pca --cost 0
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np               # noqa: E402
import pandas as pd              # noqa: E402
import yaml                      # noqa: E402

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(PROJECT_ROOT)
sys.path.insert(0, PROJECT_ROOT)

from src import backtest   # noqa: E402
from src import bt_tools   # noqa: E402

DEFAULT_KAPPA_MIN = 252.0 / 30.0


def load_cfg():
    with open("configs/optimise_trading_rules.yml", "r") as fh:
        return yaml.load(fh, Loader=yaml.SafeLoader)


def sector_start_date(prices_file_path):
    """First date present in a sector CSV (to skip sectors that postdate the window)."""
    idx = pd.read_csv(prices_file_path, index_col="Date", parse_dates=True, usecols=["Date"]).index
    return idx.min()


def run_sector(etf, prices_file_path, st_dt, ed_dt, defactoring, kappa_min, cost, sl, weighting):
    """Run one sector sleeve; return (daily_combined_return, trade_blotter) or (None, None)."""
    model = backtest.bt(
        prices_file_path=prices_file_path, etf_name=etf, st_dt=st_dt, ed_dt=ed_dt,
        defactoring=defactoring, kappa_min=kappa_min, performance_only=True, progress=True,
    )
    model.run(weighting_scheme=weighting, sl=sl, long_only=False,
              transaction_cost=(cost, cost))

    sector_ret = model.port_ret["cum_ret"].rename(etf)
    longs = bt_tools.extract_trades(model.pos_long_df, model.ret_df, "long")
    shorts = bt_tools.extract_trades(model.pos_short_df, model.ret_df, "short")
    trades = pd.concat([longs, shorts])
    trades.insert(0, "sector", etf)
    return sector_ret, trades


def main():
    cfg = load_cfg()
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--start", default="2007-01-03")
    p.add_argument("--end", default="2015-01-02")
    p.add_argument("--sectors", default="all",
                   help="Comma list of sector ETFs, or 'all' (default).")
    p.add_argument("--defactoring", default="etf", choices=["etf", "pca"],
                   help="'etf' = each stock hedged by its sector ETF (spec baseline).")
    p.add_argument("--kappa-min", type=float, default=DEFAULT_KAPPA_MIN)
    p.add_argument("--cost", type=float, default=0.0,
                   help="Per-side transaction cost fraction (default 0 = frictionless).")
    p.add_argument("--sl", type=float, default=-0.10, help="Stop-loss (default -0.10).")
    p.add_argument("--tag", default=None, help="Label for the output sub-folder.")
    args = p.parse_args()

    kappa_min = args.kappa_min if args.kappa_min > 0 else None
    weighting = cfg.get("weighting_scheme", "equal_weighted")

    all_sectors = list(cfg["prices_file_path"].keys())
    sectors = all_sectors if args.sectors == "all" else [s.strip() for s in args.sectors.split(",")]

    # Skip sectors whose data starts after the window start (e.g. XLC 2018, XLRE 2015).
    win_start = pd.Timestamp(args.start)
    active, skipped = [], []
    for s in sectors:
        if sector_start_date(cfg["prices_file_path"][s]) <= win_start:
            active.append(s)
        else:
            skipped.append(s)

    print(f"window {args.start}..{args.end}  defactoring={args.defactoring}  "
          f"kappa_min={kappa_min if kappa_min is not None else 'off'}  "
          f"cost={args.cost} ({'frictionless' if args.cost == 0 else f'{args.cost*1e4:g} bps'})")
    print(f"active sectors ({len(active)}): {active}")
    if skipped:
        print(f"skipped (data starts after window): {skipped}")

    sector_rets, all_trades, summary = {}, [], []
    for s in active:
        print(f"\n--- {s.upper()} ---")
        try:
            ret, trades = run_sector(s, cfg["prices_file_path"][s], args.start, args.end,
                                     args.defactoring, kappa_min, args.cost, args.sl, weighting)
        except Exception as exc:  # robust: one bad sector shouldn't kill the run
            print(f"  skipped {s}: {exc}")
            continue
        sector_rets[s] = ret
        all_trades.append(trades)
        sh = np.sqrt(252) * ret.mean() / ret.std() if ret.std() else np.nan
        summary.append({"sector": s, "n_trades": len(trades),
                        "sharpe": sh, "total_ret": (1 + ret).prod() - 1})

    if not sector_rets:
        print("No sectors produced returns; nothing to aggregate.")
        return

    # Equal-weight active sleeves each day (skipna so pre-inception sectors don't count).
    ret_panel = pd.concat(sector_rets.values(), axis=1).sort_index()
    port_ret = ret_panel.mean(axis=1).dropna()
    equity = (1 + port_ret).cumprod()
    drawdown = equity / equity.cummax() - 1
    sharpe = np.sqrt(252) * port_ret.mean() / port_ret.std() if port_ret.std() else np.nan

    print("\n=== Per-sector summary ===")
    print(pd.DataFrame(summary).round(3).to_string(index=False))
    print("\n=== Full-universe (equal-weight sectors) ===")
    print(f"Active sectors : {len(sector_rets)}")
    print(f"Trading days   : {len(port_ret)}")
    print(f"Total trades   : {sum(len(t) for t in all_trades)}")
    print(f"Sharpe ratio   : {sharpe:.3f}")
    print(f"Max drawdown   : {drawdown.min():.2%}")
    print(f"Total return   : {equity.iloc[-1] - 1:+.2%}")

    tag = args.tag or f"full_{args.defactoring}_{args.start}_{args.end}"
    out_dir = os.path.join("results", "full_universe", tag)
    os.makedirs(out_dir, exist_ok=True)
    perf = pd.DataFrame({"port_ret": port_ret, "equity": equity, "drawdown": drawdown})
    perf.to_csv(os.path.join(out_dir, "equity_curve.csv"))
    ret_panel.to_csv(os.path.join(out_dir, "sector_returns.csv"))
    blotter = pd.concat(all_trades).sort_values("entry").reset_index(drop=True)
    blotter.to_csv(os.path.join(out_dir, "trades.csv"), index=False)

    fig, axs = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    equity.plot(ax=axs[0], title=f"Full-universe sector-hedged equity ({args.defactoring})")
    axs[0].axhline(1.0, lw=0.6, ls="--", color="k")
    axs[0].text(0.02, 0.04, f"Sharpe {sharpe:.2f}\nMaxDD {drawdown.min():.1%}\n"
                f"Total {equity.iloc[-1]-1:+.1%}", transform=axs[0].transAxes,
                va="bottom", bbox=dict(boxstyle="round", fc="white", ec="0.7"))
    drawdown.plot(ax=axs[1], color="firebrick", title="Drawdown")
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "performance.png"), dpi=110)
    plt.close(fig)
    print(f"\nSaved equity_curve.csv, sector_returns.csv, trades.csv, performance.png to: {out_dir}")


if __name__ == "__main__":
    main()
