"""
Post-hoc analysis of a full-universe run produced by run_full_universe.py.

  --mode aug2007 : daily-equity zoom on Aug 1-31 2007 to look for the
                   Khandani-Lo (Aug 6-9 2007) stat-arb meltdown.
  --mode break2009 : locate the start of the 2009 drawdown, pull the 20 worst
                     trades in that window, and break them down by sector/ticker.
  --mode both : run both (default).

Reads results/full_universe/<tag>/{equity_curve.csv, trades.csv}.
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd              # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def load(tag):
    base = os.path.join(HERE, "results", "full_universe", tag)
    eq = pd.read_csv(os.path.join(base, "equity_curve.csv"), index_col=0, parse_dates=True)
    tr = pd.read_csv(os.path.join(base, "trades.csv"), parse_dates=["entry", "exit"])
    return base, eq, tr


def aug2007(base, eq):
    win = eq.loc["2007-08-01":"2007-08-31"]
    if win.empty:
        print("No data in Aug 2007 for this run.")
        return
    # Rebase equity to Aug 1 = 1.0 for a clean intra-month view.
    rebased = (1 + win["port_ret"]).cumprod()
    print("\n=== Aug 2007 daily zoom (Khandani-Lo window) ===")
    print(f"{'date':<12}{'daily_ret':>12}{'rebased_eq':>14}")
    for d, r in win["port_ret"].items():
        print(f"{d.date()!s:<12}{r:>12.4%}{rebased.loc[d]:>14.4f}")
    klo = win.loc["2007-08-06":"2007-08-09", "port_ret"]
    if not klo.empty:
        cum = (1 + klo).prod() - 1
        print(f"\nAug 6-9 cumulative return: {cum:+.2%}  (Khandani-Lo meltdown window)")
        print(f"Aug 2007 worst single day: {win['port_ret'].min():+.2%} "
              f"on {win['port_ret'].idxmin().date()}")

    fig, ax = plt.subplots(figsize=(11, 5))
    rebased.plot(ax=ax, marker="o", ms=3)
    ax.axhline(1.0, lw=0.6, ls="--", color="k")
    for d in ("2007-08-06", "2007-08-09"):
        if pd.Timestamp(d) in rebased.index:
            ax.axvline(pd.Timestamp(d), lw=0.8, ls=":", color="firebrick")
    ax.set_title("Full-universe equity, Aug 2007 (rebased Aug 1 = 1.0)")
    ax.set_ylabel("equity")
    plt.tight_layout()
    out = os.path.join(base, "aug2007_zoom.png")
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"Saved {out}")


def break2009(base, eq, tr):
    # Deepest drawdown point in 2009, and the equity peak that preceded it.
    dd_2009 = eq.loc["2009-01-01":"2009-12-31", "drawdown"]
    if dd_2009.empty:
        print("No 2009 data for this run.")
        return
    trough = dd_2009.idxmin()
    peak = eq.loc[:trough, "equity"].idxmax()
    print("\n=== 2009 drawdown decomposition ===")
    print(f"Drawdown START (prior equity peak): {peak.date()}  (week of {peak.to_period('W')})")
    print(f"Drawdown TROUGH (deepest point)   : {trough.date()}  "
          f"({eq.loc[trough, 'drawdown']:.2%})")
    print(f"Peak->trough portfolio return     : "
          f"{eq.loc[trough,'equity']/eq.loc[peak,'equity']-1:+.2%}")

    # Trades that closed inside the drawdown window.
    win_tr = tr[(tr["exit"] >= peak) & (tr["exit"] <= trough)].copy()
    print(f"\nTrades closed in {peak.date()}..{trough.date()}: {len(win_tr)}")
    if win_tr.empty:
        return
    worst = win_tr.nsmallest(20, "trade_ret")
    view = worst.copy()
    view["entry"] = view["entry"].dt.date
    view["exit"] = view["exit"].dt.date
    view["ret_%"] = (view["trade_ret"] * 100).round(2)
    print("\n--- 20 worst trades in the drawdown window ---")
    print(view[["sector", "ticker", "side", "entry", "exit", "days", "ret_%"]].to_string(index=False))

    print("\n--- worst-20 by sector ---")
    print(worst["sector"].value_counts().to_string())
    print("\n--- all-window losers by sector (share of negative trades) ---")
    losers = win_tr[win_tr["trade_ret"] < 0]
    print(losers["sector"].value_counts().to_string())
    print(f"\nWindow win rate: {(win_tr['trade_ret'] > 0).mean():.1%}  "
          f"mean trade: {win_tr['trade_ret'].mean():+.2%}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tag", default="baseline_2007_2015")
    p.add_argument("--mode", default="both", choices=["aug2007", "break2009", "both"])
    args = p.parse_args()

    base, eq, tr = load(args.tag)
    print(f"Loaded {args.tag}: {len(eq)} days "
          f"({eq.index.min().date()}..{eq.index.max().date()}), {len(tr)} trades")
    if args.mode in ("aug2007", "both"):
        aug2007(base, eq)
    if args.mode in ("break2009", "both"):
        break2009(base, eq, tr)


if __name__ == "__main__":
    main()
