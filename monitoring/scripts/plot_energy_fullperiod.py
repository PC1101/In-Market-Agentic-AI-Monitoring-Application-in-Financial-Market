"""Full-period energy performance: baseline vs AL PCA vs JT momentum, $1M account.

Continuous 2017-2025 (the strategies' common range), not per-window slices.
The agentic monitor is a crisis-window overlay (news 2017+, heavy inference) and is
shown in the case-study figures, not applied continuously here.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "monitoring"))
START_CAP = 1_000_000
CURVES = {
    "AL PCA": (REPO / "Stat Arb" / "statsArb-dev" / "results" / "equity_curve_energy_al.csv", "#1d4ed8"),
    "JT momentum": (REPO / "XSectional" / "results" / "equity_curve_energy.csv", "#c2410c"),
}
OUT = REPO / "monitoring" / "results" / "energy_fullperiod_2017_2025.png"


def _returns(path: Path) -> pd.Series:
    return pd.read_csv(path, index_col=0, parse_dates=True).sort_index()["port_ret"].astype(float)


def main() -> None:
    rets = {name: _returns(p) for name, (p, _) in CURVES.items()}
    lo = max(r.index.min() for r in rets.values())
    hi = min(r.index.max() for r in rets.values())
    idx = None
    series = {}
    for name, r in rets.items():
        r = r.loc[lo:hi]
        idx = r.index if idx is None else idx
        series[name] = r

    from providers.energy.universe import load_prices
    px = load_prices(start=(lo - pd.Timedelta(days=10)).strftime("%Y-%m-%d"),
                     end=hi.strftime("%Y-%m-%d")).dropna(axis=1, how="all")
    base = px.pct_change().mean(axis=1).reindex(idx).fillna(0.0)

    def acct(r):
        return START_CAP * (1 + r.fillna(0.0)).cumprod()

    plotted = {"Baseline (buy & hold energy)": ("#6b7280", acct(base))}
    for name, r in series.items():
        plotted[name] = (CURVES[name][1], acct(r))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 7.5), height_ratios=[3, 1],
                                   sharex=True, gridspec_kw={"hspace": 0.1})
    for name, (c, a) in plotted.items():
        ret = a.iloc[-1] / START_CAP - 1
        dd = (a / a.cummax() - 1).min()
        yrs = (a.index[-1] - a.index[0]).days / 365.25
        cagr = (a.iloc[-1] / START_CAP) ** (1 / yrs) - 1
        ax1.plot(a.index, a, color=c, lw=1.8,
                 label=f"{name}:  ${a.iloc[-1]:,.0f}  ({ret:+.0%} tot, {cagr:+.1%} CAGR, maxDD {dd:.0%})")
        ax2.fill_between(a.index, (a/a.cummax()-1)*100, 0, color=c, alpha=0.25)

    ax1.axhline(START_CAP, color="#9ca3af", lw=1.0, ls="--")
    ax1.set_title(f"Global energy — {lo.date()} to {hi.date()}: baseline vs strategies\n"
                  "$1,000,000 account (continuous, full period)", fontsize=13, fontweight="bold")
    ax1.set_ylabel("Account value"); ax1.grid(True, alpha=0.25)
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x/1e6:.2f}M"))
    ax1.legend(loc="upper left", fontsize=9, frameon=False)
    ax2.set_ylabel("Drawdown %"); ax2.set_xlabel("Date"); ax2.grid(True, alpha=0.25)

    fig.savefig(OUT, dpi=140, bbox_inches="tight")
    print(f"wrote {OUT}")
    for name, (_, a) in plotted.items():
        print(f"  {name}: ${a.iloc[-1]:,.0f} ({a.iloc[-1]/START_CAP-1:+.1%}), maxDD {(a/a.cummax()-1).min():.1%}")


if __name__ == "__main__":
    main()
