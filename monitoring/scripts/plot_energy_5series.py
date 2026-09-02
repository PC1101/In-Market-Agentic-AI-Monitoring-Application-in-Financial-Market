"""5-series energy performance chart for one window ($1M account).

  baseline (buy & hold) · AL PCA · AL PCA + agentic · JT momentum · JT momentum + agentic

Usage: python scripts/plot_energy_5series.py --window oil_crash_2020
The agentic overlay scales exposure by the qwen2.5:3b monitor state
(NORMAL=full, WATCH=half, ALERT=cash), applied with a 1-day lag (no look-ahead).
"""
from __future__ import annotations

import argparse
import json
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
EXPOSURE = {"NORMAL": 1.0, "WATCH": 0.5, "ALERT": 0.0}
STRAT_CURVE = {
    "AL_PCA": REPO / "Stat Arb" / "statsArb-dev" / "results" / "equity_curve_energy_al.csv",
    "JT_MOM": REPO / "XSectional" / "results" / "equity_curve_energy.csv",
}


def _window_dates(name: str) -> tuple[str, str]:
    from providers.energy.windows import ALL_WINDOWS
    for w in ALL_WINDOWS:
        if w.name == name:
            return w.start, w.end
    raise SystemExit(f"unknown window {name}")


def _strat_returns(strat: str, s: str, e: str) -> pd.Series:
    df = pd.read_csv(STRAT_CURVE[strat], index_col=0, parse_dates=True).sort_index()
    return df["port_ret"].astype(float).loc[s:e]


def _exposure(window: str, strat: str, index) -> pd.Series:
    log = REPO / "monitoring" / "results" / f"agentic_{window}_{strat}.jsonl"
    sig = {}
    if log.exists():
        for line in open(log):
            r = json.loads(line)
            if r.get("agent") == "performance_supervisor":
                st = (r.get("assessment") or {}).get("state", "NORMAL")
                sig[pd.Timestamp(r["as_of"])] = EXPOSURE.get(st, 1.0)
    s = pd.Series(sig).sort_index()
    return s.reindex(index).ffill().fillna(1.0).shift(1).fillna(1.0)


def _baseline(index, s: str, e: str) -> pd.Series:
    from providers.energy.universe import load_prices
    start_pad = (pd.Timestamp(s) - pd.Timedelta(days=10)).strftime("%Y-%m-%d")
    px = load_prices(start=start_pad, end=e).dropna(axis=1, how="all")
    return px.pct_change().mean(axis=1).reindex(index).fillna(0.0)


def _acct(r: pd.Series) -> pd.Series:
    return START_CAP * (1 + r.fillna(0.0)).cumprod()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", default="oil_crash_2020")
    args = ap.parse_args()
    s, e = _window_dates(args.window)

    al = _strat_returns("AL_PCA", s, e); idx = al.index
    jt = _strat_returns("JT_MOM", s, e).reindex(idx).fillna(0.0)
    al_ag = al * _exposure(args.window, "AL_PCA", idx)
    jt_ag = jt * _exposure(args.window, "JT_MOM", idx)
    base = _baseline(idx, s, e)

    series = {
        "Baseline (buy & hold energy)": ("#6b7280", "-", _acct(base)),
        "AL PCA": ("#1d4ed8", "-", _acct(al)),
        "AL PCA + agentic": ("#1d4ed8", "--", _acct(al_ag)),
        "JT momentum": ("#c2410c", "-", _acct(jt)),
        "JT momentum + agentic": ("#c2410c", "--", _acct(jt_ag)),
    }

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7.5), height_ratios=[3, 1],
                                   sharex=True, gridspec_kw={"hspace": 0.1})
    for name, (c, ls, acct) in series.items():
        ret = acct.iloc[-1] / START_CAP - 1
        dd = (acct / acct.cummax() - 1).min()
        ax1.plot(acct.index, acct, color=c, ls=ls, lw=1.7,
                 label=f"{name}:  {ret:+.1%}, maxDD {dd:.1%}")
        ax2.fill_between(acct.index, (acct/acct.cummax()-1)*100, 0, color=c, alpha=0.15)
    ax1.axhline(START_CAP, color="#9ca3af", lw=1.0, ls=":")
    ax1.set_title(f"Global energy — {args.window}: baseline vs strategies vs strategies + agentic monitor\n"
                  "$1,000,000 account (dashed = with qwen2.5:3b monitor overlay)",
                  fontsize=12, fontweight="bold")
    ax1.set_ylabel("Account value"); ax1.grid(True, alpha=0.25)
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x/1e6:.2f}M"))
    ax1.legend(loc="best", fontsize=9, frameon=False)
    ax2.set_ylabel("Drawdown %"); ax2.set_xlabel("Date"); ax2.grid(True, alpha=0.25)

    out = REPO / "monitoring" / "results" / f"energy_5series_{args.window}.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"wrote {out}")
    for name, (_, _, acct) in series.items():
        print(f"  {name}: {acct.iloc[-1]/START_CAP-1:+.1%}, maxDD {(acct/acct.cummax()-1).min():.1%}")


if __name__ == "__main__":
    main()
