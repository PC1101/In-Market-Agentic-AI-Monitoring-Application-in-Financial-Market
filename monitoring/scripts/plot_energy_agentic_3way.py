"""3-way energy performance comparison over the 2020 oil-crash window ($1M account).

  1. Baseline           — equal-weight buy-and-hold of the energy universe
  2. Strategy           — AL PCA stat-arb alone
  3. Strategy + agentic — AL PCA with exposure scaled by the qwen2.5:3b monitor's
                          regime signal (NORMAL=full, WATCH=half, ALERT=cash),
                          applied with a 1-day lag (act the day AFTER the signal —
                          no lookahead).

Renders monitoring/results/energy_pnl_3way.png.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
START_CAP = 1_000_000
WIN_START, WIN_END = "2020-02-14", "2020-05-29"

#: per-strategy curve file + display label
CURVES = {
    "AL_PCA": (REPO / "Stat Arb" / "statsArb-dev" / "results" / "equity_curve_energy_al.csv",
               "AL PCA"),
    "JT_MOM": (REPO / "XSectional" / "results" / "equity_curve_energy.csv",
               "JT momentum"),
}

#: monitor state -> fraction of capital deployed
EXPOSURE = {"NORMAL": 1.0, "WATCH": 0.5, "ALERT": 0.0}

# Set by main() from --strategy.
STRATEGY = "AL_PCA"
AGENTIC_LOG = REPO / "monitoring" / "results" / "agentic_oil_crash_2020_AL_PCA.jsonl"
OUT = REPO / "monitoring" / "results" / "energy_pnl_3way.png"


def _strategy_returns() -> pd.Series:
    curve = CURVES[STRATEGY][0]
    df = pd.read_csv(curve, index_col=0, parse_dates=True).sort_index()
    r = df["port_ret"].astype(float)
    return r.loc[WIN_START:WIN_END]


def _agent_exposure(index: pd.DatetimeIndex) -> pd.Series:
    """Daily exposure from the monitor's state, forward-filled, acted on next day."""
    sig = {}
    for line in open(AGENTIC_LOG):
        rec = json.loads(line)
        if rec.get("agent") == "performance_supervisor":
            a = rec.get("assessment") or {}
            st = a.get("state", "NORMAL")
            sig[pd.Timestamp(rec["as_of"])] = EXPOSURE.get(st, 1.0)
    s = pd.Series(sig).sort_index()
    # Reindex to trading days, forward-fill the last signal, default full exposure
    # before the first assessment. Shift by 1 day so we act AFTER seeing the signal.
    expo = s.reindex(index).ffill().fillna(1.0).shift(1).fillna(1.0)
    return expo


def _baseline_returns(index: pd.DatetimeIndex) -> pd.Series:
    """Equal-weight buy-and-hold of the energy universe over the window."""
    import sys
    sys.path.insert(0, str(REPO / "monitoring"))
    from providers.energy.universe import load_prices
    px = load_prices(start="2020-02-01", end=WIN_END).dropna(axis=1, how="all")
    rets = px.pct_change().mean(axis=1)  # equal-weight daily return
    return rets.reindex(index).fillna(0.0)


def _to_account(returns: pd.Series) -> pd.Series:
    return START_CAP * (1 + returns.fillna(0.0)).cumprod()


def main() -> None:
    global STRATEGY, AGENTIC_LOG, OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategy", default="AL_PCA", choices=list(CURVES))
    args = ap.parse_args()
    STRATEGY = args.strategy
    AGENTIC_LOG = REPO / "monitoring" / "results" / f"agentic_oil_crash_2020_{STRATEGY}.jsonl"
    suffix = "" if STRATEGY == "AL_PCA" else f"_{STRATEGY}"
    OUT = REPO / "monitoring" / "results" / f"energy_pnl_3way{suffix}.png"

    strat_r = _strategy_returns()
    idx = strat_r.index
    expo = _agent_exposure(idx)
    agentic_r = strat_r * expo
    base_r = _baseline_returns(idx)

    slabel = CURVES[STRATEGY][1]
    curves = {
        "Baseline (buy & hold energy)": ("#6b7280", _to_account(base_r)),
        f"Strategy ({slabel})": ("#1d4ed8", _to_account(strat_r)),
        f"{slabel} + agentic (qwen2.5:3b)": ("#c2410c", _to_account(agentic_r)),
    }

    fig, (ax1, ax2, ax3) = plt.subplots(
        3, 1, figsize=(12, 9), height_ratios=[3, 1, 0.8], sharex=True,
        gridspec_kw={"hspace": 0.12})

    for name, (color, acct) in curves.items():
        final = acct.iloc[-1]; ret = final / START_CAP - 1
        dd = (acct / acct.cummax() - 1).min()
        ax1.plot(acct.index, acct, color=color, lw=1.8,
                 label=f"{name}:  ${final:,.0f}  ({ret:+.1%}, maxDD {dd:.1%})")
        ax2.fill_between(acct.index, (acct/acct.cummax()-1)*100, 0, color=color, alpha=0.30)

    ax1.axhline(START_CAP, color="#9ca3af", lw=1.0, ls="--")
    ax1.set_title(f"Global energy — 2020 oil crash: baseline vs {slabel} vs {slabel} + agentic monitor\n"
                  "$1,000,000 account", fontsize=13, fontweight="bold")
    ax1.set_ylabel("Account value"); ax1.grid(True, alpha=0.25)
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x/1e6:.2f}M"))
    ax1.legend(loc="lower left", fontsize=9, frameon=False)
    ax2.set_ylabel("Drawdown %"); ax2.grid(True, alpha=0.25)

    # Exposure band: show when the monitor cut risk.
    ax3.fill_between(expo.index, expo.values, 0, step="pre", color="#c2410c", alpha=0.35)
    ax3.set_ylabel("Agent\nexposure"); ax3.set_ylim(0, 1.05); ax3.set_xlabel("Date")
    ax3.grid(True, alpha=0.25)
    ax3.set_yticks([0, 0.5, 1.0]); ax3.set_yticklabels(["cash", "half", "full"])

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=140, bbox_inches="tight")
    print(f"wrote {OUT}")
    for name, (_, acct) in curves.items():
        print(f"  {name}: ${acct.iloc[-1]:,.0f} ({acct.iloc[-1]/START_CAP-1:+.1%}), "
              f"maxDD {(acct/acct.cummax()-1).min():.1%}")


if __name__ == "__main__":
    main()
