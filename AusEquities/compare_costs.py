"""
compare_costs.py — Pre vs post transaction cost equity curves for both AU strategies.

Loads cached data/results and produces one figure per strategy showing gross (no TC)
vs net (full TC) equity curves, plus a cumulative cost drag panel.
"""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

import data_norgate as dn
import run_jt_momentum as jt
import run_al_statarb as al

INITIAL_VALUE = 1_000_000
OUT_DIR = HERE / "results" / "cost_comparison"
OUT_DIR.mkdir(parents=True, exist_ok=True)

C_GROSS = "#2176AE"   # blue  — gross / pre-TC
C_NET   = "#E8553A"   # red   — net / post-TC
C_DRAG  = "#888888"   # grey  — cost drag


# ─────────────────────────────────────────────────────────────────────────────
# JT Momentum — run backtest capturing both gross and net
# ─────────────────────────────────────────────────────────────────────────────

def jt_gross_net(weights: pd.DataFrame, prices: pd.DataFrame):
    """Return (gross_equity, net_equity, daily_tc) series."""
    daily_ret = prices.pct_change(fill_method=None)
    w = weights.astype(float).reindex(columns=daily_ret.columns)
    w_daily = w.reindex(daily_ret.index, method="ffill").shift(1)
    w0 = w_daily.fillna(0)
    delta_w = w0.diff().abs()
    gross_ret = (w0 * daily_ret.fillna(0)).sum(axis=1)

    # Gross track: portfolio grows without any TC
    V_gross = INITIAL_VALUE
    gross_port, gross_V = [], []
    for i in range(len(daily_ret)):
        r = gross_ret.iloc[i]
        gross_port.append(r)
        V_gross = max(V_gross * (1 + r), 1.0)
        gross_V.append(V_gross)

    # Net track: TC subtracted, V used to scale $10 minimum
    V_net = INITIAL_VALUE
    net_port, tc_list = [], []
    for i in range(len(daily_ret)):
        tc = jt._tiered_tc_frac(delta_w.iloc[i], V_net)
        r = gross_ret.iloc[i] - tc
        net_port.append(r)
        tc_list.append(tc)
        V_net = max(V_net * (1 + r), 1.0)

    idx = daily_ret.index

    # Trim to first active day
    active = w0.abs().sum(axis=1) > 0
    start = active.idxmax() if active.any() else idx[0]

    gross_eq = (1 + pd.Series(gross_port, index=idx)).cumprod() * INITIAL_VALUE
    net_eq   = (1 + pd.Series(net_port,   index=idx)).cumprod() * INITIAL_VALUE
    tc_ser   = pd.Series(tc_list, index=idx)

    return gross_eq.loc[start:], net_eq.loc[start:], tc_ser.loc[start:]


# ─────────────────────────────────────────────────────────────────────────────
# AL Stat Arb — uses existing portfolio_returns which returns (gross, net)
# ─────────────────────────────────────────────────────────────────────────────

def al_gross_net(prices, membership, pca_factors):
    """Return (gross_equity, net_equity, daily_tc_plus_borrow) series."""
    s_path = HERE / "results" / "al_statarb" / "s_scores.csv"
    s_scores = pd.read_csv(s_path, index_col=0, parse_dates=True)

    ret = prices.astype(float).ffill().pct_change()
    s_aligned = s_scores.reindex(ret.index).ffill()

    long_entry  = s_aligned < al.S_BO
    long_exit   = s_aligned > al.S_BC
    short_entry = s_aligned > al.S_SO
    short_exit  = s_aligned < al.S_SC

    common = [t for t in s_scores.columns if t in ret.columns]
    ret_c = ret[common]
    long_entry  = long_entry[common];  long_exit  = long_exit[common]
    short_entry = short_entry[common]; short_exit = short_exit[common]

    pos_long  = al.build_position_df(long_entry,  long_exit,  ret_c)
    pos_short = al.build_position_df(short_entry, short_exit, ret_c)

    long_gross,  long_net  = al.portfolio_returns(ret_c, pos_long,  "long")
    short_gross, short_net = al.portfolio_returns(ret_c, pos_short, "short")

    # Combined equity curves (long_ret + short_ret, both sign-adjusted profits)
    gross_combined = (long_gross + short_gross).fillna(0)
    net_combined   = (long_net   + short_net).fillna(0)

    gross_eq = (1 + gross_combined).cumprod() * INITIAL_VALUE
    net_eq   = (1 + net_combined).cumprod()   * INITIAL_VALUE

    # Daily cost drag = gross - net
    daily_cost = gross_combined - net_combined

    # Trim to first active day
    active = (pos_long.abs().sum(axis=1) + pos_short.abs().sum(axis=1)) > 0
    start = active.idxmax() if active.any() else gross_eq.index[0]

    return gross_eq.loc[start:], net_eq.loc[start:], daily_cost.loc[start:]


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

def plot_comparison(gross_eq, net_eq, daily_cost, strategy_label, out_path):
    """
    2-panel figure:
      Top   : gross vs net equity curve ($)
      Bottom: cumulative cost drag ($)
    """
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True,
                             gridspec_kw={"height_ratios": [3, 1.5]})

    # ── Panel 1: equity curves ──────────────────────────────────────────────
    axes[0].plot(gross_eq.index, gross_eq.values,
                 color=C_GROSS, linewidth=1.4, label="Gross (no TC)")
    axes[0].plot(net_eq.index, net_eq.values,
                 color=C_NET, linewidth=1.4, label="Net (after TC)")
    axes[0].axhline(INITIAL_VALUE, color="black", linewidth=0.4, linestyle="--")

    fmt = mticker.FuncFormatter(lambda x, _: f"${x:,.0f}")
    axes[0].yaxis.set_major_formatter(fmt)
    axes[0].set_ylabel("Portfolio Value ($)", fontsize=11)
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)

    gross_final = gross_eq.iloc[-1]
    net_final   = net_eq.iloc[-1]
    drag_total  = gross_final - net_final
    gross_ret   = gross_final / INITIAL_VALUE - 1
    net_ret     = net_final   / INITIAL_VALUE - 1
    axes[0].set_title(
        f"{strategy_label}  —  Pre vs Post Transaction Cost\n"
        f"Gross total return: {gross_ret:+.1%}   "
        f"Net total return: {net_ret:+.1%}   "
        f"Total TC drag: ${drag_total:,.0f}  ({drag_total/INITIAL_VALUE:.1%} of capital)",
        fontsize=12, fontweight="bold",
    )

    # ── Panel 2: cumulative cost drag ───────────────────────────────────────
    cum_drag = daily_cost.cumsum() * INITIAL_VALUE
    axes[1].fill_between(cum_drag.index, cum_drag.values, 0,
                         color=C_DRAG, alpha=0.35)
    axes[1].plot(cum_drag.index, cum_drag.values, color=C_DRAG, linewidth=0.9)
    axes[1].yaxis.set_major_formatter(fmt)
    axes[1].set_ylabel("Cumulative TC Drag ($)", fontsize=10)
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("Loading prices and membership...")
    prices     = dn.load_prices()
    membership = dn.load_membership(prices)

    # ── JT Momentum ──────────────────────────────────────────────────────────
    print("\n[JT Momentum] computing gross/net curves...")
    scores  = jt.compute_momentum_scores(prices, membership)
    weights = jt.build_weights(scores)
    jt_gross, jt_net, jt_tc = jt_gross_net(weights, prices)
    plot_comparison(
        jt_gross, jt_net, jt_tc,
        "ASX 200 — JT Momentum 12-1",
        OUT_DIR / "jt_pre_post_tc.png",
    )
    print(f"  Gross: ${jt_gross.iloc[-1]:,.0f}  Net: ${jt_net.iloc[-1]:,.0f}  "
          f"TC drag: ${(jt_gross.iloc[-1] - jt_net.iloc[-1]):,.0f}")

    # ── AL PCA Stat Arb ──────────────────────────────────────────────────────
    print("\n[AL PCA Stat Arb] computing gross/net curves...")
    pca_factors = dn.compute_pca_factors(prices, membership)
    al_gross, al_net, al_tc = al_gross_net(prices, membership, pca_factors)
    plot_comparison(
        al_gross, al_net, al_tc,
        "ASX 200 — AL PCA Stat Arb",
        OUT_DIR / "al_pre_post_tc.png",
    )
    print(f"  Gross: ${al_gross.iloc[-1]:,.0f}  Net: ${al_net.iloc[-1]:,.0f}  "
          f"TC drag: ${(al_gross.iloc[-1] - al_net.iloc[-1]):,.0f}")

    print(f"\nDone. Charts in: {OUT_DIR}")


if __name__ == "__main__":
    main()
