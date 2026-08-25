"""
run_jt_momentum.py — Jegadeesh-Titman cross-sectional momentum for S&P 500 C&P.

Strategy:
  Formation : 12-month cumulative return, skipping the most recent month (12-1).
  Portfolio : Top/bottom 20% by score → equal-weighted long/short.
  Rebalance : Month-end.
  Universe  : S&P 500 Current & Past (PIT membership mask per month-end).
  Cost      : max($1, 0.05%) brokerage + ~5 bps market impact per trade.

Outputs (USEquities/results/jt_momentum/):
  equity_curve.csv   — daily [port_ret, equity, drawdown]
  performance.png    — equity curve + drawdown panel
"""

import os
import sys
import logging

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import data_norgate as dn

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

# ── parameters ────────────────────────────────────────────────────────────────
LOOKBACK_MONTHS  = 12
SKIP_MONTHS      = 1
TOP_QUANTILE     = 0.20
BOTTOM_QUANTILE  = 0.20
MIN_STOCKS_LEG   = 5

# US transaction costs
MIN_BROKERAGE  = 1.00        # $1 minimum (vs $10 ASX)
BROKERAGE_RATE = 0.0005      # 0.05% of trade value (vs 0.11% ASX)
MARKET_IMPACT  = 0.0005      # ~5 bps market impact (vs 7 bps ASX)
INITIAL_VALUE  = 1_000_000

OUT_DIR = os.path.join(HERE, "results", "jt_momentum")


# ─────────────────────────────────────────────────────────────────────────────
# Signal
# ─────────────────────────────────────────────────────────────────────────────

def compute_momentum_scores(prices, membership):
    """12-1 cross-sectional momentum, masked by PIT S&P 500 membership."""
    monthly = prices.resample("ME").last()
    log_ret = np.log(1 + monthly.pct_change().clip(lower=-0.9999))
    mom_log = log_ret.shift(SKIP_MONTHS).rolling(LOOKBACK_MONTHS).sum()
    scores  = np.exp(mom_log) - 1

    monthly_mask = membership.resample("ME").last().reindex(scores.index)
    monthly_mask = monthly_mask.reindex(columns=scores.columns).fillna(False)
    scores[~monthly_mask] = np.nan
    return scores


# ─────────────────────────────────────────────────────────────────────────────
# Portfolio construction
# ─────────────────────────────────────────────────────────────────────────────

def build_weights(scores):
    """Equal-weighted long-short portfolio from cross-sectional momentum scores."""
    weights = pd.DataFrame(np.nan, index=scores.index, columns=scores.columns,
                           dtype=float)
    n_skip = 0
    for dt in scores.index:
        valid = scores.loc[dt].dropna()
        if valid.empty:
            continue
        n_long  = max(1, int(len(valid) * TOP_QUANTILE))
        n_short = max(1, int(len(valid) * BOTTOM_QUANTILE))
        if n_long < MIN_STOCKS_LEG or n_short < MIN_STOCKS_LEG:
            n_skip += 1
            log.debug("Skip %s — only %d valid stocks", dt.date(), len(valid))
            continue
        longs  = valid.nlargest(n_long).index
        shorts = valid.nsmallest(n_short).index
        weights.loc[dt, longs]  =  1.0 / n_long
        weights.loc[dt, shorts] = -1.0 / n_short

    n_active = int(weights.notna().any(axis=1).sum())
    log.info("Portfolio: %d rebalancing periods with active positions", n_active)
    if n_skip:
        log.info("  (skipped %d months with fewer than %d stocks per leg)",
                 n_skip, MIN_STOCKS_LEG)
    return weights


# ─────────────────────────────────────────────────────────────────────────────
# Backtest
# ─────────────────────────────────────────────────────────────────────────────

def _tiered_tc_frac(delta_w, portfolio_value):
    """Total TC for one day as fraction of portfolio value."""
    active = delta_w[delta_w > 0]
    if active.empty:
        return 0.0
    min_fee_frac = MIN_BROKERAGE / portfolio_value
    cost = np.maximum(min_fee_frac, BROKERAGE_RATE * active.values) \
           + MARKET_IMPACT * active.values
    return float(cost.sum())


def run_backtest(weights, prices):
    """Daily equity curve from monthly portfolio weights with tiered US costs."""
    daily_ret = prices.pct_change(fill_method=None)
    w = weights.astype(float).reindex(columns=daily_ret.columns)
    w_daily = w.reindex(daily_ret.index, method="ffill").shift(1)
    w0 = w_daily.fillna(0)

    delta_w   = w0.diff().abs()
    gross_ret = (w0 * daily_ret.fillna(0)).sum(axis=1)

    V = INITIAL_VALUE
    port_rets = []
    for i in range(len(daily_ret)):
        tc = _tiered_tc_frac(delta_w.iloc[i], V)
        r  = gross_ret.iloc[i] - tc
        port_rets.append(r)
        V = max(V * (1 + r), 1.0)

    port_ret = pd.Series(port_rets, index=daily_ret.index)
    active = w0.abs().sum(axis=1) > 0
    if active.any():
        port_ret = port_ret.loc[active.idxmax():]

    equity   = (1 + port_ret).cumprod()
    drawdown = equity / equity.cummax() - 1
    out = pd.DataFrame({"port_ret": port_ret, "equity": equity, "drawdown": drawdown})
    out.index.name = "Date"
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Metrics & plot
# ─────────────────────────────────────────────────────────────────────────────

def summarise(curve):
    r = curve["port_ret"]
    sharpe  = np.sqrt(252) * r.mean() / r.std() if r.std() > 0 else np.nan
    ann_ret = (curve["equity"].iloc[-1] ** (252 / len(r))) - 1
    return {
        "sharpe":       round(float(sharpe), 3),
        "ann_return":   round(float(ann_ret), 4),
        "total_return": round(float(curve["equity"].iloc[-1] - 1), 4),
        "max_drawdown": round(float(curve["drawdown"].min()), 4),
        "n_days":       int(len(r)),
    }


def plot_performance(curve, m, out_dir):
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    curve["equity"].plot(ax=axes[0], color="steelblue", lw=1.5)
    axes[0].axhline(1.0, lw=0.6, ls="--", color="k", alpha=0.6)
    axes[0].set_title(
        f"S&P 500  |  JT Momentum 12-1  |  "
        f"Sharpe {m['sharpe']:.2f}  "
        f"Ann.Ret {m['ann_return']:.1%}  "
        f"MaxDD {m['max_drawdown']:.1%}  "
        f"(cost max($1, 0.05%) + 5bps impact)"
    )
    axes[0].set_ylabel("Cumulative PnL")
    curve["drawdown"].plot(ax=axes[1], color="firebrick", lw=1.0)
    axes[1].axhline(0, lw=0.6, ls="--", color="k", alpha=0.6)
    axes[1].set_ylabel("Drawdown")
    fig.tight_layout()
    path = os.path.join(out_dir, "performance.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    log.info("Chart saved: %s", path)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    log.info("=== JT Momentum — S&P 500 C&P ===")

    log.info("Step 1/4  Loading prices...")
    prices = dn.load_prices()
    log.info("  %d trading days × %d tickers", *prices.shape)

    log.info("Step 2/4  Loading PIT membership mask...")
    membership = dn.load_membership(prices)

    log.info("Step 3/4  Computing 12-1 momentum scores...")
    scores = compute_momentum_scores(prices, membership)
    log.info("  Months with at least one valid score: %d",
             int(scores.notna().any(axis=1).sum()))

    log.info("Step 4/4  Building weights and running backtest...")
    weights = build_weights(scores)
    curve   = run_backtest(weights, prices)
    curve.to_csv(os.path.join(OUT_DIR, "equity_curve.csv"))

    m = summarise(curve)
    print("\n=== Results ===")
    print(f"  Sharpe ratio   : {m['sharpe']:.3f}")
    print(f"  Ann. return    : {m['ann_return']:.2%}")
    print(f"  Total return   : {m['total_return']:.2%}")
    print(f"  Max drawdown   : {m['max_drawdown']:.2%}")
    print(f"  Trading days   : {m['n_days']}")

    plot_performance(curve, m, OUT_DIR)
    print(f"\nOutputs in: {OUT_DIR}")


if __name__ == "__main__":
    main()
