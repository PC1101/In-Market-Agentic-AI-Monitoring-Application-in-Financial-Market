"""
run_jt_momentum.py — Jegadeesh-Titman cross-sectional momentum for ASX 200.

Replicates the XSectional/ JT strategy on the Australian market using Norgate's
point-in-time S&P ASX 200 C&P universe (no survivorship bias).

Strategy:
  Formation : 12-month cumulative return, skipping the most recent month (12-1).
  Portfolio : Top/bottom 20% by score → equal-weighted long/short.
  Rebalance : Month-end.
  Universe  : S&P ASX 200 Current & Past (PIT membership mask per month-end).
  Cost      : Tiered ASX brokerage — max($10, 0.11% × trade_$) per trade,
              plus ~7 bps market impact for ASX 200 liquid stocks.

NOTE: With ~24 months of total data the 13-month formation window leaves ~11
rebalancing periods. Results are directionally indicative; statistical
significance requires a longer history.

Outputs (AusEquities/results/jt_momentum/):
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
import data_norgate as dn   # noqa: E402 (after sys.path tweak)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

# ── parameters ────────────────────────────────────────────────────────────────
LOOKBACK_MONTHS  = 12     # formation window
SKIP_MONTHS      = 1      # short-term reversal skip
TOP_QUANTILE     = 0.20   # fraction to go long
BOTTOM_QUANTILE  = 0.20   # fraction to go short
MIN_STOCKS_LEG   = 5      # skip month-end if either leg has fewer stocks

# ASX tiered transaction costs (per trade, both entry and exit)
MIN_BROKERAGE  = 10.00    # $10 minimum brokerage (competitive online broker)
BROKERAGE_RATE = 0.0011   # 0.11% of trade value for trades up to $25k
MARKET_IMPACT  = 0.0007   # ~7 bps market impact for ASX 200 liquid stocks
INITIAL_VALUE  = 1_000_000  # $1M starting capital

OUT_DIR = os.path.join(HERE, "results", "jt_momentum")


# ─────────────────────────────────────────────────────────────────────────────
# Signal
# ─────────────────────────────────────────────────────────────────────────────

def compute_momentum_scores(
    prices: pd.DataFrame,
    membership: pd.DataFrame,
) -> pd.DataFrame:
    """
    12-1 cross-sectional momentum, masked by PIT ASX 200 membership.

    At each month-end t the score for ticker i is the log-cumulative return
    over [t-13, t-2] (12 months, skipping the most recent month).
    Tickers not in the ASX 200 on that month-end receive NaN.
    """
    monthly = prices.resample("ME").last()
    log_ret = np.log(1 + monthly.pct_change().clip(lower=-0.9999))
    mom_log = log_ret.shift(SKIP_MONTHS).rolling(LOOKBACK_MONTHS).sum()
    scores = np.exp(mom_log) - 1

    # apply PIT membership at month-end (use last available constituent flag)
    monthly_mask = membership.resample("ME").last().reindex(scores.index)
    monthly_mask = monthly_mask.reindex(columns=scores.columns).fillna(False)
    scores[~monthly_mask] = np.nan
    return scores


# ─────────────────────────────────────────────────────────────────────────────
# Portfolio construction
# ─────────────────────────────────────────────────────────────────────────────

def build_weights(scores: pd.DataFrame) -> pd.DataFrame:
    """Equal-weighted long-short portfolio from cross-sectional momentum scores."""
    weights = pd.DataFrame(np.nan, index=scores.index, columns=scores.columns,
                           dtype="float64")
    traded = 0
    for dt, row in scores.iterrows():
        valid = row.dropna()
        if len(valid) == 0:
            continue
        n_long  = max(1, int(len(valid) * TOP_QUANTILE))
        n_short = max(1, int(len(valid) * BOTTOM_QUANTILE))
        if n_long < MIN_STOCKS_LEG or n_short < MIN_STOCKS_LEG:
            log.warning("Skip %s — only %d valid stocks", dt.date(), len(valid))
            continue
        longs  = valid.nlargest(n_long).index
        shorts = valid.nsmallest(n_short).index
        weights.loc[dt, longs]  =  1.0 / n_long
        weights.loc[dt, shorts] = -1.0 / n_short
        traded += 1
    log.info("Portfolio: %d rebalancing periods with active positions", traded)
    return weights


# ─────────────────────────────────────────────────────────────────────────────
# Backtest
# ─────────────────────────────────────────────────────────────────────────────

def _tiered_tc_frac(delta_w: pd.Series, portfolio_value: float) -> float:
    """
    Total transaction cost for one day as a fraction of portfolio value.

    For each active trade i with |weight change| = delta_w_i:
      dollar_trade = delta_w_i × portfolio_value
      brokerage    = max(MIN_BROKERAGE, BROKERAGE_RATE × dollar_trade)
      impact       = MARKET_IMPACT × dollar_trade
      cost_frac_i  = (brokerage + impact) / portfolio_value

    Equivalently:
      cost_frac_i = max(MIN_BROKERAGE / portfolio_value, BROKERAGE_RATE × delta_w_i)
                    + MARKET_IMPACT × delta_w_i
    """
    active = delta_w[delta_w > 0]
    if active.empty:
        return 0.0
    min_fee_frac = MIN_BROKERAGE / portfolio_value
    # per-trade cost as portfolio fraction: max of min-fee-frac vs rate×Δw, plus impact×Δw
    cost = np.maximum(min_fee_frac, BROKERAGE_RATE * active.values) \
           + MARKET_IMPACT * active.values
    return float(cost.sum())


def run_backtest(weights: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    """
    Daily equity curve from monthly portfolio weights with tiered ASX costs.

    Month-end weights are forward-filled and shifted one trading day (no lookahead).
    Portfolio value is tracked day-by-day so the $10 minimum brokerage per trade
    correctly scales with the evolving portfolio size.
    """
    daily_ret = prices.pct_change(fill_method=None)
    w = weights.astype(float).reindex(columns=daily_ret.columns)
    w_daily = w.reindex(daily_ret.index, method="ffill").shift(1)
    w0 = w_daily.fillna(0)

    delta_w  = w0.diff().abs()
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

    out = pd.DataFrame(
        {"port_ret": port_ret, "equity": equity, "drawdown": drawdown}
    )
    out.index.name = "Date"
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Metrics & plot
# ─────────────────────────────────────────────────────────────────────────────

def summarise(curve: pd.DataFrame) -> dict:
    r = curve["port_ret"]
    sharpe    = np.sqrt(252) * r.mean() / r.std() if r.std() > 0 else np.nan
    total_ret = curve["equity"].iloc[-1] - 1
    max_dd    = curve["drawdown"].min()
    ann_ret   = (curve["equity"].iloc[-1] ** (252 / len(r))) - 1
    return {
        "sharpe":       round(float(sharpe), 3),
        "ann_return":   round(float(ann_ret), 4),
        "total_return": round(float(total_ret), 4),
        "max_drawdown": round(float(max_dd), 4),
        "n_days":       int(len(r)),
    }


def plot_performance(curve: pd.DataFrame, m: dict, out_dir: str) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)

    curve["equity"].plot(ax=axes[0], color="steelblue", lw=1.5)
    axes[0].axhline(1.0, lw=0.6, ls="--", color="k", alpha=0.6)
    axes[0].set_title(
        f"ASX 200  |  JT Momentum 12-1  |  "
        f"Sharpe {m['sharpe']:.2f}  "
        f"Ann.Ret {m['ann_return']:.1%}  "
        f"MaxDD {m['max_drawdown']:.1%}  "
        f"(cost max($10, 0.11%) + 7bps impact)"
    )
    axes[0].set_ylabel("Cumulative PnL")

    curve["drawdown"].plot(ax=axes[1], color="firebrick", lw=1.0)
    axes[1].axhline(0, lw=0.6, ls="--", color="k", alpha=0.6)
    axes[1].set_ylabel("Drawdown")
    axes[1].set_xlabel("")

    fig.tight_layout()
    path = os.path.join(out_dir, "performance.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    log.info("Chart saved: %s", path)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)

    log.info("=== JT Momentum — S&P ASX 200 C&P ===")

    log.info("Step 1/4  Loading prices...")
    prices = dn.load_prices()
    log.info("  %d trading days × %d tickers", *prices.shape)

    log.info("Step 2/4  Loading PIT membership mask...")
    membership = dn.load_membership(prices)

    log.info("Step 3/4  Computing 12-1 momentum scores...")
    scores = compute_momentum_scores(prices, membership)
    n_active = int(scores.notna().any(axis=1).sum())
    log.info("  Months with at least one valid score: %d", n_active)

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
