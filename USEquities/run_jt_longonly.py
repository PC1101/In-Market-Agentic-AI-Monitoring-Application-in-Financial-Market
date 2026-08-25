"""
run_jt_longonly.py — JT 12-1 Momentum LONG-ONLY for S&P 500 C&P (Norgate).

Same signal as run_jt_momentum.py but only takes the top-quintile long leg.
Portfolio is fully invested (weights sum to 1.0).

Outputs: USEquities/results/jt_longonly/
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
import run_jt_momentum as jt

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

OUT_DIR = os.path.join(HERE, "results", "jt_longonly")


def build_long_weights(scores: pd.DataFrame) -> pd.DataFrame:
    """Equal-weighted LONG-ONLY portfolio — top quantile only."""
    weights = pd.DataFrame(np.nan, index=scores.index, columns=scores.columns,
                           dtype="float64")
    traded = 0
    for dt, row in scores.iterrows():
        valid = row.dropna()
        if len(valid) == 0:
            continue
        n_long = max(1, int(len(valid) * jt.TOP_QUANTILE))
        if n_long < jt.MIN_STOCKS_LEG:
            log.warning("Skip %s — only %d valid stocks", dt.date(), len(valid))
            continue
        longs = valid.nlargest(n_long).index
        weights.loc[dt, longs] = 1.0 / n_long
        traded += 1
    log.info("Portfolio: %d rebalancing periods with active positions", traded)
    return weights


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    log.info("=== JT Momentum LONG-ONLY — S&P 500 C&P ===")

    prices = dn.load_prices()
    log.info("  %d trading days × %d tickers", *prices.shape)
    membership = dn.load_membership(prices)

    scores  = jt.compute_momentum_scores(prices, membership)
    weights = build_long_weights(scores)
    curve   = jt.run_backtest(weights, prices)
    curve.to_csv(os.path.join(OUT_DIR, "equity_curve.csv"))

    m = jt.summarise(curve)
    print("\n=== Results (Long-Only) ===")
    print(f"  Sharpe ratio   : {m['sharpe']:.3f}")
    print(f"  Ann. return    : {m['ann_return']:.2%}")
    print(f"  Total return   : {m['total_return']:.2%}")
    print(f"  Max drawdown   : {m['max_drawdown']:.2%}")
    print(f"  Trading days   : {m['n_days']}")

    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    curve["equity"].plot(ax=axes[0], color="steelblue", lw=1.5)
    axes[0].axhline(1.0, lw=0.6, ls="--", color="k", alpha=0.6)
    axes[0].set_title(
        f"S&P 500  |  JT Momentum 12-1  LONG-ONLY  |  "
        f"Sharpe {m['sharpe']:.2f}  Ann.Ret {m['ann_return']:.1%}  MaxDD {m['max_drawdown']:.1%}"
    )
    axes[0].set_ylabel("Cumulative PnL")
    curve["drawdown"].plot(ax=axes[1], color="firebrick", lw=1.0)
    axes[1].axhline(0, lw=0.6, ls="--", color="k", alpha=0.6)
    axes[1].set_ylabel("Drawdown")
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "performance.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    log.info("Chart saved: %s", path)
    print(f"\nOutputs in: {OUT_DIR}")


if __name__ == "__main__":
    main()
