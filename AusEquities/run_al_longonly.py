"""
run_al_longonly.py — AL PCA Stat Arb LONG-ONLY for ASX 200 C&P (Norgate).

Same OU signal as run_al_statarb.py but only takes long positions
(s-score < -1.25, close when s-score > -0.50). No short leg, no borrow cost.

Outputs: AusEquities/results/al_longonly/
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
import run_al_statarb as al

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

OUT_DIR = os.path.join(HERE, "results", "al_longonly")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    log.info("=== AL PCA Stat Arb LONG-ONLY — S&P ASX 200 C&P ===")

    prices = dn.load_prices()
    log.info("  %d trading days × %d tickers", *prices.shape)
    membership = dn.load_membership(prices)

    pca_factors = dn.compute_pca_factors(prices, membership)
    log.info("  PCA factors: %d rows × %d components", *pca_factors.shape)

    # Use cached s-scores if available (shared with long-short run)
    s_path_cache = os.path.join(HERE, "results", "al_statarb", "s_scores.csv")
    s_path_own   = os.path.join(OUT_DIR, "s_scores.csv")
    if os.path.exists(s_path_cache):
        log.info("Loading s-scores from cache: %s", s_path_cache)
        s_scores = pd.read_csv(s_path_cache, index_col=0, parse_dates=True)
    elif os.path.exists(s_path_own):
        log.info("Loading s-scores from: %s", s_path_own)
        s_scores = pd.read_csv(s_path_own, index_col=0, parse_dates=True)
    else:
        log.info("Computing s-scores (no cache found)...")
        s_scores = al.compute_s_scores(prices, membership, pca_factors)
        s_scores.to_csv(s_path_own)
        log.info("  s-scores saved: %s", s_path_own)

    ret = prices.astype(float).ffill().pct_change()
    s_aligned = s_scores.reindex(ret.index).ffill()
    common = [t for t in s_scores.columns if t in ret.columns]
    ret_c = ret[common]
    s_c   = s_aligned[common]

    long_entry = s_c < al.S_BO
    long_exit  = s_c > al.S_BC

    log.info("Building long positions...")
    pos_long = al.build_position_df(long_entry, long_exit, ret_c)

    gross_ret, net_ret = al.portfolio_returns(ret_c, pos_long, direction="long")

    active = pos_long.abs().sum(axis=1) > 0
    if not active.any():
        log.error("No active positions found.")
        return
    start = active.idxmax()
    net_ret = net_ret.loc[start:]

    equity   = (1 + net_ret).cumprod()
    drawdown = equity / equity.cummax() - 1
    curve = pd.DataFrame({"net_ret": net_ret, "equity": equity, "drawdown": drawdown})
    curve.to_csv(os.path.join(OUT_DIR, "equity_curve.csv"))

    trades = al.extract_trades(pos_long, ret_c, "long")
    trades.to_csv(os.path.join(OUT_DIR, "trades.csv"), index=False)

    r = net_ret
    sharpe  = float(np.sqrt(252) * r.mean() / r.std()) if r.std() > 0 else np.nan
    ann_ret = float(equity.iloc[-1] ** (252 / len(r)) - 1)
    max_dd  = float(drawdown.min())
    win_rate = float((trades["trade_ret"] > 0).mean()) if not trades.empty else np.nan

    print("\n=== Results (Long-Only) ===")
    print(f"  Sharpe ratio   : {sharpe:.3f}")
    print(f"  Ann. return    : {ann_ret:.2%}")
    print(f"  End PnL (×)    : {equity.iloc[-1]:.3f}")
    print(f"  Max drawdown   : {max_dd:.2%}")
    print(f"  Total trades   : {len(trades)}")
    if not trades.empty:
        print(f"  Win rate       : {win_rate:.1%}")
        print(f"  Avg trade ret  : {trades['trade_ret'].mean():+.2%}")
        print(f"  Avg hold (days): {trades['days'].mean():.1f}")

    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    equity.plot(ax=axes[0], color="steelblue", lw=1.5)
    axes[0].axhline(1.0, lw=0.6, ls="--", color="k", alpha=0.6)
    axes[0].set_title(
        f"ASX 200  |  AL PCA Stat Arb  LONG-ONLY  |  "
        f"Sharpe {sharpe:.2f}  Ann {ann_ret:.1%}  MaxDD {max_dd:.1%}  "
        f"({len(trades)} trades)"
    )
    axes[0].set_ylabel("Cumulative PnL")
    drawdown.plot(ax=axes[1], color="firebrick", lw=1.0)
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
