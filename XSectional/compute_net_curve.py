"""Compute net-of-cost equity curves for JT momentum from daily weights.

Reads the gross equity curve and daily weight matrix, computes one-way turnover
(rebalance turnover vs drifted prior weights), then applies era-dependent
transaction costs via the shared frictions module.

Outputs: results/equity_curve_daily_long_only_net_{low,base,high}.csv

Usage (from XSectional/):
    python compute_net_curve.py
    python compute_net_curve.py --weights results/weights_daily_long_only.csv
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add monitoring/ to path for frictions module
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "monitoring"))

from frictions import apply_costs, COST_SCHEDULE  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

RESULTS = Path(__file__).resolve().parent / "results"


def compute_turnover(w_daily: pd.DataFrame, daily_returns: pd.DataFrame) -> pd.Series:
    """Compute daily one-way turnover from applied weights and asset returns.

    Only charges turnover at actual monthly rebalance points (when the underlying
    monthly target weights change) and on day 1 (initial buy-in). Between rebalances
    positions are assumed to drift passively — the constant-weight backtest
    approximation slightly overstates returns but we do NOT charge for that
    implied daily rebalancing, as a real trader would not rebalance intra-month.

    Turnover on rebalance days = sum of |w_new_target - w_drifted_from_prior|.
    """
    w = w_daily.fillna(0.0)
    r = daily_returns.reindex(index=w.index, columns=w.columns).fillna(0.0)

    turnover = pd.Series(0.0, index=w.index)

    # Detect actual rebalance days: where the ffilled TARGET weights change.
    # w_daily is forward-filled from monthly weights then shifted 1 day, so
    # consecutive non-rebalance days have identical weight vectors.
    w_diff = w.diff(1).abs().sum(axis=1)
    is_rebalance = w_diff > 1e-8

    # Track drifted weights between rebalances
    w_held = None  # weights actually held (drift between rebalances)

    for i in range(len(w)):
        dt = w.index[i]
        w_target = w.iloc[i].values

        if i == 0:
            # Initial buy: full position
            turnover.iloc[i] = np.abs(w_target).sum()
            w_held = w_target.copy()
            continue

        # Drift held weights by today's returns
        ret_today = r.iloc[i].values
        w_drifted_raw = w_held * (1 + ret_today)
        denom = w_drifted_raw.sum()
        if abs(denom) > 1e-12:
            w_drifted = w_drifted_raw / denom
        else:
            w_drifted = w_held.copy()

        if is_rebalance.iloc[i]:
            # Monthly rebalance: charge turnover vs drifted position
            turnover.iloc[i] = np.abs(w_target - w_drifted).sum()
            w_held = w_target.copy()
        else:
            # No rebalance: just update held weights to drifted
            w_held = w_drifted

    return turnover


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weights", default=str(RESULTS / "weights_daily_long_only.csv"),
                        help="Path to daily weights CSV")
    parser.add_argument("--curve", default=str(RESULTS / "equity_curve_daily_long_only.csv"),
                        help="Path to gross equity curve CSV")
    parser.add_argument("--prices", default=None,
                        help="Path to prices CSV (for drift calc; auto-detected if omitted)")
    args = parser.parse_args()

    # Load weights
    logger.info("Loading weights: %s", args.weights)
    w_daily = pd.read_csv(args.weights, index_col="Date", parse_dates=True)
    logger.info("  %d days × %d tickers", *w_daily.shape)

    # Load gross curve
    logger.info("Loading gross curve: %s", args.curve)
    gross = pd.read_csv(args.curve, index_col="Date", parse_dates=True)
    gross_ret = gross["port_ret"]
    logger.info("  %d trading days, terminal equity %.3f", len(gross), gross["equity"].iloc[-1])

    # Load prices for drift computation
    if args.prices:
        prices_path = args.prices
    else:
        # Auto-detect: look for the PIT prices file
        pit_path = Path(__file__).resolve().parent / "data" / "sp500_prices_pit.csv"
        if pit_path.exists():
            prices_path = str(pit_path)
        else:
            prices_path = str(Path(__file__).resolve().parent / "data" / "sp500_prices.csv")

    logger.info("Loading prices: %s", prices_path)
    prices = pd.read_csv(prices_path, index_col=0, parse_dates=True)
    prices.index.name = "Date"
    daily_returns = prices.pct_change()
    logger.info("  %d days × %d tickers", *prices.shape)

    # Compute turnover
    logger.info("Computing daily one-way turnover...")
    turnover = compute_turnover(w_daily, daily_returns)

    # Align turnover to gross return index
    turnover = turnover.reindex(gross_ret.index).fillna(0.0)

    # Summary stats
    rebalance_days = (turnover > 1e-6).sum()
    mean_turnover = turnover[turnover > 1e-6].mean() if rebalance_days > 0 else 0
    annual_turnover = turnover.sum() / (len(turnover) / 252)
    logger.info("  Rebalance days: %d / %d", rebalance_days, len(turnover))
    logger.info("  Mean turnover on rebalance days: %.2f%%", mean_turnover * 100)
    logger.info("  Annualised one-way turnover: %.1f%%", annual_turnover * 100)

    # Generate net curves for each scenario
    for scenario in ["low", "base", "high"]:
        net_ret, daily_cost = apply_costs(gross_ret, turnover, scenario)
        net_equity = (1.0 + net_ret).cumprod()
        net_dd = net_equity / net_equity.cummax() - 1.0

        out_df = pd.DataFrame({
            "port_ret": net_ret,
            "equity": net_equity,
            "drawdown": net_dd,
            "turnover": turnover,
            "cost": daily_cost,
        })
        out_df.index.name = "Date"

        out_path = RESULTS / f"equity_curve_daily_long_only_net_{scenario}.csv"
        out_df.to_csv(out_path)

        annual_drag_bps = daily_cost.sum() / (len(daily_cost) / 252) * 1e4
        terminal_gross = gross["equity"].iloc[-1]
        terminal_net = net_equity.iloc[-1]
        logger.info("  %s: terminal %.3f (gross %.3f, drag %.0f bps/yr, "
                     "cost %.2f%% total)", scenario, terminal_net, terminal_gross,
                     annual_drag_bps, (terminal_gross - terminal_net) / terminal_gross * 100)

    # Verify net ≤ gross
    base_net = pd.read_csv(RESULTS / "equity_curve_daily_long_only_net_base.csv",
                           index_col="Date", parse_dates=True)
    assert (base_net["equity"] <= gross["equity"] + 1e-10).all(), \
        "VERIFICATION FAILED: net equity exceeds gross"
    logger.info("Verification passed: net ≤ gross at all dates")

    logger.info("Done — 3 net curves written to %s", RESULTS)


if __name__ == "__main__":
    main()
