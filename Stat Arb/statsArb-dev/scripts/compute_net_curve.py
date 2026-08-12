"""Compute net-of-cost equity curves for AL_PCA stat-arb from the trades blotter.

Reconstructs daily positions and weights from trades.csv, computes turnover,
and applies era-dependent transaction costs via the shared frictions module.

The AL portfolio is an equal-weight average of 9 sector sleeves, each running
an OU-process stat-arb on sector constituents.  Within each sleeve, active
positions are equal-weighted (1/N').

Verification gate: reconstructed gross return must match equity_curve.csv
(correlation > 0.999, terminal equity within 1%).

Usage (from Stat Arb/statsArb-dev/):
    python scripts/compute_net_curve.py
    python scripts/compute_net_curve.py --tag pit_pca_long_only
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "monitoring"))

from frictions import apply_costs, COST_SCHEDULE  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

STAT_ARB = Path(__file__).resolve().parents[1]


def reconstruct_positions(trades: pd.DataFrame, calendar: pd.DatetimeIndex,
                          sector: str) -> pd.DataFrame:
    """Build a binary position matrix (1=holding, 0=not) for one sector.

    Args:
        trades: trades blotter filtered to one sector.
        calendar: full trading-day index.
        sector: sector name for logging.

    Returns:
        DataFrame (calendar × tickers) with 1 while holding, 0 otherwise.
    """
    sector_trades = trades[trades["sector"] == sector].copy()
    tickers = sorted(sector_trades["ticker"].unique())
    pos = pd.DataFrame(0, index=calendar, columns=tickers)

    for ticker in tickers:
        tk_trades = sector_trades[sector_trades["ticker"] == ticker]
        for _, row in tk_trades.iterrows():
            entry = pd.Timestamp(row["entry"])
            exit_dt = pd.Timestamp(row["exit"])
            # Position held from entry to exit (inclusive)
            mask = (calendar >= entry) & (calendar <= exit_dt)
            pos.loc[mask, ticker] = 1

    return pos


def compute_sector_turnover(pos: pd.DataFrame) -> pd.Series:
    """Compute daily one-way turnover for one sector.

    Turnover = sum of absolute weight changes. When a stock enters/exits,
    its weight changes by ~1/N'. When N' changes, all existing weights shift.
    """
    n_active = pos.sum(axis=1).replace(0, np.nan)
    w = pos.div(n_active, axis=0).fillna(0.0)
    w_change = w.diff(1).abs()
    turnover = w_change.sum(axis=1)
    # First day: full buy-in
    turnover.iloc[0] = w.iloc[0].abs().sum()
    return turnover


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tag", default="pit_pca_long_only",
                        help="Result sub-folder tag")
    args = parser.parse_args()

    result_dir = STAT_ARB / "results" / "full_universe" / args.tag
    trades_path = result_dir / "trades.csv"
    curve_path = result_dir / "equity_curve.csv"
    sector_ret_path = result_dir / "sector_returns.csv"

    # Load data
    logger.info("Loading trades: %s", trades_path)
    trades = pd.read_csv(trades_path)
    logger.info("  %d trades across %d sectors", len(trades), trades["sector"].nunique())

    logger.info("Loading sector returns: %s", sector_ret_path)
    sector_rets = pd.read_csv(sector_ret_path, index_col="Date", parse_dates=True)
    sectors = list(sector_rets.columns)
    calendar = sector_rets.index
    logger.info("  %d trading days, %d sectors: %s", len(calendar), len(sectors), sectors)

    logger.info("Loading gross equity curve: %s", curve_path)
    gross = pd.read_csv(curve_path, index_col="Date", parse_dates=True)
    gross_ret = gross["port_ret"]

    # Reconstruct positions and compute per-sector turnover
    logger.info("Reconstructing positions from trades blotter...")
    sector_turnovers = {}
    for s in sectors:
        s_trades = trades[trades["sector"] == s]
        if s_trades.empty:
            logger.info("  %s: no trades, skipping", s)
            continue
        pos = reconstruct_positions(trades, calendar, s)
        turnover = compute_sector_turnover(pos)
        sector_turnovers[s] = turnover
        n_trades = len(s_trades)
        mean_to = turnover[turnover > 1e-8].mean()
        logger.info("  %s: %d trades, mean turnover on active days %.2f%%",
                     s, n_trades, mean_to * 100)

    # Portfolio-level turnover = equal-weight mean of sector turnovers
    turnover_panel = pd.DataFrame(sector_turnovers)
    port_turnover = turnover_panel.mean(axis=1).fillna(0.0)

    annual_turnover = port_turnover.sum() / (len(port_turnover) / 252)
    rebalance_days = (port_turnover > 1e-8).sum()
    logger.info("Portfolio: %d active turnover days / %d total, annualised %.1f%%",
                rebalance_days, len(port_turnover), annual_turnover * 100)

    # Verification gate: check that mean(sector_rets) matches equity_curve
    reconstructed_ret = sector_rets.mean(axis=1)
    corr = reconstructed_ret.corr(gross_ret)
    terminal_recon = (1 + reconstructed_ret).cumprod().iloc[-1]
    terminal_gross = gross["equity"].iloc[-1]
    terminal_pct_diff = abs(terminal_recon - terminal_gross) / terminal_gross * 100
    logger.info("Verification: corr=%.6f, terminal recon=%.3f vs gross=%.3f (%.2f%% diff)",
                corr, terminal_recon, terminal_gross, terminal_pct_diff)
    if corr < 0.999:
        logger.error("VERIFICATION FAILED: correlation %.6f < 0.999", corr)
        sys.exit(1)
    if terminal_pct_diff > 1.0:
        logger.error("VERIFICATION FAILED: terminal diff %.2f%% > 1%%", terminal_pct_diff)
        sys.exit(1)
    logger.info("Verification PASSED")

    # Generate net curves for each scenario
    for scenario in ["low", "base", "high"]:
        net_ret, daily_cost = apply_costs(gross_ret, port_turnover, scenario)
        net_equity = (1.0 + net_ret).cumprod()
        net_dd = net_equity / net_equity.cummax() - 1.0

        out_df = pd.DataFrame({
            "port_ret": net_ret,
            "equity": net_equity,
            "drawdown": net_dd,
            "turnover": port_turnover,
            "cost": daily_cost,
        })
        out_df.index.name = "Date"

        out_path = result_dir / f"equity_curve_net_{scenario}.csv"
        out_df.to_csv(out_path)

        annual_drag_bps = daily_cost.sum() / (len(daily_cost) / 252) * 1e4
        logger.info("  %s: terminal %.3f (gross %.3f, drag %.0f bps/yr, "
                     "cost %.2f%% total)", scenario, net_equity.iloc[-1],
                     terminal_gross, annual_drag_bps,
                     (terminal_gross - net_equity.iloc[-1]) / terminal_gross * 100)

    # Verify net ≤ gross
    base_net = pd.read_csv(result_dir / "equity_curve_net_base.csv",
                           index_col="Date", parse_dates=True)
    assert (base_net["equity"] <= gross["equity"] + 1e-10).all(), \
        "VERIFICATION FAILED: net equity exceeds gross"
    logger.info("Verification passed: net ≤ gross at all dates")

    logger.info("Done — 3 net curves written to %s", result_dir)


if __name__ == "__main__":
    main()
