"""Live trading backtest: strategy + agentic risk-overlay simulation.

Simulates a portfolio starting at $CAPITAL invested in a strategy (JT momentum
or AL stat arb, long-only), with the agentic monitoring layer dynamically
scaling exposure based on supervisor decisions.  Compares against an unmanaged
(buy-and-hold strategy) curve and an S&P 500 (SPY) benchmark.

Usage (from monitoring/):
    python run_live_backtest.py \\
        --curve ../XSectional/results/equity_curve_daily_long_only.csv \\
        --strategy JT_MOM \\
        --logs-glob "results/agentic_*_JT_MOM.jsonl" \\
        --capital 1000000 --tcost-bps 10 --benchmark SPY

Outputs (to results/live_backtest/<STRATEGY>/):
    fig_a_portfolio_value.png       — 3 equity curves ($)
    fig_b_pnl_drawdown_exposure.png — PnL, drawdown, exposure panels
    metrics.json / metrics_table.png — performance summary
    portfolio_daily.csv             — full daily audit trail
    exposure.csv                    — daily exposure series

Limitations (disclose in any reporting):
    1. Hybrid coverage: agentic decisions exist only in 12 evaluation windows;
       100% exposure assumed elsewhere (favorable selection bias).
    2. Curve mismatch: logs were generated on long-short curve; applied to
       long-only (regime breaks are market-wide, but it's an approximation).
    3. When using net curves: costs are applied post-hoc to gross curves;
       agent decisions were generated on gross curves (justified: cost drag
       ~1-3 bps/day barely shifts drawdown/vol features used by the agent).
"""
from __future__ import annotations

import argparse
import glob
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from pnl_loader import load_pnl, returns_series
from windows import ALL_WINDOWS_FULL
from live_backtest.policy import PolicyConfig, load_decisions, build_exposure
from live_backtest.engine import EngineConfig, simulate
from live_backtest.benchmark import fetch_benchmark, buy_and_hold
from live_backtest.report import (
    compute_metrics, write_metrics_table,
    fig_portfolio_value, fig_pnl_drawdown,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Live trading backtest with agentic risk overlay.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--curve", required=True,
                        help="Path to strategy equity_curve CSV (Date,port_ret,equity,drawdown)")
    parser.add_argument("--strategy", required=True,
                        help="Strategy label (e.g. JT_MOM, AL_PCA)")
    parser.add_argument("--logs-glob", required=True,
                        help="Glob pattern for agentic JSONL logs (e.g. 'results/agentic_*_JT_MOM.jsonl')")
    parser.add_argument("--capital", type=float, default=1_000_000.0,
                        help="Starting capital (default: 1000000)")
    parser.add_argument("--tcost-bps", type=float, default=10.0,
                        help="Transaction cost in bps on exposure changes (default: 10)")
    parser.add_argument("--strat-tcost-bps", type=float, default=0.0,
                        help="Annualised strategy friction in bps (default: 0; use net curves instead)")
    parser.add_argument("--cash-rate", type=float, default=0.0,
                        help="Annual rate earned on uninvested cash (default: 0; e.g. 0.014 for ~1.4%%)")
    parser.add_argument("--benchmark", default="SPY",
                        help="Benchmark ticker (default: SPY)")
    parser.add_argument("--cooldown", type=int, default=5,
                        help="Cooldown days after de-risk trigger (default: 5)")
    parser.add_argument("--lag", type=int, default=1,
                        help="Lag days for exposure application (default: 1)")
    parser.add_argument("--start", default=None, help="Override start date")
    parser.add_argument("--end", default=None, help="Override end date")
    parser.add_argument("--outdir", default=None,
                        help="Output directory (default: results/live_backtest/<STRATEGY>)")
    parser.add_argument("--no-fetch", action="store_true",
                        help="Do not fetch benchmark data (use cache only)")
    args = parser.parse_args()

    # --- Load strategy curve ---
    logger.info("Loading strategy curve: %s", args.curve)
    pnl_df = load_pnl(args.curve)
    strat_ret = returns_series(pnl_df)

    if args.start:
        strat_ret = strat_ret.loc[args.start:]
    if args.end:
        strat_ret = strat_ret.loc[:args.end]

    logger.info("  %d trading days: %s to %s",
                len(strat_ret), strat_ret.index[0].date(), strat_ret.index[-1].date())

    # --- Load agentic decisions ---
    log_paths = sorted(Path(p) for p in glob.glob(args.logs_glob))
    if not log_paths:
        # Try relative to script directory
        script_dir = Path(__file__).resolve().parent
        log_paths = sorted(script_dir.glob(args.logs_glob))
    logger.info("Found %d agentic log files", len(log_paths))
    decisions = load_decisions(log_paths)
    logger.info("  %d supervisor decisions loaded (%d unique dates)",
                len(decisions), decisions.index.nunique())

    # Count by state
    if not decisions.empty:
        state_counts = decisions["state"].value_counts()
        logger.info("  State distribution: %s",
                    ", ".join(f"{k}={v}" for k, v in state_counts.items()))

    # --- Build exposure ---
    policy_cfg = PolicyConfig(cooldown_days=args.cooldown, lag_days=args.lag)
    exposure = build_exposure(strat_ret.index, decisions, policy_cfg)
    n_reduced = (exposure < 1.0).sum()
    logger.info("  Exposure < 100%% on %d / %d days (%.1f%%)",
                n_reduced, len(exposure), 100 * n_reduced / len(exposure))

    # --- Simulate managed and unmanaged ---
    strat_annual = args.strat_tcost_bps / 1e4  # convert bps to decimal
    engine_cfg = EngineConfig(capital=args.capital, tcost_bps=args.tcost_bps,
                              strat_tcost_annual=strat_annual,
                              cash_rate_annual=args.cash_rate)

    logger.info("Simulating managed portfolio (strategy + agent)...")
    managed = simulate(strat_ret, exposure, engine_cfg)

    logger.info("Simulating unmanaged portfolio (buy-and-hold strategy)...")
    full_exposure = pd.Series(1.0, index=strat_ret.index)
    unmanaged = simulate(strat_ret, full_exposure,
                         EngineConfig(capital=args.capital, tcost_bps=0.0,
                                     strat_tcost_annual=strat_annual))

    # --- Benchmark ---
    benchmark_value = None
    try:
        logger.info("Loading benchmark: %s", args.benchmark)
        bench_ret = fetch_benchmark(
            args.benchmark,
            start=str(strat_ret.index[0].date()),
            end=str(strat_ret.index[-1].date()),
            no_fetch=args.no_fetch,
        )
        benchmark_value = buy_and_hold(bench_ret, args.capital)
        # Align to strategy calendar
        benchmark_value = benchmark_value.reindex(
            strat_ret.index, method="ffill"
        ).dropna()
        logger.info("  Benchmark: %d days loaded", len(benchmark_value))
    except Exception as e:
        logger.warning("Could not load benchmark: %s", e)

    # --- Output ---
    outdir = Path(args.outdir) if args.outdir else (
        Path(__file__).resolve().parent / "results" / "live_backtest" / args.strategy
    )
    outdir.mkdir(parents=True, exist_ok=True)
    logger.info("Output directory: %s", outdir)

    # Metrics
    metrics_results = {
        f"{args.strategy} + Agent": compute_metrics(managed["value"]),
        f"{args.strategy} (unmanaged)": compute_metrics(unmanaged["value"]),
    }
    if benchmark_value is not None and len(benchmark_value) > 1:
        metrics_results[f"{args.benchmark} B&H"] = compute_metrics(benchmark_value)

    # Add alpha row
    if benchmark_value is not None:
        bench_m = metrics_results.get(f"{args.benchmark} B&H", {})
        for label in [f"{args.strategy} + Agent", f"{args.strategy} (unmanaged)"]:
            m = metrics_results[label]
            m["alpha_vs_benchmark"] = round(
                m.get("cagr", 0) - bench_m.get("cagr", 0), 6
            )

    json_path = write_metrics_table(metrics_results, outdir)
    logger.info("Wrote %s", json_path)

    # Print metrics
    print("\n" + "=" * 72)
    print("LIVE TRADING BACKTEST RESULTS")
    print("=" * 72)
    for label, m in metrics_results.items():
        print(f"\n  {label}:")
        print(f"    CAGR:            {m['cagr']:.2%}")
        print(f"    Ann. Volatility: {m['ann_vol']:.2%}")
        print(f"    Sharpe:          {m['sharpe']:.2f}")
        print(f"    Max Drawdown:    {m['max_dd']:.2%}")
        print(f"    Calmar:          {m['calmar']:.2f}")
        print(f"    Terminal Value:  ${m['terminal_value']:,.0f}")
        print(f"    Total Return:    {m['total_return']:.2%}")
        if "alpha_vs_benchmark" in m:
            print(f"    Alpha vs {args.benchmark}:   {m['alpha_vs_benchmark']:.2%}")

    # Figures
    logger.info("Generating figures...")
    fig_a = fig_portfolio_value(
        managed["value"], unmanaged["value"], benchmark_value,
        ALL_WINDOWS_FULL, args.strategy, outdir,
    )
    logger.info("  %s", fig_a.name)

    fig_b = fig_pnl_drawdown(managed, ALL_WINDOWS_FULL, args.strategy, outdir)
    logger.info("  %s", fig_b.name)

    # Audit CSVs
    managed.to_csv(outdir / "portfolio_daily.csv")
    exposure.to_frame("exposure").to_csv(outdir / "exposure.csv")
    logger.info("Wrote portfolio_daily.csv and exposure.csv")

    total_tcost = managed["tcost"].sum()
    logger.info("Total transaction costs: $%.2f (%.2f bps of initial capital)",
                total_tcost, total_tcost / args.capital * 1e4)

    print(f"\nDone. Results in: {outdir}")


if __name__ == "__main__":
    main()
