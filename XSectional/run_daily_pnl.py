"""Generate the JT momentum strategy's DAILY PnL curve for classical monitoring.

Runs the XSectional pipeline on the POINT-IN-TIME S&P 500 universe (survivorship-bias
control): prices for every historical index member available from yfinance, with the
candidate set at each monthly rebalance restricted to that date's actual members via
the sp500-master membership table.

Outputs (schema Date,port_ret,equity,drawdown):
  * results/equity_curve_daily.csv        — PIT universe (canonical; monitored curve)
  * results/pit_coverage.csv              — per-rebalance member coverage (bias audit)

Use --survivorship to reproduce the old current-members-only curve
(written to results/equity_curve_daily_survivorship.csv) for comparison.

First PIT run downloads ~1,000 tickers from yfinance (a few minutes); cached thereafter.
"""

import argparse
import logging
from pathlib import Path

from data import load_prices, load_prices_pit
from signals import compute_momentum_scores
from portfolio import construct_portfolio, to_long_only
from backtest import run_backtest_daily, write_daily_equity_curve
import universe

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

RESULTS = Path(__file__).resolve().parent / "results"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--survivorship", action="store_true",
                        help="use the biased current-members universe (comparison only)")
    parser.add_argument("--long-only", action="store_true",
                        help="use only the long leg (positive weights, renormalised)")
    args = parser.parse_args()

    if args.survivorship:
        logger.info("Loading CURRENT-MEMBERS prices (survivorship-biased comparison run)...")
        prices = load_prices()
        out = RESULTS / "equity_curve_daily_survivorship.csv"
    else:
        logger.info("Loading POINT-IN-TIME universe prices...")
        prices = load_prices_pit()
        out = RESULTS / "equity_curve_daily.csv"
    logger.info("  %d tickers, %d daily observations", prices.shape[1], len(prices))

    logger.info("Computing momentum scores...")
    scores = compute_momentum_scores(prices)

    if not args.survivorship:
        logger.info("Applying point-in-time membership mask...")
        membership = universe.load_membership()
        scores = universe.apply_membership(scores, membership)
        cov = universe.coverage_report(scores, membership)
        cov.to_csv(RESULTS / "pit_coverage.csv")
        recent = cov["coverage"].dropna()
        logger.info("  member coverage: mean %.0f%%, min %.0f%% (see results/pit_coverage.csv)",
                    100 * recent.mean(), 100 * recent.min())

    logger.info("Constructing portfolio + simulating daily PnL...")
    weights = construct_portfolio(scores)
    if args.long_only:
        logger.info("Filtering to long-only weights...")
        weights = to_long_only(weights)
    daily, w_daily = run_backtest_daily(weights, prices, return_weights=True)
    if args.long_only:
        out = out.with_name(out.stem + "_long_only" + out.suffix)
    write_daily_equity_curve(daily, str(out))
    logger.info("Wrote %s (%d trading days, final equity %.3f)",
                out, len(daily), daily["equity"].iloc[-1])

    # Write daily applied weights for post-hoc turnover/cost analysis
    weights_path = out.with_name(out.stem.replace("equity_curve_daily", "weights_daily") + out.suffix)
    w_daily.to_csv(weights_path)
    logger.info("Wrote %s (%d days × %d tickers)", weights_path, *w_daily.shape)


if __name__ == "__main__":
    main()
