# main.py — orchestrates the full XSectional momentum pipeline

import logging

from data import load_prices
from signals import compute_momentum_scores
from portfolio import construct_portfolio
from backtest import run_backtest
from report import generate_report

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def main() -> None:
    logger.info("Step 1/5 — Loading prices...")
    prices = load_prices()
    logger.info("  %d tickers, %d daily observations", prices.shape[1], len(prices))

    logger.info("Step 2/5 — Computing momentum scores...")
    scores = compute_momentum_scores(prices)

    logger.info("Step 3/5 — Constructing long-short portfolio...")
    weights = construct_portfolio(scores)

    logger.info("Step 4/5 — Running backtest...")
    returns = run_backtest(weights, prices)
    logger.info("  %d monthly periods simulated", len(returns))

    logger.info("Step 5/5 — Generating tearsheet...")
    generate_report(returns)


if __name__ == "__main__":
    main()
