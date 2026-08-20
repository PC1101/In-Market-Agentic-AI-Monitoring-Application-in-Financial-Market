"""Generate the JT momentum strategy's DAILY PnL curve for the global-energy universe.

Runs the same market-agnostic XSectional pipeline used for S&P 500
(``run_daily_pnl.py``) but against the curated global-energy universe
(``monitoring/providers/energy/universe.py``). The energy universe is a fixed
curated set (see that module's docstring for the survivorship rationale), so
unlike the S&P 500 driver there is NO point-in-time membership mask applied.

Output (schema Date,port_ret,equity,drawdown):
  * results/equity_curve_energy.csv

Downloads energy universe prices via yfinance (network, ~30-60s; cached by
yfinance's own session cache thereafter).
"""

import logging
import sys
from pathlib import Path

XSECTIONAL_DIR = Path(__file__).resolve().parent
REPO_ROOT = XSECTIONAL_DIR.parent
MONITORING_DIR = REPO_ROOT / "monitoring"

def _prepend_to_syspath(path: Path) -> None:
    """Force `path` to the front of sys.path, deduping any prior occurrence.

    Running `python XSectional/run_energy_pnl.py` auto-inserts XSECTIONAL_DIR
    into sys.path[0] *before* this module's code runs, so a plain
    ``if str(p) not in sys.path: sys.path.insert(0, ...)`` guard silently
    no-ops for XSECTIONAL_DIR and leaves MONITORING_DIR first instead —
    which shadows XSectional's own top-level config.py with the unrelated
    monitoring/config/ package. Removing-then-reinserting guarantees order.
    """
    p = str(path)
    if p in sys.path:
        sys.path.remove(p)
    sys.path.insert(0, p)


# The energy provider lives under monitoring/providers/energy/.
_prepend_to_syspath(MONITORING_DIR)
# XSectional's own modules (signals, portfolio, backtest, config) import as
# top-level names from within XSectional/, so it must be on sys.path — put
# *after* MONITORING_DIR above so it ends up first and wins name collisions
# (e.g. XSectional's config.py vs. monitoring/config/).
_prepend_to_syspath(XSECTIONAL_DIR)

from signals import compute_momentum_scores
from portfolio import construct_portfolio
from backtest import run_backtest_daily, write_daily_equity_curve
from providers.energy import universe as energy_universe

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

RESULTS = XSECTIONAL_DIR / "results"


def main() -> None:
    logger.info("Loading global-energy universe prices...")
    prices = energy_universe.load_prices(start="2016-06-01", end="2022-12-31")
    prices = prices.dropna(axis=1, how="all")
    logger.info("  %d tickers, %d daily observations", prices.shape[1], len(prices))

    logger.info("Computing momentum scores...")
    scores = compute_momentum_scores(prices)

    logger.info("Constructing portfolio + simulating daily PnL (no membership mask)...")
    # The shared config defaults (20% quantiles, min 10/leg) are tuned for the
    # ~500-name S&P universe; against the curated ~34-name energy universe they
    # cap every month at 6 long / 6 short candidates, below the 10-stock floor,
    # so every rebalance gets skipped and the curve is degenerate (flat, zero
    # PnL). Widen the quantiles and lower the per-leg floor for this smaller,
    # fixed universe: 30% of 34 names ~= 10 per leg, a real long/short book.
    # This does not touch config.py or the S&P 500 runner's behavior.
    weights = construct_portfolio(
        scores, top_quantile=0.30, bottom_quantile=0.30, min_stocks_per_leg=6
    )
    daily = run_backtest_daily(weights, prices)

    out = RESULTS / "equity_curve_energy.csv"
    write_daily_equity_curve(daily, str(out))
    logger.info("Wrote %s (%d trading days, final equity %.3f)",
                out, len(daily), daily["equity"].iloc[-1])


if __name__ == "__main__":
    main()
