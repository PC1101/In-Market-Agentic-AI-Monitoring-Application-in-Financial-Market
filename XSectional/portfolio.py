# portfolio.py — construct equal-weighted long-short portfolio from momentum scores

import logging

import pandas as pd
import numpy as np

import config

logger = logging.getLogger(__name__)


def construct_portfolio(
    scores: pd.DataFrame,
    top_quantile: float | None = None,
    bottom_quantile: float | None = None,
    min_stocks_per_leg: int | None = None,
) -> pd.DataFrame:
    """
    Build equal-weighted long-short portfolio weights from monthly momentum scores.

    Each month:
      - Long:  top `top_quantile` fraction (equal weight = 1/n_long each)
      - Short: bottom `bottom_quantile` fraction (equal weight = -1/n_short each)
      - All others: pd.NA (unassigned / neutral)

    If fewer than `min_stocks_per_leg` stocks qualify for either leg,
    that month's weights are left as pd.NA and a warning is logged.

    Args:
        scores: monthly momentum scores (tickers as columns).
        top_quantile: fraction of stocks to go long; defaults to
            `config.TOP_QUANTILE` when None (S&P 500 / default behavior).
        bottom_quantile: fraction of stocks to short; defaults to
            `config.BOTTOM_QUANTILE` when None.
        min_stocks_per_leg: minimum names required per leg for a month to be
            traded; defaults to `config.MIN_STOCKS_PER_LEG` when None.

    Passing no overrides reproduces the exact prior behavior (reads the
    module-level `config` constants), so existing callers — e.g. the S&P 500
    daily runner — are unaffected. Overrides let smaller fixed universes
    (e.g. the curated ~34-name energy universe) use a wider quantile / lower
    per-leg minimum without changing the shared defaults.

    Uses Float64 (nullable float) dtype so that pd.NA comparisons propagate
    correctly through pandas boolean operations.

    Returns a DataFrame of weights with the same shape as scores.
    """
    top_quantile = config.TOP_QUANTILE if top_quantile is None else top_quantile
    bottom_quantile = config.BOTTOM_QUANTILE if bottom_quantile is None else bottom_quantile
    min_stocks_per_leg = (
        config.MIN_STOCKS_PER_LEG if min_stocks_per_leg is None else min_stocks_per_leg
    )

    weights = pd.DataFrame(
        pd.NA,
        index=scores.index,
        columns=scores.columns,
        dtype="Float64",
    )

    for date, row in scores.iterrows():
        valid = row.dropna()
        if valid.empty:
            continue

        n_long = max(1, int(len(valid) * top_quantile))
        n_short = max(1, int(len(valid) * bottom_quantile))

        if n_long < min_stocks_per_leg or n_short < min_stocks_per_leg:
            logger.warning(
                "Skipping %s: only %d long / %d short candidates (min %d required)",
                date.date(),
                n_long,
                n_short,
                min_stocks_per_leg,
            )
            continue

        long_tickers = valid.nlargest(n_long).index
        short_tickers = valid.nsmallest(n_short).index

        weights.loc[date, long_tickers] = 1.0 / n_long
        weights.loc[date, short_tickers] = -1.0 / n_short

    return weights
