# portfolio.py — construct equal-weighted long-short portfolio from momentum scores

import logging

import pandas as pd
import numpy as np

import config

logger = logging.getLogger(__name__)


def construct_portfolio(scores: pd.DataFrame) -> pd.DataFrame:
    """
    Build equal-weighted long-short portfolio weights from monthly momentum scores.

    Each month:
      - Long:  top TOP_QUANTILE fraction (equal weight = 1/n_long each)
      - Short: bottom BOTTOM_QUANTILE fraction (equal weight = -1/n_short each)
      - All others: pd.NA (unassigned / neutral)

    If fewer than MIN_STOCKS_PER_LEG stocks qualify for either leg,
    that month's weights are left as pd.NA and a warning is logged.

    Uses Float64 (nullable float) dtype so that pd.NA comparisons propagate
    correctly through pandas boolean operations.

    Returns a DataFrame of weights with the same shape as scores.
    """
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

        n_long = max(1, int(len(valid) * config.TOP_QUANTILE))
        n_short = max(1, int(len(valid) * config.BOTTOM_QUANTILE))

        if n_long < config.MIN_STOCKS_PER_LEG or n_short < config.MIN_STOCKS_PER_LEG:
            logger.warning(
                "Skipping %s: only %d long / %d short candidates (min %d required)",
                date.date(),
                n_long,
                n_short,
                config.MIN_STOCKS_PER_LEG,
            )
            continue

        long_tickers = valid.nlargest(n_long).index
        short_tickers = valid.nsmallest(n_short).index

        weights.loc[date, long_tickers] = 1.0 / n_long
        weights.loc[date, short_tickers] = -1.0 / n_short

    return weights
