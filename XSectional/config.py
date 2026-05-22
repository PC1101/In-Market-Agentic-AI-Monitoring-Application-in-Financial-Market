# config.py — all tunable parameters for the XSectional momentum model

START_DATE = "2000-01-01"
END_DATE   = "2025-12-31"

LOOKBACK_MONTHS = 12   # Momentum formation window (months)
SKIP_MONTHS     = 1    # Months to skip before lookback (short-term reversal avoidance)

TOP_QUANTILE    = 0.20  # Fraction of stocks to go long (top performers)
BOTTOM_QUANTILE = 0.20  # Fraction of stocks to short (bottom performers)

REBALANCE_FREQ  = "ME"  # Month-end rebalancing (pandas resample alias)

DATA_DIR = "data"

MISSING_DATA_THRESHOLD = 0.20  # Drop tickers with > 20% missing data
MIN_STOCKS_PER_LEG     = 10    # Minimum stocks required per long/short leg
