# universe.py — point-in-time S&P 500 membership (survivorship-bias control)
#
# Loads the membership-interval table produced by sp500-master
# (ticker,start_date,end_date; blank end_date = still a member) and provides
# causal membership queries: "who was in the index on date t" uses only
# information that was public on date t (index changes are announced before
# their effective date).
#
# Delisted-name handling (documented per VRI Week 1):
#   * A ticker's membership interval ends on its removal date; after that it is
#     no longer a candidate at any rebalance.
#   * If a held name stops trading mid-hold (bankruptcy/acquisition), its price
#     series ends; the daily backtest books 0 return from the day after its last
#     price — i.e. the position is implicitly exited at the last traded price.
#     This slightly understates losses on names that gapped to zero in OTC
#     trading after delisting, and is noted as a limitation.

import logging
import os

import pandas as pd

import config

logger = logging.getLogger(__name__)

PIT_TABLE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "sp500-master",
    "sp500_ticker_start_end.csv",
)


def load_membership(path: str = PIT_TABLE) -> pd.DataFrame:
    """Load the membership-interval table.

    Returns a DataFrame with columns ticker (str), start_date (Timestamp),
    end_date (Timestamp or NaT for current members). A ticker may appear in
    multiple rows (left and re-entered the index).
    """
    m = pd.read_csv(path, parse_dates=["start_date", "end_date"])
    m["ticker"] = m["ticker"].str.strip().str.upper()
    return m


def members_on(date, membership: pd.DataFrame) -> set:
    """Return the set of tickers that were index members on ``date``."""
    ts = pd.Timestamp(date)
    active = membership[
        (membership["start_date"] <= ts)
        & (membership["end_date"].isna() | (membership["end_date"] >= ts))
    ]
    return set(active["ticker"])


def tickers_active_between(start, end, membership: pd.DataFrame) -> list[str]:
    """All tickers that were members at any point in [start, end] (sorted)."""
    lo, hi = pd.Timestamp(start), pd.Timestamp(end)
    active = membership[
        (membership["start_date"] <= hi)
        & (membership["end_date"].isna() | (membership["end_date"] >= lo))
    ]
    return sorted(set(active["ticker"]))


def membership_mask(dates: pd.DatetimeIndex, tickers, membership: pd.DataFrame) -> pd.DataFrame:
    """Boolean DataFrame (dates x tickers): True where ticker was a member on that date."""
    tickers = list(tickers)
    mask = pd.DataFrame(False, index=dates, columns=tickers)
    ticker_set = set(tickers)
    for _, row in membership.iterrows():
        t = row["ticker"]
        if t not in ticker_set:
            continue
        lo = row["start_date"]
        hi = row["end_date"] if pd.notna(row["end_date"]) else dates.max()
        in_range = (dates >= lo) & (dates <= hi)
        if in_range.any():
            mask.loc[in_range, t] = True
    return mask


def apply_membership(scores: pd.DataFrame, membership: pd.DataFrame) -> pd.DataFrame:
    """Mask a (rebalance-dates x tickers) score matrix to point-in-time members.

    At each rebalance date, tickers that were NOT index members on that date get
    pd.NA — they are simply not candidates that month. Scores themselves are
    unchanged where the ticker was a member.
    """
    mask = membership_mask(scores.index, scores.columns, membership)
    masked = scores.where(mask)
    n_before = scores.notna().sum(axis=1)
    n_after = masked.notna().sum(axis=1)
    dropped = (n_before - n_after).sum()
    logger.info(
        "PIT membership mask: removed %d non-member ticker-months "
        "(avg candidates/month %.0f -> %.0f)",
        int(dropped), n_before.mean(), n_after.mean(),
    )
    return masked


def coverage_report(scores: pd.DataFrame, membership: pd.DataFrame) -> pd.DataFrame:
    """Per-rebalance-date coverage: members on date vs members we have scores for.

    Returns a DataFrame indexed by rebalance date with columns
    n_members, n_covered, coverage. Low coverage means missing price history
    (typically delisted names unavailable from the data vendor) — the honest
    measure of residual survivorship bias.
    """
    rows = []
    for ts in scores.index:
        mem = members_on(ts, membership)
        have = set(scores.columns[scores.loc[ts].notna()]) & mem
        rows.append({
            "n_members": len(mem),
            "n_covered": len(have),
            "coverage": len(have) / len(mem) if mem else float("nan"),
        })
    return pd.DataFrame(rows, index=scores.index)
