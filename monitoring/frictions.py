"""Realistic trading-friction model for post-hoc cost application.

Era-dependent per-side cost schedule (commission + half-spread) applied to
one-way turnover.  Three scenarios (low / base / high) for sensitivity.

Cost model:
    net_ret(t) = gross_ret(t) − turnover_oneway(t) × bps(t) / 1e4

Frictions included:
  - Commission: era-dependent (pre-2019 retail commissions, post-2019 zero)
  - Bid-ask half-spread: era- and liquidity-dependent

Frictions excluded (disclosed):
  - Slippage / market impact: <1 bps at $1M in S&P 500 names (folded into "high")
  - Borrow cost: N/A (long-only strategies)
  - Dividends: already in adjusted close
"""
from __future__ import annotations

import pandas as pd
import numpy as np


# ---------------------------------------------------------------------------
# Era-dependent per-side cost schedule (bps of notional traded)
# ---------------------------------------------------------------------------
#   Combines commission + bid-ask half-spread for retail, long-only,
#   S&P 500 large-cap / SPDR sector-ETF trading.
#
#   Era breakpoints:
#     2004–2009: pre-decimalization maturity, wider spreads, ~$5-10 commissions
#     2010–2018: tighter spreads, lower commissions
#     2019–2025: zero-commission retail (Schwab Oct 2019), tight spreads
#
COST_SCHEDULE = {
    "low":  [("2004-01-01", "2009-12-31", 4.0),
             ("2010-01-01", "2018-12-31", 2.0),
             ("2019-01-01", "2099-12-31", 1.0)],
    "base": [("2004-01-01", "2009-12-31", 8.0),
             ("2010-01-01", "2018-12-31", 5.0),
             ("2019-01-01", "2099-12-31", 2.5)],
    "high": [("2004-01-01", "2009-12-31", 15.0),
             ("2010-01-01", "2018-12-31", 10.0),
             ("2019-01-01", "2099-12-31", 5.0)],
}


def cost_schedule(scenario: str = "base") -> list[tuple[str, str, float]]:
    """Return the cost schedule for a scenario as (start, end, bps_per_side) tuples."""
    if scenario not in COST_SCHEDULE:
        raise ValueError(f"Unknown scenario {scenario!r}; choose from {list(COST_SCHEDULE)}")
    return COST_SCHEDULE[scenario]


def bps_series(index: pd.DatetimeIndex, scenario: str = "base") -> pd.Series:
    """Map each date in *index* to its per-side cost in basis points."""
    schedule = cost_schedule(scenario)
    bps = pd.Series(np.nan, index=index, dtype=float)
    for start, end, val in schedule:
        mask = (index >= start) & (index <= end)
        bps[mask] = val
    # Fill any dates outside the schedule with the last era's cost
    bps = bps.ffill().bfill()
    return bps


def apply_costs(
    gross_ret: pd.Series,
    turnover_oneway: pd.Series,
    scenario: str = "base",
) -> tuple[pd.Series, pd.Series]:
    """Apply era-dependent transaction costs to a gross return series.

    Args:
        gross_ret: daily gross portfolio returns (index = DatetimeIndex).
        turnover_oneway: daily one-way turnover as a fraction of portfolio value
            (e.g. 0.05 = 5% of portfolio traded that day).
        scenario: cost scenario ('low', 'base', 'high').

    Returns:
        (net_ret, daily_cost) where daily_cost is in return-fraction units.
    """
    bps = bps_series(gross_ret.index, scenario)
    daily_cost = turnover_oneway * bps / 1e4
    net_ret = gross_ret - daily_cost
    return net_ret, daily_cost


def net_curve_df(
    gross_ret: pd.Series,
    turnover_oneway: pd.Series,
    scenario: str = "base",
) -> pd.DataFrame:
    """Build a full net equity curve DataFrame from gross returns and turnover.

    Returns DataFrame with columns: port_ret, equity, drawdown, turnover, cost.
    Index named 'Date'.
    """
    net_ret, daily_cost = apply_costs(gross_ret, turnover_oneway, scenario)

    equity = (1.0 + net_ret).cumprod()
    drawdown = equity / equity.cummax() - 1.0

    out = pd.DataFrame({
        "port_ret": net_ret,
        "equity": equity,
        "drawdown": drawdown,
        "turnover": turnover_oneway,
        "cost": daily_cost,
    })
    out.index.name = "Date"
    return out
