"""Portfolio simulation engine: strategy returns + exposure overlay -> $-value curve."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class EngineConfig:
    """Simulation parameters."""
    capital: float = 1_000_000.0
    tcost_bps: float = 10.0          # transaction cost on |delta-exposure| * notional
    strat_tcost_annual: float = 0.0  # annualised strategy friction (bps/year / 1e4)
    cash_rate_annual: float = 0.0    # annual rate earned on uninvested cash sleeve


def simulate(strategy_returns: pd.Series, exposure: pd.Series,
             cfg: EngineConfig | None = None) -> pd.DataFrame:
    """Simulate a portfolio that scales strategy exposure daily.

    Args:
        strategy_returns: daily simple returns of the underlying strategy.
        exposure: daily target exposure in [0, 1], aligned to strategy_returns index.
        cfg: simulation parameters.

    Returns:
        DataFrame with columns: value, pnl, exposure, strat_ret, port_ret, tcost, drawdown.
    """
    if cfg is None:
        cfg = EngineConfig()

    # Align
    idx = strategy_returns.index.intersection(exposure.index)
    ret = strategy_returns.reindex(idx).fillna(0.0).values
    exp = exposure.reindex(idx).fillna(1.0).values
    n = len(idx)

    cash_daily = (1 + cfg.cash_rate_annual) ** (1 / 252) - 1
    strat_drag_daily = cfg.strat_tcost_annual / 252  # daily friction deducted from strategy return

    values = np.empty(n)
    pnl = np.empty(n)
    port_ret = np.empty(n)
    tcost = np.empty(n)

    prev_value = cfg.capital
    prev_exp = exp[0]  # no tcost on first day's initial position

    for i in range(n):
        e = exp[i]
        r = ret[i]

        # Gross return: invested portion earns strategy minus friction, rest earns cash
        gross = e * (r - strat_drag_daily) + (1 - e) * cash_daily

        # Transaction cost on exposure change
        delta_exp = abs(e - prev_exp) if i > 0 else 0.0
        tc = delta_exp * prev_value * cfg.tcost_bps / 1e4

        v = prev_value * (1 + gross) - tc

        values[i] = v
        pnl[i] = v - prev_value
        port_ret[i] = (v - prev_value) / prev_value if prev_value != 0 else 0.0
        tcost[i] = tc

        prev_value = v
        prev_exp = e

    # Drawdown
    peak = np.maximum.accumulate(values)
    drawdown = (values - peak) / peak

    return pd.DataFrame({
        "value": values,
        "pnl": pnl,
        "exposure": exp,
        "strat_ret": ret,
        "port_ret": port_ret,
        "tcost": tcost,
        "drawdown": drawdown,
    }, index=idx)
