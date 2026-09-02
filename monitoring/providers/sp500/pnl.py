"""S&P 500 strategy PnL.

Canonical source of truth for the S&P 500 curve paths. ``run_classical.py`` and
``run_agentic.py`` import ``STRATEGY_CURVES`` from here so there is exactly one
definition of which curves each strategy uses. The path logic mirrors the
original ``run_classical._prefer_pit`` exactly (point-in-time curve if built,
else the 2020-snapshot one) to guarantee bit-identical behaviour after the
provider refactor.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from pnl_loader import load_pnl, returns_series, as_of as _as_of

ROOT = Path(__file__).resolve().parents[3]  # repo root
STAT_ARB = ROOT / "Stat Arb" / "statsArb-dev" / "results" / "full_universe"
JT_DAILY = ROOT / "XSectional" / "results" / "equity_curve_daily.csv"


def _prefer_pit(tag: str) -> Path:
    """Use the point-in-time curve for a window tag if built, else the snapshot."""
    pit = STAT_ARB / f"{tag}_pit" / "equity_curve.csv"
    return pit if pit.exists() else STAT_ARB / tag / "equity_curve.csv"


# Each strategy is a list of continuous daily PnL curves (identical to the
# original run_classical.STRATEGY_CURVES).
STRATEGY_CURVES: dict[str, list[Path]] = {
    "AL_PCA": [
        _prefer_pit("baseline_2007_2015"),
        _prefer_pit("calm_2004_2006"),
    ],
    "JT_MOM": [JT_DAILY],
}

#: Convenience alias used by the provider contract test.
AL_CURVE_PATHS = STRATEGY_CURVES["AL_PCA"]


class SP500PnL:
    def strategies(self) -> list[str]:
        return list(STRATEGY_CURVES)

    def returns(self, strategy: str, as_of=None) -> pd.Series:
        paths = [p for p in STRATEGY_CURVES[strategy] if p.exists()]
        parts = [returns_series(load_pnl(p)) for p in paths]
        s = pd.concat(parts).sort_index()
        s = s[~s.index.duplicated(keep="first")]
        s.name = "port_ret"
        return _as_of(s, as_of) if as_of is not None else s
