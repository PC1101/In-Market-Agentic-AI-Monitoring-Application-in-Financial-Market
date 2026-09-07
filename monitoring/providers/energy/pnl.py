"""Energy strategy PnL.

Canonical source of truth for the energy curve paths (mirrors
``providers/sp500/pnl.py``). Two committed daily curves:

  * JT momentum: ``XSectional/results/equity_curve_energy.csv``
  * AL PCA:      ``Stat Arb/statsArb-dev/results/equity_curve_energy_al.csv``
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from pnl_loader import load_pnl, returns_series, as_of as _as_of

ROOT = Path(__file__).resolve().parents[3]  # repo root

CURVES: dict[str, Path] = {
    "JT_MOM": ROOT / "XSectional" / "results" / "equity_curve_energy.csv",
    "AL_PCA": ROOT / "Stat Arb" / "statsArb-dev" / "results" / "equity_curve_energy_al.csv",
}


class EnergyPnL:
    def strategies(self) -> list[str]:
        return list(CURVES)

    def returns(self, strategy: str, as_of=None) -> pd.Series:
        s = returns_series(load_pnl(CURVES[strategy]))
        s.name = "port_ret"
        return _as_of(s, as_of) if as_of is not None else s
