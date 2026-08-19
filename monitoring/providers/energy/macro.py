"""Energy commodity macro provider: as-of-correct WTI/Brent/natgas + risk block.

Energy regime breaks are commodity-driven, so the energy market's MacroProvider
carries oil/gas price curves rather than just rates (design §6). All series are
unrevised daily FRED prints (like VIX), so the point-in-time query is the existing
``asof_daily`` (last print on/before the decision date) — no ALFRED vintages needed.

Series are fetched with the existing ``macro.fetch_macro.fetch_series`` (reuses the
FRED integration + FRED_API_KEY) into a separate ``data/macro_energy`` dir so the
US macro cache is untouched.
"""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from macro.asof import asof_daily
from macro.fetch_macro import fetch_series

#: FRED series id -> friendly context key. All unrevised daily prints.
ENERGY_SERIES: dict[str, str] = {
    "DCOILWTICO": "wti_crude",        # WTI crude, $/bbl
    "DCOILBRENTEU": "brent_crude",    # Brent crude, $/bbl
    "DHHNGSP": "henry_hub_natgas",    # Henry Hub natural gas, $/MMBtu
    "VIXCLS": "vix",                  # risk-on/off (shared with US macro)
}

DATA_DIR = Path(__file__).resolve().parents[2].parent / "data" / "macro_energy"


def fetch_energy_macro(api_key: str | None = None, data_dir: str | Path = DATA_DIR) -> dict[str, int]:
    """Fetch the energy commodity series from FRED into per-series parquets.

    Reuses ``macro.fetch_macro.fetch_series`` (unrevised daily). Requires a free
    FRED_API_KEY. Returns {series_id: row count}.
    """
    api_key = api_key or os.environ.get("FRED_API_KEY")
    if not api_key:
        raise SystemExit(
            "Set FRED_API_KEY (free: https://fred.stlouisfed.org/docs/api/api_key.html)")
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    for sid in ENERGY_SERIES:
        df = fetch_series(sid, api_key, vintages=False)
        df.to_parquet(data_dir / f"{sid}.parquet")
        counts[sid] = len(df)
    return counts


class EnergyMacro:
    def __init__(self, data_dir: str | Path = DATA_DIR):
        self._dir = Path(data_dir)

    def context(self, as_of) -> dict:
        """As-of-correct commodity block (mirrors macro.context.macro_context)."""
        out: dict = {}
        for sid, name in ENERGY_SERIES.items():
            path = self._dir / f"{sid}.parquet"
            if path.exists():
                v = asof_daily(pd.read_parquet(path), as_of)
                if v is not None:
                    out[name] = v
        if "wti_crude" in out and "brent_crude" in out:
            out["brent_wti_spread"] = round(
                out["brent_crude"]["value"] - out["wti_crude"]["value"], 3)
        return out
