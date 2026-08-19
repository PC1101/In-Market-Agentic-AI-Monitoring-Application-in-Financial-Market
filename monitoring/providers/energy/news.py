"""Energy NewsProvider: reads a GDELT-built store via the existing NewsStore.

The GDELT adapter (``news.gdelt.build_gdelt_store``) writes the same per-year
parquet layout FNSPID uses, so this provider is a thin wrapper over ``NewsStore``
pointed at the energy store root — identical read path to the S&P 500 provider.
Build the store once with ``build_energy_store`` before querying.
"""
from __future__ import annotations

from pathlib import Path

from news.store import NewsStore
from news import gdelt
from providers.energy.universe import UNIVERSE

DEFAULT_ROOT = Path(__file__).resolve().parents[2].parent / "data" / "news" / "energy_gdelt"


def energy_symbol_queries() -> dict[str, list[str]]:
    """One GDELT query per universe ticker: [company name, ticker]."""
    return {ticker: [name, ticker] for ticker, name in UNIVERSE.items()}


def build_energy_store(start="2017-01-01", end="2020-12-31",
                       out_dir: str | Path = DEFAULT_ROOT, **kw) -> dict[str, int]:
    """Ingest GDELT news for the whole energy universe into the store."""
    return gdelt.build_gdelt_store(energy_symbol_queries(), start, end, out_dir, **kw)


class EnergyNews:
    def __init__(self, root: str | Path = DEFAULT_ROOT):
        self._store = NewsStore(root)

    def query(self, start, end, tickers=None):
        return self._store.query(start, end, tickers=tickers)
