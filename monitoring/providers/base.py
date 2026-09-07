"""Provider interfaces every market implements.

Each method is *as-of dated*: an ``as_of`` date means "return only what was
knowable on that date". Concrete providers wrap a market's real sources; the
pipeline depends only on these Protocols, never on a specific market.
"""
from __future__ import annotations

from datetime import date
from typing import Protocol, runtime_checkable

import pandas as pd


@runtime_checkable
class StrategyPnLProvider(Protocol):
    def strategies(self) -> list[str]:
        """Names of strategies this market exposes (e.g. ['AL_PCA', 'JT_MOM'])."""
        ...

    def returns(self, strategy: str, as_of: date | None = None) -> pd.Series:
        """Daily port_ret series for ``strategy``, sliced to <= as_of if given."""
        ...


@runtime_checkable
class NewsProvider(Protocol):
    def query(self, start, end, tickers: list[str] | None = None) -> pd.DataFrame:
        """News rows with publication date in [start, end], optionally by ticker."""
        ...


@runtime_checkable
class MacroProvider(Protocol):
    def context(self, as_of: date) -> dict:
        """As-of-correct JSON-serialisable macro block for the agent."""
        ...
