"""S&P 500 news: wraps the FNSPID NewsStore at its default root."""
from __future__ import annotations

from pathlib import Path

from news.store import NewsStore

DEFAULT_ROOT = Path(__file__).resolve().parents[2] / "news" / "store"


class SP500News:
    def __init__(self, root: str | Path = DEFAULT_ROOT):
        self._store = NewsStore(root)

    def query(self, start, end, tickers=None):
        return self._store.query(start, end, tickers=tickers)
