"""S&P 500 macro: wraps FRED/ALFRED as-of context."""
from __future__ import annotations

from macro.context import macro_context


class SP500Macro:
    def context(self, as_of) -> dict:
        return macro_context(as_of)
