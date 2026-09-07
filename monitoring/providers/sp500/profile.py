"""Build + register the sp500 MarketProfile (import side-effect)."""
from __future__ import annotations

from config.profiles import MarketProfile, register
from providers.sp500.pnl import SP500PnL
from providers.sp500.news import SP500News
from providers.sp500.macro import SP500Macro

try:
    from windows import ALL_WINDOWS
except Exception:  # pragma: no cover - windows module always importable in-repo
    ALL_WINDOWS = []

register(MarketProfile(
    key="sp500",
    timezone="America/New_York",
    base_ccy="USD",
    pnl=SP500PnL(),
    news=SP500News(),
    macro=SP500Macro(),
    windows=list(ALL_WINDOWS),
))
