"""Build + register the energy MarketProfile (import side-effect)."""
from __future__ import annotations

from config.profiles import MarketProfile, register
from providers.energy.pnl import EnergyPnL
from providers.energy.news import EnergyNews
from providers.energy.macro import EnergyMacro
from providers.energy.windows import ALL_WINDOWS

register(MarketProfile(
    key="energy",
    timezone="UTC",
    base_ccy="USD",
    pnl=EnergyPnL(),
    news=EnergyNews(),
    macro=EnergyMacro(),
    windows=list(ALL_WINDOWS),
))
