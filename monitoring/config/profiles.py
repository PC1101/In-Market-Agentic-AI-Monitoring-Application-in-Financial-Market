"""MarketProfile: names the concrete providers + calendar for one market."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class MarketProfile:
    key: str
    timezone: str
    base_ccy: str
    pnl: Any            # StrategyPnLProvider
    news: Any           # NewsProvider
    macro: Any          # MacroProvider
    windows: list = field(default_factory=list)


#: Populated by each market package's register() call (import side-effect).
REGISTRY: dict[str, MarketProfile] = {}


def register(profile: MarketProfile) -> None:
    REGISTRY[profile.key] = profile


def get_profile(key: str) -> MarketProfile:
    if key not in REGISTRY:
        # Import known markets lazily so entry points need not import each package.
        if key == "sp500":
            import providers.sp500.profile  # noqa: F401  (registers on import)
        elif key == "energy":
            import providers.energy.profile  # noqa: F401  (registers on import)
    if key not in REGISTRY:
        raise KeyError(f"unknown market '{key}'; registered: {sorted(REGISTRY)}")
    return REGISTRY[key]
