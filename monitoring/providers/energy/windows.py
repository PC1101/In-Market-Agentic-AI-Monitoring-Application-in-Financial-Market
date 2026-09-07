"""Evaluation windows for the global-energy market.

Reuses the market-agnostic ``Window`` dataclass. Onsets are commodity-regime
breaks (the energy market's analogue of the US quant/credit events). Windows are
chosen inside the GDELT DOC 2.0 coverage range (2017+) so the agentic layer has
news for every window; pre-2017 energy events (2014-16 oil crash, 2016 OPEC cut)
need the GKG raw files and can be added once that ingest exists.

Tentative per design §11.5 — confirm onset dates against the commodity curves
when the FRED energy series are fetched.
"""
from __future__ import annotations

from windows import Window

EVENT_WINDOWS: tuple[Window, ...] = (
    Window(
        name="oil_crash_2020",
        kind="event",
        start="2020-02-14",
        end="2020-05-29",
        onset="2020-03-09",  # OPEC+ collapse + COVID demand shock; WTI -25% in a day
        description="COVID demand collapse + Saudi-Russia price war; WTI later went "
                    "negative on 2020-04-20.",
    ),
    Window(
        name="energy_spike_2022",
        kind="event",
        start="2022-01-14",
        end="2022-06-30",
        onset="2022-02-24",  # Russian invasion of Ukraine -> energy price spike
        description="Russia's invasion of Ukraine drives a sharp global energy price "
                    "spike (crude + natural gas).",
    ),
)

CALM_WINDOWS: tuple[Window, ...] = (
    Window(
        name="calm_energy_2017",
        kind="calm",
        start="2017-04-03",
        end="2017-09-29",
        onset=None,
        description="Range-bound crude through mid-2017; no regime break.",
    ),
    Window(
        name="calm_energy_2019",
        kind="calm",
        start="2019-04-01",
        end="2019-09-30",
        onset=None,
        description="Relatively stable energy prices mid-2019; no regime break.",
    ),
)

ALL_WINDOWS: tuple[Window, ...] = EVENT_WINDOWS + CALM_WINDOWS
