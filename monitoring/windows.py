"""Six-window evaluation design: 4 event windows + 2 calm windows.

Each window is a labelled slice of the backtest timeline. Event windows carry a
ground-truth ``onset`` date — the day the regime break is considered to *begin* —
which the metrics module uses to compute detection latency, precision, and recall.
Calm windows carry no onset; any alarm inside a calm window is a false positive.

These are the working definitions for Week 2. The two gating events named in the
VRI plan (Aug-2007 quant meltdown, Apr-2009 momentum crash) are fixed; the other
two event windows (2008 GFC/Lehman, 2011 downgrade) are the natural additions for
stat-arb + momentum strategies over 2007-2016 and should be confirmed with the
supervisor before the Week 5 write-up.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import pandas as pd


@dataclass(frozen=True)
class Window:
    """A labelled evaluation window.

    Attributes:
        name:   short unique identifier (used in result paths / tables).
        kind:   "event" or "calm".
        start:  inclusive first date of the window (ISO string).
        end:    inclusive last date of the window (ISO string).
        onset:  ground-truth regime-break date (ISO string) for event windows;
                None for calm windows.
        description: human-readable note on what the window represents.
    """

    name: str
    kind: str  # "event" | "calm"
    start: str
    end: str
    onset: str | None
    description: str

    def __post_init__(self) -> None:
        if self.kind not in ("event", "calm"):
            raise ValueError(f"kind must be 'event' or 'calm', got {self.kind!r}")
        if self.kind == "event" and self.onset is None:
            raise ValueError(f"event window {self.name!r} requires an onset date")
        if self.kind == "calm" and self.onset is not None:
            raise ValueError(f"calm window {self.name!r} must not have an onset date")
        # Validate ordering.
        s, e = date.fromisoformat(self.start), date.fromisoformat(self.end)
        if s > e:
            raise ValueError(f"window {self.name!r}: start {self.start} after end {self.end}")
        if self.onset is not None:
            o = date.fromisoformat(self.onset)
            if not (s <= o <= e):
                raise ValueError(
                    f"window {self.name!r}: onset {self.onset} outside [{self.start}, {self.end}]"
                )

    @property
    def start_ts(self) -> pd.Timestamp:
        return pd.Timestamp(self.start)

    @property
    def end_ts(self) -> pd.Timestamp:
        return pd.Timestamp(self.end)

    @property
    def onset_ts(self) -> pd.Timestamp | None:
        return pd.Timestamp(self.onset) if self.onset else None

    def contains(self, ts) -> bool:
        """True if timestamp ts falls within [start, end] inclusive."""
        ts = pd.Timestamp(ts)
        return self.start_ts <= ts <= self.end_ts


# --- The canonical six windows -------------------------------------------------

EVENT_WINDOWS: tuple[Window, ...] = (
    Window(
        name="quant_meltdown_2007",
        kind="event",
        start="2007-07-16",
        end="2007-09-14",
        onset="2007-08-06",
        description="August 2007 quant meltdown — simultaneous de-risking of "
        "statistical-arbitrage books; gating test for the AL PCA strategy.",
    ),
    Window(
        name="gfc_lehman_2008",
        kind="event",
        start="2008-08-15",
        end="2008-11-28",
        onset="2008-09-15",
        description="Global Financial Crisis / Lehman bankruptcy (15 Sep 2008) and "
        "the ensuing volatility spike.",
    ),
    Window(
        name="momentum_crash_2009",
        kind="event",
        start="2009-02-16",
        end="2009-05-29",
        onset="2009-03-09",
        description="Spring-2009 momentum crash — sharp reversal off the 9 Mar 2009 "
        "market bottom; gating test for the JT momentum strategy.",
    ),
    Window(
        name="downgrade_2011",
        kind="event",
        start="2011-07-18",
        end="2011-10-14",
        onset="2011-08-05",
        description="August 2011 US sovereign-debt downgrade and euro-area stress; "
        "broad cross-sectional dislocation.",
    ),
)

CALM_WINDOWS: tuple[Window, ...] = (
    Window(
        name="calm_2004_2006",
        kind="calm",
        start="2004-01-02",
        end="2006-12-29",
        onset=None,
        description="Low-volatility pre-crisis expansion; control window with no "
        "known regime break.",
    ),
    Window(
        name="calm_2013_2014",
        kind="calm",
        start="2013-01-02",
        end="2014-12-31",
        onset=None,
        description="Post-crisis low-volatility recovery; control window with no "
        "known regime break.",
    ),
)

ALL_WINDOWS: tuple[Window, ...] = EVENT_WINDOWS + CALM_WINDOWS


def get_window(name: str) -> Window:
    """Look up a window by name."""
    for w in ALL_WINDOWS:
        if w.name == name:
            return w
    raise KeyError(f"unknown window {name!r}; known: {[w.name for w in ALL_WINDOWS]}")
