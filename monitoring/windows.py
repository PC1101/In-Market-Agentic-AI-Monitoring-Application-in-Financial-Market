"""Evaluation window design: 6 development windows + 6 confirmatory test windows.

Development set (6 windows — used for calibration, prompt iteration, and H1/H2/H3
baseline measurement; treated as "contaminated" because calibration was done on them):
  * 4 event windows: quant_meltdown_2007, gfc_lehman_2008, momentum_crash_2009,
    downgrade_2011
  * 2 calm windows: calm_2004_2006, calm_2013_2014

Confirmatory test set (6 windows — BLIND; no parameter tuning, no prompt changes
may occur after seeing outcomes; access is gated by the eval-freeze-v1 tag):
  * 4 event windows: flash_crash_2010, china_deval_2015, volmageddon_2018, covid_2020
  * 2 calm windows: calm_2012, calm_2017

Each window is a labelled slice of the backtest timeline. Event windows carry a
ground-truth ``onset`` date — the day the regime break is considered to *begin* —
which the metrics module uses to compute detection latency, precision, and recall.
Calm windows carry no onset; any alarm inside a calm window is a false positive.

See preregistration §3 for the full dev/test split rationale and §11 for the
pre-freeze checklist that gates access to the test set.
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

# --- Confirmatory test-set windows (BLIND — no tuning against these) -----------
# Access is gated by the eval-freeze-v1 git tag (preregistration §3.2 + §11).

TEST_EVENT_WINDOWS: tuple[Window, ...] = (
    Window(
        name="flash_crash_2010",
        kind="event",
        start="2010-04-15",
        end="2010-07-15",
        onset="2010-05-06",
        description="2010 Flash Crash — sudden intraday collapse and partial recovery "
        "on 6 May 2010; first modern microstructure-driven crash.",
    ),
    Window(
        name="china_deval_2015",
        kind="event",
        start="2015-07-15",
        end="2015-10-16",
        onset="2015-08-11",
        description="Chinese yuan devaluation (11 Aug 2015) and the ensuing global "
        "equity selloff; AL PCA conditional on PIT curve extending to 2015-10-16.",
    ),
    Window(
        name="volmageddon_2018",
        kind="event",
        start="2018-01-15",
        end="2018-04-13",
        onset="2018-02-05",
        description="Volmageddon — XIV collapse and VIX spike on 5 Feb 2018; "
        "forced short-vol unwind; JT-only (AL PCA curve ends 2014).",
    ),
    Window(
        name="covid_2020",
        kind="event",
        start="2020-02-03",
        end="2020-05-29",
        onset="2020-02-24",
        description="COVID-19 market crash — 24 Feb 2020 first major selloff day "
        "following weekend case escalation; JT-only.",
    ),
)

TEST_CALM_WINDOWS: tuple[Window, ...] = (
    Window(
        name="calm_2012",
        kind="calm",
        start="2012-01-03",
        end="2012-12-31",
        onset=None,
        description="Low-volatility post-GFC recovery year; no known market regime break.",
    ),
    Window(
        name="calm_2017",
        kind="calm",
        start="2017-01-03",
        end="2017-12-29",
        onset=None,
        description="Historically low-VIX year (VIX ~11); no known regime break; JT-only.",
    ),
)

# Strategy-window eligibility for the confirmatory test set.
# AL PCA curve ends 2014-12-31; china_deval_2015 is conditional (§11 checklist).
# volmageddon_2018, covid_2020, calm_2017 are JT-only (beyond AL PCA coverage).
TEST_STRATEGY_WINDOWS: dict[str, list[str]] = {
    "AL_PCA": ["flash_crash_2010", "china_deval_2015", "calm_2012"],
    "JT_MOM": ["flash_crash_2010", "china_deval_2015", "volmageddon_2018",
               "covid_2020", "calm_2012", "calm_2017"],
}

# --- Partition aliases ----------------------------------------------------------

#: Development set: 6 windows used for calibration and baseline measurement.
DEV_WINDOWS: tuple[Window, ...] = EVENT_WINDOWS + CALM_WINDOWS

#: Confirmatory test set: 6 blind windows (no tuning against these).
TEST_WINDOWS: tuple[Window, ...] = TEST_EVENT_WINDOWS + TEST_CALM_WINDOWS

#: All 12 windows (dev + test).
ALL_WINDOWS_FULL: tuple[Window, ...] = DEV_WINDOWS + TEST_WINDOWS

#: Backward-compatible alias: points to the development set only.
ALL_WINDOWS: tuple[Window, ...] = DEV_WINDOWS


def get_window(name: str) -> Window:
    """Look up a window by name (searches dev + test sets)."""
    for w in ALL_WINDOWS_FULL:
        if w.name == name:
            return w
    raise KeyError(
        f"unknown window {name!r}; known dev: {[w.name for w in DEV_WINDOWS]}, "
        f"known test: {[w.name for w in TEST_WINDOWS]}"
    )
