import pandas as pd
import pytest

import universe


@pytest.fixture
def membership():
    """Small synthetic membership table.

    ALIVE: member the whole time.
    DEAD:  member 2005-01-01 .. 2008-06-30 (delisted).
    LATE:  joined 2010-01-01, still a member.
    BACK:  two intervals — 2004..2006 and re-entered 2012.
    """
    return pd.DataFrame({
        "ticker": ["ALIVE", "DEAD", "LATE", "BACK", "BACK"],
        "start_date": pd.to_datetime(
            ["2000-01-01", "2005-01-01", "2010-01-01", "2004-01-01", "2012-01-01"]),
        "end_date": pd.to_datetime(
            [pd.NaT, "2008-06-30", pd.NaT, "2006-12-31", pd.NaT]),
    })


def test_members_on(membership):
    assert universe.members_on("2005-06-01", membership) == {"ALIVE", "DEAD", "BACK"}
    assert universe.members_on("2009-01-01", membership) == {"ALIVE"}
    assert universe.members_on("2013-01-01", membership) == {"ALIVE", "LATE", "BACK"}


def test_members_on_boundary_dates_inclusive(membership):
    assert "DEAD" in universe.members_on("2005-01-01", membership)  # start inclusive
    assert "DEAD" in universe.members_on("2008-06-30", membership)  # end inclusive
    assert "DEAD" not in universe.members_on("2008-07-01", membership)


def test_tickers_active_between(membership):
    got = universe.tickers_active_between("2007-01-01", "2011-01-01", membership)
    # DEAD active until mid-2008, LATE joined 2010; BACK inactive 2007-2011
    assert got == ["ALIVE", "DEAD", "LATE"]


def test_membership_mask_and_apply(membership):
    dates = pd.DatetimeIndex(["2005-06-30", "2009-06-30", "2013-06-30"])
    cols = ["ALIVE", "DEAD", "LATE", "BACK"]
    scores = pd.DataFrame(1.0, index=dates, columns=cols)

    masked = universe.apply_membership(scores, membership)

    assert masked.loc["2005-06-30"].notna().sum() == 3   # ALIVE, DEAD, BACK
    assert pd.isna(masked.loc["2005-06-30", "LATE"])
    assert masked.loc["2009-06-30"].notna().sum() == 1   # only ALIVE
    assert pd.isna(masked.loc["2009-06-30", "DEAD"])     # delisted -> excluded
    assert masked.loc["2013-06-30"].notna().sum() == 3   # ALIVE, LATE, BACK (re-entry)
    # surviving scores unchanged
    assert masked.loc["2005-06-30", "ALIVE"] == 1.0


def test_reentry_gap_excluded(membership):
    dates = pd.DatetimeIndex(["2008-06-30"])
    scores = pd.DataFrame(1.0, index=dates, columns=["BACK"])
    masked = universe.apply_membership(scores, membership)
    assert pd.isna(masked.loc["2008-06-30", "BACK"])  # between its two intervals


def test_coverage_report(membership):
    dates = pd.DatetimeIndex(["2005-06-30"])
    # We only have score data for ALIVE — DEAD and BACK are members but uncovered.
    scores = pd.DataFrame({"ALIVE": [1.0], "DEAD": [None], "BACK": [None]}, index=dates)
    cov = universe.coverage_report(scores, membership)
    assert cov.loc[dates[0], "n_members"] == 3
    assert cov.loc[dates[0], "n_covered"] == 1
    assert cov.loc[dates[0], "coverage"] == pytest.approx(1 / 3)


def test_real_table_loads_and_is_sane():
    m = universe.load_membership()
    assert {"ticker", "start_date", "end_date"} <= set(m.columns)
    assert len(m) > 1000
    # roughly 500 members on a mid-sample date
    n = len(universe.members_on("2015-06-01", m))
    assert 480 <= n <= 520
