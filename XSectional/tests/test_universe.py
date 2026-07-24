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


def test_crisis_era_removals():
    """Regression: crisis-era delistings must have correct end_dates (survivorship-bias fix)."""
    m = universe.load_membership()

    # AIG removed 2008-09-22 (govt bailout; S&P index removal)
    aig = m[m["ticker"] == "AIG"]
    assert len(aig) >= 1, "AIG must appear in membership table"
    aig_row = aig.sort_values("start_date").iloc[0]
    assert pd.notna(aig_row["end_date"]), "AIG must have an end_date (removed Sept 2008)"
    assert aig_row["end_date"] == pd.Timestamp("2008-09-22")
    assert "AIG" in universe.members_on("2008-09-01", m)
    assert "AIG" not in universe.members_on("2008-09-23", m)

    # GM: two membership intervals — original ending bankruptcy removal, re-added 2013
    gm = m[m["ticker"] == "GM"].sort_values("start_date").reset_index(drop=True)
    assert len(gm) >= 2, "GM must have at least 2 membership intervals (pre-2009 + re-IPO)"
    assert pd.notna(gm.iloc[0]["end_date"]), "GM original interval must have end_date"
    assert gm.iloc[0]["end_date"] == pd.Timestamp("2009-06-08")
    assert "GM" in universe.members_on("2008-01-01", m)    # original membership
    assert "GM" not in universe.members_on("2010-06-01", m)  # gap (bankrupt/delisted)
    assert "GM" in universe.members_on("2014-01-01", m)    # re-addition

    # WM removed ~2008-09-30 (FDIC seizure Sept 25, 2008)
    wm = m[m["ticker"] == "WM"]
    assert len(wm) >= 1, "WM must appear in membership table"
    wm_row = wm.sort_values("start_date").iloc[0]
    assert pd.notna(wm_row["end_date"]), "WM must have an end_date (removed Sept 2008)"
    assert wm_row["end_date"] == pd.Timestamp("2008-09-30")
    assert "WM" in universe.members_on("2008-09-01", m)
    assert "WM" not in universe.members_on("2008-10-01", m)
