"""Energy evaluation windows: well-formed, onset-in-range, GDELT-era coverage."""
import pandas as pd

from providers.energy import windows as ew


def test_windows_are_wellformed_and_unique():
    names = [w.name for w in ew.ALL_WINDOWS]
    assert len(names) == len(set(names)), "window names must be unique"
    assert len(ew.EVENT_WINDOWS) >= 2 and len(ew.CALM_WINDOWS) >= 2


def test_event_windows_have_onset_in_range():
    for w in ew.EVENT_WINDOWS:
        assert w.onset is not None
        assert w.start_ts <= w.onset_ts <= w.end_ts


def test_calm_windows_have_no_onset():
    for w in ew.CALM_WINDOWS:
        assert w.onset is None


def test_all_windows_within_gdelt_doc_coverage():
    # DOC 2.0 covers ~2017+; every window must start in-range for agentic news.
    for w in ew.ALL_WINDOWS:
        assert w.start_ts >= pd.Timestamp("2017-01-01"), f"{w.name} predates GDELT DOC 2.0"
