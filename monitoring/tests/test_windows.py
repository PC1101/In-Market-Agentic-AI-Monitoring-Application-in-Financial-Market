import pytest

from windows import ALL_WINDOWS, EVENT_WINDOWS, CALM_WINDOWS, get_window, Window


def test_six_windows_four_event_two_calm():
    assert len(ALL_WINDOWS) == 6
    assert len(EVENT_WINDOWS) == 4
    assert len(CALM_WINDOWS) == 2


def test_event_windows_have_onset_inside_range():
    for w in EVENT_WINDOWS:
        assert w.onset is not None
        assert w.start_ts <= w.onset_ts <= w.end_ts


def test_calm_windows_have_no_onset():
    for w in CALM_WINDOWS:
        assert w.onset is None


def test_names_unique():
    names = [w.name for w in ALL_WINDOWS]
    assert len(names) == len(set(names))


def test_contains():
    w = get_window("quant_meltdown_2007")
    assert w.contains("2007-08-06")
    assert not w.contains("2007-06-01")


def test_invalid_event_without_onset_rejected():
    with pytest.raises(ValueError):
        Window("bad", "event", "2007-01-01", "2007-02-01", None, "no onset")


def test_onset_outside_range_rejected():
    with pytest.raises(ValueError):
        Window("bad", "event", "2007-01-01", "2007-02-01", "2007-03-01", "onset outside")
