import pytest

from windows import (
    ALL_WINDOWS, ALL_WINDOWS_FULL, DEV_WINDOWS, TEST_WINDOWS,
    EVENT_WINDOWS, CALM_WINDOWS,
    TEST_EVENT_WINDOWS, TEST_CALM_WINDOWS, TEST_STRATEGY_WINDOWS,
    get_window, Window,
)


# ---------------------------------------------------------------------------
# Development-set tests (backward-compat with original 6-window design)
# ---------------------------------------------------------------------------

def test_dev_set_counts():
    assert len(DEV_WINDOWS) == 6
    assert len(EVENT_WINDOWS) == 4
    assert len(CALM_WINDOWS) == 2


def test_all_windows_alias_is_dev_windows():
    assert ALL_WINDOWS == DEV_WINDOWS


def test_event_windows_have_onset_inside_range():
    for w in EVENT_WINDOWS:
        assert w.onset is not None
        assert w.start_ts <= w.onset_ts <= w.end_ts


def test_calm_windows_have_no_onset():
    for w in CALM_WINDOWS:
        assert w.onset is None


def test_dev_names_unique():
    names = [w.name for w in DEV_WINDOWS]
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


# ---------------------------------------------------------------------------
# Confirmatory test-set tests
# ---------------------------------------------------------------------------

def test_test_set_counts():
    assert len(TEST_WINDOWS) == 6
    assert len(TEST_EVENT_WINDOWS) == 4
    assert len(TEST_CALM_WINDOWS) == 2


def test_test_event_windows_have_onset():
    for w in TEST_EVENT_WINDOWS:
        assert w.onset is not None
        assert w.start_ts <= w.onset_ts <= w.end_ts


def test_test_calm_windows_have_no_onset():
    for w in TEST_CALM_WINDOWS:
        assert w.onset is None


def test_expected_test_window_names():
    names = {w.name for w in TEST_WINDOWS}
    assert "flash_crash_2010" in names
    assert "china_deval_2015" in names
    assert "volmageddon_2018" in names
    assert "covid_2020" in names
    assert "calm_2012" in names
    assert "calm_2017" in names


# ---------------------------------------------------------------------------
# Full-set and partition integrity tests
# ---------------------------------------------------------------------------

def test_all_windows_full_count():
    assert len(ALL_WINDOWS_FULL) == 12


def test_no_name_overlap_between_dev_and_test():
    dev_names = {w.name for w in DEV_WINDOWS}
    test_names = {w.name for w in TEST_WINDOWS}
    overlap = dev_names & test_names
    assert not overlap, f"dev/test name overlap: {overlap}"


def test_all_names_unique_full():
    names = [w.name for w in ALL_WINDOWS_FULL]
    assert len(names) == len(set(names))


def test_get_window_finds_dev_windows():
    for w in DEV_WINDOWS:
        found = get_window(w.name)
        assert found.name == w.name


def test_get_window_finds_test_windows():
    for w in TEST_WINDOWS:
        found = get_window(w.name)
        assert found.name == w.name


def test_get_window_raises_for_unknown():
    with pytest.raises(KeyError):
        get_window("this_window_does_not_exist")


# ---------------------------------------------------------------------------
# Strategy-window mapping
# ---------------------------------------------------------------------------

def test_strategy_windows_keys():
    assert "AL_PCA" in TEST_STRATEGY_WINDOWS
    assert "JT_MOM" in TEST_STRATEGY_WINDOWS


def test_strategy_windows_all_exist():
    for strategy, names in TEST_STRATEGY_WINDOWS.items():
        for name in names:
            w = get_window(name)
            assert w.name == name, f"{strategy}: {name} not found"


def test_jt_has_more_windows_than_al():
    # JT covers beyond 2014 (volmageddon_2018, covid_2020, calm_2017)
    assert len(TEST_STRATEGY_WINDOWS["JT_MOM"]) > len(TEST_STRATEGY_WINDOWS["AL_PCA"])


def test_al_pca_windows_subset_of_jt():
    al = set(TEST_STRATEGY_WINDOWS["AL_PCA"])
    jt = set(TEST_STRATEGY_WINDOWS["JT_MOM"])
    assert al.issubset(jt), f"AL_PCA windows not a subset of JT_MOM: {al - jt}"


# ---------------------------------------------------------------------------
# Date ordering within test windows
# ---------------------------------------------------------------------------

def test_test_windows_start_before_end():
    for w in TEST_WINDOWS:
        assert w.start_ts < w.end_ts, f"{w.name}: start >= end"


def test_test_windows_no_overlap_with_dev():
    """Test windows should not temporally overlap with dev windows."""
    for tw in TEST_WINDOWS:
        for dw in DEV_WINDOWS:
            overlap = tw.start_ts <= dw.end_ts and tw.end_ts >= dw.start_ts
            assert not overlap, (
                f"Temporal overlap: test window {tw.name} "
                f"[{tw.start}, {tw.end}] overlaps dev window {dw.name} "
                f"[{dw.start}, {dw.end}]"
            )
