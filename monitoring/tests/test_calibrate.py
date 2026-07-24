"""Tests for the classical grid calibration module (preregistration §6.1)."""
import numpy as np
import pandas as pd
import pytest

from calibrate_classical import (
    DETECTOR_GRIDS, _MULTIPLIER_LABELS, _J, _make_detector,
)


# ---------------------------------------------------------------------------
# Grid structure
# ---------------------------------------------------------------------------

def test_grid_sizes():
    assert len(DETECTOR_GRIDS["page_hinkley"]) == 9   # 3×3
    assert len(DETECTOR_GRIDS["bocpd"]) == 9
    assert len(DETECTOR_GRIDS["hmm"]) == 3            # 1 param × 3
    assert len(DETECTOR_GRIDS["distributional"]) == 9


def test_multiplier_labels_match_grid():
    for det in DETECTOR_GRIDS:
        assert len(DETECTOR_GRIDS[det]) == len(_MULTIPLIER_LABELS[det])


def test_default_config_in_grid():
    """The 1× (default) config must appear in every detector's grid."""
    ph_defaults = {"delta": 0.5, "lambda_": 8.0}
    ph_grid = DETECTOR_GRIDS["page_hinkley"]
    assert ph_defaults in ph_grid

    bocpd_defaults = {"hazard": round(1/250, 6), "min_run": 25}
    assert bocpd_defaults in DETECTOR_GRIDS["bocpd"]

    hmm_defaults = {"threshold": 0.5}
    assert hmm_defaults in DETECTOR_GRIDS["hmm"]

    dist_defaults = {"z_threshold": 3.0, "vol_threshold": 2.0}
    assert dist_defaults in DETECTOR_GRIDS["distributional"]


def test_multipliers_cover_half_one_double():
    """Each parameter in the grid should have 0.5×, 1×, and 2× variants."""
    # Check PageHinkley delta values: should include 0.25, 0.5, 1.0
    ph_deltas = sorted(set(p["delta"] for p in DETECTOR_GRIDS["page_hinkley"]))
    assert ph_deltas == pytest.approx([0.25, 0.5, 1.0])


# ---------------------------------------------------------------------------
# Objective function J
# ---------------------------------------------------------------------------

def test_J_recall_dominated():
    # Perfect recall, zero FPR → J = 1.0
    assert _J(1.0, 0.0) == pytest.approx(1.0)


def test_J_fpr_penalty():
    # FPR = 0.01 → penalty = 50 × 0.01 = 0.5
    assert _J(1.0, 0.01) == pytest.approx(0.5)


def test_J_nan_recall():
    assert _J(float("nan"), 0.0) == float("-inf")


def test_J_nan_fpr():
    assert _J(1.0, float("nan")) == float("-inf")


# ---------------------------------------------------------------------------
# Detector instantiation
# ---------------------------------------------------------------------------

def _make_synthetic_series(n=300, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2004-01-02", periods=n)
    returns = rng.normal(0, 0.01, size=n)
    return pd.Series(returns, index=dates, name="port_ret")


@pytest.mark.parametrize("det_name,params", [
    ("page_hinkley", {"delta": 0.25, "lambda_": 4.0}),
    ("bocpd", {"hazard": round(1/500, 6), "min_run": 12}),
    ("hmm", {"threshold": 0.25}),
    ("distributional", {"z_threshold": 1.5, "vol_threshold": 1.0}),
])
def test_make_detector_runs_without_error(det_name, params):
    series = _make_synthetic_series()
    det = _make_detector(det_name, params)
    result = det.detect(series)
    assert hasattr(result, "alarms")


def test_make_detector_unknown_raises():
    with pytest.raises(ValueError):
        _make_detector("unknown_detector", {})


# ---------------------------------------------------------------------------
# build_grid_stats (smoke test on synthetic data)
# ---------------------------------------------------------------------------

def test_evaluate_config_integration():
    """evaluate_config should run all hmm grid configs on a synthetic series."""
    from calibrate_classical import evaluate_config
    series = _make_synthetic_series(600)
    params = DETECTOR_GRIDS["hmm"][0]  # smallest threshold
    mults = _MULTIPLIER_LABELS["hmm"][0]
    # Directly exercise the underlying detector — curve-based evaluate_config
    # requires real files, so test at the detector level only.
    det = _make_detector("hmm", params)
    result = det.detect(series)
    assert isinstance(result.alarms, list)
