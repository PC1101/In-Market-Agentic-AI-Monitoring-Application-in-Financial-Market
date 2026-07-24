"""Tests for the pre-freeze checklist verifier (preregistration §9 + §11)."""
import pytest

from freeze_gate import (
    check_significance_importable,
    check_leakage_importable,
    check_alarm_extraction_importable,
    check_prompt_versions,
    check_triage_constants,
    check_model_default,
    check_window_partition,
    check_freeze_readiness,
)


def test_significance_importable():
    r = check_significance_importable()
    assert r["passed"] is True


def test_leakage_importable():
    r = check_leakage_importable()
    assert r["passed"] is True


def test_alarm_extraction_importable():
    r = check_alarm_extraction_importable()
    assert r["passed"] is True


def test_prompt_versions_match():
    r = check_prompt_versions()
    assert r["passed"] is True, r.get("detail")


def test_triage_constants_match():
    r = check_triage_constants()
    assert r["passed"] is True, r.get("detail")


def test_model_default_documented():
    r = check_model_default()
    # Must pass — either exact match or documented deviation
    assert r["passed"] is True, r.get("detail")


def test_window_partition_disjoint():
    r = check_window_partition()
    assert r["passed"] is True, r.get("detail")


def test_freeze_readiness_returns_dict():
    r = check_freeze_readiness()
    assert "passed" in r
    assert "checks" in r
    assert isinstance(r["checks"], dict)


def test_freeze_readiness_check_names():
    r = check_freeze_readiness()
    expected = {
        "significance.py importable",
        "leakage harness importable",
        "alarm_extraction importable",
        "prompt versions match §6.2",
        "triage constants match §6.2",
        "model default documented",
        "dev/test window partition",
    }
    # All expected check names must be present
    actual = set(r["checks"].keys())
    assert expected.issubset(actual)


def test_freeze_passed_field_consistent():
    r = check_freeze_readiness()
    computed = all(v["passed"] for v in r["checks"].values())
    assert r["passed"] == computed
