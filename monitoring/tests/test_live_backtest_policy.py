"""Tests for live_backtest.policy — exposure ladder, cooldown, re-entry."""
import json
from pathlib import Path

import pandas as pd
import pytest

from live_backtest.policy import (
    PolicyConfig, load_decisions, build_exposure, _target_exposure_single,
    STATE_RANK, ACTION_RANK,
)


# ---------------------------------------------------------------------------
# target_exposure_single
# ---------------------------------------------------------------------------

class TestTargetExposure:
    cfg = PolicyConfig()

    def test_normal_hold(self):
        assert _target_exposure_single("NORMAL", "HOLD", self.cfg) == 1.0

    def test_watch_investigate(self):
        assert _target_exposure_single("WATCH", "INVESTIGATE", self.cfg) == 1.0

    def test_alert_investigate(self):
        assert _target_exposure_single("ALERT", "INVESTIGATE", self.cfg) == 0.50

    def test_alert_reduce(self):
        assert _target_exposure_single("ALERT", "REDUCE", self.cfg) == 0.25

    def test_critical_any(self):
        assert _target_exposure_single("CRITICAL", "HOLD", self.cfg) == 0.0

    def test_any_halt(self):
        assert _target_exposure_single("WATCH", "HALT", self.cfg) == 0.0

    def test_unknown_state_treated_normal(self):
        assert _target_exposure_single("UNKNOWN", "HOLD", self.cfg) == 1.0

    def test_unknown_action_state_fallback(self):
        # ALERT with unknown action -> exposure_alert (action rank 0)
        assert _target_exposure_single("ALERT", "UNKNOWN", self.cfg) == 0.50


# ---------------------------------------------------------------------------
# load_decisions
# ---------------------------------------------------------------------------

class TestLoadDecisions:
    def _write_jsonl(self, tmp_path, records):
        p = tmp_path / "test.jsonl"
        with open(p, "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")
        return p

    def test_parses_supervisor_records(self, tmp_path):
        records = [
            {"agent": "triage", "as_of": "2020-01-01"},
            {"agent": "performance_supervisor", "as_of": "2020-01-02",
             "assessment": {"state": "ALERT", "action": "REDUCE", "confidence": 0.9}},
            {"agent": "performance_supervisor", "as_of": "2020-01-03",
             "assessment": {"state": "WATCH", "action": "INVESTIGATE", "confidence": 0.7}},
        ]
        p = self._write_jsonl(tmp_path, records)
        df = load_decisions([p])
        assert len(df) == 2
        assert df.loc[pd.Timestamp("2020-01-02"), "state"] == "ALERT"

    def test_duplicate_dates_keeps_most_severe(self, tmp_path):
        records = [
            {"agent": "performance_supervisor", "as_of": "2020-01-02",
             "assessment": {"state": "WATCH", "action": "HOLD", "confidence": 0.5}},
            {"agent": "performance_supervisor", "as_of": "2020-01-02",
             "assessment": {"state": "ALERT", "action": "REDUCE", "confidence": 0.9}},
        ]
        p = self._write_jsonl(tmp_path, records)
        df = load_decisions([p])
        assert len(df) == 1
        assert df.iloc[0]["state"] == "ALERT"

    def test_malformed_lines_skipped(self, tmp_path):
        p = tmp_path / "bad.jsonl"
        with open(p, "w") as f:
            f.write("not valid json\n")
            f.write(json.dumps({
                "agent": "performance_supervisor", "as_of": "2020-01-02",
                "assessment": {"state": "WATCH", "action": "HOLD", "confidence": 0.5},
            }) + "\n")
        df = load_decisions([p])
        assert len(df) == 1

    def test_empty_paths(self):
        df = load_decisions([])
        assert df.empty

    def test_nonexistent_path(self, tmp_path):
        df = load_decisions([tmp_path / "does_not_exist.jsonl"])
        assert df.empty


# ---------------------------------------------------------------------------
# build_exposure
# ---------------------------------------------------------------------------

class TestBuildExposure:
    def test_no_decisions_all_normal(self):
        days = pd.bdate_range("2020-01-01", periods=10)
        decisions = pd.DataFrame(columns=["state", "action", "confidence"])
        exp = build_exposure(days, decisions)
        assert (exp == 1.0).all()

    def test_alert_reduces_exposure(self):
        days = pd.bdate_range("2020-01-01", periods=20)
        decisions = pd.DataFrame([
            {"state": "ALERT", "action": "INVESTIGATE", "confidence": 0.8},
        ], index=[days[5]])
        decisions.index.name = "date"
        exp = build_exposure(days, decisions, PolicyConfig(cooldown_days=0,
                                                           lag_days=0,
                                                           reentry_steps=()))
        assert exp.iloc[5] == 0.50

    def test_lag_shifts_exposure(self):
        days = pd.bdate_range("2020-01-01", periods=20)
        decisions = pd.DataFrame([
            {"state": "ALERT", "action": "INVESTIGATE", "confidence": 0.8},
        ], index=[days[5]])
        decisions.index.name = "date"
        cfg = PolicyConfig(cooldown_days=0, lag_days=1, reentry_steps=())
        exp = build_exposure(days, decisions, cfg)
        # Day 5 should still be 1.0 (lag); day 6 should be reduced
        assert exp.iloc[5] == 1.0
        assert exp.iloc[6] == 0.50

    def test_cooldown_holds_min_exposure(self):
        days = pd.bdate_range("2020-01-01", periods=20)
        decisions = pd.DataFrame([
            {"state": "ALERT", "action": "REDUCE", "confidence": 0.9},
        ], index=[days[3]])
        decisions.index.name = "date"
        cfg = PolicyConfig(cooldown_days=3, lag_days=0, reentry_steps=())
        exp = build_exposure(days, decisions, cfg)
        # Days 3,4,5,6 should all be 0.25 (trigger + 3 cooldown)
        for i in [3, 4, 5, 6]:
            assert exp.iloc[i] == 0.25
