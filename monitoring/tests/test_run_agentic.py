"""End-to-end agentic loop on synthetic data with the offline stub."""
import json

import numpy as np
import pandas as pd
import pytest

from run_agentic import run_window, _log_run_meta
from agentic.model import OfflineStubModel, DeadRunnerError
from agentic.logging_utils import RunLogger
from agentic.schemas import validate_assessment
from news.store import build_store, NewsStore
from windows import Window


def _synthetic(tmp_path):
    # 150 calm days then a 10-day crash inside the window
    rng = np.random.default_rng(7)
    dates = pd.bdate_range("2007-01-02", periods=160)
    rets = rng.normal(0.0003, 0.004, 160)
    rets[150:] = rng.normal(-0.03, 0.02, 10)
    series = pd.Series(rets, index=dates)

    raw = pd.DataFrame({
        "Date": [f"{d.date()} 10:00:00 UTC" for d in dates[145:155]],
        "Article_title": (["Ordinary market story"] * 5 +
                          ["Meltdown: quant funds hit by margin calls and liquidation"] * 5),
        "Stock_symbol": [""] * 10, "Lsa_summary": [""] * 10,
        "Publisher": ["P"] * 10, "Url": ["u"] * 10,
    })
    raw_path = tmp_path / "raw.csv"
    raw.to_csv(raw_path, index=False)
    build_store(raw_path, tmp_path / "store", start="2003-01-01", end="2016-12-31")

    window = Window(name="synthetic_event", kind="event",
                    start=str(dates[140].date()), end=str(dates[-1].date()),
                    onset=str(dates[150].date()), description="synthetic crash")
    return series, NewsStore(tmp_path / "store"), window


def test_run_meta_header_structure(tmp_path):
    logger = RunLogger(tmp_path / "meta_test.jsonl")
    _log_run_meta(logger, model_name="stub", curve="/some/path.csv",
                  window="quant_meltdown_2007", strategy="AL_PCA", condition="A")
    lines = (tmp_path / "meta_test.jsonl").read_text().splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["agent"] == "run_meta"
    assert record["model"] == "stub"
    assert record["curve"] == "/some/path.csv"
    assert record["window"] == "quant_meltdown_2007"
    assert record["strategy"] == "AL_PCA"
    assert record["condition"] == "A"
    assert "git_sha" in record  # value may be "unknown" in CI; just check key exists


def test_run_window_end_to_end_stub(tmp_path):
    series, store, window = _synthetic(tmp_path)
    logger = RunLogger(tmp_path / "log.jsonl")
    records = run_window(series, window, store, OfflineStubModel(),
                         logger=logger, macro_dir=tmp_path / "no_macro")
    assert records, "expected at least one triage record"
    modes = {r["triage_mode"] for r in records}
    assert "skip" in modes                       # calm days skipped
    assert modes & {"cheap", "thinking", "classical_escalation"}  # crash escalates

    # every non-skip day produced a schema-valid supervisor assessment
    assessed = [r for r in records if r["triage_mode"] != "skip"]
    assert assessed
    for r in assessed:
        validate_assessment(r["assessment"])     # raises on invalid
        assert r["assessment"]["as_of"] <= window.end

    # the JSONL log is replayable
    lines = [json.loads(l) for l in (tmp_path / "log.jsonl").read_text().splitlines()]
    assert any(rec["agent"] == "news_context" for rec in lines)
    assert any(rec["agent"] == "performance_supervisor" for rec in lines)


def test_resume_skip_dates_strict(tmp_path):
    """Day-level resume: only skip-triage and supervisor days count as done.
    Bare triage lines (aborted mid-call) must be reprocessed."""
    jsonl = tmp_path / "resume_test.jsonl"
    lines = [
        # run_meta — never counted
        json.dumps({"agent": "run_meta", "model": "stub"}),
        # day A: skip-triage → DONE
        json.dumps({"agent": "triage", "as_of": "2020-01-02", "triage_mode": "skip"}),
        # day B: triage + supervisor → DONE
        json.dumps({"agent": "triage", "as_of": "2020-01-03", "triage_mode": "thinking"}),
        json.dumps({"agent": "performance_supervisor", "as_of": "2020-01-03"}),
        # day C: bare triage only (aborted) → NOT DONE
        json.dumps({"agent": "triage", "as_of": "2020-01-06", "triage_mode": "thinking"}),
        # day D: triage + news_context but no supervisor (aborted) → NOT DONE
        json.dumps({"agent": "triage", "as_of": "2020-01-07", "triage_mode": "classical_escalation"}),
        json.dumps({"agent": "news_context", "as_of": "2020-01-07"}),
    ]
    jsonl.write_text("\n".join(lines) + "\n")

    # Simulate the resume scan from run_agentic.py main()
    _supervisor_dates: set[str] = set()
    _skip_triage_dates: set[str] = set()
    with open(jsonl) as fh:
        for line in fh:
            rec = json.loads(line)
            agent = rec.get("agent", "")
            d = rec.get("as_of") or rec.get("date")
            if not d or agent == "run_meta":
                continue
            if agent == "performance_supervisor":
                _supervisor_dates.add(d)
            elif agent == "triage" and rec.get("triage_mode") == "skip":
                _skip_triage_dates.add(d)
    skip_dates = _supervisor_dates | _skip_triage_dates

    assert "2020-01-02" in skip_dates   # skip-triage day
    assert "2020-01-03" in skip_dates   # supervisor day
    assert "2020-01-06" not in skip_dates  # bare triage
    assert "2020-01-07" not in skip_dates  # triage+news but no supervisor


class _DeadRunnerModel(OfflineStubModel):
    """Stub that raises DeadRunnerError on every complete() call."""
    def complete(self, system, user, json_schema=None):
        raise DeadRunnerError("simulated dead runner")


def test_dead_runner_error_skips_day(tmp_path):
    """DeadRunnerError is caught at the day level — the day is skipped with an
    error recorded, and the run continues rather than aborting."""
    series, store, window = _synthetic(tmp_path)
    logger = RunLogger(tmp_path / "log.jsonl")
    records = run_window(series, window, store, _DeadRunnerModel(),
                         logger=logger, macro_dir=tmp_path / "no_macro")
    # Days that would have called the model should have error entries
    errors = [r for r in records if "news_error" in r or "supervisor_error" in r]
    assert len(errors) > 0, "expected at least one day with a dead_runner error"
    assert any("dead_runner" in (r.get("news_error") or "") for r in errors)
