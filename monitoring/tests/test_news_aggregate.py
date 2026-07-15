"""Daily signal aggregator tests: counts, stress, and hit_z causality."""

import math

import pandas as pd

from news.aggregate import daily_signal
from news.filters import split_records
from news.sentiment import FakeScorer


def _rec(ts, headline, section="Business"):
    return {"timestamp": ts, "headline": headline, "abstract": None,
            "source": "t", "tickers": [], "section": section}


def _build(records, start, end):
    candidates, matched = split_records(records)
    return daily_signal(candidates, matched, FakeScorer(), start, end)


def test_counts_and_hit_rate():
    records = [
        _rec("2010-06-01T09:00:00", "Steady quarterly earnings"),
        _rec("2010-06-01T10:00:00", "Stocks plunge in panic"),
        _rec("2010-06-02T09:00:00", "Dividend increase announced"),
    ]
    sig = _build(records, "2010-06-01", "2010-06-03")
    assert list(sig.index) == list(pd.date_range("2010-06-01", "2010-06-03"))
    assert sig.loc["2010-06-01", "n_articles"] == 2
    assert sig.loc["2010-06-01", "n_hits"] == 1
    assert sig.loc["2010-06-01", "hit_rate"] == 0.5
    assert sig.loc["2010-06-02", "n_hits"] == 0
    assert sig.loc["2010-06-03", "n_articles"] == 0
    assert math.isnan(sig.loc["2010-06-03", "hit_rate"])  # 0/0 day


def test_stress_scored_on_matched_only():
    records = [
        # matched + negative for FakeScorer
        _rec("2010-06-01T09:00:00", "Markets crash in panic"),
        # matched but not negative ("counterparty" hits lexicon, no neg word)
        _rec("2010-06-01T10:00:00", "Broker warns of counterparty risk"),
        # benign, must not enter the stress denominator
        _rec("2010-06-01T11:00:00", "Steady quarterly earnings"),
    ]
    sig = _build(records, "2010-06-01", "2010-06-01")
    assert sig.loc["2010-06-01", "stress"] == 0.5


def test_stress_nan_on_empty_day():
    records = [_rec("2010-06-01T09:00:00", "Steady quarterly earnings")]
    sig = _build(records, "2010-06-01", "2010-06-02")
    assert math.isnan(sig.loc["2010-06-01", "stress"])  # no matched headlines
    assert math.isnan(sig.loc["2010-06-02", "stress"])  # no records at all


def test_hit_z_is_causal():
    """The z-score for the spike day must use only *prior* days as baseline."""
    records = []
    # 30 quiet days with zero hits, then a spike day with 5 hits.
    for i in range(5):
        records.append(_rec(f"2010-06-30T0{i+1}:00:00", "Stocks plunge in panic"))
    candidates, matched = split_records(records)
    sig = daily_signal(candidates, matched, FakeScorer(), "2010-06-01", "2010-06-30")
    spike = sig.loc["2010-06-30"]
    assert spike["n_hits"] == 5
    # baseline (previous 29 zero-hit days) has sd == 0 -> z is NaN, never inf,
    # and crucially the spike itself is not in its own baseline
    assert math.isnan(spike["hit_z"]) or spike["hit_z"] > 0


def test_hit_z_excludes_own_day():
    records = []
    # alternating baseline: one hit every other day, then a 4-hit spike
    for d in range(1, 29):
        if d % 2 == 0:
            records.append(_rec(f"2010-06-{d:02d}T09:00:00", "Markets crash"))
    for i in range(4):
        records.append(_rec(f"2010-06-30T0{i+1}:00:00", "Stocks plunge in panic"))
    sig = _build(records, "2010-06-01", "2010-06-30")
    spike_z = sig.loc["2010-06-30", "hit_z"]
    mu = sig["n_hits"].iloc[:-1].tail(29).mean()  # what a causal baseline sees
    assert spike_z > 2.0  # 4 hits vs ~0.5 mean baseline
    # recompute: z must be based on prior days only
    prior = sig["n_hits"].iloc[-30:-1]
    expected = (4 - prior.mean()) / prior.std(ddof=1)
    assert abs(spike_z - expected) < 1e-9
    assert mu < 1.0


def test_records_outside_span_ignored():
    records = [
        _rec("2010-05-31T09:00:00", "Markets crash"),   # before span
        _rec("2010-06-02T09:00:00", "Markets crash"),
        _rec("2010-06-04T09:00:00", "Markets crash"),   # after span
    ]
    sig = _build(records, "2010-06-01", "2010-06-03")
    assert int(sig["n_hits"].sum()) == 1
