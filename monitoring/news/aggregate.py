"""Quantitative signal aggregator — daily news-stress table.

Turns the filtered record stream into a per-calendar-day signal frame. Sentiment
is scored on *matched* (keyword-hit) headlines only: on dense days this keeps the
FinBERT batch small, and stress over stress-candidate stories is the quantity the
triage thresholds are written against.

Causality: ``hit_z`` for day *t* is computed against the trailing baseline of the
*previous* ``baseline_days`` days (day *t* excluded), so the z-score never sees
its own day, let alone the future. The frame itself must only ever be built from
records already cut off at T-1 by ``pipeline.build_news_block``.
"""

from __future__ import annotations

import math

import pandas as pd

BASELINE_DAYS = 30
_MIN_BASELINE_OBS = 5


def daily_signal(candidates: list[dict], matched: list[dict], scorer,
                 start, end, baseline_days: int = BASELINE_DAYS) -> pd.DataFrame:
    """Build the daily signal frame over calendar days [start, end].

    Args:
        candidates: records forming the daily article-count denominator.
        matched:    keyword-hit records (subset of candidates).
        scorer:     object with ``daily_stress(list[str]) -> float`` (FinBERT or Fake).
        start/end:  inclusive calendar-day bounds.

    Returns:
        DataFrame indexed by normalized daily DatetimeIndex with columns
        ``n_articles``, ``n_hits``, ``hit_rate``, ``stress``, ``hit_z``.
    """
    days = pd.date_range(pd.Timestamp(start).normalize(),
                         pd.Timestamp(end).normalize(), freq="D")

    def _by_day(records):
        out: dict[pd.Timestamp, list[dict]] = {}
        for r in records:
            day = pd.Timestamp(r["timestamp"]).normalize()
            if days[0] <= day <= days[-1]:
                out.setdefault(day, []).append(r)
        return out

    cand_by_day = _by_day(candidates)
    match_by_day = _by_day(matched)

    n_articles = pd.Series({d: len(cand_by_day.get(d, [])) for d in days})
    n_hits = pd.Series({d: len(match_by_day.get(d, [])) for d in days})
    hit_rate = (n_hits / n_articles.astype(float).replace(0.0, math.nan)).astype(float)
    stress = pd.Series(
        {d: scorer.daily_stress([r["headline"] for r in match_by_day.get(d, [])])
         for d in days},
        dtype=float,
    )

    # Causal z-score of n_hits vs the previous `baseline_days` days.
    base = n_hits.shift(1).rolling(baseline_days, min_periods=_MIN_BASELINE_OBS)
    mu, sd = base.mean(), base.std(ddof=1)
    hit_z = ((n_hits - mu) / sd.replace(0.0, math.nan)).astype(float)

    return pd.DataFrame({
        "n_articles": n_articles.astype(int),
        "n_hits": n_hits.astype(int),
        "hit_rate": hit_rate,
        "stress": stress,
        "hit_z": hit_z,
    }, index=days)
