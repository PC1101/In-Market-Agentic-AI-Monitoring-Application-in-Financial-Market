"""News pipeline orchestration: records -> T-1 cutoff -> filter -> signal -> triage.

``build_news_block`` is the single entry point the monitoring runner uses. The
returned block is what the News Context Agent is allowed to see for a decision
date, and it is causal by construction:

* **T-1 calendar-day cutoff** — only records published strictly before the
  decision date are used (``filter_news_by_timestamp`` at ``as_of - 1 day``).
  This matches the T-1 convention of the statsArb ``RegimeFilter``, is
  conservative for NYT's exact timestamps, correct for FNSPID's day-granularity
  midnight stamps, and uses calendar days so a Monday decision sees weekend news.
* **Fail-closed text guard** — any record whose text contains an ISO date after
  the cutoff is dropped and counted (never silently forwarded).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from agentic.guardrails import (
    LookaheadError,
    assert_no_lookahead,
    filter_news_by_timestamp,
)

from .aggregate import BASELINE_DAYS, daily_signal
from .filters import split_records
from .records import load_parquet_news
from .triage import RECENT_DAYS, TriageMode, triage

DEFAULT_DATA_DIR = Path(__file__).resolve().parent / "data"
DEFAULT_SOURCES = ("nyt_archive.parquet", "fnspid_windows.parquet")


@dataclass
class NewsConfig:
    """Configuration for building news blocks.

    ``scorer`` must provide ``daily_stress(list[str]) -> float`` — pass
    ``sentiment.FinBERTScorer()`` for real runs, ``sentiment.FakeScorer()``
    for offline/deterministic use.
    """

    scorer: object
    data_dir: Path = DEFAULT_DATA_DIR
    sources: tuple[str, ...] = DEFAULT_SOURCES
    lookback_days: int = BASELINE_DAYS + 15  # signal frame span before cutoff
    max_headlines: int = 25

    source_paths: tuple[Path, ...] = field(init=False)

    def __post_init__(self):
        self.data_dir = Path(self.data_dir)
        self.source_paths = tuple(self.data_dir / s for s in self.sources)


def _json_safe(x):
    if isinstance(x, float) and math.isnan(x):
        return None
    return x


def build_news_block(as_of, cfg: NewsConfig, records: list[dict] | None = None,
                     n_recent_alarms: int = 0,
                     aggregate_alarm: bool = False) -> dict:
    """Build the causal news block for a decision date.

    Args:
        as_of:   decision date (the agent decides on the morning of this date).
        cfg:     pipeline configuration.
        records: optional pre-loaded record list (tests / reuse across dates);
                 when None, records are loaded from ``cfg.source_paths``.
        n_recent_alarms / aggregate_alarm: classical-detector state visible as of
                 the decision date, used by triage only.

    Returns:
        JSON-serialisable dict: ``as_of``, ``cutoff``, ``triage_mode``,
        ``signal`` (latest day + recent aggregates), ``headlines`` (most recent
        matched, capped), ``coverage_note``.
    """
    as_of = pd.Timestamp(as_of)
    cutoff = as_of.normalize() - pd.Timedelta(days=1)          # T-1 calendar day
    cutoff_ts = as_of.normalize() - pd.Timedelta(seconds=1)    # end of T-1
    frame_start = cutoff - pd.Timedelta(days=cfg.lookback_days)

    if records is None:
        records = load_parquet_news(cfg.source_paths, start=frame_start, end=cutoff_ts)

    # Causal cut: everything published up to end-of-day T-1, nothing from day T.
    # (Idempotent for loaded records, essential for injected ones.)
    visible = filter_news_by_timestamp(records, cutoff_ts)

    # Fail-closed text guard: no future ISO date may ride in on a record's text.
    clean: list[dict] = []
    n_dropped_lookahead = 0
    for r in visible:
        try:
            assert_no_lookahead(
                {"headline": r.get("headline"), "abstract": r.get("abstract")}, cutoff
            )
            clean.append(r)
        except LookaheadError:
            n_dropped_lookahead += 1

    candidates, matched = split_records(clean)
    signal = daily_signal(candidates, matched, cfg.scorer, frame_start, cutoff)
    mode = triage(signal, n_recent_alarms=n_recent_alarms,
                  aggregate_alarm=aggregate_alarm)

    recent = signal.tail(RECENT_DAYS)
    last = signal.iloc[-1]
    signal_summary = {
        "last_day": str(signal.index[-1].date()),
        "n_articles_last_day": int(last["n_articles"]),
        "n_hits_last_day": int(last["n_hits"]),
        "stress_last_day": _json_safe(float(last["stress"])),
        "hit_z_last_day": _json_safe(float(last["hit_z"])),
        f"n_hits_{RECENT_DAYS}d": int(recent["n_hits"].sum()),
        f"max_stress_{RECENT_DAYS}d": _json_safe(
            float(recent["stress"].max()) if recent["stress"].notna().any() else math.nan
        ),
        "lookback_days": int(cfg.lookback_days),
        "lookback_n_articles": int(signal["n_articles"].sum()),
        "lookback_n_hits": int(signal["n_hits"].sum()),
    }

    matched_sorted = sorted(matched, key=lambda r: r["timestamp"])
    headlines = [
        {
            "date": str(pd.Timestamp(r["timestamp"]).date()),
            "headline": r["headline"],
            "categories": r["categories"],
            "source": r["source"],
        }
        for r in matched_sorted[-cfg.max_headlines:]
    ]

    coverage_note = (
        f"{cfg.lookback_days}-day lookback to {cutoff.date()} (T-1 of {as_of.date()}): "
        f"{signal_summary['lookback_n_articles']} articles, "
        f"{signal_summary['lookback_n_hits']} stress-keyword hits"
        + (f"; {n_dropped_lookahead} record(s) dropped by lookahead text guard"
           if n_dropped_lookahead else "")
    )

    block = {
        "as_of": str(as_of.date()),
        "cutoff": str(cutoff.date()),
        "triage_mode": mode.value,
        "signal": signal_summary,
        "headlines": headlines,
        "coverage_note": coverage_note,
    }
    # Final belt-and-braces: nothing in the block may post-date the decision date
    # (headline timestamps are additionally <= cutoff by construction above).
    assert_no_lookahead(block, as_of)
    return block
