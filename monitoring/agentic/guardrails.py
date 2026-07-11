"""Information-parity guardrails.

The central methodological requirement of this project: the agentic monitor, evaluated
at a decision date ``as_of``, may only see information available on or before ``as_of``.
These helpers build agent context from causal slices and assert that nothing in the
context post-dates the decision date. They are the framework-level enforcement of the
"no look-ahead / no hindsight" spine described in CLAUDE.md.
"""

from __future__ import annotations

import re

import pandas as pd

_ISO_DATE_RE = re.compile(r"\d{4}-\d{2}-\d{2}")


class LookaheadError(AssertionError):
    """Raised when context contains information dated after the decision date."""


def as_of_context(returns: pd.Series, as_of, detector_alarms=None,
                  lookback: int = 60) -> dict:
    """Assemble the causal context an agent is allowed to see at ``as_of``.

    Args:
        returns: full daily return series (indexed by date).
        as_of:   decision date; only data on/before this date is included.
        detector_alarms: optional mapping detector_name -> list[Timestamp]; alarms
                         after ``as_of`` are dropped.
        lookback: number of trailing observations to summarise in the telemetry block.

    Returns:
        A JSON-serialisable context dict with as_of, telemetry summary, and the
        classical-detector firings visible so far.
    """
    as_of = pd.Timestamp(as_of)
    visible = returns.loc[:as_of].astype(float)
    if visible.empty:
        raise ValueError(f"no return data on or before {as_of.date()}")

    recent = visible.tail(lookback)
    equity_visible = (1.0 + visible).cumprod()
    running_peak = equity_visible.cummax()
    drawdown = equity_visible.iloc[-1] / running_peak.iloc[-1] - 1.0

    telemetry = {
        "last_date": str(visible.index[-1].date()),
        "n_obs": int(len(visible)),
        "recent_mean_daily_return": float(recent.mean()),
        "recent_daily_vol": float(recent.std(ddof=1)) if len(recent) > 1 else 0.0,
        "recent_cum_return": float((1.0 + recent).prod() - 1.0),
        "current_drawdown": float(drawdown),
        "recent_worst_day": float(recent.min()),
    }

    visible_alarms: dict[str, list[str]] = {}
    if detector_alarms:
        for name, alarms in detector_alarms.items():
            visible_alarms[name] = [
                str(pd.Timestamp(a).date()) for a in alarms if pd.Timestamp(a) <= as_of
            ]

    context = {
        "as_of": str(as_of.date()),
        "telemetry": telemetry,
        "detector_alarms": visible_alarms,
    }
    assert_no_lookahead(context, as_of)
    return context


def assert_no_lookahead(context: dict, as_of) -> None:
    """Verify no date anywhere in ``context`` exceeds ``as_of``.

    Recursively scans strings that parse as ISO dates and any pandas Timestamps.
    """
    as_of = pd.Timestamp(as_of)

    def _scan(node, path="context"):
        if isinstance(node, dict):
            for k, v in node.items():
                _scan(v, f"{path}.{k}")
        elif isinstance(node, (list, tuple)):
            for i, v in enumerate(node):
                _scan(v, f"{path}[{i}]")
        elif isinstance(node, pd.Timestamp):
            if node > as_of:
                raise LookaheadError(f"{path}={node} post-dates as_of={as_of}")
        elif isinstance(node, str):
            # Catch both a bare ISO date and any ISO date embedded in free text
            # (e.g. a news snippet), so no future-dated string leaks through.
            for match in _ISO_DATE_RE.findall(node):
                ts = _try_date(match)
                if ts is not None and ts > as_of:
                    raise LookaheadError(
                        f"{path}={node!r} contains {match} which post-dates "
                        f"as_of={as_of.date()}"
                    )

    _scan(context)


def _try_date(s: str):
    try:
        return pd.Timestamp(s)
    except (ValueError, TypeError):
        return None


def filter_news_by_timestamp(records, as_of, ts_field: str = "timestamp") -> list:
    """Return only news records published on or before ``as_of`` (Week-3 news pipeline).

    Args:
        records: iterable of dicts each carrying a publication timestamp.
        as_of:   decision date/time.
        ts_field: key holding the publication timestamp.

    Returns:
        List of records with ts_field <= as_of. Records missing/with unparseable
        timestamps are dropped (fail-closed — never leak an undated item forward).
    """
    as_of = pd.Timestamp(as_of)
    kept = []
    for r in records:
        ts = _try_date(str(r.get(ts_field, "")))
        if ts is not None and ts <= as_of:
            kept.append(r)
    return kept
