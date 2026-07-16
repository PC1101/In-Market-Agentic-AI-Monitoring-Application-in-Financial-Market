"""Point-in-time queries over the fetched macro parquets."""

from __future__ import annotations

import pandas as pd


def asof_vintage(df: pd.DataFrame, as_of) -> dict | None:
    """Latest observation *as it was known* at ``as_of`` (ALFRED vintage frame).

    Returns {"value", "obs_date", "published"} or None if nothing was
    published yet. Uses the newest vintage whose realtime_start <= as_of for
    the newest observation date published by then — i.e. first prints before
    their revisions arrive, revisions once they do.
    """
    as_of = pd.Timestamp(as_of)
    known = df[df["realtime_start"] <= as_of].dropna(subset=["value"])
    if known.empty:
        return None
    latest_obs = known["date"].max()
    row = (known[known["date"] == latest_obs]
           .sort_values("realtime_start").iloc[-1])
    return {"value": float(row["value"]),
            "obs_date": str(row["date"].date()),
            "published": str(row["realtime_start"].date())}


def asof_daily(df: pd.DataFrame, as_of) -> dict | None:
    """Last non-null daily print dated on/before ``as_of`` (unrevised series)."""
    as_of = pd.Timestamp(as_of)
    known = df[(df["date"] <= as_of)].dropna(subset=["value"])
    if known.empty:
        return None
    row = known.sort_values("date").iloc[-1]
    return {"value": float(row["value"]), "obs_date": str(row["date"].date())}
