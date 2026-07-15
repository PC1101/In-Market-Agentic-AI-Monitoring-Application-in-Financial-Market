"""Fetch macro series from FRED/ALFRED into data/macro/*.parquet.

Point-in-time discipline:
  * VINTAGE_SERIES (monthly releases that get revised — UNRATE, CPIAUCSL, INDPRO)
    are fetched from ALFRED with the full realtime history, so ``asof_vintage``
    can reconstruct exactly what was published by any decision date. This is the
    point-in-time-correct route to BLS data (BLS's own API serves only the
    latest revisions — using it would be lookahead).
  * DAILY_SERIES (market prints, unrevised — VIX; Treasury constant-maturity
    yields DGS10/DGS2, i.e. the Treasury integration; fed funds; TED spread)
    are fetched as plain FRED series.

Usage (from monitoring/, with FRED_API_KEY exported):
    python -m macro.fetch_macro
"""

from __future__ import annotations

import json
import os
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

FRED_OBS_URL = "https://api.stlouisfed.org/fred/series/observations"

VINTAGE_SERIES = ["UNRATE", "CPIAUCSL", "INDPRO"]
DAILY_SERIES = ["VIXCLS", "DGS10", "DGS2", "DFF", "TEDRATE"]

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "macro"
_LIMIT = 100_000


def parse_observations(payload: dict) -> pd.DataFrame:
    """FRED JSON payload -> DataFrame(date, value[, realtime_start, realtime_end])."""
    obs = payload["observations"]
    df = pd.DataFrame(obs)
    df["date"] = pd.to_datetime(df["date"])
    df["value"] = pd.to_numeric(df["value"].replace(".", np.nan), errors="coerce")
    for col in ("realtime_start", "realtime_end"):
        if col in df.columns:
            # ALFRED marks the current vintage with "9999-12-31", which overflows
            # datetime64[ns]; coerce it to NaT and pin to Timestamp.max.
            df[col] = pd.to_datetime(df[col], errors="coerce")
            if col == "realtime_end":
                df[col] = df[col].fillna(pd.Timestamp.max)
    return df


def fetch_series(series_id: str, api_key: str, vintages: bool = False) -> pd.DataFrame:
    params = {
        "series_id": series_id, "api_key": api_key, "file_type": "json",
        "limit": _LIMIT,
    }
    if vintages:
        # Full realtime history => one row per (obs date, vintage span).
        params["realtime_start"] = "1990-01-01"
        params["realtime_end"] = "9999-12-31"
    url = f"{FRED_OBS_URL}?{urllib.parse.urlencode(params)}"
    with urllib.request.urlopen(url, timeout=60) as resp:
        payload = json.loads(resp.read().decode())
    if payload.get("count", 0) >= _LIMIT:
        raise RuntimeError(f"{series_id}: hit the {_LIMIT}-row limit; add offset paging")
    df = parse_observations(payload)
    keep = ["date", "value"] + (["realtime_start", "realtime_end"] if vintages else [])
    return df[keep]


def main() -> None:
    api_key = os.environ.get("FRED_API_KEY")
    if not api_key:
        raise SystemExit("Set FRED_API_KEY (free key: https://fred.stlouisfed.org/docs/api/api_key.html)")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for sid in VINTAGE_SERIES:
        df = fetch_series(sid, api_key, vintages=True)
        df.to_parquet(DATA_DIR / f"{sid}.parquet")
        print(f"{sid:10} {len(df):6,} vintage rows -> {DATA_DIR / (sid + '.parquet')}")
    for sid in DAILY_SERIES:
        df = fetch_series(sid, api_key, vintages=False)
        df.to_parquet(DATA_DIR / f"{sid}.parquet")
        print(f"{sid:10} {len(df):6,} daily rows   -> {DATA_DIR / (sid + '.parquet')}")


if __name__ == "__main__":
    main()
