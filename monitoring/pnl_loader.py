"""Load strategy daily PnL into a standard, causal return series.

Both strategies emit the same daily schema:

    Date,port_ret,equity,drawdown

This module normalises that CSV into a ``pandas`` object the detectors can consume,
and provides a strict *as-of* slice so that no downstream code can accidentally read
returns from the future (information-parity guardrail #1).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

REQUIRED_COLUMNS = ("port_ret", "equity", "drawdown")


def load_pnl(path: str | Path) -> pd.DataFrame:
    """Load a daily equity_curve.csv into a DatetimeIndex-ed DataFrame.

    Handles both schema variants seen in the repo:
      * an explicit ``Date`` column, or
      * an unnamed first column holding the date index (arbitragelab driver).

    Args:
        path: path to the equity_curve.csv.

    Returns:
        DataFrame indexed by a sorted DatetimeIndex named "Date", with columns
        port_ret, equity, drawdown.

    Raises:
        FileNotFoundError: if path does not exist.
        ValueError: if required columns are missing or the index is not unique/sorted-able.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"PnL file not found: {path}")

    df = pd.read_csv(path)

    # Identify the date column: explicit "Date", else the first column.
    if "Date" in df.columns:
        date_col = "Date"
    else:
        date_col = df.columns[0]
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.rename(columns={date_col: "Date"}).set_index("Date").sort_index()

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"{path.name} missing required columns {missing}; found {list(df.columns)}"
        )
    if df.index.has_duplicates:
        raise ValueError(f"{path.name} has duplicate dates in the index")

    return df[list(REQUIRED_COLUMNS)]


def returns_series(df: pd.DataFrame) -> pd.Series:
    """Extract the daily return series (port_ret) as a clean float Series."""
    s = df["port_ret"].astype(float)
    s.name = "port_ret"
    return s


def as_of(obj: pd.Series | pd.DataFrame, ts) -> pd.Series | pd.DataFrame:
    """Return only the observations dated <= ts (causal / no-lookahead slice).

    This is the fundamental information-parity guard: any monitor evaluated at a
    decision date ``ts`` may only see data on or before ``ts``.
    """
    ts = pd.Timestamp(ts)
    return obj.loc[:ts]
