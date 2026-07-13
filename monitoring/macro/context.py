"""Assemble the JSON-serialisable macro block for agent context at a decision date.

Every value is as-of correct by construction (vintage or last-print-<=-as_of),
so the block passes ``guardrails.assert_no_lookahead`` — which run_agentic.py
still re-checks on the full context, fail-closed.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from macro.asof import asof_daily, asof_vintage
from macro.fetch_macro import DAILY_SERIES, VINTAGE_SERIES, DATA_DIR


def macro_context(as_of, data_dir: str | Path = DATA_DIR) -> dict:
    data_dir = Path(data_dir)
    out: dict = {}
    for sid in DAILY_SERIES:
        path = data_dir / f"{sid}.parquet"
        if path.exists():
            v = asof_daily(pd.read_parquet(path), as_of)
            if v is not None:
                out[sid.lower()] = v
    for sid in VINTAGE_SERIES:
        path = data_dir / f"{sid}.parquet"
        if path.exists():
            v = asof_vintage(pd.read_parquet(path), as_of)
            if v is not None:
                out[sid.lower()] = v
    if "dgs10" in out and "dgs2" in out:
        out["yield_curve_10y_2y"] = round(out["dgs10"]["value"] - out["dgs2"]["value"], 4)
    return out
