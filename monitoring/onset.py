"""Curve-derived onset computation (sensitivity analysis, preregistration §4/§7).

The primary scoring always uses the hardcoded onset in windows.py.  curve_onset()
provides a deterministic alternative anchored to the *strategy curve* rather than
the market-event date — used only as a secondary sensitivity check to confirm that
the agentic advantage is not an artefact of news being indexed to the market-event
date while the strategy break slightly leads or lags.

Rule (pre-registered): within [window.start_ts, window.end_ts], find the deepest
peak-to-trough drawdown of the cumulative-return curve.  The onset is the *peak*
date immediately preceding that trough.  No detector, no tunable parameters.
"""

from __future__ import annotations

import pandas as pd

from windows import Window


def curve_onset(series: pd.Series, window: Window) -> pd.Timestamp:
    """Return the curve-derived onset for an event window.

    Args:
        series: Daily returns with a DatetimeIndex (any frequency).
        window: Must be an event window (kind == "event").

    Returns:
        pd.Timestamp — the peak date preceding the deepest within-window trough.

    Raises:
        ValueError: if window.kind != "event", the window slice is empty, or no
                    data exists before the trough.
    """
    if window.kind != "event":
        raise ValueError(
            f"curve_onset requires an event window; got kind={window.kind!r}"
        )

    mask = (series.index >= window.start_ts) & (series.index <= window.end_ts)
    s = series.loc[mask]
    if s.empty:
        raise ValueError(
            f"no data for window {window.name!r} in series "
            f"(series spans {series.index.min()} – {series.index.max()})"
        )

    cum = (1.0 + s).cumprod()
    running_max = cum.cummax()
    drawdown = cum / running_max - 1.0   # 0 at peaks, negative in drawdown

    trough_date = drawdown.idxmin()      # date of deepest drawdown

    # Peak = last date on or before trough where drawdown touched 0 (a local peak)
    pre_trough = drawdown.loc[:trough_date]
    at_peak = pre_trough[pre_trough >= 0.0]   # floating-point safe: >= 0 captures 0.0
    if at_peak.empty:
        # Fall back to the date of the maximum running-max value before the trough
        peak_date = running_max.loc[:trough_date].idxmax()
    else:
        peak_date = at_peak.index[-1]

    return pd.Timestamp(peak_date)
