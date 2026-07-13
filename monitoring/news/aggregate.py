"""Stage 2 of the news pipeline: daily quantitative news signals.

Turns article-level risk matches into a causal daily intensity series:
  n_articles   — all articles that calendar day
  n_risk       — articles with >=1 risk-term hit
  intensity    — total risk-term hits (a risk-language volume proxy)
  intensity_z  — intensity vs a TRAILING baseline (mean/std over the prior
                 ``baseline_days`` days, shifted one day so day t never
                 baselines on itself — causal by construction)
"""

from __future__ import annotations

import pandas as pd


def daily_signals(all_news: pd.DataFrame, risk_news: pd.DataFrame,
                  baseline_days: int = 60, min_baseline: int = 20) -> pd.DataFrame:
    """Aggregate article- to day-level signals on a continuous calendar index."""
    if all_news.empty:
        return pd.DataFrame(columns=["n_articles", "n_risk", "intensity", "intensity_z"])

    day_all = all_news["date"].dt.normalize()
    idx = pd.date_range(day_all.min(), day_all.max(), freq="D")

    out = pd.DataFrame(index=idx)
    out["n_articles"] = day_all.value_counts().reindex(idx, fill_value=0)
    if risk_news.empty:
        out["n_risk"] = 0
        out["intensity"] = 0
    else:
        day_risk = risk_news["date"].dt.normalize()
        out["n_risk"] = day_risk.value_counts().reindex(idx, fill_value=0)
        out["intensity"] = (risk_news.groupby(day_risk)["n_risk_terms"].sum()
                            .reindex(idx, fill_value=0))

    base = out["intensity"].rolling(baseline_days, min_periods=min_baseline)
    mean, std = base.mean().shift(1), base.std(ddof=1).shift(1)
    # A zero-variance quiet baseline must not mute a spike: floor the std at one
    # risk-term hit. Warm-up days stay NaN via the NaN mean.
    std_eff = std.where(std > 0, 1.0)
    out["intensity_z"] = (out["intensity"] - mean) / std_eff
    return out
