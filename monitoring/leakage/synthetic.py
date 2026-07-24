"""Synthetic event window generator for pretraining-leakage Condition C (§8).

M=10 fabricated windows are created by:
1. Block-bootstrapping calm returns (~120 trading days) as a no-event baseline.
2. Injecting a crash template (drawn from dev event windows) at a random onset.
3. Generating generic stress headlines at fabricated dates (2099-XX-XX range).

The returned SyntheticWindow objects can be fed to run_leakage.run_condition_c()
using the same detector + supervisor pipeline as real windows, but with no real
event names, dates, or company mentions that an LLM could have memorised.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from windows import Window

# Headline templates: generic stress language, no real event names or dates.
_STRESS_TEMPLATES = [
    "Markets face broad selling pressure amid heightened uncertainty",
    "Volatility spikes sharply as risk assets sell off across sectors",
    "Credit spreads widen on concerns over systematic deleveraging",
    "Investors reduce equity exposure amid deteriorating conditions",
    "Trading volumes surge as portfolio managers reassess positions",
    "Market stress indicators reach elevated levels not seen recently",
    "Systematic strategies face challenging cross-sectional conditions",
    "Risk-off sentiment dominates as correlations spike",
    "Liquidity conditions tighten; bid-ask spreads widen significantly",
    "Factor exposures reverse sharply in broad-market selloff",
    "Margin calls reported as leveraged positions come under pressure",
    "Flight to safety as equities and risk assets decline in tandem",
]

_CALM_TEMPLATES = [
    "Equity markets trade mixed in low-volume session",
    "Volatility remains subdued as trading activity slows",
    "Sector rotation continues with no clear directional trend",
    "Central bank commentary leaves markets relatively unchanged",
    "Credit conditions remain stable with narrow spreads",
]

# Fabricated base year — far future, cannot match any real event.
_FAKE_BASE_YEAR = 2099


@dataclass
class SyntheticWindow:
    """A fabricated event window with injected crash returns and fake headlines."""

    window: Window               # Window dataclass with fabricated onset/dates
    returns: pd.Series           # block-bootstrapped calm + crash injection
    articles: list[dict]         # template-generated headlines with fake dates
    crash_params: dict           # {drawdown_pct, vol_mult, duration_days, template_name}
    onset_idx: int               # index within returns where crash was injected


def _measure_crash_template(returns: pd.Series, window: Window) -> dict:
    """Extract drawdown magnitude, vol multiplier, and duration from a dev event window."""
    onset = window.onset_ts
    pre_onset = returns.loc[:onset]
    post_onset = returns.loc[onset:]

    # Pre-event vol (60-day window before onset, or all available)
    pre_vol = float(pre_onset.tail(60).std(ddof=1)) if len(pre_onset) > 1 else 0.01

    # Post-onset: find peak-to-trough drawdown over first 42 trading days
    event_slice = post_onset.iloc[:42] if len(post_onset) >= 42 else post_onset
    equity = (1 + event_slice).cumprod()
    peak = equity.cummax()
    drawdown = float((equity / peak - 1).min())

    # Post-event vol multiplier
    post_vol = float(event_slice.std(ddof=1)) if len(event_slice) > 1 else pre_vol
    vol_mult = (post_vol / pre_vol) if pre_vol > 0 else 2.0

    # Duration: trading days from onset to trough
    trough_idx = (equity / peak - 1).idxmin()
    trough_td = len(returns.loc[onset:trough_idx])
    duration = max(5, min(trough_td, 30))  # clamp to [5, 30] trading days

    return {
        "template_name": window.name,
        "drawdown_pct": drawdown,     # negative number, e.g. -0.15
        "vol_mult": max(1.5, vol_mult),
        "duration_days": duration,
    }


def _block_bootstrap(returns: pd.Series, target_len: int, block_size: int,
                     rng: np.random.Generator) -> np.ndarray:
    """Moving-block bootstrap: sample blocks with replacement to fill target_len."""
    arr = returns.values.astype(float)
    n = len(arr)
    if n < block_size:
        # Fallback: resample with replacement directly
        return rng.choice(arr, size=target_len, replace=True)

    blocks = [arr[i:i + block_size] for i in range(n - block_size + 1)]
    out: list[float] = []
    while len(out) < target_len:
        blk = blocks[rng.integers(0, len(blocks))]
        out.extend(blk.tolist())
    return np.array(out[:target_len])


def _inject_crash(baseline: np.ndarray, template: dict, onset_idx: int,
                  rng: np.random.Generator) -> np.ndarray:
    """Overlay a scaled crash onto the block-bootstrapped baseline returns.

    The crash is injected at onset_idx. Returns are scaled to achieve:
    - Approximate drawdown matching template["drawdown_pct"]
    - Vol multiplied by template["vol_mult"] in the crash zone
    """
    result = baseline.copy()
    duration = template["duration_days"]
    end_idx = min(onset_idx + duration, len(result))
    crash_len = end_idx - onset_idx
    if crash_len <= 0:
        return result

    target_dd = template["drawdown_pct"]   # e.g. -0.15
    vol_mult = template["vol_mult"]
    baseline_vol = float(np.std(baseline[:onset_idx], ddof=1)) if onset_idx > 1 else 0.01

    # Generate crash returns: scaled negative drift + elevated noise
    crash_vol = baseline_vol * vol_mult
    # Daily drift to reach target drawdown over duration
    daily_drift = target_dd / crash_len
    noise = rng.normal(0, crash_vol, size=crash_len)
    crash_returns = daily_drift + noise * 0.5  # blend drift and noise
    result[onset_idx:end_idx] = crash_returns

    return result


def _fabricate_dates(n_days: int, month_offset: int) -> pd.DatetimeIndex:
    """Generate a fake business-day DatetimeIndex in the far future."""
    start = pd.Timestamp(f"{_FAKE_BASE_YEAR}-{(month_offset % 12) + 1:02d}-02")
    return pd.bdate_range(start, periods=n_days)


def _fabricate_headlines(n_total_days: int, onset_idx: int, crash_severity: str,
                         fake_dates: pd.DatetimeIndex,
                         rng: np.random.Generator) -> list[dict]:
    """Generate template stress headlines around the synthetic onset."""
    articles = []
    for i, dt in enumerate(fake_dates):
        date_str = str(dt.date())
        days_from_onset = i - onset_idx
        if days_from_onset < -5:
            # Pre-crash: calm news
            if rng.random() < 0.3:
                tmpl = _CALM_TEMPLATES[rng.integers(0, len(_CALM_TEMPLATES))]
                articles.append({"date": date_str, "ticker": "MARKET",
                                 "title": tmpl, "summary": ""})
        elif -5 <= days_from_onset <= 21:
            # Crash zone: stress news (density increases near onset)
            n_articles = rng.integers(2, 5) if days_from_onset >= 0 else rng.integers(0, 2)
            for _ in range(n_articles):
                tmpl = _STRESS_TEMPLATES[rng.integers(0, len(_STRESS_TEMPLATES))]
                articles.append({"date": date_str, "ticker": "MARKET",
                                 "title": tmpl, "summary": ""})
        else:
            # Post-crash calm
            if rng.random() < 0.2:
                tmpl = _CALM_TEMPLATES[rng.integers(0, len(_CALM_TEMPLATES))]
                articles.append({"date": date_str, "ticker": "MARKET",
                                 "title": tmpl, "summary": ""})
    return articles


def generate_synthetic_windows(
    calm_returns: pd.Series,
    dev_event_windows: list[Window],
    dev_event_returns: dict[str, pd.Series],
    M: int = 10,
    window_len: int = 120,      # trading days per synthetic window
    onset_range: tuple = (30, 60),  # onset injected between day 30 and 60
    block_size: int = 20,
    rng_seed: int = 42,
) -> list[SyntheticWindow]:
    """Generate M synthetic event windows for Condition C leakage testing.

    Args:
        calm_returns: A continuous daily return series from development calm windows
                      (e.g. calm_2004_2006 or calm_2013_2014) used as the baseline
                      for block-bootstrapping.
        dev_event_windows: List of dev event Window objects to extract crash templates.
        dev_event_returns: {window.name: pd.Series} of returns for each dev event window.
        M: Number of synthetic windows to generate.
        window_len: Total trading days per synthetic window (pre + event + post).
        onset_range: (min, max) trading day index for crash injection.
        block_size: Block size for moving-block bootstrap.
        rng_seed: RNG seed for reproducibility.

    Returns:
        List of M SyntheticWindow objects, each with fabricated dates, synthetic
        returns, and template-generated news headlines.
    """
    rng = np.random.default_rng(rng_seed)

    # Measure crash templates from all dev event windows
    templates = []
    for w in dev_event_windows:
        if w.name in dev_event_returns:
            t = _measure_crash_template(dev_event_returns[w.name], w)
            templates.append(t)
    if not templates:
        # Fallback template if no dev event returns available
        templates = [{"template_name": "fallback", "drawdown_pct": -0.12,
                      "vol_mult": 2.0, "duration_days": 15}]

    synthetic_windows = []
    for i in range(M):
        # 1. Block-bootstrap calm returns
        baseline = _block_bootstrap(calm_returns, window_len, block_size, rng)

        # 2. Pick crash template and onset
        template = templates[rng.integers(0, len(templates))]
        onset_idx = int(rng.integers(onset_range[0], onset_range[1]))

        # 3. Inject crash
        synthetic_rets = _inject_crash(baseline, template, onset_idx, rng)

        # 4. Build fake DatetimeIndex (far future, month offset per window)
        fake_dates = _fabricate_dates(window_len, month_offset=i)
        fake_series = pd.Series(synthetic_rets, index=fake_dates, name="port_ret")

        # 5. Build fake Window dataclass
        onset_date = str(fake_dates[onset_idx].date())
        fake_window = Window(
            name=f"synthetic_{i:02d}",
            kind="event",
            start=str(fake_dates[0].date()),
            end=str(fake_dates[-1].date()),
            onset=onset_date,
            description=f"Synthetic event window {i} (leakage Condition C)",
        )

        # 6. Fabricate headlines
        severity = "high" if template["drawdown_pct"] < -0.10 else "moderate"
        articles = _fabricate_headlines(
            window_len, onset_idx, severity, fake_dates, rng
        )

        crash_params = dict(template, onset_idx=onset_idx)
        synthetic_windows.append(SyntheticWindow(
            window=fake_window,
            returns=fake_series,
            articles=articles,
            crash_params=crash_params,
            onset_idx=onset_idx,
        ))

    return synthetic_windows
