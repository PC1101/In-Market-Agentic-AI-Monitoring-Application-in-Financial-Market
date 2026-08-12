"""Map agentic supervisor decisions to daily portfolio exposure [0, 1].

Graded risk-overlay ladder grounded in published practice:
  - QuantPedia, "An Introduction to Volatility Targeting"
  - Bollerslev et al., "Conditional Volatility Targeting", FAJ 2020
  - CSSA, "Building a Risk Control Index with Drawdown Protection"
  - Man Group, "The Impact of Volatility Targeting"

The exposure mapping keys on the supervisor *state* (NORMAL < WATCH < ALERT
< CRITICAL) and refines by *action* (HOLD < INVESTIGATE < REDUCE < HALT).
Only ALERT+ states trigger de-risking; WATCH/INVESTIGATE alone do NOT reduce
exposure — this is critical to avoid destroying returns in calm windows where
WATCH/INVESTIGATE dominate.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

# Severity ranking (from monitoring/agentic/schemas.py + model.py:109)
STATE_RANK = {"NORMAL": 0, "WATCH": 1, "ALERT": 2, "CRITICAL": 3}
ACTION_RANK = {"HOLD": 0, "INVESTIGATE": 1, "REDUCE": 2, "HALT": 3}


@dataclass(frozen=True)
class PolicyConfig:
    """Risk-overlay policy parameters."""
    exposure_normal: float = 1.0        # NORMAL/WATCH, or any uncovered day
    exposure_alert: float = 0.75        # ALERT with action below REDUCE
    exposure_alert_reduce: float = 0.50 # ALERT + REDUCE
    exposure_halt: float = 0.0          # CRITICAL state or HALT action
    cooldown_days: int = 3              # hold min exposure N trading days after trigger
    reentry_consecutive: int = 2        # consecutive non-ALERT days per re-entry step
    reentry_steps: tuple[float, ...] = (0.75, 1.0)  # staged ramp back up
    lag_days: int = 1                   # decision at close t -> exposure from t+1


def load_decisions(log_paths: list[Path]) -> pd.DataFrame:
    """Parse performance_supervisor records from agentic JSONLs.

    Returns DataFrame indexed by date with columns: state, action, confidence.
    On duplicate dates (overlapping windows), keeps the most severe state.
    """
    records = []
    for p in sorted(log_paths):
        if not p.exists():
            continue
        with open(p) as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if rec.get("agent") != "performance_supervisor":
                    continue
                assessment = rec.get("assessment") or rec.get("raw_output", {})
                state = assessment.get("state", "")
                action = assessment.get("action", "")
                confidence = assessment.get("confidence", None)
                as_of = rec.get("as_of")
                if not as_of or not state:
                    continue
                records.append({
                    "date": pd.Timestamp(as_of),
                    "state": state,
                    "action": action,
                    "confidence": confidence,
                })
    if not records:
        return pd.DataFrame(columns=["state", "action", "confidence"])

    df = pd.DataFrame(records)
    # On duplicate dates, keep the most severe state
    df["state_rank"] = df["state"].map(STATE_RANK).fillna(0)
    df = df.sort_values("state_rank", ascending=False).drop_duplicates("date", keep="first")
    df = df.set_index("date").sort_index()
    df = df.drop(columns=["state_rank"])
    return df


def _target_exposure_single(state: str, action: str, cfg: PolicyConfig) -> float:
    """Map a single (state, action) pair to target exposure."""
    sr = STATE_RANK.get(state, 0)
    ar = ACTION_RANK.get(action, 0)

    if sr >= 3 or ar >= 3:  # CRITICAL or HALT
        return cfg.exposure_halt
    if sr >= 2:  # ALERT
        if ar >= 2:  # REDUCE
            return cfg.exposure_alert_reduce
        return cfg.exposure_alert
    return cfg.exposure_normal


def build_exposure(trading_days: pd.DatetimeIndex,
                   decisions: pd.DataFrame,
                   cfg: PolicyConfig | None = None) -> pd.Series:
    """Build full-period exposure series in [0, 1].

    Days without decision records get exposure_normal (hybrid coverage).
    Applies: ladder -> cooldown -> staged re-entry -> lag shift.
    """
    if cfg is None:
        cfg = PolicyConfig()

    exposure = pd.Series(cfg.exposure_normal, index=trading_days, dtype=float)

    if decisions.empty:
        return exposure

    # Step 1: apply ladder on covered days
    for dt in decisions.index:
        if dt in exposure.index:
            row = decisions.loc[dt]
            # Handle case where loc returns a DataFrame (shouldn't after dedup, but safe)
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            exposure.at[dt] = _target_exposure_single(
                row["state"], row["action"], cfg
            )

    # Step 2: cooldown — after any day where exposure dropped below normal,
    # hold the minimum exposure for cooldown_days
    if cfg.cooldown_days > 0:
        triggered_exposure = None
        cooldown_remaining = 0
        for i, dt in enumerate(exposure.index):
            if cooldown_remaining > 0:
                exposure.iloc[i] = min(exposure.iloc[i], triggered_exposure)
                cooldown_remaining -= 1
            elif exposure.iloc[i] < cfg.exposure_normal:
                triggered_exposure = exposure.iloc[i]
                cooldown_remaining = cfg.cooldown_days

    # Step 3: staged re-entry — after cooldown expires, require
    # reentry_consecutive clean days per step before ramping back up
    if cfg.reentry_steps:
        i = 0
        while i < len(exposure.index):
            if exposure.iloc[i] < cfg.exposure_normal:
                # Find end of reduced-exposure block
                j = i
                while j < len(exposure.index) and exposure.iloc[j] < cfg.exposure_normal:
                    j += 1
                # j is the first "clean" day after the block
                min_exp = exposure.iloc[i:j].min()
                # Now do staged re-entry from j onward
                current_target = min_exp
                for step_target in cfg.reentry_steps:
                    if step_target <= current_target:
                        continue
                    # Need reentry_consecutive clean days at current level
                    clean_count = 0
                    while j < len(exposure.index):
                        if exposure.iloc[j] < cfg.exposure_normal:
                            # New trigger during re-entry, break
                            break
                        clean_count += 1
                        exposure.iloc[j] = current_target
                        if clean_count >= cfg.reentry_consecutive:
                            j += 1
                            current_target = step_target
                            break
                        j += 1
                    else:
                        break
                # Fill remaining clean days at final target (which may be 1.0)
                while j < len(exposure.index) and exposure.iloc[j] >= cfg.exposure_normal:
                    exposure.iloc[j] = current_target
                    j += 1
                i = j
            else:
                i += 1

    # Step 4: lag shift — decision at close t -> exposure from t+1
    if cfg.lag_days > 0:
        exposure = exposure.shift(cfg.lag_days).fillna(cfg.exposure_normal)

    return exposure
