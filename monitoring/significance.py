"""Statistical analysis plan implementation (preregistration §7).

All analyses use NumPy + SciPy only. This module is written and tested against
*development* logs before any test-set run exists (§9 freeze mechanic).

Analyses implemented (in function order):
  1. mcnemar_test          — Recall (secondary): exact McNemar on paired detection outcomes
  2. permutation_test_latency — Latency (co-primary): exact permutation test (2^n)
  3. block_bootstrap_fpr   — FPR (co-primary): moving-block bootstrap CI + Fisher's exact
  4. strategy_effects      — H2/H3: per-strategy signs + interaction
  5. placebo_onsets        — Empirical null for "permanently alarmed" baseline
  6. bayesian_recall       — Beta-Binomial Jeffreys posteriors on recall
  7. holm_correction       — Multiplicity correction for two co-primary endpoints

All p-values are two-sided unless stated. Multiplicity correction (Holm) is applied
only to the two co-primary endpoints (latency + FPR); secondary/companion analyses
are reported unadjusted.
"""

from __future__ import annotations

import itertools
from typing import Sequence

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# 1. McNemar test — Recall (secondary)
# ---------------------------------------------------------------------------

def mcnemar_test(
    classical_detected: Sequence[bool],
    agentic_detected: Sequence[bool],
) -> dict:
    """Exact McNemar test on paired detection outcomes.

    Args:
        classical_detected: bool per test window×strategy pair (n ≤ 6 for 4 test events
                            × up to 2 strategies).
        agentic_detected:   same length.

    Returns dict with keys:
        n, a, b, c, d, p_value
        (a=both, b=classical-only, c=agentic-only, d=neither;
         McNemar uses only the off-diagonal b, c.)
    """
    cd = list(classical_detected)
    ad = list(agentic_detected)
    if len(cd) != len(ad):
        raise ValueError("classical_detected and agentic_detected must have the same length")
    n = len(cd)
    a = sum(1 for c, a_ in zip(cd, ad) if c and a_)
    b = sum(1 for c, a_ in zip(cd, ad) if c and not a_)
    c = sum(1 for c, a_ in zip(cd, ad) if not c and a_)
    d = sum(1 for c, a_ in zip(cd, ad) if not c and not a_)

    # Exact McNemar: binomial test on min(b, c) successes out of b+c trials, p=0.5
    bc = b + c
    if bc == 0:
        p_value = 1.0  # no discordant pairs → no evidence either way
    else:
        from scipy.stats import binomtest
        result = binomtest(min(b, c), bc, p=0.5, alternative="two-sided")
        p_value = float(result.pvalue)

    return {"n": n, "a": a, "b": b, "c": c, "d": d, "p_value": p_value}


# ---------------------------------------------------------------------------
# 2. Permutation test — Latency (co-primary)
# ---------------------------------------------------------------------------

def permutation_test_latency(
    classical_latencies: Sequence[float | None],
    agentic_latencies: Sequence[float | None],
    censor_at: int = 21,
) -> dict:
    """Exact permutation test on paired restricted mean detection time.

    Non-detections (None or NaN) are censored at ``censor_at`` days.
    Tests whether the mean paired difference (agentic − classical) is
    significantly negative (agentic faster).

    With n ≤ 6 pairs the full 2^n enumeration is exact and fast.

    Returns dict: observed_diff, p_value_one_sided (agentic faster),
                  p_value_two_sided, n_pairs, n_permutations.
    """
    def _censor(x):
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return float(censor_at)
        return min(float(x), float(censor_at))

    cl = np.array([_censor(x) for x in classical_latencies])
    al = np.array([_censor(x) for x in agentic_latencies])
    if len(cl) != len(al):
        raise ValueError("latency lists must have the same length")
    n = len(cl)
    diffs = al - cl  # positive = agentic slower; negative = agentic faster
    observed = float(diffs.mean())

    # Enumerate all 2^n sign assignments.
    n_perms = 2 ** n
    null_stats = np.empty(n_perms)
    for i, signs in enumerate(itertools.product([-1, 1], repeat=n)):
        null_stats[i] = float((diffs * np.array(signs)).mean())

    p_one_sided = float((null_stats <= observed).mean())   # agentic faster
    p_two_sided = float((np.abs(null_stats) >= abs(observed)).mean())

    return {
        "observed_diff_mean": observed,
        "p_value_one_sided": p_one_sided,
        "p_value": p_two_sided,
        "n_pairs": n,
        "n_permutations": n_perms,
        "censor_at": censor_at,
    }


# ---------------------------------------------------------------------------
# 3. Block bootstrap — FPR (co-primary)
# ---------------------------------------------------------------------------

def block_bootstrap_fpr(
    classical_alarm_days: Sequence[np.ndarray],
    agentic_alarm_days: Sequence[np.ndarray],
    block_size: int = 10,
    n_resamples: int = 10_000,
    rng_seed: int = 42,
) -> dict:
    """Moving-block bootstrap CI on FPR/day difference (agentic − classical).

    Args:
        classical_alarm_days: one binary (0/1) array per test calm window.
        agentic_alarm_days:   same structure (same window order and lengths).
        block_size:  trading-day block length (default 10).
        n_resamples: bootstrap iterations (default 10,000).
        rng_seed:    for reproducibility.

    Returns dict: observed_diff, ci_95 [lo, hi], p_value (two-sided),
                  fisher_p, n_resamples.
    """
    c_streams = [np.asarray(x, dtype=float) for x in classical_alarm_days]
    a_streams = [np.asarray(x, dtype=float) for x in agentic_alarm_days]
    if len(c_streams) != len(a_streams):
        raise ValueError("classical and agentic stream lists must have the same length")

    c_cat = np.concatenate(c_streams) if c_streams else np.array([])
    a_cat = np.concatenate(a_streams) if a_streams else np.array([])
    N = len(c_cat)
    if N == 0:
        return {"observed_diff": 0.0, "ci_95": [0.0, 0.0], "p_value": 1.0,
                "fisher_p": 1.0, "n_resamples": n_resamples}

    obs_c_fpr = float(c_cat.mean())
    obs_a_fpr = float(a_cat.mean())
    observed_diff = obs_a_fpr - obs_c_fpr

    rng = np.random.default_rng(rng_seed)
    boot_diffs = np.empty(n_resamples)
    for i in range(n_resamples):
        # Moving-block resample (same blocks for both arms → paired resampling).
        indices = _moving_block_indices(N, block_size, rng)
        c_boot = c_cat[indices]
        a_boot = a_cat[indices]
        boot_diffs[i] = a_boot.mean() - c_boot.mean()

    ci_lo, ci_hi = float(np.percentile(boot_diffs, 2.5)), float(np.percentile(boot_diffs, 97.5))
    # p-value: fraction of bootstrap samples as or more extreme than observed (two-sided)
    p_value = float(np.mean(np.abs(boot_diffs) >= abs(observed_diff)))

    # Fisher's exact test as companion
    fisher_p = _fisher_exact_fpr(c_cat, a_cat)

    return {
        "observed_diff": round(observed_diff, 6),
        "classical_fpr": round(obs_c_fpr, 6),
        "agentic_fpr": round(obs_a_fpr, 6),
        "ci_95": [round(ci_lo, 6), round(ci_hi, 6)],
        "p_value": round(p_value, 4),
        "fisher_p": round(fisher_p, 4),
        "n_resamples": n_resamples,
        "block_size": block_size,
    }


def _moving_block_indices(N: int, block_size: int, rng: np.random.Generator) -> np.ndarray:
    """Sample a moving-block bootstrap index array of length N."""
    n_blocks = int(np.ceil(N / block_size))
    starts = rng.integers(0, max(1, N - block_size + 1), size=n_blocks)
    idx = np.concatenate([np.arange(s, min(s + block_size, N)) for s in starts])
    return idx[:N]


def _fisher_exact_fpr(c: np.ndarray, a: np.ndarray) -> float:
    """Fisher's exact test on alarm/non-alarm counts per arm."""
    from scipy.stats import fisher_exact
    c_alarm = int(c.sum())
    a_alarm = int(a.sum())
    c_no = len(c) - c_alarm
    a_no = len(a) - a_alarm
    table = [[c_alarm, c_no], [a_alarm, a_no]]
    _, p = fisher_exact(table, alternative="two-sided")
    return float(p)


# ---------------------------------------------------------------------------
# 4. H2/H3 — Per-strategy effects and interaction
# ---------------------------------------------------------------------------

def strategy_effects(
    results_by_strategy: dict[str, dict],
) -> dict:
    """Per-strategy effect signs (H2) and strategy × method interaction (H3).

    Args:
        results_by_strategy: dict mapping strategy name →
            {"classical": {"latencies": [...], "fpr_days": [...]},
             "agentic":   {"latencies": [...], "fpr_days": [...]}}

    Returns dict with H2 (per-strategy direction) and H3 (interaction sign test).
    """
    h2 = {}
    latency_diffs_by_strat = {}
    fpr_diffs_by_strat = {}

    for strategy, arms in results_by_strategy.items():
        cl = arms["classical"]
        ag = arms["agentic"]

        def _mean_censor(lats):
            return float(np.mean([min(x, 21) if x is not None else 21 for x in lats]))

        lat_diff = _mean_censor(ag["latencies"]) - _mean_censor(cl["latencies"])
        fpr_diff = float(ag.get("fpr_per_day", 0.0) - cl.get("fpr_per_day", 0.0))

        h2[strategy] = {
            "latency_diff_agentic_minus_classical": round(lat_diff, 2),
            "fpr_diff_agentic_minus_classical": round(fpr_diff, 6),
            "latency_direction": "agentic_faster" if lat_diff < 0 else "classical_faster_or_equal",
            "fpr_direction": "agentic_lower" if fpr_diff < 0 else "classical_lower_or_equal",
        }
        latency_diffs_by_strat[strategy] = lat_diff
        fpr_diffs_by_strat[strategy] = fpr_diff

    # H3 interaction: do the diffs have the same sign across strategies?
    # (If agentic is better on one strategy but worse on the other → interaction.)
    strats = list(latency_diffs_by_strat)
    h3 = {}
    if len(strats) == 2:
        s1, s2 = strats
        lat_interaction = latency_diffs_by_strat[s1] - latency_diffs_by_strat[s2]
        fpr_interaction = fpr_diffs_by_strat[s1] - fpr_diffs_by_strat[s2]
        h3 = {
            "latency_interaction": round(lat_interaction, 2),
            "fpr_interaction": round(fpr_interaction, 6),
            "note": "Interaction = diff_strategy1 − diff_strategy2 on (agentic − classical). "
                    "Non-zero implies H3: advantage differs by strategy.",
        }

    return {"H2": h2, "H3": h3}


# ---------------------------------------------------------------------------
# 5. Placebo onsets
# ---------------------------------------------------------------------------

def placebo_onsets(
    classical_alarms: Sequence[pd.Timestamp],
    agentic_alarms: Sequence[pd.Timestamp],
    calm_windows,
    trading_dates: pd.DatetimeIndex,
    K: int = 1_000,
    tolerance: int = 21,
    min_gap_td: int = 42,
    edge_buffer_td: int = 21,
    rng_seed: int = 42,
) -> dict:
    """K=1,000 pseudo-onsets uniformly in test calm windows.

    Constraints (§7.5):
        ≥42 trading days apart; ≥21 trading days from window edges.

    Returns dict: classical_placebo_rate, agentic_placebo_rate, K,
                  n_sampled, pseudo_onsets (first 20 as ISO strings).
    """
    # Build candidate pool: trading days ≥edge_buffer from each calm window edge.
    candidates: list[pd.Timestamp] = []
    for w in calm_windows:
        w_td = trading_dates[(trading_dates >= w.start_ts) & (trading_dates <= w.end_ts)]
        if len(w_td) < 2 * edge_buffer_td:
            continue
        candidates.extend(w_td[edge_buffer_td:-edge_buffer_td])

    if not candidates:
        return {"classical_placebo_rate": float("nan"),
                "agentic_placebo_rate": float("nan"),
                "K": K, "n_sampled": 0, "pseudo_onsets": []}

    rng = np.random.default_rng(rng_seed)
    cand_arr = np.array(candidates)
    # Build trading-day index for gap check.
    td_idx = {ts: i for i, ts in enumerate(trading_dates)}

    pseudo: list[pd.Timestamp] = []
    tries = 0
    max_tries = K * 100
    while len(pseudo) < K and tries < max_tries:
        tries += 1
        onset = cand_arr[rng.integers(len(cand_arr))]
        # Gap check: ≥min_gap_td trading days from all existing pseudo onsets.
        if pseudo:
            dists = [abs(td_idx.get(onset, 0) - td_idx.get(p, 0)) for p in pseudo]
            if min(dists) < min_gap_td:
                continue
        pseudo.append(onset)

    c_alarms = sorted(pd.Timestamp(a) for a in classical_alarms)
    a_alarms = sorted(pd.Timestamp(a) for a in agentic_alarms)

    def _detected(alarms, onset):
        zone_end = trading_dates[
            min(len(trading_dates) - 1,
                np.searchsorted(trading_dates, onset) + tolerance)
        ]
        return any(onset <= a <= zone_end for a in alarms)

    c_rate = float(np.mean([_detected(c_alarms, o) for o in pseudo])) if pseudo else float("nan")
    a_rate = float(np.mean([_detected(a_alarms, o) for o in pseudo])) if pseudo else float("nan")

    return {
        "classical_placebo_rate": round(c_rate, 4) if not np.isnan(c_rate) else None,
        "agentic_placebo_rate": round(a_rate, 4) if not np.isnan(a_rate) else None,
        "K": K,
        "n_sampled": len(pseudo),
        "pseudo_onsets": [str(o.date()) for o in pseudo[:20]],
    }


# ---------------------------------------------------------------------------
# 6. Bayesian companion — Beta-Binomial posteriors on recall
# ---------------------------------------------------------------------------

def bayesian_recall(
    classical_detected: int,
    classical_total: int,
    agentic_detected: int,
    agentic_total: int,
    prior_alpha: float = 0.5,
    prior_beta: float = 0.5,
    n_samples: int = 100_000,
    rng_seed: int = 42,
) -> dict:
    """Beta-Binomial posteriors on recall (Jeffreys prior α=β=0.5).

    Returns dict: classical_ci95, agentic_ci95, p_agentic_better,
                  classical_mean, agentic_mean.
    """
    rng = np.random.default_rng(rng_seed)

    def _posterior(detected, total):
        a = prior_alpha + detected
        b = prior_beta + (total - detected)
        return rng.beta(a, b, size=n_samples)

    c_samples = _posterior(classical_detected, classical_total)
    a_samples = _posterior(agentic_detected, agentic_total)

    p_better = float((a_samples > c_samples).mean())
    c_ci = [float(np.percentile(c_samples, 2.5)), float(np.percentile(c_samples, 97.5))]
    a_ci = [float(np.percentile(a_samples, 2.5)), float(np.percentile(a_samples, 97.5))]

    return {
        "classical_mean": round(float(c_samples.mean()), 4),
        "agentic_mean": round(float(a_samples.mean()), 4),
        "classical_ci95": [round(c_ci[0], 4), round(c_ci[1], 4)],
        "agentic_ci95": [round(a_ci[0], 4), round(a_ci[1], 4)],
        "p_agentic_better": round(p_better, 4),
        "prior": {"alpha": prior_alpha, "beta": prior_beta},
        "n_samples": n_samples,
    }


# ---------------------------------------------------------------------------
# 7. Holm correction — multiplicity across co-primary endpoints
# ---------------------------------------------------------------------------

def holm_correction(p_values: dict[str, float]) -> dict[str, dict]:
    """Holm step-down correction across co-primary endpoints.

    Args:
        p_values: mapping endpoint name → raw p-value. Expected keys:
                  "latency" and "fpr" (the two co-primary endpoints per §7.7).

    Returns dict: endpoint → {raw_p, adjusted_p, reject_005, rank}.
    """
    names = list(p_values)
    raw = [p_values[n] for n in names]
    k = len(raw)
    # Sort ascending by p-value.
    order = sorted(range(k), key=lambda i: raw[i])
    adjusted = [None] * k
    running_max = 0.0
    for rank, i in enumerate(order):
        adj = raw[i] * (k - rank)
        running_max = max(running_max, adj)
        adjusted[i] = min(running_max, 1.0)

    return {
        name: {
            "raw_p": round(raw[i], 4),
            "adjusted_p": round(adjusted[i], 4),
            "reject_005": adjusted[i] < 0.05,
            "rank": order.index(i) + 1,
        }
        for i, name in enumerate(names)
    }


# ---------------------------------------------------------------------------
# Top-level runner
# ---------------------------------------------------------------------------

def run_full_analysis(
    classical_results: dict,
    agentic_results: dict,
    test_windows,
    trading_dates: pd.DatetimeIndex | None = None,
    rng_seed: int = 42,
) -> dict:
    """Execute all 7 §7 analyses and return the complete results structure.

    Args:
        classical_results: dict {"window_name": {"detected": bool, "latency": int|None, ...}}
        agentic_results:   same structure
        test_windows:      list of Window objects (test set only)
        trading_dates:     full DatetimeIndex covering test period (for placebo onsets)
    """
    # Extract paired outcomes across all test windows × strategy pairs.
    keys = list(classical_results.keys())
    c_detected = [bool(classical_results[k].get("detected", False)) for k in keys]
    a_detected = [bool(agentic_results[k].get("detected", False)) for k in keys]
    c_latencies = [classical_results[k].get("latency") for k in keys]
    a_latencies = [agentic_results[k].get("latency") for k in keys]

    event_keys = [k for k in keys if classical_results[k].get("kind") == "event"]
    calm_keys = [k for k in keys if classical_results[k].get("kind") == "calm"]

    # FPR binary streams (one per calm window)
    c_fpr_streams = [np.array(classical_results[k].get("alarm_stream", []), dtype=float)
                     for k in calm_keys]
    a_fpr_streams = [np.array(agentic_results[k].get("alarm_stream", []), dtype=float)
                     for k in calm_keys]

    # Bayesian recall uses event pairs only.
    n_test_events = len(event_keys)
    c_n_det = sum(1 for k in event_keys if classical_results[k].get("detected"))
    a_n_det = sum(1 for k in event_keys if agentic_results[k].get("detected"))

    mcnemar = mcnemar_test(c_detected, a_detected)
    perm = permutation_test_latency(c_latencies, a_latencies)
    bootstrap = block_bootstrap_fpr(c_fpr_streams, a_fpr_streams, rng_seed=rng_seed)
    bayes = bayesian_recall(c_n_det, n_test_events, a_n_det, n_test_events, rng_seed=rng_seed)
    holm = holm_correction({"latency": perm["p_value"], "fpr": bootstrap["p_value"]})

    placebo: dict = {}
    if trading_dates is not None:
        calm_wins = [w for w in test_windows if w.kind == "calm"]
        c_all_alarms = [pd.Timestamp(a) for k in keys
                        for a in classical_results[k].get("alarms", [])]
        a_all_alarms = [pd.Timestamp(a) for k in keys
                        for a in agentic_results[k].get("alarms", [])]
        placebo = placebo_onsets(c_all_alarms, a_all_alarms, calm_wins,
                                 trading_dates, rng_seed=rng_seed)

    return {
        "1_mcnemar_recall": mcnemar,
        "2_permutation_latency": perm,
        "3_bootstrap_fpr": bootstrap,
        "6_bayesian_recall": bayes,
        "7_holm_correction": holm,
        "5_placebo_onsets": placebo,
        "meta": {
            "n_test_pairs": len(keys),
            "n_event_pairs": len(event_keys),
            "n_calm_windows": len(calm_keys),
        },
    }
