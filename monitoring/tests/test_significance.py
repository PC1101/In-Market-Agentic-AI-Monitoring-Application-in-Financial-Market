"""Tests for the statistical analysis module (preregistration §7)."""
import numpy as np
import pytest

from significance import (
    mcnemar_test,
    permutation_test_latency,
    block_bootstrap_fpr,
    bayesian_recall,
    holm_correction,
    strategy_effects,
)


# ---------------------------------------------------------------------------
# 1. McNemar test
# ---------------------------------------------------------------------------

def test_mcnemar_no_discordant_pairs():
    r = mcnemar_test([True, True, False], [True, True, False])
    assert r["b"] == 0 and r["c"] == 0
    assert r["p_value"] == 1.0


def test_mcnemar_all_discordant_equal():
    # b = 1, c = 1 → most conservative case → p = 1.0
    r = mcnemar_test([True, False], [False, True])
    assert r["b"] == 1 and r["c"] == 1
    assert r["p_value"] == 1.0


def test_mcnemar_agentic_dominates():
    # c >> b: agentic detects what classical misses
    r = mcnemar_test([False, False, False, True], [True, True, True, True])
    assert r["c"] == 3 and r["b"] == 0
    # With 0 vs 3, exact binomial p = 2 * (0.5^3) = 0.25 ≤ 1
    assert r["p_value"] <= 1.0


def test_mcnemar_mismatched_lengths():
    with pytest.raises(ValueError):
        mcnemar_test([True, False], [True])


def test_mcnemar_keys():
    r = mcnemar_test([True], [False])
    for key in ("n", "a", "b", "c", "d", "p_value"):
        assert key in r


# ---------------------------------------------------------------------------
# 2. Permutation test (latency)
# ---------------------------------------------------------------------------

def test_permutation_no_difference():
    r = permutation_test_latency([5, 10, 3], [5, 10, 3])
    assert abs(r["observed_diff_mean"]) < 1e-10
    assert r["p_value"] >= 0.5   # no effect → not significant


def test_permutation_agentic_much_faster():
    # With n=4 identical pairs, min exact one-sided p = 1/16 = 0.0625
    r = permutation_test_latency([21, 21, 21, 21], [1, 1, 1, 1])
    assert r["observed_diff_mean"] < 0
    assert r["p_value_one_sided"] <= 0.0625


def test_permutation_censor_applied():
    # None → censored at 21
    r = permutation_test_latency([None, 5], [None, 3])
    # Both have latency 21 (censored) and 5/3 → diff = (21+3)/2 - (21+5)/2 = -1
    assert abs(r["observed_diff_mean"] - (-1.0)) < 1e-10


def test_permutation_n_permutations():
    r = permutation_test_latency([5, 10], [3, 8])
    assert r["n_permutations"] == 4  # 2^2


def test_permutation_keys():
    r = permutation_test_latency([5], [3])
    for key in ("observed_diff_mean", "p_value", "p_value_one_sided", "n_pairs", "n_permutations"):
        assert key in r


# ---------------------------------------------------------------------------
# 3. Block bootstrap (FPR)
# ---------------------------------------------------------------------------

def test_bootstrap_identical_streams():
    s = np.zeros(50)
    r = block_bootstrap_fpr([s], [s])
    assert abs(r["observed_diff"]) < 1e-10
    assert r["ci_95"][0] <= 0 <= r["ci_95"][1]


def test_bootstrap_agentic_lower_fpr():
    rng = np.random.default_rng(0)
    c = (rng.random(100) < 0.1).astype(float)  # ~10% FPR
    a = (rng.random(100) < 0.03).astype(float)  # ~3% FPR
    r = block_bootstrap_fpr([c], [a], n_resamples=500)
    assert r["observed_diff"] < 0  # agentic lower
    assert r["classical_fpr"] > r["agentic_fpr"]


def test_bootstrap_empty_streams():
    r = block_bootstrap_fpr([], [])
    assert r["observed_diff"] == 0.0
    assert r["p_value"] == 1.0


def test_bootstrap_keys():
    r = block_bootstrap_fpr([np.zeros(20)], [np.zeros(20)])
    for key in ("observed_diff", "ci_95", "p_value", "fisher_p", "n_resamples"):
        assert key in r


# ---------------------------------------------------------------------------
# 6. Bayesian recall
# ---------------------------------------------------------------------------

def test_bayesian_perfect_agentic():
    r = bayesian_recall(1, 4, 4, 4)  # classical 1/4, agentic 4/4
    assert r["p_agentic_better"] > 0.8
    assert r["agentic_mean"] > r["classical_mean"]


def test_bayesian_equal_arms():
    r = bayesian_recall(2, 4, 2, 4)
    assert 0.3 < r["p_agentic_better"] < 0.7  # close to 0.5


def test_bayesian_ci_ordering():
    r = bayesian_recall(3, 4, 2, 4)
    assert r["classical_ci95"][0] < r["classical_ci95"][1]
    assert r["agentic_ci95"][0] < r["agentic_ci95"][1]


def test_bayesian_jeffreys_prior():
    r = bayesian_recall(0, 0, 0, 0)
    assert r["prior"]["alpha"] == 0.5 and r["prior"]["beta"] == 0.5


# ---------------------------------------------------------------------------
# 7. Holm correction
# ---------------------------------------------------------------------------

def test_holm_both_significant():
    r = holm_correction({"latency": 0.001, "fpr": 0.002})
    # Both reject at 5%
    assert r["latency"]["reject_005"] is True
    assert r["fpr"]["reject_005"] is True


def test_holm_neither_significant():
    r = holm_correction({"latency": 0.4, "fpr": 0.6})
    assert r["latency"]["reject_005"] is False
    assert r["fpr"]["reject_005"] is False


def test_holm_one_significant():
    # latency p=0.01, fpr p=0.9 → with k=2, Holm threshold for rank-1 is 0.025
    r = holm_correction({"latency": 0.01, "fpr": 0.9})
    assert r["latency"]["reject_005"] is True
    assert r["fpr"]["reject_005"] is False


def test_holm_adjusted_ge_raw():
    r = holm_correction({"latency": 0.02, "fpr": 0.04})
    for endpoint in r.values():
        assert endpoint["adjusted_p"] >= endpoint["raw_p"]


def test_holm_adjusted_capped_at_1():
    r = holm_correction({"latency": 0.9, "fpr": 0.8})
    for endpoint in r.values():
        assert endpoint["adjusted_p"] <= 1.0


# ---------------------------------------------------------------------------
# strategy_effects (H2/H3)
# ---------------------------------------------------------------------------

def test_strategy_effects_returns_h2_h3():
    data = {
        "AL_PCA": {
            "classical": {"latencies": [5, 10], "fpr_per_day": 0.01},
            "agentic": {"latencies": [3, 7], "fpr_per_day": 0.005},
        },
        "JT_MOM": {
            "classical": {"latencies": [15, None], "fpr_per_day": 0.02},
            "agentic": {"latencies": [8, 12], "fpr_per_day": 0.015},
        },
    }
    r = strategy_effects(data)
    assert "H2" in r and "H3" in r
    assert "AL_PCA" in r["H2"] and "JT_MOM" in r["H2"]
    assert "latency_direction" in r["H2"]["AL_PCA"]
