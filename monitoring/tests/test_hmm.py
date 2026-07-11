from detectors import HMMDetector
from detectors.hmm import _GaussianHMM2


def _split(alarms, change_ts):
    return ([a for a in alarms if a < change_ts],
            [a for a in alarms if a >= change_ts])


def test_fires_after_vol_shift(vol_shift, change_ts):
    res = HMMDetector().detect(vol_shift)
    before, after = _split(res.alarms, change_ts)
    assert after, "HMM should flag the stressed regime after a volatility jump"
    assert (after[0] - change_ts).days <= 60
    assert len(before) <= 2


def test_learns_two_variance_regimes(vol_shift):
    model = _GaussianHMM2().fit(vol_shift.to_numpy())
    lo, hi = sorted(model.var)
    assert hi > 3 * lo, "should separate a low- and a high-variance regime"


def test_filter_is_causal_probability(vol_shift):
    model = _GaussianHMM2().fit(vol_shift.to_numpy())
    post = model.filter(vol_shift.to_numpy())
    assert post.shape == (len(vol_shift), 2)
    # rows are proper distributions
    assert abs(post.sum(axis=1) - 1.0).max() < 1e-8


def test_quiet_on_stationary(stationary):
    res = HMMDetector().detect(stationary)
    # Single regime: allow a few spurious crossings but not a storm of them.
    assert len(res.alarms) <= 6, f"too many false alarms: {len(res.alarms)}"
