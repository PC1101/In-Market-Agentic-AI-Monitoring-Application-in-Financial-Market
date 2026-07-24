"""Two-state Gaussian Hidden Markov Model regime detector.

A daily return is modelled as emitted by one of two latent regimes — a *calm* state
(low variance) and a *stressed* state (high variance). Regime probabilities are obtained
by the **forward filter**, which at day *t* uses observations up to *t* — strictly causal
— so the day-to-day monitoring signal never looks ahead.

Parameters (initial, transition, per-state Gaussian mean/variance) are learned by
Baum-Welch (EM). For strict information parity, pass ``train_returns`` = a historical
span that predates the evaluation window and contains both regimes; the fixed model is
then filtered causally over the evaluation window. If ``train_returns`` is omitted, the
parameters are estimated in-sample on the evaluation series itself — a documented
compromise (the *parameters* are stationary constants, but they do touch the whole
window); the emitted probability signal remains causal. ``run_classical.py`` supplies a
pre-2007 training period (calm_2004_2006 curve for AL PCA, 2001-2006 slice for JT) so
all four event-window results are out-of-sample. The calm_2004_2006 evaluation window
is in-sample for AL PCA (parameters fit on that same curve) — a documented trade-off.

An alarm fires on the rising edge where the filtered probability of the stressed state
crosses ``threshold`` from below, subject to a cooldown. The stressed state is the one
with the larger fitted variance.

All recursions run in log-space for numerical stability.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import logsumexp

from .base import Detector, DetectionResult

_VAR_FLOOR = 1e-10


def _gauss_logpdf(x: np.ndarray, mu: float, var: float) -> np.ndarray:
    var = max(var, _VAR_FLOOR)
    return -0.5 * (np.log(2.0 * np.pi * var) + (x - mu) ** 2 / var)


class _GaussianHMM2:
    """Minimal 2-state Gaussian HMM (fit via Baum-Welch, causal forward filter)."""

    def __init__(self, n_iter: int = 50, tol: float = 1e-4, seed_split: float = 0.5):
        self.n_iter = int(n_iter)
        self.tol = float(tol)
        self.seed_split = float(seed_split)
        self.pi: np.ndarray | None = None
        self.A: np.ndarray | None = None
        self.mu: np.ndarray | None = None
        self.var: np.ndarray | None = None

    def _log_b(self, x: np.ndarray) -> np.ndarray:
        """(T, 2) matrix of Gaussian log-emission probabilities."""
        return np.column_stack([
            _gauss_logpdf(x, self.mu[0], self.var[0]),
            _gauss_logpdf(x, self.mu[1], self.var[1]),
        ])

    def fit(self, x: np.ndarray) -> "_GaussianHMM2":
        x = np.asarray(x, float)
        T = len(x)
        # Initialise: state 0 = low variance, state 1 = high variance, split on |deviation|.
        dev = np.abs(x - np.median(x))
        thr = np.quantile(dev, self.seed_split)
        low, high = x[dev <= thr], x[dev > thr]
        if len(low) < 2 or len(high) < 2:
            low, high = x, x
        self.mu = np.array([low.mean(), high.mean()])
        self.var = np.array([max(low.var(), _VAR_FLOOR), max(high.var(), _VAR_FLOOR)])
        self.pi = np.array([0.5, 0.5])
        self.A = np.array([[0.95, 0.05], [0.10, 0.90]])

        prev_ll = -np.inf
        for _ in range(self.n_iter):
            log_b = self._log_b(x)
            log_A = np.log(self.A)
            log_pi = np.log(self.pi)

            # Forward.
            log_alpha = np.zeros((T, 2))
            log_alpha[0] = log_pi + log_b[0]
            for t in range(1, T):
                for j in range(2):
                    log_alpha[t, j] = log_b[t, j] + logsumexp(log_alpha[t - 1] + log_A[:, j])
            ll = logsumexp(log_alpha[-1])

            # Backward.
            log_beta = np.zeros((T, 2))
            for t in range(T - 2, -1, -1):
                for i in range(2):
                    log_beta[t, i] = logsumexp(log_A[i, :] + log_b[t + 1] + log_beta[t + 1])

            # Posteriors.
            log_gamma = log_alpha + log_beta
            log_gamma -= logsumexp(log_gamma, axis=1, keepdims=True)
            gamma = np.exp(log_gamma)

            # Pairwise posteriors xi, summed over t.
            xi_sum = np.zeros((2, 2))
            for t in range(T - 1):
                m = (log_alpha[t][:, None] + log_A
                     + (log_b[t + 1] + log_beta[t + 1])[None, :])
                m -= logsumexp(m)
                xi_sum += np.exp(m)

            # M-step.
            self.pi = gamma[0] / gamma[0].sum()
            self.A = xi_sum / xi_sum.sum(axis=1, keepdims=True)
            for j in range(2):
                w = gamma[:, j]
                wsum = w.sum()
                if wsum < 1e-8:
                    continue
                self.mu[j] = (w * x).sum() / wsum
                self.var[j] = max((w * (x - self.mu[j]) ** 2).sum() / wsum, _VAR_FLOOR)

            if abs(ll - prev_ll) < self.tol:
                break
            prev_ll = ll

        return self

    def filter(self, x: np.ndarray) -> np.ndarray:
        """Causal filtered posteriors P(state_t | x_1..t); shape (T, 2)."""
        x = np.asarray(x, float)
        T = len(x)
        log_b = self._log_b(x)
        log_A = np.log(self.A)
        log_alpha = np.zeros((T, 2))
        log_alpha[0] = np.log(self.pi) + log_b[0]
        log_alpha[0] -= logsumexp(log_alpha[0])  # normalise (filtering)
        for t in range(1, T):
            for j in range(2):
                log_alpha[t, j] = log_b[t, j] + logsumexp(log_alpha[t - 1] + log_A[:, j])
            log_alpha[t] -= logsumexp(log_alpha[t])
        return np.exp(log_alpha)

    @property
    def stressed_state(self) -> int:
        return int(np.argmax(self.var))


class HMMDetector(Detector):
    name = "hmm"

    def __init__(self, threshold: float = 0.5, min_samples: int = 120,
                 warmup: int = 20, n_iter: int = 50, cooldown: int = 10,
                 train_returns: pd.Series | None = None):
        """
        Args:
            threshold:     filtered P(stressed) whose upward crossing fires an alarm.
            min_samples:   minimum series length required to run; below this, no alarms.
            warmup:        suppress alarms for the first ``warmup`` days (filter settling).
            n_iter:        Baum-Welch iterations.
            cooldown:      days to suppress further alarms after firing.
            train_returns: optional historical return series to fit parameters on (for
                           strict information parity). If None, parameters are fit
                           in-sample on the evaluation series (documented compromise).
        """
        self.threshold = float(threshold)
        self.min_samples = int(min_samples)
        self.warmup = int(warmup)
        self.n_iter = int(n_iter)
        self.cooldown = int(cooldown)
        self.train_returns = train_returns

    def detect(self, returns: pd.Series) -> DetectionResult:
        r = self._validate(returns)
        x = r.to_numpy()
        idx = r.index
        n = len(x)

        empty_stat = pd.Series(np.zeros(n), index=idx, name=self.name)
        if n < self.min_samples:
            return DetectionResult(self.name, [], empty_stat)

        if self.train_returns is not None:
            x_train = self._validate(self.train_returns).to_numpy()
        else:
            x_train = x  # in-sample parameter estimation (documented compromise)
        model = _GaussianHMM2(n_iter=self.n_iter).fit(x_train)

        post = model.filter(x)  # (T, 2) causal
        p_stress = post[:, model.stressed_state]

        alarms: list[pd.Timestamp] = []
        cooldown_until = -1
        for t in range(1, n):
            crossed_up = p_stress[t] > self.threshold >= p_stress[t - 1]
            if crossed_up and t >= self.warmup and t > cooldown_until:
                alarms.append(idx[t])
                cooldown_until = t + self.cooldown

        return DetectionResult(
            self.name, alarms, pd.Series(p_stress, index=idx, name=self.name)
        )
