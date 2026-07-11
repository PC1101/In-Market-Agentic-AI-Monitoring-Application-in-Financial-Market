"""Bayesian Online Change-Point Detection (Adams & MacKay, 2007).

Maintains, online, the posterior distribution over the current *run length* r_t (the
number of steps since the last change point). A constant hazard H = 1/λ encodes the
prior probability of a change on any given day. The observation model is Gaussian with
unknown mean and unknown variance, giving a Normal-inverse-Gamma conjugate prior and a
Student-t posterior predictive.

Detection rule (causal): the MAP run length grows by 1 each day a regime is stable and
collapses toward 0 when a change occurs. We fire an alarm on the day the MAP run length
collapses, provided the prior run had grown to at least ``min_run`` days (which filters
out noise-driven resets). This makes the detection latency ≈ ``min_run`` — an honest,
online-detectable delay.

Reference: Adams & MacKay (2007), "Bayesian Online Changepoint Detection", arXiv:0710.3742.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import t as student_t

from .base import Detector, DetectionResult


class BOCPD(Detector):
    name = "bocpd"

    def __init__(self, hazard: float = 1 / 250, mu0: float = 0.0,
                 kappa0: float = 1.0, alpha0: float = 1.0, beta0: float = 5e-4,
                 min_run: int = 25, max_run_length: int = 500, cooldown: int = 10):
        """
        Args:
            hazard:         constant per-day change probability H = 1/λ.
            mu0, kappa0, alpha0, beta0: Normal-inverse-Gamma prior hyperparameters.
                            beta0 sets the prior scale of the variance; the default is
                            tuned for daily-return magnitudes (~1e-2 std).
            min_run:        minimum grown run length before a collapse counts as a change.
            max_run_length: truncate the run-length vector for bounded memory.
            cooldown:       days to suppress further alarms after firing.
        """
        self.hazard = float(hazard)
        self.mu0 = float(mu0)
        self.kappa0 = float(kappa0)
        self.alpha0 = float(alpha0)
        self.beta0 = float(beta0)
        self.min_run = int(min_run)
        self.max_run_length = int(max_run_length)
        self.cooldown = int(cooldown)

    def _predictive(self, x, mu, kappa, alpha, beta):
        """Student-t posterior predictive pdf of x for each run-length's params."""
        df = 2.0 * alpha
        scale = np.sqrt(beta * (kappa + 1.0) / (alpha * kappa))
        return student_t.pdf(x, df=df, loc=mu, scale=scale)

    def detect(self, returns: pd.Series) -> DetectionResult:
        r = self._validate(returns)
        x = r.to_numpy()
        idx = r.index
        n = len(x)

        H = self.hazard
        # Parameter vectors indexed by run length (start with the prior only).
        mu = np.array([self.mu0])
        kappa = np.array([self.kappa0])
        alpha = np.array([self.alpha0])
        beta = np.array([self.beta0])
        # Run-length posterior R (starts certain that r_0 = 0).
        R = np.array([1.0])

        map_run = np.zeros(n, dtype=int)
        alarms: list[pd.Timestamp] = []
        prev_map = 0
        cooldown_until = -1

        for t in range(n):
            xt = x[t]
            pred = self._predictive(xt, mu, kappa, alpha, beta)

            growth = R * pred * (1.0 - H)
            cp = np.sum(R * pred * H)

            new_R = np.empty(len(R) + 1)
            new_R[0] = cp
            new_R[1:] = growth
            total = new_R.sum()
            new_R = new_R / total if total > 0 else np.concatenate(([1.0], np.zeros(len(R))))

            # Update sufficient statistics (prepend prior for the r=0 case).
            new_mu = np.concatenate(([self.mu0], (kappa * mu + xt) / (kappa + 1.0)))
            new_kappa = np.concatenate(([self.kappa0], kappa + 1.0))
            new_alpha = np.concatenate(([self.alpha0], alpha + 0.5))
            new_beta = np.concatenate((
                [self.beta0],
                beta + (kappa * (xt - mu) ** 2) / (2.0 * (kappa + 1.0)),
            ))

            # Truncate for bounded memory.
            if len(new_R) > self.max_run_length:
                k = self.max_run_length
                new_R, new_mu = new_R[:k], new_mu[:k]
                new_kappa, new_alpha, new_beta = new_kappa[:k], new_alpha[:k], new_beta[:k]
                s = new_R.sum()
                if s > 0:
                    new_R = new_R / s

            R, mu, kappa, alpha, beta = new_R, new_mu, new_kappa, new_alpha, new_beta

            cur_map = int(np.argmax(R))
            map_run[t] = cur_map

            # Collapse of a grown run => detected change point.
            if cur_map < prev_map and prev_map >= self.min_run and t > cooldown_until:
                alarms.append(idx[t])
                cooldown_until = t + self.cooldown
            prev_map = cur_map

        return DetectionResult(
            self.name, alarms,
            pd.Series(map_run.astype(float), index=idx, name=self.name),
        )
