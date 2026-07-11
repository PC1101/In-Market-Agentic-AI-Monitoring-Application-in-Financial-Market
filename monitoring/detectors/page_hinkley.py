"""Page-Hinkley sequential change-point test (two-sided, volatility-normalized).

The Page-Hinkley test tracks the cumulative difference between observations and their
running mean, and fires when that cumulative difference departs from its running
extremum by more than ``lambda_``. It is a classic, cheap, online mean-shift detector.

To make the test **scale-free across strategies** (a stat-arb book and a momentum book
run at very different daily volatilities), each observation is standardised by a *causal*
running standard deviation (Welford) before entering the cumulative sums. ``delta`` and
``lambda_`` are therefore expressed in units of daily standard deviation, not raw return,
so one set of defaults transfers across strategies.

Two accumulators are maintained so that both an upward and a downward shift in the mean
standardised return are detected:

    increase:  m⁺_t = Σ (z_i − δ),   PH⁺_t = m⁺_t − min_{s≤t} m⁺_s
    decrease:  m⁻_t = Σ (z_i + δ),   PH⁻_t = max_{s≤t} m⁻_s − m⁻_t

where z_i is x_i standardised by the running mean/std known *before* seeing x_i. An alarm
fires when PH⁺_t > λ or PH⁻_t > λ; after an alarm the detector resets.

Reference: Page (1954); Gama et al. (2013), "On evaluating stream learning algorithms".
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .base import Detector, DetectionResult


class PageHinkley(Detector):
    name = "page_hinkley"

    def __init__(self, delta: float = 0.5, lambda_: float = 8.0,
                 min_samples: int = 20):
        """
        Args:
            delta:       magnitude tolerance in std units (per-step slack).
            lambda_:     alarm threshold on the Page-Hinkley statistic (std units).
            min_samples: warm-up period before alarms may fire (stabilise mean/std).
        """
        self.delta = float(delta)
        self.lambda_ = float(lambda_)
        self.min_samples = int(min_samples)

    def detect(self, returns: pd.Series) -> DetectionResult:
        r = self._validate(returns)
        x = r.to_numpy()
        idx = r.index

        alarms: list[pd.Timestamp] = []
        stat = np.zeros(len(x))

        # Welford running mean/variance (reset on alarm).
        n = 0
        mean = 0.0
        m2 = 0.0  # sum of squared deviations
        m_pos = 0.0
        min_pos = 0.0
        m_neg = 0.0
        max_neg = 0.0

        for t, xt in enumerate(x):
            # Standardise using stats known BEFORE incorporating xt (causal).
            std_prev = np.sqrt(m2 / (n - 1)) if n >= 2 else 0.0
            z = (xt - mean) / std_prev if std_prev > 1e-12 else 0.0

            # Update Welford stats with xt.
            n += 1
            delta_x = xt - mean
            mean += delta_x / n
            m2 += delta_x * (xt - mean)

            m_pos += z - self.delta
            min_pos = min(min_pos, m_pos)
            ph_pos = m_pos - min_pos

            m_neg += z + self.delta
            max_neg = max(max_neg, m_neg)
            ph_neg = max_neg - m_neg

            ph = max(ph_pos, ph_neg)
            stat[t] = ph

            if n >= self.min_samples and ph > self.lambda_:
                alarms.append(idx[t])
                # reset
                n = 0
                mean = m2 = 0.0
                m_pos = min_pos = m_neg = max_neg = 0.0

        return DetectionResult(self.name, alarms, pd.Series(stat, index=idx, name=self.name))
