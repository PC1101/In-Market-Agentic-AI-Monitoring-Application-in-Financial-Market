"""Distributional-threshold detector.

Compares the return distribution of a short *recent* window against a longer trailing
*baseline* window and fires when the recent window looks distributionally different.
Two complementary statistics are monitored, both standardised by the baseline so the
threshold is scale-free:

    mean shift:      z = (mean_recent − mean_baseline) / (std_baseline / √recent)
    volatility jump: f = std_recent / std_baseline

An alarm fires when |z| > ``z_threshold`` OR f > ``vol_threshold``. The two windows do
not overlap (baseline ends where recent begins), so the comparison is between disjoint
samples. A cooldown suppresses repeated alarms from the same episode. Fully causal: at
day *t* only returns up to *t* are used.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .base import Detector, DetectionResult


class DistributionalThreshold(Detector):
    name = "distributional"

    def __init__(self, baseline: int = 120, recent: int = 20,
                 z_threshold: float = 3.0, vol_threshold: float = 2.0,
                 cooldown: int = 10):
        """
        Args:
            baseline:      length of the trailing baseline window (trading days).
            recent:        length of the recent window (trading days).
            z_threshold:   absolute z-score threshold for a mean shift.
            vol_threshold: ratio threshold for a volatility jump (recent/baseline std).
            cooldown:      days to suppress further alarms after firing.
        """
        self.baseline = int(baseline)
        self.recent = int(recent)
        self.z_threshold = float(z_threshold)
        self.vol_threshold = float(vol_threshold)
        self.cooldown = int(cooldown)

    def detect(self, returns: pd.Series) -> DetectionResult:
        r = self._validate(returns)
        x = r.to_numpy()
        idx = r.index
        n = len(x)

        alarms: list[pd.Timestamp] = []
        stat = np.zeros(n)

        need = self.baseline + self.recent
        cooldown_until = -1

        for t in range(n):
            if t + 1 < need:
                continue
            recent_win = x[t + 1 - self.recent : t + 1]
            baseline_win = x[t + 1 - need : t + 1 - self.recent]

            mu_b = baseline_win.mean()
            sd_b = baseline_win.std(ddof=1)
            sd_r = recent_win.std(ddof=1)
            mu_r = recent_win.mean()

            if sd_b <= 1e-12:
                continue

            z = (mu_r - mu_b) / (sd_b / np.sqrt(self.recent))
            vol_ratio = sd_r / sd_b
            # Combined severity for the diagnostic series.
            stat[t] = max(abs(z) / self.z_threshold, vol_ratio / self.vol_threshold)

            fired = abs(z) > self.z_threshold or vol_ratio > self.vol_threshold
            if fired and t > cooldown_until:
                alarms.append(idx[t])
                cooldown_until = t + self.cooldown

        return DetectionResult(self.name, alarms, pd.Series(stat, index=idx, name=self.name))
