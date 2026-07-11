"""Detector interface shared by all classical change-point detectors.

Every detector is **online / causal**: the statistic emitted at day *t* is a function
of the return series up to and including *t* only. ``detect`` runs the detector over a
full daily return series and returns the alarm dates plus a per-day diagnostic series.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import pandas as pd


@dataclass
class DetectionResult:
    """Output of running a detector over a return series.

    Attributes:
        detector:   the detector's name.
        alarms:     sorted list of pd.Timestamp dates on which the detector fired.
        statistic:  per-day diagnostic Series (same index as the input), e.g. the
                    test statistic or filtered probability; for plotting / debugging.
    """

    detector: str
    alarms: list[pd.Timestamp]
    statistic: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))

    def alarm_index(self) -> pd.DatetimeIndex:
        return pd.DatetimeIndex(sorted(self.alarms))


class Detector(ABC):
    """Abstract base class for online change-point detectors."""

    #: short, stable identifier used in result tables and aggregation.
    name: str = "detector"

    @abstractmethod
    def detect(self, returns: pd.Series) -> DetectionResult:
        """Run the detector over a daily return Series (DatetimeIndex).

        Args:
            returns: daily portfolio returns indexed by trading date, ascending.

        Returns:
            DetectionResult with alarm timestamps and a per-day statistic.
        """
        raise NotImplementedError

    @staticmethod
    def _validate(returns: pd.Series) -> pd.Series:
        """Coerce/validate the input return series."""
        if not isinstance(returns, pd.Series):
            raise TypeError("returns must be a pandas Series")
        if not isinstance(returns.index, pd.DatetimeIndex):
            raise TypeError("returns must be indexed by a DatetimeIndex")
        if not returns.index.is_monotonic_increasing:
            returns = returns.sort_index()
        return returns.astype(float).dropna()
