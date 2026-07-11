"""Classical change-point detectors (NumPy/SciPy, from scratch, online/causal)."""

from .base import Detector, DetectionResult
from .page_hinkley import PageHinkley
from .bocpd import BOCPD
from .hmm import HMMDetector
from .distributional import DistributionalThreshold
from .aggregate import aggregate_alarms

__all__ = [
    "Detector",
    "DetectionResult",
    "PageHinkley",
    "BOCPD",
    "HMMDetector",
    "DistributionalThreshold",
    "aggregate_alarms",
    "default_detectors",
]


def default_detectors():
    """Return the standard Week-2 detector suite with default parameters."""
    return [PageHinkley(), BOCPD(), HMMDetector(), DistributionalThreshold()]
