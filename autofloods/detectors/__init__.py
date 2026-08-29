# autofloods/detectors/__init__.py

"""
Flood detection backends for autofloods.

- ZScoreDetector: Z-score thresholding on VV/VH backscatter against a
  dry-season baseline. requires_baseline_fitting = True. The pipeline's
  default detector.
- OtsuDetector: single-scene Otsu thresholding; needs no dry-season
  baseline. requires_baseline_fitting = False.

New backends implement the FloodDetector interface in .base; the
requires_baseline_fitting flag lets a detector (e.g. a pretrained
model loading weights once, globally) skip fit_baseline() entirely.
"""

from .base import FloodDetector
from .zscore import ZScoreDetector
from .otsu import OtsuDetector

__all__ = ["FloodDetector", "ZScoreDetector", "OtsuDetector"]
