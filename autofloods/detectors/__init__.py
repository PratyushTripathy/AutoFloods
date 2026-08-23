# autofloods/detectors/__init__.py

"""
Flood detection backends for autofloods.

Currently implemented: ZScoreDetector, the pipeline's default and only
field-validated method (Z-score thresholding on VV/VH backscatter,
adapted from Global Flood Mapper).

Future work, not yet implemented: a deep-learning backend. The
FloodDetector interface's requires_baseline_fitting flag exists
specifically so a pretrained DL detector (which loads weights once,
globally, rather than fitting a per-tile dry-season baseline) can skip
fit_baseline() entirely without a redesign of this interface.
"""

from .base import FloodDetector
from .zscore import ZScoreDetector

__all__ = ["FloodDetector", "ZScoreDetector"]
