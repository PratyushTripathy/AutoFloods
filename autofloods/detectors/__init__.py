# autofloods/detectors/__init__.py

"""
Flood detection backends for autofloods.

Implemented:

- ZScoreDetector: the pipeline's default and only field-validated
  method (Z-score thresholding on VV/VH backscatter, adapted from
  Global Flood Mapper). requires_baseline_fitting = True. Used for
  every result in the SoftwareX manuscript.
- OtsuDetector: single-scene Otsu thresholding, needs no dry-season
  baseline at all. requires_baseline_fitting = False -- the first
  detector to actually exercise that flag's skip path through
  flood_mapper (see flood_mapper.generate_mean_std_by_aoi's docstring).
  Not the default; see OtsuDetector's own docstring for why.

Future work, not yet implemented: a deep-learning backend. The
FloodDetector interface's requires_baseline_fitting flag exists
specifically so a pretrained DL detector (which loads weights once,
globally, rather than fitting a per-tile dry-season baseline) can skip
fit_baseline() entirely without a redesign of this interface.
"""

from .base import FloodDetector
from .zscore import ZScoreDetector
from .otsu import OtsuDetector

__all__ = ["FloodDetector", "ZScoreDetector", "OtsuDetector"]
