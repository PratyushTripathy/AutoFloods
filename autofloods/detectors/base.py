# autofloods/detectors/base.py

from abc import ABC, abstractmethod

import xarray as xr


class FloodDetector(ABC):
    """
    Contract a flood-classification method must satisfy so that
    autofloods.flood_mapper can run it without knowing which detection
    method it is.
    """

    @abstractmethod
    def fit_baseline(self, vv_stack: xr.DataArray, vh_stack: xr.DataArray) -> xr.DataArray:
        """
        Given the stacked dry-season VV and VH DataArrays for one tile
        (as produced by autofloods.preprocessing.stack_images), return
        whatever per-pixel baseline this method needs (Z-score: mean+std
        per band). If requires_baseline_fitting is False, this may be a
        no-op returning an empty/marker DataArray -- callers should check
        that flag rather than assume this always does real work.
        """

    @abstractmethod
    def detect(self, baseline: xr.DataArray, wet_scene: xr.DataArray) -> xr.DataArray:
        """
        Given the baseline from fit_baseline() and one wet-season scene
        (a DataArray with band=['vv_ds', 'vh_ds']), return a classified
        DataArray using this package's existing 0/1/2/3
        (none/VH/VV/high-confidence) encoding.
        """

    @property
    @abstractmethod
    def requires_slope_mask(self) -> bool:
        """
        Whether detect()'s output should still go through the existing
        slope-masking post-step.
        """

    @property
    @abstractmethod
    def requires_baseline_fitting(self) -> bool:
        """
        Whether this detector needs a per-tile dry-season baseline at
        all. False for e.g. a pretrained deep-learning backend that loads
        weights once, globally, rather than fitting per-AOI. When False,
        callers should skip fit_baseline() entirely rather than call it
        and discard the result.
        """
