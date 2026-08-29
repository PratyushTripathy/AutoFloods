# autofloods/detectors/zscore.py

import xarray as xr

from .base import FloodDetector


class ZScoreDetector(FloodDetector):
    """
    Z-score anomaly detection on VV/VH backscatter, adapted from Tripathy
    & Malladi (2022) / Global Flood Mapper. A pixel is flagged as flooded
    where its wet-season Z-score (relative to the dry-season per-pixel
    mean and standard deviation) falls at or below a threshold. Pixels
    flagged in both VV and VH are high-confidence floods (class 3);
    pixels flagged in only one band are low-confidence floods (class 1
    or 2). This is the pipeline's default and only field-validated
    detector.
    """

    def __init__(self, vv_thd: float = -2.5, vh_thd: float = -2.5):
        """
        Parameters
        ----------
        vv_thd, vh_thd : float
            Z-score thresholds for the VV and VH bands; a pixel is
            flagged flooded in that band when its anomaly is <= this
            value (more negative = stricter).
        """
        self.vv_thd = vv_thd
        self.vh_thd = vh_thd

    def fit_baseline(self, vv_stack, vh_stack):
        """
        Per-pixel dry-season mean and standard deviation for each band,
        concatenated along a new `band` coordinate
        ['vv_mean', 'vv_std', 'vh_mean', 'vh_std'].
        """
        return xr.concat(
            [
                vv_stack.mean(axis=0),
                vv_stack.std(axis=0),
                vh_stack.mean(axis=0),
                vh_stack.std(axis=0),
            ],
            dim='band',
        ).assign_coords(band=['vv_mean', 'vv_std', 'vh_mean', 'vh_std'])

    def detect(self, baseline, wet_scene):
        """
        Classify `wet_scene` against `baseline` using per-band Z-score
        thresholds (self.vv_thd, self.vh_thd). Returns the 0/1/2/3
        (none/VH/VV/high-confidence) encoding described in
        FloodDetector.detect. Pixels where either band's anomaly is NaN
        (e.g. a tile-edge artifact or missing input data) are set to NaN
        in the output rather than 0, so gaps aren't misread as "not
        flooded".
        """
        # calculate anomaly and flood cells for VV band
        anomaly_vv = (wet_scene.loc['vv_ds'] - baseline.loc['vv_mean']) / baseline.loc['vv_std']
        floods_vv = (anomaly_vv < self.vv_thd).astype(int)

        # calculate anomaly and flood cells for VH band
        anomaly_vh = (wet_scene.loc['vh_ds'] - baseline.loc['vh_mean']) / baseline.loc['vh_std']
        floods_vh = (anomaly_vh < self.vh_thd).astype(int)

        # here's what numbers in the flood map mean
        # 1. flood cells identified in the VH band
        # 2. flood cells identified in the VV band
        # 3. flood cells identified in both VV and VH bands
        combined_floods = floods_vv + floods_vh
        combined_floods = combined_floods.where(floods_vh.values != 1, 1)
        combined_floods = combined_floods.where(floods_vv.values != 1, 2)
        combined_floods = combined_floods.where((floods_vv + floods_vh).values != 2, 3)

        # `NaN < threshold` evaluates to False, so a NaN baseline or
        # wet-scene pixel (e.g. a tile-edge interpolation artifact, or a
        # genuine data gap) would otherwise silently resolve to a
        # valid-looking 0 ("not flooded") via .astype(int) above --
        # indistinguishable from a real negative observation. Re-mask
        # with NaN wherever either band's input was actually invalid, so
        # it exports and aggregates as nodata instead of a false zero.
        invalid = anomaly_vv.isnull() | anomaly_vh.isnull()
        combined_floods = combined_floods.where(~invalid.values)

        return combined_floods

    @property
    def requires_slope_mask(self):
        return True

    @property
    def requires_baseline_fitting(self):
        return True
