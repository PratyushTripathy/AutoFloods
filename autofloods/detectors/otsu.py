# autofloods/detectors/otsu.py

import logging

import numpy as np
import xarray as xr
from skimage.filters import threshold_otsu

from .base import FloodDetector

logger = logging.getLogger(__name__)

# Minimum number of valid (non-NaN) pixels required before we trust a
# per-band histogram enough to fit an Otsu threshold to it at all. Below
# this, a "threshold" is really just noise -- chosen as a low, permissive
# floor (not a statistically derived value) meant to catch genuinely
# tiny/empty inputs, not to second-guess otherwise-valid tiles.
MIN_VALID_PIXELS = 100

# Bimodality check parameters: before trusting a threshold, we require
# the band's own smoothed histogram to show at least two real local
# peaks -- Otsu always returns *some* threshold even for a single-mode
# histogram (it doesn't test its own two-class assumption), so this is
# what actually catches that case. Otsu's between-class/total-variance
# separability ratio was tried first and rejected: a plain single
# Gaussian split at its mean already has a separability around 0.6-0.7
# (verified numerically), so it doesn't distinguish "genuinely bimodal"
# from "smooth unimodal" at all -- counting real histogram peaks does.
# N_HIST_BINS/PEAK_SMOOTHING_WINDOW are tuned for realistic tile pixel
# counts (thousands+); MIN_PEAK_HEIGHT_FRAC filters bins-of-few-counts
# noise bumps out of the peak count.
N_HIST_BINS = 32
PEAK_SMOOTHING_WINDOW = 5
MIN_PEAK_HEIGHT_FRAC = 0.05


class OtsuDetector(FloodDetector):
    """
    Otsu thresholding on each wet-season scene's own VV/VH histogram
    (skimage.filters.threshold_otsu), classifying pixels below the
    threshold as water -- same low-backscatter-is-water logic as
    ZScoreDetector, just without a dry-season reference. This is a
    single-scene method: it needs no dry-season baseline at all, so
    requires_baseline_fitting is False and fit_baseline() is a genuine
    no-op that the orchestrator never calls.

    Output uses the same 0/1/2/3 (none/VH/VV/high-confidence) encoding
    as ZScoreDetector, so every downstream step (merge_floods_by_date,
    generate_number_of_scenes, monthly_sum, the mosaicking scripts)
    works unchanged.

    Not the default detector, and not used for any result in the
    SoftwareX manuscript -- see ZScoreDetector's docstring and
    Section 2.1 of the paper for why Z-score is preferred in production:
    single-scene thresholding is more prone to false positives over
    smooth, non-water surfaces (bare soil, paved areas) than a method
    that compares each scene against a multi-scene dry-season baseline
    for the same pixel.

    Degenerate per-band cases (handled by returning an all-"not flooded"
    classification for that band, with a logged warning, rather than
    raising):
      - fewer than MIN_VALID_PIXELS valid (non-NaN) pixels in the band
        (empty/all-nodata tile, or too small a sample to trust a
        histogram at all);
      - a constant band (zero or one unique value -- no threshold is
        possible; skimage.filters.threshold_otsu itself raises on this);
      - a histogram with fewer than two real local peaks after light
        smoothing (see _is_bimodal), meaning the band doesn't actually
        show two distinct populations even though Otsu still returned a
        number for it.
    """

    def fit_baseline(self, vv_stack, vh_stack):
        """No-op: Otsu needs no dry-season baseline. Per requires_base
        line_fitting=False, the orchestrator never calls this -- it
        exists only to satisfy the FloodDetector interface. Returns an
        empty marker DataArray, per the interface's documented no-op
        contract (see FloodDetector.fit_baseline's docstring)."""
        return xr.DataArray()

    def detect(self, baseline, wet_scene):
        vv = wet_scene.loc['vv_ds']
        vh = wet_scene.loc['vh_ds']

        floods_vv = self._below_otsu_threshold(vv).astype(int)
        floods_vh = self._below_otsu_threshold(vh).astype(int)

        # Same combination logic as ZScoreDetector.detect(): 1 = VH-only,
        # 2 = VV-only, 3 = both bands agree (high confidence).
        combined_floods = floods_vv + floods_vh
        combined_floods = combined_floods.where(floods_vh.values != 1, 1)
        combined_floods = combined_floods.where(floods_vv.values != 1, 2)
        combined_floods = combined_floods.where((floods_vv + floods_vh).values != 2, 3)

        invalid = vv.isnull() | vh.isnull()
        combined_floods = combined_floods.where(~invalid.values)

        return combined_floods

    def _below_otsu_threshold(self, band):
        """
        Boolean DataArray (True = below this band's own Otsu threshold,
        i.e. candidate water), same shape/coords as `band`. Falls back
        to all-False (logged) for any of the degenerate cases in this
        class's docstring, instead of raising.

        OPERA RTC-S1 gamma0 is delivered in linear power (confirmed by
        direct inspection: real tile 318 VV/VH values cluster near 1.0
        with a long right tail, never negative -- not dB). Otsu
        thresholding on SAR backscatter is conventionally done in dB:
        linear power compresses the water and land populations into one
        right-skewed distribution, which is exactly what this method's
        own bimodality check is designed to reject, and does -- an
        unguarded linear-scale threshold on real tile 318 data flagged
        99.95% of one band as "flooded". Converting to dB restores the
        separation the guard needs to find. Values <= 0 are not expected
        for real gamma0 but would produce -inf under log10; they are
        masked to NaN instead, consistent with how the rest of this
        method treats invalid pixels.
        """
        with np.errstate(divide='ignore', invalid='ignore'):
            values = np.where(band.values > 0, 10 * np.log10(band.values), np.nan)
        valid = ~np.isnan(values)
        n_valid = int(valid.sum())

        if n_valid < MIN_VALID_PIXELS:
            logger.warning(
                'OtsuDetector: only %d valid pixel(s) (< %d), skipping this band '
                '(returning all not-flooded).', n_valid, MIN_VALID_PIXELS,
            )
            return xr.zeros_like(band, dtype=bool)

        valid_values = values[valid]
        if np.unique(valid_values).size < 2:
            logger.warning(
                'OtsuDetector: band is constant (a single value across all valid '
                'pixels), no threshold is possible, skipping this band.',
            )
            return xr.zeros_like(band, dtype=bool)

        if not self._is_bimodal(valid_values):
            logger.warning(
                'OtsuDetector: band histogram has fewer than two real peaks '
                '(likely a unimodal histogram), skipping this band.',
            )
            return xr.zeros_like(band, dtype=bool)

        threshold = threshold_otsu(valid_values)
        below = valid_values < threshold

        w0 = below.mean()
        if w0 == 0.0 or w0 == 1.0:
            logger.warning(
                'OtsuDetector: Otsu threshold assigned every valid pixel to one '
                'class, skipping this band.',
            )
            return xr.zeros_like(band, dtype=bool)

        mask = np.zeros(values.shape, dtype=bool)
        mask[valid] = below
        return xr.DataArray(mask, dims=band.dims, coords=band.coords)

    @staticmethod
    def _is_bimodal(valid_values):
        """
        Whether valid_values' histogram shows at least two real local
        peaks (see N_HIST_BINS/PEAK_SMOOTHING_WINDOW/MIN_PEAK_HEIGHT_FRAC
        above) -- reliable at realistic tile pixel counts (thousands+);
        deliberately not tuned for small (e.g. < 1000-pixel) arrays,
        where histogram noise makes any peak-counting approach unstable.
        """
        counts, _ = np.histogram(valid_values, bins=N_HIST_BINS)
        counts = counts.astype(float)
        if PEAK_SMOOTHING_WINDOW > 1:
            # Edge-replicate padding before smoothing, not the zero-
            # padding np.convolve(..., mode='same') does implicitly:
            # zero-padding here invents a fake decline immediately past
            # the histogram's edge, which can flatten a real tail-end
            # mode into an exact-tie plateau (observed directly: a tight
            # cluster concentrated in the last two bins smoothed into
            # three bins tied at precisely the same value, and a strict
            # local-max test finds no peak on a plateau). Edge
            # replication assumes no such decline, which is the neutral
            # choice when what lies beyond the observed range is
            # actually unknown.
            half_window = PEAK_SMOOTHING_WINDOW // 2
            edge_padded = np.pad(counts, half_window, mode='edge')
            kernel = np.ones(PEAK_SMOOTHING_WINDOW) / PEAK_SMOOTHING_WINDOW
            counts = np.convolve(edge_padded, kernel, mode='valid')
        if counts.max() == 0:
            return False
        min_peak_height = counts.max() * MIN_PEAK_HEIGHT_FRAC
        # Pad with an implicit zero on each side before finding local
        # maxima, so a real mode sitting in the first or last bin (its
        # only neighbour is the edge of the observed range, not another
        # bin) can still register as a peak. Without this, a two-mode
        # histogram where one mode happens to be pinned at the range's
        # own min/max -- which is exactly what a tight, near-saturated
        # land cluster can look like -- was undercounted to one peak and
        # wrongly rejected as unimodal (caught by this class's own test
        # suite, not a hypothetical).
        padded = np.concatenate(([0.0], counts, [0.0]))
        is_peak = (
            (padded[1:-1] > padded[:-2])
            & (padded[1:-1] > padded[2:])
            & (padded[1:-1] >= min_peak_height)
        )
        return int(is_peak.sum()) >= 2

    @property
    def requires_slope_mask(self):
        # Otsu is, if anything, more prone to false positives on terrain
        # than Z-score (no dry-season baseline to help distinguish a
        # genuinely low-backscatter surface from steep-slope layover/
        # shadow), so slope masking still applies.
        return True

    @property
    def requires_baseline_fitting(self):
        return False
