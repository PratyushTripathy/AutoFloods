# tests/test_otsu_detector.py

"""
OtsuDetector-specific behaviour, including the degenerate cases from its
class docstring. No network access -- every fixture is synthetic.

Fixtures are built in dB, then exponentiated back to linear power
(_db_to_linear) before being handed to the detector -- OtsuDetector's
detect() now does its own linear-to-dB conversion internally (matching
real OPERA RTC-S1 gamma0, which arrives in linear power; see otsu.py),
so a synthetic fixture in raw dB values would be converted a second
time and silently double-transformed.
"""
import numpy as np
import xarray as xr

from autofloods.detectors import OtsuDetector


def _db_to_linear(values_db):
    return 10 ** (values_db / 10.0)


def _wet_scene_from_band(band):
    """Wrap one 2D array as a wet_scene DataArray with both bands set to
    the same values, so a detection in one band implies the other."""
    data = np.stack([band, band])
    size_y, size_x = band.shape
    return xr.DataArray(
        data, dims=('band', 'y', 'x'),
        coords={'band': ['vv_ds', 'vh_ds'], 'y': np.arange(size_y), 'x': np.arange(size_x)},
    )


def test_bimodal_scene_flags_water_as_high_confidence():
    # 200x200 so OtsuDetector's own bimodality check (tuned for
    # realistic, thousands-of-pixels tile counts -- see otsu.py) has a
    # stable histogram to work with.
    size = 200
    half = size // 2
    rng = np.random.default_rng(11)
    land_db = rng.normal(-10.0, 1.0, size=(size, half))
    water_db = rng.normal(-22.0, 0.5, size=(size, size - half))
    band = _db_to_linear(np.concatenate([land_db, water_db], axis=1)).astype('float32')

    detector = OtsuDetector()
    result = detector.detect(baseline=None, wet_scene=_wet_scene_from_band(band))

    assert (result.values[:, half:] == 3).mean() > 0.95
    assert (result.values[:, :half] == 0).mean() > 0.95


def test_all_nodata_band_returns_all_nan_not_an_exception():
    size = 10
    band = np.full((size, size), np.nan, dtype='float32')

    detector = OtsuDetector()
    result = detector.detect(baseline=None, wet_scene=_wet_scene_from_band(band))

    # All-invalid input: detect()'s final invalid-pixel re-mask NaNs out
    # every pixel regardless of the (all-False) per-band fallback
    # underneath -- no valid observation means no classification, not a
    # false "not flooded" 0.
    assert np.isnan(result.values).all()


def test_too_few_valid_pixels_returns_all_zero_not_an_exception():
    # Below MIN_VALID_PIXELS but not all-NaN, so the invalid-mask doesn't
    # itself NaN out the result -- this exercises the n_valid floor
    # specifically, distinct from the all-nodata case above.
    size = 5  # 25 px < MIN_VALID_PIXELS (100)
    rng = np.random.default_rng(12)
    band = _db_to_linear(rng.normal(-12.0, 2.0, size=(size, size))).astype('float32')

    detector = OtsuDetector()
    result = detector.detect(baseline=None, wet_scene=_wet_scene_from_band(band))

    assert (result.values == 0).all()


def test_unimodal_scene_returns_all_zero_not_an_exception():
    # A single, smooth Gaussian population (one surface type, e.g. bare
    # soil with no water at all) -- large enough for the bimodality
    # check to be stable, per its own docstring.
    size = 200
    rng = np.random.default_rng(13)
    band = _db_to_linear(rng.normal(-12.0, 1.5, size=(size, size))).astype('float32')

    detector = OtsuDetector()
    result = detector.detect(baseline=None, wet_scene=_wet_scene_from_band(band))

    assert (result.values == 0).all()


def test_constant_band_returns_all_zero_not_an_exception():
    size = 20
    band = np.full((size, size), _db_to_linear(-12.0), dtype='float32')

    detector = OtsuDetector()
    result = detector.detect(baseline=None, wet_scene=_wet_scene_from_band(band))

    assert (result.values == 0).all()
