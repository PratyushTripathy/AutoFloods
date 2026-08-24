# tests/test_detectors_contract.py

"""
Parametrised contract test for autofloods.detectors.FloodDetector
implementations.

This is a fresh file, not an existing one being generalised -- there was
no test suite in this repo before OtsuDetector (see this session's other
tests for the same finding re: the orchestrator's baseline-skip path,
which had never been exercised either). It's written parametrised over
DETECTOR_FACTORIES specifically so the FloodDetector abstraction claim
(Section 2.1: "a different detection method could be added without
touching orchestration, tiling, or caching") is something a future third
detector can be checked against by adding one line here, not by writing
a new file.

No network access, no real Sentinel-1 data -- every fixture below is
synthetic.
"""
import numpy as np
import xarray as xr
import pytest

from autofloods.detectors import ZScoreDetector, OtsuDetector

DETECTOR_FACTORIES = [ZScoreDetector, OtsuDetector]


def _synthetic_dry_stack(n_scenes=4, size=200, band_mean=-12.0, band_std=2.0, seed=0):
    rng = np.random.default_rng(seed)
    data = rng.normal(band_mean, band_std, size=(n_scenes, size, size)).astype('float32')
    return xr.DataArray(
        data, dims=('band', 'y', 'x'),
        coords={'band': np.arange(n_scenes), 'y': np.arange(size), 'x': np.arange(size)},
    )


def _synthetic_wet_scene(size=200, seed=1):
    # Two clearly separated populations per band (land ~ -10 dB, water ~
    # -20 dB) at a pixel count realistic enough for OtsuDetector's own
    # bimodality check (see test_otsu_detector.py) as well as
    # ZScoreDetector's simpler thresholding -- this fixture isn't tuned
    # to either detector specifically.
    rng = np.random.default_rng(seed)
    half = size // 2
    land = rng.normal(-10.0, 1.0, size=(size, half))
    water = rng.normal(-20.0, 1.0, size=(size, size - half))
    band = np.concatenate([land, water], axis=1).astype('float32')
    data = np.stack([band, band])
    return xr.DataArray(
        data, dims=('band', 'y', 'x'),
        coords={'band': ['vv_ds', 'vh_ds'], 'y': np.arange(size), 'x': np.arange(size)},
    )


@pytest.mark.parametrize('detector_cls', DETECTOR_FACTORIES)
def test_flags_are_bool(detector_cls):
    d = detector_cls()
    assert isinstance(d.requires_slope_mask, bool)
    assert isinstance(d.requires_baseline_fitting, bool)


@pytest.mark.parametrize('detector_cls', DETECTOR_FACTORIES)
def test_fit_baseline_is_callable(detector_cls):
    # Part of the abstract contract regardless of whether the
    # orchestrator ever actually calls it for this detector (it doesn't,
    # for one with requires_baseline_fitting=False -- see
    # test_baseline_skip.py).
    d = detector_cls()
    dry = _synthetic_dry_stack()
    baseline = d.fit_baseline(dry, dry)
    assert baseline is not None


@pytest.mark.parametrize('detector_cls', DETECTOR_FACTORIES)
def test_detect_returns_valid_encoding(detector_cls):
    d = detector_cls()
    dry = _synthetic_dry_stack()
    baseline = d.fit_baseline(dry, dry) if d.requires_baseline_fitting else None
    wet_scene = _synthetic_wet_scene()

    result = d.detect(baseline, wet_scene)

    assert result.shape == wet_scene.shape[1:]
    valid_values = np.unique(result.values[~np.isnan(result.values)])
    assert set(valid_values.tolist()).issubset({0, 1, 2, 3}), (
        f'{detector_cls.__name__}.detect() returned values outside the '
        f'0/1/2/3 encoding: {valid_values}'
    )
    # The fixture's land/water split is clean and identical in both
    # bands, so every detector should find *some* high-confidence (3)
    # water on the water half and mostly 0 on the land half -- a coarse
    # sanity check that detect() is doing something, not just returning
    # a constant.
    half = wet_scene.sizes['x'] // 2
    assert (result.values[:, half:] == 3).mean() > 0.8
    assert (result.values[:, :half] == 0).mean() > 0.8
