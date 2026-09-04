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
synthetic, and built directly in linear power (matching real OPERA
RTC-S1 gamma0; see otsu.py). Values are NOT built by generating a dB
Gaussian and exponentiating it back -- that round-trip was tried first
and rejected: exponentiating a Gaussian produces a log-normal, heavy-
right-tailed distribution, which inflates the dry-season stack's linear
std enough to corrupt ZScoreDetector's anomaly calculation (verified:
it dropped the water/land separation from >95% to ~49%, indistinguishable
from noise). Real dry-season linear power is itself tight (confirmed
directly against a real production baseline: std ~0.7% of the mean),
so the fixture below is built that way from the start.
OtsuDetector.detect() does its own linear-to-dB conversion internally;
ZScoreDetector's anomaly calculation only needs a low-variance linear
baseline, which this fixture provides directly.
"""
import numpy as np
import xarray as xr
import pytest

from autofloods.detectors import ZScoreDetector, OtsuDetector

DETECTOR_FACTORIES = [ZScoreDetector, OtsuDetector]


def _synthetic_dry_stack(n_scenes=4, size=200, band_mean=1.0, band_std=0.02, seed=0):
    rng = np.random.default_rng(seed)
    data = rng.normal(band_mean, band_std, size=(n_scenes, size, size)).astype('float32')
    return xr.DataArray(
        data, dims=('band', 'y', 'x'),
        coords={'band': np.arange(n_scenes), 'y': np.arange(size), 'x': np.arange(size)},
    )


def _synthetic_dry_stats(**kwargs):
    """fit_baseline()'s current interface takes {'mean':.., 'std':..}
    dicts (see autofloods.preprocessing.compute_dry_baseline_stats),
    computed here the same way the old stack-then-reduce path did --
    this contract test isn't about that reduction itself (see
    tests/test_preprocessing.py::TestComputeDryBaselineStats for that),
    just about detectors accepting the current interface shape."""
    stack = _synthetic_dry_stack(**kwargs)
    return {'mean': stack.mean(axis=0), 'std': stack.std(axis=0)}


def _synthetic_wet_scene(size=200, seed=1):
    # Two clearly separated populations per band, in linear power: land
    # matches the dry baseline's own mean (~1.0, so ZScoreDetector's
    # anomaly is ~0, not flagged), water sits an order of magnitude
    # lower (~0.1, ~10 dB down -- well past both ZScoreDetector's -2.5 SD
    # threshold and OtsuDetector's bimodality check once it converts to
    # dB internally). Not tuned to either detector specifically.
    rng = np.random.default_rng(seed)
    half = size // 2
    land = rng.normal(1.0, 0.02, size=(size, half))
    water = rng.normal(0.1, 0.01, size=(size, size - half))
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
    dry = _synthetic_dry_stats()
    baseline = d.fit_baseline(dry, dry)
    assert baseline is not None


@pytest.mark.parametrize('detector_cls', DETECTOR_FACTORIES)
def test_detect_returns_valid_encoding(detector_cls):
    d = detector_cls()
    dry = _synthetic_dry_stats()
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
