# tests/test_baseline_skip.py

"""
Verifies flood_mapper.generate_mean_std_by_aoi() actually skips baseline
fitting for a detector with requires_baseline_fitting=False -- the path
OtsuDetector is the first detector to exercise, and which turned out to
be broken (see the docstring on generate_mean_std_by_aoi() and the
"else" branch inside it: an earlier version set mean_std_by_aoi[id] to
None here, which crashed the very first downstream reader of it --
prepare_slope()'s DEM clip -- with an AttributeError on None.rio.bounds()
the first time any detector with requires_baseline_fitting=False was
actually run end to end).

Network- and GDAL-reprojection-heavy calls (preprocessing.reproject_
clip_stac, preprocessing.compute_dry_baseline_stats) are monkeypatched
out with small synthetic, CRS-bearing DataArrays -- this test is about
generate_mean_std_by_aoi()'s own control flow, not about real Sentinel-1
reprojection (see fig_bihar_floods.py's end-to-end OtsuDetector run,
reported separately, for that).
"""
import os

import numpy as np
import pytest
import xarray as xr

import autofloods
import autofloods.preprocessing
from autofloods.detectors import OtsuDetector, ZScoreDetector

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GRID_PATH = os.path.join(REPO_ROOT, 'resources', 'india_utm_fishnet_buffer.gpkg')
TILE_ID = 318


def _synthetic_grid_array(size=24, cell_size=30.0):
    """A CRS-bearing DataArray, standing in for a reprojected dry-season
    VV or VH per-pixel mean/std or grid-reference scene without doing
    any real reprojection."""
    y = 3_000_000 - np.arange(size) * cell_size
    x = 400_000 + np.arange(size) * cell_size
    data = np.random.rand(size, size).astype('float32')
    da = xr.DataArray(data, dims=('y', 'x'), coords={'y': y, 'x': x})
    return da.rio.write_crs('EPSG:32645')


def _synthetic_dry_stats():
    """Stands in for compute_dry_baseline_stats()'s real return value."""
    return {
        'vv': {'mean': _synthetic_grid_array(), 'std': _synthetic_grid_array()},
        'vh': {'mean': _synthetic_grid_array(), 'std': _synthetic_grid_array()},
        'grid_ref': _synthetic_grid_array().expand_dims(band=[0]),
    }


def _make_flood_mapper(tmp_path, detector):
    fm = autofloods.flood_mapper(
        grid_shapefile=GRID_PATH,
        grid_id_list=[TILE_ID],
        dry_years=[2024, 2024],
        wet_duration=['2024/07', '2024/10'],
        detector=detector,
        output_dir=str(tmp_path),
    )
    # get_s1_items(dry_wet='dry')/read_scenes(dry_wet='dry') would
    # normally populate these from a real STAC search; monkeypatching
    # reproject_clip_stac below makes their actual content irrelevant.
    fm.s1_dry_dict = {}
    fm.dry_aoi_scene_dict = {TILE_ID: []}
    return fm


def test_baseline_fitting_is_skipped_and_no_cache_written(tmp_path, monkeypatch):
    fit_baseline_calls = []

    class SpyOtsuDetector(OtsuDetector):
        def fit_baseline(self, vv_stats, vh_stats):
            fit_baseline_calls.append(1)
            return super().fit_baseline(vv_stats, vh_stats)

    detector = SpyOtsuDetector()
    fm = _make_flood_mapper(tmp_path, detector)

    monkeypatch.setattr(autofloods.preprocessing, 'reproject_clip_stac', lambda *a, **k: {})
    monkeypatch.setattr(
        autofloods.preprocessing, 'compute_dry_baseline_stats',
        lambda *a, **k: _synthetic_dry_stats(),
    )

    fm.generate_mean_std_by_aoi()

    assert fit_baseline_calls == [], (
        'OtsuDetector.fit_baseline() must never be called when '
        'requires_baseline_fitting is False'
    )

    mean_std_dir = os.path.join(fm.output_base, 'mean_std')
    written = [f for f in os.listdir(mean_std_dir) if f.endswith('.nc')]
    assert written == [], f'no baseline .nc should be written for this detector, found: {written}'

    # This is exactly what crashed before the fix: mean_std_by_aoi[id]
    # was None, and prepare_slope()/map_floods()/merge_floods_by_date()/
    # generate_number_of_scenes() all read .rio.crs / .rio.bounds() /
    # .coords / .dims off it unconditionally, regardless of detector.
    assert fm.mean_std_by_aoi[TILE_ID] is not None
    assert fm.mean_std_by_aoi[TILE_ID].rio.crs is not None
    assert 'y' in fm.mean_std_by_aoi[TILE_ID].coords
    assert 'x' in fm.mean_std_by_aoi[TILE_ID].coords


def test_baseline_fitting_still_runs_for_zscore(tmp_path, monkeypatch):
    # Control case: confirm the monkeypatched harness itself is sound by
    # checking the True-flag path still behaves as before -- fit_baseline
    # IS called and a .nc cache file IS written.
    fit_baseline_calls = []

    class SpyZScoreDetector(ZScoreDetector):
        def fit_baseline(self, vv_stats, vh_stats):
            fit_baseline_calls.append(1)
            return super().fit_baseline(vv_stats, vh_stats)

    detector = SpyZScoreDetector()
    fm = _make_flood_mapper(tmp_path, detector)

    monkeypatch.setattr(autofloods.preprocessing, 'reproject_clip_stac', lambda *a, **k: {})
    monkeypatch.setattr(
        autofloods.preprocessing, 'compute_dry_baseline_stats',
        lambda *a, **k: _synthetic_dry_stats(),
    )

    fm.generate_mean_std_by_aoi()

    assert fit_baseline_calls == [1]
    mean_std_dir = os.path.join(fm.output_base, 'mean_std')
    written = [f for f in os.listdir(mean_std_dir) if f.endswith('.nc')]
    assert len(written) == 1
