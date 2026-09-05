# tests/test_keep_intermediate_in_memory.py

"""
Tests for flood_mapper's keep_intermediate_in_memory flag (default
False): confirms the default behavior still deletes self.s1_dry_dict
(generate_mean_std_by_aoi()) and self.slope (prepare_slope()), and that
setting the flag True instead keeps them resident AND makes
map_floods() use the in-memory self.slope instead of re-reading the
same data from disk (the actual point of the flag -- see __init__'s
docstring for the memory/IO tradeoff).

No real network/STAC/GDAL calls: matching tests/test_pipeline_logging.py's
and tests/test_baseline_skip.py's existing convention for this class,
each method's expensive/network-bound internals are monkeypatched out
so only this flag's own control flow is under test.
"""
import os
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import xarray as xr

import autofloods
import autofloods.preprocessing
import autofloods.utils as utils
from autofloods.detectors import ZScoreDetector

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GRID_PATH = os.path.join(REPO_ROOT, 'resources', 'india_utm_fishnet_buffer.gpkg')
TILE_ID = 318


def _make_flood_mapper(tmp_path, **kwargs):
    defaults = dict(
        grid_shapefile=GRID_PATH,
        grid_id_list=[TILE_ID],
        dry_years=[2024, 2024],
        wet_duration=['2024/07', '2024/10'],
        detector=ZScoreDetector(),
        output_dir=str(tmp_path),
    )
    defaults.update(kwargs)
    return autofloods.flood_mapper(**defaults)


def _synthetic_grid_array(size=3, cell_size=30.0, n_band=1):
    y = 3_000_000 - np.arange(size) * cell_size
    x = 400_000 + np.arange(size) * cell_size
    data = np.random.rand(n_band, size, size).astype('float32')
    da = xr.DataArray(
        data, dims=('band', 'y', 'x'), coords={'band': np.arange(n_band), 'y': y, 'x': x},
    )
    return da.rio.write_crs('EPSG:32645')


def _run_generate_mean_std_by_aoi(fm, monkeypatch):
    fm.s1_dry_dict = {}
    fm.dry_aoi_scene_dict = {TILE_ID: []}
    monkeypatch.setattr(autofloods.preprocessing, 'reproject_clip_stac', lambda *a, **k: {})
    monkeypatch.setattr(
        autofloods.preprocessing, 'compute_dry_baseline_stats',
        lambda *a, **k: {
            'vv': {'mean': _synthetic_grid_array().squeeze('band', drop=True),
                   'std': _synthetic_grid_array().squeeze('band', drop=True)},
            'vh': {'mean': _synthetic_grid_array().squeeze('band', drop=True),
                   'std': _synthetic_grid_array().squeeze('band', drop=True)},
            'grid_ref': _synthetic_grid_array(),
        },
    )
    fm.generate_mean_std_by_aoi()


def _run_prepare_slope(fm, monkeypatch):
    fm.mean_std_by_aoi = {TILE_ID: _synthetic_grid_array()}
    monkeypatch.setattr(utils, 'gpd_to_json', lambda **k: [{'dummy': 'bbox'}])
    monkeypatch.setattr(autofloods.utils, 'download_nasadem', lambda *a, **k: MagicMock())
    monkeypatch.setattr(
        autofloods.preprocessing, 'compute_slope', lambda *a, **k: _synthetic_grid_array(),
    )
    monkeypatch.setattr(
        autofloods.preprocessing, 'clip_xarray_using_id', lambda *a, **k: _synthetic_grid_array(),
    )
    monkeypatch.setattr(autofloods.utils, 'export_xarray', lambda *a, **k: None)
    fm.prepare_slope()


class TestDefaultDeletesIntermediates:
    def test_s1_dry_dict_deleted_after_generate_mean_std_by_aoi(self, tmp_path, monkeypatch):
        fm = _make_flood_mapper(tmp_path)
        assert fm.keep_intermediate_in_memory is False

        _run_generate_mean_std_by_aoi(fm, monkeypatch)

        assert not hasattr(fm, 's1_dry_dict')

    def test_slope_deleted_after_prepare_slope(self, tmp_path, monkeypatch):
        fm = _make_flood_mapper(tmp_path)
        assert fm.keep_intermediate_in_memory is False

        _run_prepare_slope(fm, monkeypatch)

        assert not hasattr(fm, 'slope')


class TestKeepIntermediateInMemoryRetainsAttributes:
    def test_s1_dry_dict_retained_when_flag_true(self, tmp_path, monkeypatch):
        fm = _make_flood_mapper(tmp_path, keep_intermediate_in_memory=True)

        _run_generate_mean_std_by_aoi(fm, monkeypatch)

        assert hasattr(fm, 's1_dry_dict')
        assert fm.s1_dry_dict == {}

    def test_slope_retained_when_flag_true(self, tmp_path, monkeypatch):
        fm = _make_flood_mapper(tmp_path, keep_intermediate_in_memory=True)

        _run_prepare_slope(fm, monkeypatch)

        assert hasattr(fm, 'slope')
        assert TILE_ID in fm.slope


class TestMapFloodsUsesInMemorySlope:
    def _setup_for_map_floods(self, tmp_path, keep_in_memory, slope_available):
        fm = _make_flood_mapper(tmp_path, keep_intermediate_in_memory=keep_in_memory)
        fm.mean_std_by_aoi = {TILE_ID: _synthetic_grid_array()}
        fm.wet_scenes_by_aoi = {TILE_ID: {'scene_a': MagicMock()}}

        classified = xr.DataArray(np.array([[3, 0, 0], [0, 0, 0], [0, 0, 0]]))
        fm.detector = MagicMock()
        fm.detector.requires_slope_mask = True
        fm.detector.detect.return_value = classified

        if slope_available:
            fm.slope = {TILE_ID: _synthetic_grid_array()}

        return fm

    def test_skips_disk_read_when_in_memory_slope_available(self, tmp_path):
        fm = self._setup_for_map_floods(tmp_path, keep_in_memory=True, slope_available=True)

        with patch('autofloods.xr.load_dataarray') as mock_load:
            fm.map_floods(export_raster=False, export_vector=False, export_maps=False)

        mock_load.assert_not_called()

    def test_falls_back_to_disk_read_when_in_memory_slope_missing(self, tmp_path):
        fm = self._setup_for_map_floods(tmp_path, keep_in_memory=True, slope_available=False)

        with patch('autofloods.xr.load_dataarray', return_value=_synthetic_grid_array()) as mock_load:
            fm.map_floods(export_raster=False, export_vector=False, export_maps=False)

        mock_load.assert_called_once()

    def test_default_mode_always_reads_from_disk_even_if_slope_attribute_exists(self, tmp_path):
        # Simulates a stale self.slope left over from some other path --
        # default (False) mode must never trust it, always re-reading.
        fm = self._setup_for_map_floods(tmp_path, keep_in_memory=False, slope_available=True)

        with patch('autofloods.xr.load_dataarray', return_value=_synthetic_grid_array()) as mock_load:
            fm.map_floods(export_raster=False, export_vector=False, export_maps=False)

        mock_load.assert_called_once()
