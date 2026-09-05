# tests/test_pipeline_logging.py

"""
Confirms each flood_mapper pipeline stage emits its one-line INFO status
summary via the 'autofloods' logger (see autofloods/__init__.py's logging
setup, and the per-method logger.info() calls). These tests only check
that the expected message fires and, where cheap, that the numbers in it
are correct -- they are not full output-content/formatting assertions,
and they don't exercise real network/STAC/GDAL calls: each method's
expensive or network-bound internals are monkeypatched out, matching
tests/test_baseline_skip.py's existing convention for this same class,
so only the pipeline method's own control flow (and, for
generate_number_of_scenes/map_floods, real small in-memory arrays) is
under test.
"""
import logging
import os
from unittest.mock import MagicMock

import numpy as np
import pytest
import xarray as xr

import autofloods
import autofloods.preprocessing
import autofloods.postprocessing
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


class TestGetDryDates:
    def test_logs_dry_months_by_aoi(self, tmp_path, caplog):
        fm = _make_flood_mapper(tmp_path)

        with caplog.at_level(logging.INFO, logger='autofloods'):
            fm.get_dry_dates()

        assert f'Dry season set for {len(fm.dry_months)} AOI(s)' in caplog.text
        assert str(fm.dry_months) in caplog.text


class TestGenerateDryDateRanges:
    def test_logs_search_range(self, tmp_path, caplog):
        fm = _make_flood_mapper(tmp_path)
        fm.get_dry_dates()

        with caplog.at_level(logging.INFO, logger='autofloods'):
            fm.generate_dry_date_ranges()

        assert 'Dry-season search range:' in caplog.text
        assert f'({len(fm.dry_years)} year(s))' in caplog.text

    def test_logs_skip_message_when_all_aois_already_processed(self, tmp_path, caplog):
        fm = _make_flood_mapper(tmp_path)
        fm.dry_months = {}  # simulates every requested AOI already having a baseline

        with caplog.at_level(logging.INFO, logger='autofloods'):
            fm.generate_dry_date_ranges()

        assert 'already have a completed dry-season baseline' in caplog.text


class TestGetS1Items:
    def test_logs_scene_count_found(self, tmp_path, monkeypatch, caplog):
        fm = _make_flood_mapper(tmp_path)
        fm.get_dry_dates()
        fm.generate_dry_date_ranges()

        monkeypatch.setattr(fm.source, 'search_sentinel1', lambda **k: ['s1', 's2', 's3'])
        monkeypatch.setattr(
            utils, 'seggregate_sentinel_search',
            lambda aoi_list, search_items: (
                {TILE_ID: ['s1', 's2', 's3']},
                {'s1': [TILE_ID], 's2': [TILE_ID], 's3': [TILE_ID]},
            ),
        )

        with caplog.at_level(logging.INFO, logger='autofloods'):
            fm.get_s1_items(dry_wet='dry')

        assert 'Found 3 dry-season scene(s)' in caplog.text


class TestReadScenes:
    def test_logs_scenes_read(self, tmp_path, monkeypatch, caplog):
        fm = _make_flood_mapper(tmp_path)
        fm.dry_s1_scenes = ['s1', 's2', 's3']

        monkeypatch.setattr(
            autofloods.preprocessing, 'read_sentinel1_stac',
            lambda item, source, overview_level, bbox=None: (item, {'vv_ds': MagicMock(), 'vh_ds': MagicMock()}),
        )

        with caplog.at_level(logging.INFO, logger='autofloods'):
            fm.read_scenes(dry_wet='dry')

        assert 'Read 3 dry-season scene(s)' in caplog.text


class TestGenerateMeanStdByAoi:
    def test_logs_baseline_fit_complete(self, tmp_path, monkeypatch, caplog):
        fm = _make_flood_mapper(tmp_path)
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

        with caplog.at_level(logging.INFO, logger='autofloods'):
            fm.generate_mean_std_by_aoi()

        assert f'Baseline fit complete for {len(fm.mean_std_by_aoi)} AOI(s)' in caplog.text
        assert str(sorted(fm.mean_std_by_aoi.keys())) in caplog.text


class TestPrepareSlope:
    def test_logs_slope_prepared(self, tmp_path, monkeypatch, caplog):
        fm = _make_flood_mapper(tmp_path)
        fm.mean_std_by_aoi = {TILE_ID: _synthetic_grid_array()}

        monkeypatch.setattr(utils, 'gpd_to_json', lambda **k: [{'dummy': 'bbox'}])
        monkeypatch.setattr(autofloods.utils, 'download_nasadem', lambda *a, **k: MagicMock())
        monkeypatch.setattr(autofloods.preprocessing, 'compute_slope', lambda *a, **k: MagicMock())
        monkeypatch.setattr(autofloods.preprocessing, 'clip_xarray_using_id', lambda *a, **k: MagicMock())
        monkeypatch.setattr(autofloods.utils, 'export_xarray', lambda *a, **k: None)

        with caplog.at_level(logging.INFO, logger='autofloods'):
            fm.prepare_slope()

        assert 'Slope prepared for 1 AOI(s) (1 newly computed)' in caplog.text


class TestPrepareWetScenes:
    def test_logs_wet_scenes_prepared(self, tmp_path, monkeypatch, caplog):
        fm = _make_flood_mapper(tmp_path)
        fm.mean_std_by_aoi = {TILE_ID: _synthetic_grid_array()}
        fm.wet_dates = fm.wet_dates  # already set by __init__

        monkeypatch.setattr(fm.source, 'search_sentinel1', lambda **k: ['s1', 's2', 's3'])
        monkeypatch.setattr(
            utils, 'seggregate_sentinel_search',
            lambda aoi_list, search_items: (
                {TILE_ID: ['s1', 's2', 's3']},
                {'s1': [TILE_ID], 's2': [TILE_ID], 's3': [TILE_ID]},
            ),
        )
        monkeypatch.setattr(
            autofloods.preprocessing, 'read_sentinel1_stac',
            lambda item, source, overview_level, bbox=None: (item, {'vv_ds': MagicMock(), 'vh_ds': MagicMock()}),
        )
        monkeypatch.setattr(
            autofloods.preprocessing, 'clip_xarray_using_id',
            lambda **k: _synthetic_grid_array(n_band=1).squeeze('band', drop=True),
        )

        with caplog.at_level(logging.INFO, logger='autofloods'):
            fm.prepare_wet_scenes()

        assert 'Wet-season scenes prepared for 1 AOI(s), 3 scene(s) total' in caplog.text


class TestMapFloods:
    def test_logs_flood_maps_generated_with_correct_pixel_count(self, tmp_path, caplog):
        fm = _make_flood_mapper(tmp_path)
        fm.mean_std_by_aoi = {TILE_ID: MagicMock()}

        wet_scene_path = tmp_path / 'wetscene_318_scene_a.nc'
        xr.DataArray(
            np.zeros((2, 2, 2)), dims=('band', 'y', 'x'), coords={'band': ['vv_ds', 'vh_ds']},
        ).to_netcdf(wet_scene_path)
        fm.wet_scene_paths = {TILE_ID: {'scene_a': str(wet_scene_path)}}

        classified = xr.DataArray(
            np.array([[3, 3], [0, 1]]), dims=('y', 'x'), coords={'y': [1, 0], 'x': [0, 1]},
        )
        fm.detector = MagicMock()
        fm.detector.requires_slope_mask = False
        fm.detector.detect.return_value = classified

        with caplog.at_level(logging.INFO, logger='autofloods'):
            fm.map_floods(export_vector=False, export_maps=False)

        assert 'Flood maps generated for 1 AOI(s), 1 scene(s): 2 high-confidence flooded pixels' in caplog.text


class TestMergeFloodsByDate:
    def test_logs_dates_merged(self, tmp_path, caplog):
        fm = _make_flood_mapper(tmp_path)
        fm.mean_std_by_aoi = {TILE_ID: _synthetic_grid_array(size=5).squeeze('band', drop=True)}

        classified = _synthetic_grid_array(size=5).squeeze('band', drop=True)
        path_a = tmp_path / 'floodextent_scene_20240701_a.tif'
        path_b = tmp_path / 'floodextent_scene_20240702_b.tif'
        utils.export_xarray(classified, str(path_a))
        utils.export_xarray(classified, str(path_b))
        fm.flood_dict = {TILE_ID: {'scene_20240701_a': str(path_a), 'scene_20240702_b': str(path_b)}}

        with caplog.at_level(logging.INFO, logger='autofloods'):
            fm.merge_floods_by_date(export_raster=False)

        assert 'Merged floods by date for 1 AOI(s), 2 date(s) total' in caplog.text


class TestGenerateNumberOfScenes:
    def test_logs_scene_count_computed(self, tmp_path, caplog):
        fm = _make_flood_mapper(tmp_path)
        fm.mean_std_by_aoi = {TILE_ID: _synthetic_grid_array(size=3).squeeze('band', drop=True).expand_dims(band=[0])}
        fm.wet_scene_paths = {TILE_ID: {'scene_a': 'dummy_path.nc'}}
        fm._wet_scene_valid_count_by_aoi = {TILE_ID: np.array([[1, 0, 1], [1, 1, 1], [1, 1, 1]])}

        with caplog.at_level(logging.INFO, logger='autofloods'):
            fm.generate_number_of_scenes(export_raster=False)

        assert 'Valid-observation-count raster computed for 1 AOI(s)' in caplog.text


class TestMonthlySum:
    def test_logs_monthly_aggregation_complete(self, tmp_path, monkeypatch, caplog):
        fm = _make_flood_mapper(tmp_path)
        fm.flood_raster_dict = {TILE_ID: 'dummy_path.tif'}

        monkeypatch.setattr(autofloods.postprocessing, 'aggregate_monthly', lambda *a, **k: None)

        with caplog.at_level(logging.INFO, logger='autofloods'):
            fm.monthly_sum()

        assert 'Monthly aggregation complete for 1 AOI(s)' in caplog.text
