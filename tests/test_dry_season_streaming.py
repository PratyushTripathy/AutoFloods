# tests/test_dry_season_streaming.py

"""
Correctness verification for the bounded read+reproject+cache pipeline
now behind read_scenes()'s default (keep_intermediate_in_memory=False)
path, for the DRY season specifically -- confirms it reproduces the
existing (keep_intermediate_in_memory=True) bulk in-memory path's
output bit-identically, not just "runs without crashing".

Dry needed its own explicit check (separate from the wet-season
streaming tests) because dry's current bulk path does TWO regrid passes
(reproject_clip_stac()'s rough native->tile-UTM reproject+clip, then
compute_dry_baseline_stats()'s own clip_xarray_using_id align onto the
final explicit grid) -- clip_xarray_using_id's resampling is not
perfectly idempotent, so a naive single-pass streaming implementation
(reprojecting straight from native CRS, as wet's pre-existing code
already did) would NOT be bit-identical to dry's existing two-pass
output, only numerically close. The streaming implementation
deliberately replicates the exact two-pass sequence for dry to avoid
this.

Real native-CRS synthetic scenes (not mocked-away reprojection) are
used, via a real GDAL/rioxarray reproject+clip+interp chain, matching
tests/test_preprocessing.py::TestComputeDryBaselineStats's existing
"numerically equivalent, not just non-crashing" standard. Only the
network read itself (preprocessing.read_sentinel1_stac) is mocked.
"""
import os
import threading
import time
import types

import geopandas as gpd
import numpy as np
import pytest
import xarray as xr
from shapely.geometry import box

import autofloods
import autofloods.preprocessing as preprocessing
import autofloods.utils as utils
from autofloods.detectors import ZScoreDetector

TILE_ID = 1


def _make_grid_file(tmp_path, ids=(1,), zone='45R', bounds_list=None, filename='grid.gpkg'):
    """Small synthetic tile grid geopackage, matching the schema
    clip_xarray_using_id/reproject_clip_stac read (ID, zone, dry_month,
    geometry). Supports multiple tiles for the fan-out test."""
    if bounds_list is None:
        bounds_list = [(85.0, 25.0, 85.5, 25.5)] * len(ids)
    gdf = gpd.GeoDataFrame(
        {
            'ID': list(ids),
            'zone': [zone] * len(ids),
            'dry_month': ['04,05'] * len(ids),
            'geometry': [box(*b) for b in bounds_list],
        },
        crs='EPSG:4326',
    )
    path = tmp_path / filename
    gdf.to_file(path)
    return str(path)


def _make_source_dataarray(bounds=(84.9, 24.9, 85.6, 25.6), size=40, fill=None):
    """A synthetic EPSG:4326 DataArray, comfortably covering `bounds`,
    with a real incidental leading 'band' dim of size 1 -- matching
    tests/test_preprocessing.py's identical fixture (see its docstring
    for why the band dim matters)."""
    x_min, y_min, x_max, y_max = bounds
    xs = np.linspace(x_min, x_max, size)
    ys = np.linspace(y_max, y_min, size)
    data = np.ones((size, size), dtype='float64') if fill is None else fill
    da = xr.DataArray(
        data[np.newaxis, :, :], dims=('band', 'y', 'x'),
        coords={'band': [1], 'y': ys, 'x': xs},
    )
    da.rio.write_crs('EPSG:4326', inplace=True)
    return da


def _make_flood_mapper(tmp_path, grid_path, grid_id_list=(TILE_ID,)):
    return autofloods.flood_mapper(
        grid_shapefile=grid_path,
        grid_id_list=list(grid_id_list),
        dry_years=[2024, 2024],
        wet_duration=['2024/07', '2024/07'],
        detector=ZScoreDetector(vv_thd=-2.5, vh_thd=-2.5),
        output_dir=str(tmp_path),
    )


def _wire_dry_scene_mocks(fm, monkeypatch, scene_defs, aoi_scene_dict):
    """scene_defs: {scene_id: {'vv_ds': DataArray, 'vh_ds': DataArray}}
    (real native-CRS arrays). aoi_scene_dict: {aoi_id: [scene_id, ...]}."""
    scene_ids = list(scene_defs.keys())
    fake_items = [types.SimpleNamespace(id=s) for s in scene_ids]

    monkeypatch.setattr(fm.source, 'search_sentinel1', lambda **k: fake_items)
    scene_aoi_dict = {}
    for aoi_id, sids in aoi_scene_dict.items():
        for sid in sids:
            scene_aoi_dict.setdefault(sid, []).append(aoi_id)
    monkeypatch.setattr(
        utils, 'seggregate_sentinel_search',
        lambda aoi_list, search_items: (dict(aoi_scene_dict), scene_aoi_dict),
    )
    monkeypatch.setattr(
        preprocessing, 'read_sentinel1_stac',
        lambda item, source, overview_level, bbox=None: (item.id, scene_defs[item.id]),
    )


class TestDryStreamingMatchesOldBulkPath:
    def test_baseline_stats_bit_identical(self, tmp_path, monkeypatch):
        grid_path = _make_grid_file(tmp_path)
        rng = np.random.default_rng(0)
        n_scenes = 4
        scene_defs = {}
        for i in range(n_scenes):
            vv_fill = rng.normal(1.0, 0.1, size=(40, 40))
            vh_fill = rng.normal(2.0, 0.2, size=(40, 40))
            if i == 1:
                # a stray large sentinel value (upstream nodata artifact),
                # exercising the NaN-aware fold path, not just plain mean/std
                vv_fill[0:3, 0:3] = 99.0
            scene_defs[f'scene{i}'] = {
                'vv_ds': _make_source_dataarray(fill=vv_fill),
                'vh_ds': _make_source_dataarray(fill=vh_fill),
            }
        aoi_scene_dict = {TILE_ID: list(scene_defs.keys())}

        # --- OLD bulk path: real reproject_clip_stac + compute_dry_baseline_stats ---
        s1_dry_dict = {sid: scene_defs[sid] for sid in scene_defs}
        old_reprojected = preprocessing.reproject_clip_stac(
            s1_dry_dict, aoi_scene_dict, grid_path, TILE_ID, max_workers=1,
        )
        old_stats = preprocessing.compute_dry_baseline_stats(
            old_reprojected, grid_path, TILE_ID, max_workers=1, cell_size=100,
        )

        # --- NEW streaming path: real flood_mapper.read_scenes() + generate_mean_std_by_aoi() ---
        fm = _make_flood_mapper(tmp_path, grid_path)
        fm.cell_size = 100
        _wire_dry_scene_mocks(fm, monkeypatch, scene_defs, aoi_scene_dict)

        fm.get_dry_dates()
        fm.generate_dry_date_ranges()
        fm.get_s1_items(dry_wet='dry')
        fm.read_scenes(dry_wet='dry', max_workers=1, reproject_max_workers=1)
        fm.generate_mean_std_by_aoi(reproject_max_workers=1)

        new_baseline = fm.mean_std_by_aoi[TILE_ID]
        np.testing.assert_array_equal(new_baseline.sel(band='vv_mean').values, old_stats['vv']['mean'].values)
        np.testing.assert_array_equal(new_baseline.sel(band='vv_std').values, old_stats['vv']['std'].values)
        np.testing.assert_array_equal(new_baseline.sel(band='vh_mean').values, old_stats['vh']['mean'].values)
        np.testing.assert_array_equal(new_baseline.sel(band='vh_std').values, old_stats['vh']['std'].values)


class TestDryReadSceneFanOut:
    def test_scene_shared_by_two_aois_read_once(self, tmp_path, monkeypatch):
        grid_path = _make_grid_file(
            tmp_path, ids=(1, 2), bounds_list=[(85.0, 25.0, 85.5, 25.5), (85.0, 25.0, 85.5, 25.5)],
        )
        rng = np.random.default_rng(1)
        scene_defs = {
            'shared_scene': {
                'vv_ds': _make_source_dataarray(fill=rng.normal(1.0, 0.1, size=(40, 40))),
                'vh_ds': _make_source_dataarray(fill=rng.normal(2.0, 0.2, size=(40, 40))),
            },
        }
        aoi_scene_dict = {1: ['shared_scene'], 2: ['shared_scene']}

        fm = _make_flood_mapper(tmp_path, grid_path, grid_id_list=(1, 2))
        fm.cell_size = 100

        read_calls = []
        fake_items = [types.SimpleNamespace(id='shared_scene')]
        monkeypatch.setattr(fm.source, 'search_sentinel1', lambda **k: fake_items)
        scene_aoi_dict = {'shared_scene': [1, 2]}
        monkeypatch.setattr(
            utils, 'seggregate_sentinel_search',
            lambda aoi_list, search_items: (dict(aoi_scene_dict), scene_aoi_dict),
        )

        def _spy_read(item, source, overview_level, bbox=None):
            read_calls.append(item.id)
            return item.id, scene_defs[item.id]

        monkeypatch.setattr(preprocessing, 'read_sentinel1_stac', _spy_read)

        fm.get_dry_dates()
        fm.generate_dry_date_ranges()
        fm.get_s1_items(dry_wet='dry')
        fm.read_scenes(dry_wet='dry', max_workers=2, reproject_max_workers=2)

        # read from the network exactly once, despite being needed by 2 AOIs
        assert read_calls == ['shared_scene']
        assert 1 in fm.dry_scene_paths and 'shared_scene' in fm.dry_scene_paths[1]
        assert 2 in fm.dry_scene_paths and 'shared_scene' in fm.dry_scene_paths[2]
        assert os.path.exists(fm.dry_scene_paths[1]['shared_scene'])
        assert os.path.exists(fm.dry_scene_paths[2]['shared_scene'])
        # different AOIs' cache files, even for the same scene (different
        # AOI-specific clip/grid), must not collide on the same path
        assert fm.dry_scene_paths[1]['shared_scene'] != fm.dry_scene_paths[2]['shared_scene']


class TestReadScenesBoundedResidency:
    def test_never_more_than_max_workers_raw_reads_in_flight(self, tmp_path, monkeypatch):
        grid_path = _make_grid_file(tmp_path)
        rng = np.random.default_rng(2)
        n_scenes = 6
        scene_defs = {
            f'scene{i}': {
                'vv_ds': _make_source_dataarray(fill=rng.normal(1.0, 0.1, size=(40, 40))),
                'vh_ds': _make_source_dataarray(fill=rng.normal(2.0, 0.2, size=(40, 40))),
            }
            for i in range(n_scenes)
        }
        aoi_scene_dict = {TILE_ID: list(scene_defs.keys())}

        fm = _make_flood_mapper(tmp_path, grid_path)
        fm.cell_size = 100
        _wire_dry_scene_mocks(fm, monkeypatch, scene_defs, aoi_scene_dict)

        max_workers = 2
        in_flight = {'current': 0, 'max_seen': 0}
        real_read = preprocessing.read_sentinel1_stac

        def _tracked_read(item, source, overview_level, bbox=None):
            in_flight['current'] += 1
            in_flight['max_seen'] = max(in_flight['max_seen'], in_flight['current'])
            result = real_read(item, source, overview_level, bbox)
            in_flight['current'] -= 1
            return result

        # patch AFTER _wire_dry_scene_mocks so this wraps its mock, not the real function
        monkeypatch.setattr(preprocessing, 'read_sentinel1_stac', _tracked_read)

        fm.get_dry_dates()
        fm.generate_dry_date_ranges()
        fm.get_s1_items(dry_wet='dry')
        fm.read_scenes(dry_wet='dry', max_workers=max_workers, reproject_max_workers=1)

        assert in_flight['max_seen'] <= max_workers
        assert len(fm.dry_scene_paths[TILE_ID]) == n_scenes


class TestReadScenesBackpressure:
    def test_raw_residency_bounded_when_reproject_is_slow(self, tmp_path, monkeypatch):
        """
        Regression test for a real bug caught via production-scale
        before/after peak-memory measurement (2026-09-06): an earlier
        version of the read/reproject loop submitted a new network read
        the instant any read completed, with no regard for whether the
        (typically slower, CPU-bound) reproject stage could keep up.
        Fast reads raced ahead of slow reprojection and piled up an
        unbounded number of completed-but-not-yet-reprojected raw
        scenes in reproject_backlog -- silently reproducing the exact
        unbounded-residency bug read_scenes() exists to fix (measured
        peak RSS was HIGHER than the old bulk-read code it replaced).

        Forces reprojection to be much slower than reading
        (reproject_max_workers=1, serialized, with an artificial delay)
        and tracks how many raw scenes are read-but-not-yet-reprojected
        at any moment. Without backpressure this grows toward n_scenes;
        with it, it should stay bounded by roughly
        max_workers + reproject_max_workers.
        """
        grid_path = _make_grid_file(tmp_path)
        rng = np.random.default_rng(3)
        n_scenes = 8
        scene_defs = {
            f'scene{i}': {
                'vv_ds': _make_source_dataarray(fill=rng.normal(1.0, 0.1, size=(40, 40))),
                'vh_ds': _make_source_dataarray(fill=rng.normal(2.0, 0.2, size=(40, 40))),
            }
            for i in range(n_scenes)
        }
        aoi_scene_dict = {TILE_ID: list(scene_defs.keys())}

        fm = _make_flood_mapper(tmp_path, grid_path)
        fm.cell_size = 100
        _wire_dry_scene_mocks(fm, monkeypatch, scene_defs, aoi_scene_dict)

        max_workers = 3
        reproject_max_workers = 1
        lock = threading.Lock()
        state = {'pair_call': 0, 'resident': 0, 'max_resident': 0}
        real_read = preprocessing.read_sentinel1_stac
        real_clip = preprocessing.clip_xarray_using_id

        def _tracked_read(item, source, overview_level, bbox=None):
            result = real_read(item, source, overview_level, bbox)
            with lock:
                state['resident'] += 1
                state['max_resident'] = max(state['max_resident'], state['resident'])
            return result

        def _slow_clip(**kwargs):
            # reproject_max_workers=1 guarantees these calls are never
            # interleaved across threads, so a simple call-parity check
            # correctly identifies the first (vv) vs second (vh) call of
            # each scene's pair.
            with lock:
                state['pair_call'] += 1
                is_first_of_pair = state['pair_call'] % 2 == 1
            if is_first_of_pair:
                time.sleep(0.05)
            result = real_clip(**kwargs)
            if not is_first_of_pair:
                with lock:
                    state['resident'] -= 1
            return result

        monkeypatch.setattr(preprocessing, 'read_sentinel1_stac', _tracked_read)
        monkeypatch.setattr(preprocessing, 'clip_xarray_using_id', _slow_clip)

        fm.get_dry_dates()
        fm.generate_dry_date_ranges()
        fm.get_s1_items(dry_wet='dry')
        fm.read_scenes(dry_wet='dry', max_workers=max_workers, reproject_max_workers=reproject_max_workers)

        assert state['max_resident'] <= max_workers + reproject_max_workers + 1
        assert len(fm.dry_scene_paths[TILE_ID]) == n_scenes
