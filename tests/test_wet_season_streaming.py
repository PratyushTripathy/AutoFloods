# tests/test_wet_season_streaming.py

"""
Correctness verification for the wet-season streaming restructure
(prepare_wet_scenes() -> map_floods() -> merge_floods_by_date()/
generate_number_of_scenes()): confirms the new disk-cache-based,
one-scene-at-a-time pipeline produces bit-identical results to the old
in-memory (self.wet_scenes_by_aoi/self.flood_dict-as-arrays) approach,
for per-scene classification, merge-by-date max-combine, and the
per-pixel gap count -- not just "runs without crashing".

Also confirms the actual point of the restructure: map_floods() can be
called twice with different thresholds against one prepare_wet_scenes()
run without re-triggering the (expensive, network-bound) read/reproject
work -- verified via a real call-count spy on the mocked
preprocessing.clip_xarray_using_id.

No real network/STAC/GDAL calls: reproject/read internals are
monkeypatched with small deterministic synthetic arrays, but detection
(ZScoreDetector.detect(), unmodified by this restructure) and the
merge/gap-count math run for real.
"""
import os

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

# 3x3 grid, baseline mean=1.0/std=0.1 in both bands everywhere.
SIZE = 3
Y = np.arange(SIZE, 0, -1) * 30.0
X = np.arange(SIZE) * 30.0

# 4 wet scenes: two share a date (same-day multi-track, to exercise
# merge-by-date's per-pixel max-combine across *different* scenes, not
# just the same one twice), one has a sentinel-nodata gap (>=50, masked
# to NaN by the per-scene nodata handling) to exercise the gap-count
# accumulator.
SCENE_DEFS = {
    # (date 20240715) floods VV-only at (0,0)
    'S1A_IW_GRDH_1SDV_20240715T000000_20240715T000025_000001_000001_rtc': {
        'vv': [[0.5, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
        'vh': [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
    },
    # (date 20240715, different track) floods VH-only at (1,1)
    'S1A_IW_GRDH_1SDV_20240715T120000_20240715T120025_000002_000002_rtc': {
        'vv': [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
        'vh': [[1.0, 1.0, 1.0], [1.0, 0.5, 1.0], [1.0, 1.0, 1.0]],
    },
    # (date 20240716) no floods, but a sentinel-nodata gap at (0,1)
    'S1A_IW_GRDH_1SDV_20240716T000000_20240716T000025_000003_000003_rtc': {
        'vv': [[1.0, 99.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
        'vh': [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
    },
    # (date 20240717) high-confidence flood (both bands) at (1,0). Not
    # placed in the grid's outer row/col (e.g. (2,2)): export_xarray()'s
    # bounds-as-edges handling introduces a real, pre-existing (and
    # unrelated to this refactor) half-pixel shift once a slope raster
    # round-trips through disk, which reproject_match() then resolves
    # as NaN right at the outer edge on a grid this small -- a known
    # class of edge effect (see map_floods()'s own reproject_match
    # comment), not something this test is checking.
    'S1A_IW_GRDH_1SDV_20240717T000000_20240717T000025_000004_000004_rtc': {
        'vv': [[1.0, 1.0, 1.0], [0.5, 1.0, 1.0], [1.0, 1.0, 1.0]],
        'vh': [[1.0, 1.0, 1.0], [0.5, 1.0, 1.0], [1.0, 1.0, 1.0]],
    },
}


def _make_scene_dataarray(values):
    return xr.DataArray(
        np.array(values, dtype='float64'), dims=('y', 'x'), coords={'y': Y, 'x': X},
    )


def _make_baseline():
    mean_vv = xr.DataArray(np.full((SIZE, SIZE), 1.0), dims=('y', 'x'), coords={'y': Y, 'x': X})
    std_vv = xr.DataArray(np.full((SIZE, SIZE), 0.1), dims=('y', 'x'), coords={'y': Y, 'x': X})
    mean_vh = mean_vv.copy()
    std_vh = std_vv.copy()
    baseline = xr.concat(
        [mean_vv, std_vv, mean_vh, std_vh], dim='band',
    ).assign_coords(band=['vv_mean', 'vv_std', 'vh_mean', 'vh_std'])
    return baseline.rio.write_crs('EPSG:32645')


def _old_style_wet_scene(scene_id):
    """Replicates exactly what prepare_wet_scenes() used to build in
    self.wet_scenes_by_aoi[id][scene_id]: concat(vv, vh) + sentinel-nodata
    masking -- the pre-refactor in-memory reference."""
    defn = SCENE_DEFS[scene_id]
    vv = _make_scene_dataarray(defn['vv'])
    vh = _make_scene_dataarray(defn['vh'])
    scene = xr.concat([vv, vh], dim='band').assign_coords(band=['vv_ds', 'vh_ds'])
    return scene.where(scene < 50, np.nan)


def _write_dummy_slope(fm):
    """A real, on-disk, all-flat (0 degree) slope raster at the exact
    path map_floods() expects -- keeps requires_slope_mask=True's real
    logic exercised (not mocked away) while guaranteeing it never masks
    anything out (well under any reasonable rel_slope_thd), so it
    doesn't interfere with the classification-correctness comparisons
    these tests are actually checking."""
    slope = xr.DataArray(
        np.zeros((1, SIZE, SIZE)), dims=('band', 'y', 'x'), coords={'y': Y, 'x': X},
    ).rio.write_crs('EPSG:32645')
    slope_path = os.path.join(fm.slope_dir, autofloods.SLOPE_OUTFILE).replace('_id.nc', f'_{TILE_ID}.nc')
    utils.export_xarray(slope, slope_path)


def _make_flood_mapper(tmp_path):
    return autofloods.flood_mapper(
        grid_shapefile=GRID_PATH,
        grid_id_list=[TILE_ID],
        dry_years=[2024, 2024],
        wet_duration=['2024/07', '2024/07'],
        detector=ZScoreDetector(vv_thd=-2.5, vh_thd=-2.5),
        output_dir=str(tmp_path),
    )


def _wire_wet_scene_mocks(fm, monkeypatch, clip_call_log=None):
    """Monkeypatches the network/reproject chain prepare_wet_scenes()
    calls (get_s1_items -> read_scenes -> clip_xarray_using_id), so the
    real prepare_wet_scenes()/map_floods()/merge_floods_by_date()/
    generate_number_of_scenes() run against SCENE_DEFS's synthetic data
    with no network access. clip_call_log, if given, records every
    clip_xarray_using_id() call (used to verify map_floods() re-runs
    don't re-trigger reprojection)."""
    scene_ids = list(SCENE_DEFS.keys())

    monkeypatch.setattr(fm.source, 'search_sentinel1', lambda **k: scene_ids)
    monkeypatch.setattr(
        utils, 'seggregate_sentinel_search',
        lambda aoi_list, search_items: (
            {TILE_ID: scene_ids},
            {scene_id: [TILE_ID] for scene_id in scene_ids},
        ),
    )
    monkeypatch.setattr(
        autofloods.preprocessing, 'read_sentinel1_stac',
        lambda item, source, overview_level, bbox=None: (
            item, {'vv_ds': f'{item}::vv', 'vh_ds': f'{item}::vh'},
        ),
    )

    def _fake_clip(data_xarray, grid_shapefile_path, aoi_id, ref_xarray, cell_size, buffer=None):
        if clip_call_log is not None:
            clip_call_log.append(data_xarray)
        scene_id, band = data_xarray.split('::')
        return _make_scene_dataarray(SCENE_DEFS[scene_id][band])

    monkeypatch.setattr(autofloods.preprocessing, 'clip_xarray_using_id', _fake_clip)


class TestStreamingMatchesOldInMemoryComputation:
    def test_classification_matches(self, tmp_path, monkeypatch):
        fm = _make_flood_mapper(tmp_path)
        fm.mean_std_by_aoi = {TILE_ID: _make_baseline()}
        _wire_wet_scene_mocks(fm, monkeypatch)
        _write_dummy_slope(fm)

        fm.prepare_wet_scenes()
        fm.map_floods(vv_thd=-2.5, vh_thd=-2.5, export_vector=False, export_maps=False)

        baseline = fm.mean_std_by_aoi[TILE_ID]
        for scene_id in SCENE_DEFS:
            old_classified = fm.detector.detect(baseline, _old_style_wet_scene(scene_id))
            new_classified = xr.load_dataarray(fm.flood_dict[TILE_ID][scene_id], engine='rasterio').squeeze('band', drop=True)
            np.testing.assert_array_equal(new_classified.values, old_classified.values)

    def test_merge_by_date_matches(self, tmp_path, monkeypatch):
        fm = _make_flood_mapper(tmp_path)
        fm.mean_std_by_aoi = {TILE_ID: _make_baseline()}
        _wire_wet_scene_mocks(fm, monkeypatch)
        _write_dummy_slope(fm)

        fm.prepare_wet_scenes()
        fm.map_floods(vv_thd=-2.5, vh_thd=-2.5, export_vector=False, export_maps=False)
        fm.merge_floods_by_date(export_raster=False)

        # old-style reference: in-memory dict of classified arrays -> flood_data_3dstack
        baseline = fm.mean_std_by_aoi[TILE_ID]
        old_flood_dict = {
            scene_id: fm.detector.detect(baseline, _old_style_wet_scene(scene_id))
            for scene_id in SCENE_DEFS
        }
        old_dates, old_stack = utils.flood_data_3dstack(old_flood_dict)

        new_by_date = fm.flood_by_date[TILE_ID]
        assert sorted(new_by_date.date.values.tolist()) == sorted(old_dates)
        for i, date in enumerate(old_dates):
            new_slice = new_by_date.sel(date=date).values
            np.testing.assert_array_equal(new_slice, old_stack[i])

        # sanity: the two same-date scenes' floods (VV-only at (0,0),
        # VH-only at (1,1)) both survive the max-combine for 20240715
        combined_0715 = new_by_date.sel(date='20240715').values
        assert combined_0715[0, 0] == 2  # VV-only flood
        assert combined_0715[1, 1] == 1  # VH-only flood

    def test_gap_count_matches(self, tmp_path, monkeypatch):
        fm = _make_flood_mapper(tmp_path)
        fm.mean_std_by_aoi = {TILE_ID: _make_baseline()}
        _wire_wet_scene_mocks(fm, monkeypatch)

        fm.prepare_wet_scenes()
        fm.generate_number_of_scenes(export_raster=False)

        # old-style reference: stack every scene's (already nodata-masked)
        # array and sum NaN-any-band, exactly as generate_number_of_scenes()
        # used to compute it directly.
        old_gap_count = np.stack([
            np.any(np.isnan(_old_style_wet_scene(scene_id).values), axis=0)
            for scene_id in SCENE_DEFS
        ]).sum(axis=0)

        np.testing.assert_array_equal(fm.scene_count[TILE_ID].values, old_gap_count)
        # sanity: exactly one scene (20240716) had a sentinel-masked gap,
        # at pixel (0, 1)
        assert old_gap_count[0, 1] == 1
        assert old_gap_count.sum() == 1


class TestMapFloodsRerunWithoutRetriggeringPrepareWetScenes:
    def test_two_calls_different_thresholds_no_reread(self, tmp_path, monkeypatch):
        fm = _make_flood_mapper(tmp_path)
        fm.mean_std_by_aoi = {TILE_ID: _make_baseline()}
        clip_calls = []
        _wire_wet_scene_mocks(fm, monkeypatch, clip_call_log=clip_calls)
        _write_dummy_slope(fm)

        fm.prepare_wet_scenes()
        calls_after_prepare = len(clip_calls)
        assert calls_after_prepare == len(SCENE_DEFS) * 2  # vv + vh per scene

        fm.map_floods(vv_thd=-2.5, vh_thd=-2.5, export_vector=False, export_maps=False)
        lenient_result = {
            scene_id: xr.load_dataarray(path, engine='rasterio').squeeze('band', drop=True).values
            for scene_id, path in fm.flood_dict[TILE_ID].items()
        }

        fm.map_floods(vv_thd=-10, vh_thd=-10, export_vector=False, export_maps=False)
        strict_result = {
            scene_id: xr.load_dataarray(path, engine='rasterio').squeeze('band', drop=True).values
            for scene_id, path in fm.flood_dict[TILE_ID].items()
        }

        # the expensive read/reproject step must not have run again for
        # either map_floods() call -- clip_xarray_using_id call count is
        # unchanged since prepare_wet_scenes() finished.
        assert len(clip_calls) == calls_after_prepare

        # different thresholds must genuinely produce different results
        # (a much stricter threshold flags fewer/no pixels as flooded)
        vv_only_scene = 'S1A_IW_GRDH_1SDV_20240715T000000_20240715T000025_000001_000001_rtc'
        assert lenient_result[vv_only_scene][0, 0] == 2  # -2.5 threshold: flagged VV-only
        assert strict_result[vv_only_scene][0, 0] == 0   # -10 threshold: no longer flagged
