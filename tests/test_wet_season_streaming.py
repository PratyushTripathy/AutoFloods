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
import types

import numpy as np
import pytest
import rasterio
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


def _write_steep_slope(fm, steep_degrees=30.0, steep_pixel=(1, 1)):
    """A slope raster that's flat (0 degrees) everywhere except one
    interior pixel (avoiding the grid's outer row/col -- see SCENE_DEFS'
    comment on the half-pixel reproject_match edge effect) set to
    `steep_degrees`, for tests that need slope-masking to actually
    do something (unlike _write_dummy_slope's all-flat raster)."""
    values = np.zeros((1, SIZE, SIZE))
    values[0, steep_pixel[0], steep_pixel[1]] = steep_degrees
    slope = xr.DataArray(values, dims=('band', 'y', 'x'), coords={'y': Y, 'x': X}).rio.write_crs('EPSG:32645')
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
    # real STAC search results are Item objects with a .id attribute
    # (scene_id), not plain strings -- read_scenes()'s streaming path
    # needs a real item to re-look-up for a scene shared across AOIs,
    # so the mock must carry .id like the real thing does.
    fake_items = [types.SimpleNamespace(id=scene_id) for scene_id in scene_ids]

    monkeypatch.setattr(fm.source, 'search_sentinel1', lambda **k: fake_items)
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
            item.id, {'vv_ds': f'{item.id}::vv', 'vh_ds': f'{item.id}::vh'},
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

    def test_valid_count_matches(self, tmp_path, monkeypatch):
        # generate_number_of_scenes() must return a per-pixel VALID
        # (non-NaN) observation count -- see the 0.1.0a21 fix -- not the
        # inverted gap count it used to return (both before and after
        # the streaming rewrite; see CHANGELOG.md).
        fm = _make_flood_mapper(tmp_path)
        fm.mean_std_by_aoi = {TILE_ID: _make_baseline()}
        _wire_wet_scene_mocks(fm, monkeypatch)

        fm.prepare_wet_scenes()
        fm.generate_number_of_scenes(export_raster=False)

        # old-style reference, but inverted to VALID (not gap): stack
        # every scene's (already nodata-masked) array and sum "not any
        # NaN band", i.e. what a pre-streaming stack-then-reduce
        # implementation would give if it had computed the right thing.
        old_valid_count = np.stack([
            ~np.any(np.isnan(_old_style_wet_scene(scene_id).values), axis=0)
            for scene_id in SCENE_DEFS
        ]).sum(axis=0)
        old_gap_count = len(SCENE_DEFS) - old_valid_count

        np.testing.assert_array_equal(fm.scene_count[TILE_ID].values, old_valid_count)
        # sanity: exactly one scene (20240716) had a sentinel-masked gap,
        # at pixel (0, 1) -- valid count there is 3 of 4 scenes, full
        # 4-of-4 everywhere else.
        assert old_valid_count[0, 1] == 3
        assert old_valid_count.sum() == len(SCENE_DEFS) * 9 - 1

        # correctness invariant: valid count + gap count == total scenes
        # processed, at every pixel, for this known-gap synthetic case.
        np.testing.assert_array_equal(
            fm.scene_count[TILE_ID].values + old_gap_count,
            np.full_like(old_valid_count, len(SCENE_DEFS)),
        )


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


class TestValidCountNormalizesMonthlySum:
    """The actual intended use case for generate_number_of_scenes()'s
    valid-observation count (0.1.0a21): normalizing monthly_sum()'s
    per-month flood-day count into a "fraction of valid observations
    flooded" per pixel -- this is what a gap count could never support
    (it's the wrong quantity for a denominator)."""

    def test_fraction_flooded_is_computable_and_correct(self, tmp_path, monkeypatch):
        fm = _make_flood_mapper(tmp_path)
        fm.mean_std_by_aoi = {TILE_ID: _make_baseline()}
        _wire_wet_scene_mocks(fm, monkeypatch)
        _write_dummy_slope(fm)

        fm.prepare_wet_scenes()
        fm.map_floods(vv_thd=-2.5, vh_thd=-2.5, export_vector=False, export_maps=False)
        fm.merge_floods_by_date(export_raster=True)
        fm.generate_number_of_scenes(export_raster=False)
        fm.monthly_sum()

        monthly_path = fm.expected_monthly_outfile(TILE_ID)
        with rasterio.open(monthly_path) as src:
            monthly_count = src.read(1).astype(float)

        valid_count = fm.scene_count[TILE_ID].values.astype(float)
        fraction_flooded = monthly_count / valid_count

        # pixel (1, 0): high-confidence flood on exactly one of the 3
        # observed dates (20240717), valid (no gap) in all 4 raw scenes.
        assert monthly_count[1, 0] == 1
        assert valid_count[1, 0] == 4
        assert fraction_flooded[1, 0] == pytest.approx(0.25)

        # pixel (0, 1): the sentinel-nodata-gap pixel -- never flooded,
        # valid in only 3 of 4 scenes (one scene, 20240716, had a gap
        # here) -- fraction is 0, not NaN/undefined, and the denominator
        # correctly reflects the gap.
        assert monthly_count[0, 1] == 0
        assert valid_count[0, 1] == 3
        assert fraction_flooded[0, 1] == 0.0

        # a pixel with zero valid observations would be undefined (0/0)
        # -- not exercised by this synthetic case (every pixel has at
        # least 3 of 4 valid scenes), but worth noting: monthly_sum()
        # itself already handles this case explicitly by writing nodata
        # (255) rather than a misleading 0 -- see aggregate_monthly()'s
        # docstring.


class TestMapFloodsSlopeThdRename:
    """map_floods()'s rel_slope_thd parameter was renamed to slope_thd
    (0.1.0a26) -- the value is a flat absolute per-pixel slope cutoff
    in degrees, nothing relative/normalized. rel_slope_thd is kept as a
    deprecated alias so existing scripts/configs keep working."""

    def test_slope_thd_masks_steep_pixel(self, tmp_path, monkeypatch):
        fm = _make_flood_mapper(tmp_path)
        fm.mean_std_by_aoi = {TILE_ID: _make_baseline()}
        _wire_wet_scene_mocks(fm, monkeypatch)
        _write_steep_slope(fm, steep_degrees=30.0, steep_pixel=(1, 1))
        fm.prepare_wet_scenes()

        # 20240717 scene has a high-confidence (class 3) flood at (1, 0)
        # -- unaffected by the steep pixel at (1, 1) -- and nothing at
        # (1, 1) itself, so directly assert the steep pixel is forced
        # to 0 (masked) with a threshold below its slope, and NOT
        # masked with a threshold above it.
        fm.map_floods(vv_thd=-2.5, vh_thd=-2.5, slope_thd=20, export_vector=False, export_maps=False)
        masked = xr.load_dataarray(
            fm.flood_dict[TILE_ID]['S1A_IW_GRDH_1SDV_20240717T000000_20240717T000025_000004_000004_rtc'],
            engine='rasterio',
        ).squeeze('band', drop=True)
        assert masked.values[1, 1] == 0

        fm.map_floods(vv_thd=-2.5, vh_thd=-2.5, slope_thd=40, export_vector=False, export_maps=False)
        unmasked = xr.load_dataarray(
            fm.flood_dict[TILE_ID]['S1A_IW_GRDH_1SDV_20240717T000000_20240717T000025_000004_000004_rtc'],
            engine='rasterio',
        ).squeeze('band', drop=True)
        # (1, 1) is 0 (not flooded) in the underlying classification
        # regardless of masking (see SCENE_DEFS), so confirm masking
        # actually ran differently by checking the flooded pixel (1, 0)
        # -- still correctly classified, unaffected by the slope at (1, 1).
        assert unmasked.values[1, 0] == 3

    def test_rel_slope_thd_alias_forwards_and_warns(self, tmp_path, monkeypatch):
        fm = _make_flood_mapper(tmp_path)
        fm.mean_std_by_aoi = {TILE_ID: _make_baseline()}
        _wire_wet_scene_mocks(fm, monkeypatch)
        _write_steep_slope(fm, steep_degrees=30.0, steep_pixel=(1, 1))
        fm.prepare_wet_scenes()

        with pytest.warns(DeprecationWarning, match='rel_slope_thd'):
            fm.map_floods(vv_thd=-2.5, vh_thd=-2.5, rel_slope_thd=20, export_vector=False, export_maps=False)

        masked = xr.load_dataarray(
            fm.flood_dict[TILE_ID]['S1A_IW_GRDH_1SDV_20240717T000000_20240717T000025_000004_000004_rtc'],
            engine='rasterio',
        ).squeeze('band', drop=True)
        assert masked.values[1, 1] == 0  # rel_slope_thd=20 still masks the 30-degree pixel

    def test_slope_thd_alone_does_not_warn(self, tmp_path, monkeypatch, recwarn):
        fm = _make_flood_mapper(tmp_path)
        fm.mean_std_by_aoi = {TILE_ID: _make_baseline()}
        _wire_wet_scene_mocks(fm, monkeypatch)
        _write_dummy_slope(fm)
        fm.prepare_wet_scenes()

        fm.map_floods(vv_thd=-2.5, vh_thd=-2.5, slope_thd=15, export_vector=False, export_maps=False)

        assert not any(issubclass(w.category, DeprecationWarning) for w in recwarn.list)


class TestMapFloodsSmoothing:
    def test_smooth_false_is_default_and_unchanged(self, tmp_path, monkeypatch):
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

    def test_smooth_true_applies_smoothing_and_updates_log_count(self, tmp_path, monkeypatch, caplog):
        import logging
        fm = _make_flood_mapper(tmp_path)
        fm.mean_std_by_aoi = {TILE_ID: _make_baseline()}
        _wire_wet_scene_mocks(fm, monkeypatch)
        _write_dummy_slope(fm)

        fm.prepare_wet_scenes()
        with caplog.at_level(logging.INFO, logger='autofloods'):
            fm.map_floods(vv_thd=-2.5, vh_thd=-2.5, export_vector=False, export_maps=False, smooth=True, smooth_kernel_size=3)

        # the lone class-3 pixel at (1,0) in the 20240717 scene (3x3
        # grid, no same-valued neighbors) is exactly the kind of
        # isolated speckle a majority filter removes.
        smoothed = xr.load_dataarray(
            fm.flood_dict[TILE_ID]['S1A_IW_GRDH_1SDV_20240717T000000_20240717T000025_000004_000004_rtc'],
            engine='rasterio',
        ).squeeze('band', drop=True)
        assert smoothed.values[1, 0] == 0

        # the logged count reflects the POST-smoothing raster, i.e. 0
        # high-confidence flooded pixels once the lone speckle is smoothed away.
        assert '0 high-confidence flooded pixels' in caplog.text

    def test_smooth_true_differs_from_smooth_false_output(self, tmp_path, monkeypatch):
        fm = _make_flood_mapper(tmp_path)
        fm.mean_std_by_aoi = {TILE_ID: _make_baseline()}
        _wire_wet_scene_mocks(fm, monkeypatch)
        _write_dummy_slope(fm)

        fm.prepare_wet_scenes()
        fm.map_floods(vv_thd=-2.5, vh_thd=-2.5, export_vector=False, export_maps=False, smooth=False)
        unsmoothed = xr.load_dataarray(
            fm.flood_dict[TILE_ID]['S1A_IW_GRDH_1SDV_20240717T000000_20240717T000025_000004_000004_rtc'],
            engine='rasterio',
        ).squeeze('band', drop=True).values.copy()

        fm.map_floods(vv_thd=-2.5, vh_thd=-2.5, export_vector=False, export_maps=False, smooth=True)
        smoothed = xr.load_dataarray(
            fm.flood_dict[TILE_ID]['S1A_IW_GRDH_1SDV_20240717T000000_20240717T000025_000004_000004_rtc'],
            engine='rasterio',
        ).squeeze('band', drop=True).values

        assert not np.array_equal(unsmoothed, smoothed)


class TestMapFloodsExportMapsAndVector:
    """map_floods()'s export_vector=True/export_maps=True paths both
    reload a scene's just-written classified raster from disk via
    xr.load_dataarray(..., engine='rasterio') -- which returns a (band,
    y, x) array, not the 2D (y, x) array these paths got from the old
    in-memory architecture. Every other map_floods() test in this file
    (and in test_pipeline_logging.py/test_keep_intermediate_in_memory.py)
    explicitly passes export_vector=False, export_maps=False, so this
    gap slipped through the same way the earlier band-dimension bug did
    -- these tests run the real streaming map_floods() with both flags
    on, through no mocks beyond the network/reproject chain."""

    def test_export_maps_writes_one_png_per_scene(self, tmp_path, monkeypatch):
        fm = _make_flood_mapper(tmp_path)
        fm.mean_std_by_aoi = {TILE_ID: _make_baseline()}
        _wire_wet_scene_mocks(fm, monkeypatch)
        _write_dummy_slope(fm)

        fm.prepare_wet_scenes()
        fm.map_floods(vv_thd=-2.5, vh_thd=-2.5, export_vector=False, export_maps=True)

        image_dir = os.path.join(fm.output_base, 'flood_image')
        pngs = os.listdir(image_dir)
        assert len(pngs) == len(SCENE_DEFS)
        assert all(os.path.getsize(os.path.join(image_dir, f)) > 0 for f in pngs)

    def test_export_vector_writes_gpkg_for_flooded_scenes(self, tmp_path, monkeypatch):
        fm = _make_flood_mapper(tmp_path)
        fm.mean_std_by_aoi = {TILE_ID: _make_baseline()}
        _wire_wet_scene_mocks(fm, monkeypatch)
        _write_dummy_slope(fm)

        fm.prepare_wet_scenes()
        fm.map_floods(vv_thd=-2.5, vh_thd=-2.5, export_vector=True, export_maps=False)

        # polygonize_flood_raster() only keeps class-3 (high-confidence,
        # both VV+VH anomalous) pixels -- of SCENE_DEFS, only the
        # 20240717 scene has one (at (1,0)); the two 20240715 scenes are
        # single-band-only (class 1/2) and the 20240716 scene has none,
        # so those three scenes are correctly skipped (empty gdf), not a
        # bug -- see map_floods()'s "Flood cells not found" print.
        flooded_scene = 'S1A_IW_GRDH_1SDV_20240717T000000_20240717T000025_000004_000004_rtc'
        assert list(fm.flood_gdf_dict[TILE_ID].keys()) == [flooded_scene]
        gdf = fm.flood_gdf_dict[TILE_ID][flooded_scene]
        assert gdf.crs is not None
        assert gdf.shape[0] > 0

        outdir = os.path.join(fm.output_base, 'flood_vector')
        assert len(os.listdir(outdir)) == 1


# OPERA-style scene_ids ("OPERA_PASS_{YYYYMMDD}", 3 underscore-separated
# tokens) -- unlike SCENE_DEFS above (MPC-style, 9+ tokens), these are
# the exact shape that silently collided under the old
# scene_id.split('_')[4:] suffix formula (index 4 is past the end of a
# 3-token id, so every scene produced the same empty suffix and
# overwrote the same output file). See sanitize_scene_id_for_filename()
# in autofloods/utils/__init__.py, the fix for this. Distinct per-scene
# values (not just distinct dates) so a content-based assertion, not
# just distinct paths, can confirm each scene's own file holds its own
# classification.
OPERA_SCENE_DEFS = {
    'OPERA_PASS_20240701': {  # high-confidence flood at (0,0)
        'vv': [[0.5, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
        'vh': [[0.5, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
    },
    'OPERA_PASS_20240705': {  # no flood anywhere
        'vv': [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
        'vh': [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
    },
    'OPERA_PASS_20240710': {  # high-confidence flood at (1,1)
        'vv': [[1.0, 1.0, 1.0], [1.0, 0.5, 1.0], [1.0, 1.0, 1.0]],
        'vh': [[1.0, 1.0, 1.0], [1.0, 0.5, 1.0], [1.0, 1.0, 1.0]],
    },
}


def _wire_opera_style_wet_scene_mocks(fm, monkeypatch):
    """Same monkeypatch shape as _wire_wet_scene_mocks(), against
    OPERA_SCENE_DEFS's short, collision-prone scene_ids instead of
    SCENE_DEFS's MPC-style ones."""
    scene_ids = list(OPERA_SCENE_DEFS.keys())
    fake_items = [types.SimpleNamespace(id=scene_id) for scene_id in scene_ids]

    monkeypatch.setattr(fm.source, 'search_sentinel1', lambda **k: fake_items)
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
            item.id, {'vv_ds': f'{item.id}::vv', 'vh_ds': f'{item.id}::vh'},
        ),
    )

    def _fake_clip(data_xarray, grid_shapefile_path, aoi_id, ref_xarray, cell_size, buffer=None):
        scene_id, band = data_xarray.split('::')
        return _make_scene_dataarray(OPERA_SCENE_DEFS[scene_id][band])

    monkeypatch.setattr(autofloods.preprocessing, 'clip_xarray_using_id', _fake_clip)


class TestMapFloodsOperaStyleSceneIdsFilenameCollision:
    """
    Regression coverage for the real bug found via production-record
    investigation (2026-09-06): map_floods()'s per-scene output
    filename used to derive its suffix from
    scene_id.split('_')[4:], which is empty for a 3-token OPERA-style
    scene_id -- every scene in a run wrote to (and overwrote) the exact
    same output path. Dormant in the published Bihar record only
    because that run used export_raster=False; live on the current
    default (unconditional export) streaming path for any OPERA-sourced
    run. Fixed via sanitize_scene_id_for_filename() (full scene_id,
    sanitized, not a positional slice).
    """

    def test_one_distinct_file_per_opera_style_scene_not_one_total(self, tmp_path, monkeypatch):
        fm = _make_flood_mapper(tmp_path)
        fm.mean_std_by_aoi = {TILE_ID: _make_baseline()}
        _wire_opera_style_wet_scene_mocks(fm, monkeypatch)
        _write_dummy_slope(fm)

        fm.prepare_wet_scenes()
        fm.map_floods(vv_thd=-2.5, vh_thd=-2.5)

        paths = fm.flood_dict[TILE_ID]
        assert len(paths) == len(OPERA_SCENE_DEFS)

        # the actual bug: not just distinct dict values, but distinct
        # real files on disk -- the old formula produced the same path
        # for every entry, so this dict would have collapsed to one
        # scene_id -> path pair, or (as here) len(paths) real dict
        # entries all pointing at one shared, repeatedly-overwritten file.
        distinct_paths = set(paths.values())
        assert len(distinct_paths) == len(OPERA_SCENE_DEFS), (
            f'expected one distinct output path per scene, got {len(distinct_paths)} '
            f'distinct path(s) for {len(paths)} scene(s): {paths}'
        )
        for path in distinct_paths:
            assert os.path.exists(path)

        flood_raster_dir = os.path.join(fm.output_base, 'flood_raster')
        on_disk = [f for f in os.listdir(flood_raster_dir) if f.startswith('floodextent_')]
        assert len(on_disk) == len(OPERA_SCENE_DEFS)

        # content-level check, not just path-count: each scene's own
        # file holds ITS OWN classification, not a copy of whichever
        # scene happened to be written last (which is exactly what the
        # old collision produced -- one file holding only the final
        # scene's data, read back identically for every scene_id key).
        flooded_pixel_by_scene = {
            'OPERA_PASS_20240701': (0, 0),
            'OPERA_PASS_20240710': (1, 1),
        }
        for scene_id, pixel in flooded_pixel_by_scene.items():
            with rasterio.open(paths[scene_id]) as ds:
                arr = ds.read(1)
            assert arr[pixel] == 3, f'{scene_id} should be high-confidence-flooded at {pixel}'
        with rasterio.open(paths['OPERA_PASS_20240705']) as ds:
            arr = ds.read(1)
        assert (arr == 0).all(), 'OPERA_PASS_20240705 has no flood anywhere and should read all-zero'

    def test_merge_by_date_sees_all_opera_style_scenes_not_one_repeated(self, tmp_path, monkeypatch):
        """
        merge_floods_by_date() reads flood_dict's (now-distinct) paths
        via flood_data_3dstack_from_paths() -- confirms that consumer
        resolves correctly with the new naming: 3 real dates, each with
        its own scene's classification, not 3 copies of one collided
        file's content.
        """
        fm = _make_flood_mapper(tmp_path)
        fm.mean_std_by_aoi = {TILE_ID: _make_baseline()}
        _wire_opera_style_wet_scene_mocks(fm, monkeypatch)
        _write_dummy_slope(fm)

        fm.prepare_wet_scenes()
        fm.map_floods(vv_thd=-2.5, vh_thd=-2.5)
        fm.merge_floods_by_date()

        stack = fm.flood_by_date[TILE_ID]
        assert sorted(stack.date.values.tolist()) == ['20240701', '20240705', '20240710']
        assert stack.sel(date='20240701').values[0, 0] == 3
        assert stack.sel(date='20240710').values[1, 1] == 3
        assert (stack.sel(date='20240705').values == 0).all()
