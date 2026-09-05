# tests/test_visualize.py

"""
Tests for autofloods.visualize. Every function under test reads from
disk (not in-memory flood_mapper state), so these tests build small
synthetic on-disk fixtures matching the REAL file formats/paths the
pipeline writes (mean_std NetCDF, slope .nc, wet_scenes_cache .nc,
flood_raster .tif, monthlyadded .tif) directly at the paths a real
flood_mapper instance would expect -- no pipeline methods are actually
run, matching the module's own "fresh session against a completed run's
output_dir" design goal.

matplotlib uses the Agg backend (no display) throughout.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import os

import numpy as np
import pytest
import rasterio
import xarray as xr

import autofloods
import autofloods.utils as utils
import autofloods.visualize as viz
from autofloods.detectors import ZScoreDetector

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GRID_PATH = os.path.join(REPO_ROOT, 'resources', 'india_utm_fishnet_buffer.gpkg')
TILE_ID = 318
SIZE = 5
Y = np.arange(SIZE, 0, -1) * 30.0
X = np.arange(SIZE) * 30.0


def _make_flood_mapper(tmp_path):
    return autofloods.flood_mapper(
        grid_shapefile=GRID_PATH,
        grid_id_list=[TILE_ID],
        dry_years=[2024, 2024],
        wet_duration=['2024/07', '2024/07'],
        detector=ZScoreDetector(vv_thd=-2.5, vh_thd=-2.5),
        output_dir=str(tmp_path),
    )


def _write_baseline(fm):
    mean_vv = xr.DataArray(np.full((SIZE, SIZE), 0.5), dims=('y', 'x'), coords={'y': Y, 'x': X})
    std_vv = xr.DataArray(np.full((SIZE, SIZE), 0.05), dims=('y', 'x'), coords={'y': Y, 'x': X})
    mean_vh = xr.DataArray(np.full((SIZE, SIZE), 0.1), dims=('y', 'x'), coords={'y': Y, 'x': X})
    std_vh = xr.DataArray(np.full((SIZE, SIZE), 0.02), dims=('y', 'x'), coords={'y': Y, 'x': X})
    baseline = xr.concat([mean_vv, std_vv, mean_vh, std_vh], dim='band').assign_coords(
        band=['vv_mean', 'vv_std', 'vh_mean', 'vh_std']
    )
    outfile = fm.nc_outfile.replace('_id_', f'_{TILE_ID}_')
    baseline.to_netcdf(outfile)


def _write_slope(fm):
    slope = xr.DataArray(
        np.random.default_rng(0).uniform(0, 30, (1, SIZE, SIZE)), dims=('band', 'y', 'x'),
        coords={'y': Y, 'x': X},
    )
    outfile = os.path.join(fm.slope_dir, autofloods.SLOPE_OUTFILE.replace('_id.nc', f'_{TILE_ID}.nc'))
    utils.export_xarray(slope, outfile)


def _write_wet_scene(fm, scene_id, rng):
    vv = xr.DataArray(rng.uniform(0.05, 1.0, (SIZE, SIZE)), dims=('y', 'x'), coords={'y': Y, 'x': X})
    vh = xr.DataArray(rng.uniform(0.01, 0.3, (SIZE, SIZE)), dims=('y', 'x'), coords={'y': Y, 'x': X})
    scene = xr.concat([vv, vh], dim='band').assign_coords(band=['vv_ds', 'vh_ds'])
    outfile = os.path.join(fm.wet_scenes_cache_dir, f'wetscene_{TILE_ID}_{scene_id}.nc')
    scene.to_netcdf(outfile)


def _write_flood_raster(fm, scene_id, classified_values):
    classified = xr.DataArray(classified_values.astype('float64'), dims=('y', 'x'), coords={'y': Y, 'x': X})
    outfile = viz._flood_scene_raster_path(fm, TILE_ID, scene_id)
    utils.export_xarray(classified, outfile)


def _write_wet_scene_and_flood(fm, scene_id, rng, flood_value=0):
    _write_wet_scene(fm, scene_id, rng)
    classified = np.full((SIZE, SIZE), flood_value, dtype='float64')
    classified[0, 0] = np.nan  # a masked/gap pixel in every scene
    _write_flood_raster(fm, scene_id, classified)


def _mpc_scene_id(date_str, track=1):
    # MPC-style scene_id -- 4+ underscore-separated tokens before the
    # date, matching what _extract_date_token(date_index=-5) and
    # scene_id.split('_')[4:] (map_floods()'s suffix formula) expect.
    return f'S1A_IW_GRDH_1SDV_{date_str}T000000_{date_str}T000025_{track:06d}_{track:06d}_rtc'


def _write_monthly(fm, band_values_by_month):
    outfile = fm.expected_monthly_outfile(TILE_ID)
    os.makedirs(os.path.dirname(outfile), exist_ok=True)
    months = sorted(band_values_by_month)
    with rasterio.open(
        outfile, 'w', driver='GTiff', height=SIZE, width=SIZE, count=len(months),
        dtype='uint8', nodata=255,
    ) as dst:
        for i, month in enumerate(months, start=1):
            dst.write(band_values_by_month[month], i)
            dst.set_band_description(i, month)


class TestPlotBaseline:
    def test_returns_figure_with_four_panels(self, tmp_path):
        fm = _make_flood_mapper(tmp_path)
        _write_baseline(fm)

        fig = viz.plot_baseline(fm, TILE_ID)

        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) >= 4  # 4 panels + their colorbars
        plt.close(fig)


class TestPlotTerrain:
    def test_returns_single_slope_panel(self, tmp_path):
        fm = _make_flood_mapper(tmp_path)
        _write_slope(fm)

        fig = viz.plot_terrain(fm, TILE_ID)

        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) >= 1
        plt.close(fig)


class TestPlotScenesAndFloods:
    def _setup_n_scenes(self, fm, n):
        rng = np.random.default_rng(1)
        for i in range(n):
            scene_id = _mpc_scene_id(f'202407{i + 1:02d}', track=i + 1)
            _write_wet_scene_and_flood(fm, scene_id, rng, flood_value=(3 if i == 0 else 0))
        return n

    @pytest.mark.parametrize('n_scenes,expected_cols,expected_row_pairs', [
        (3, 3, 1),    # fewer than target_cols: no padding
        (6, 6, 1),    # exactly fills one row-pair
        (12, 6, 2),   # evenly divides
        (11, 6, 2),   # does NOT evenly divide -- 6 then 5
        (7, 6, 2),    # does NOT evenly divide -- 6 then 1
    ])
    def test_column_wrap_logic(self, tmp_path, n_scenes, expected_cols, expected_row_pairs):
        fm = _make_flood_mapper(tmp_path)
        self._setup_n_scenes(fm, n_scenes)

        fig = viz.plot_scenes_and_floods(fm, TILE_ID)

        expected_rows = expected_row_pairs * 2
        axes_grid = fig.axes[:expected_rows * expected_cols]  # exclude legend-only artists if any
        assert len(fig.axes) >= expected_rows * expected_cols
        assert f'{n_scenes} scene(s) shown' in fig._suptitle.get_text()
        plt.close(fig)

    def test_unused_trailing_axes_are_turned_off(self, tmp_path):
        fm = _make_flood_mapper(tmp_path)
        self._setup_n_scenes(fm, 7)  # 6 then 1 -- 5 unused columns in row-pair 2

        fig = viz.plot_scenes_and_floods(fm, TILE_ID)

        # axes are laid out row-major, 4 rows x 6 cols; the 2nd row-pair
        # (rows 2-3) has only column 0 populated -- columns 1-5 in both
        # rows 2 and 3 must be turned off.
        axes = np.array(fig.axes[:24]).reshape(4, 6)
        for col in range(1, 6):
            assert not axes[2, col].axison
            assert not axes[3, col].axison
        plt.close(fig)

    def test_max_scenes_caps_and_notes_skipped(self, tmp_path):
        fm = _make_flood_mapper(tmp_path)
        self._setup_n_scenes(fm, 5)

        fig = viz.plot_scenes_and_floods(fm, TILE_ID, max_scenes=3)

        assert '3 scene(s) shown' in fig._suptitle.get_text()
        assert '2 skipped' in fig._suptitle.get_text()
        plt.close(fig)

    def test_missing_flood_raster_is_skipped_not_crashed(self, tmp_path):
        fm = _make_flood_mapper(tmp_path)
        rng = np.random.default_rng(2)
        # one scene with both wet-scene cache and flood raster
        scene_id_ok = _mpc_scene_id('20240701', track=1)
        _write_wet_scene_and_flood(fm, scene_id_ok, rng)
        # one scene with only a wet-scene cache (map_floods() never ran)
        scene_id_missing = _mpc_scene_id('20240702', track=2)
        _write_wet_scene(fm, scene_id_missing, rng)

        fig = viz.plot_scenes_and_floods(fm, TILE_ID)

        assert '1 scene(s) shown' in fig._suptitle.get_text()
        assert '1 missing flood raster' in fig._suptitle.get_text()
        plt.close(fig)

    def test_raises_clear_error_when_no_scenes_available(self, tmp_path):
        fm = _make_flood_mapper(tmp_path)

        with pytest.raises(ValueError, match='No wet scenes'):
            viz.plot_scenes_and_floods(fm, TILE_ID)


class TestPlotFloodMap:
    def test_single_month(self, tmp_path):
        fm = _make_flood_mapper(tmp_path)
        data = np.array([[0, 1, 2, 255, 0]] * SIZE, dtype='uint8')
        _write_monthly(fm, {'202408': data})

        fig = viz.plot_flood_map(fm, TILE_ID, month='202408')

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_all_months_grid(self, tmp_path):
        fm = _make_flood_mapper(tmp_path)
        data_aug = np.full((SIZE, SIZE), 2, dtype='uint8')
        data_sep = np.full((SIZE, SIZE), 5, dtype='uint8')
        data_sep[0, 0] = 255
        _write_monthly(fm, {'202408': data_aug, '202409': data_sep})

        fig = viz.plot_flood_map(fm, TILE_ID, month=None)

        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) >= 2
        plt.close(fig)

    def test_unknown_month_raises_clear_error(self, tmp_path):
        fm = _make_flood_mapper(tmp_path)
        _write_monthly(fm, {'202408': np.zeros((SIZE, SIZE), dtype='uint8')})

        with pytest.raises(ValueError, match='not found'):
            viz.plot_flood_map(fm, TILE_ID, month='202501')
