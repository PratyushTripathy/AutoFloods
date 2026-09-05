# tests/test_preprocessing.py

"""
Tests for autofloods.preprocessing's deterministic array/geometry math
(clip_xarray_using_id's reprojection+regridding, smoothen_slope's
kernel-smoothing math) and the read/reproject/stack pipeline functions
against synthetic in-memory data (no live STAC search, no real
Sentinel-1/DEM downloads, no network access).

read_sentinel1_stac is exercised with a fully mocked `source` object
(source.read_vv_vh is mocked, matching the sources tests' own
convention) since the function itself is pure glue around whatever the
source returns.

reproject_clip_stac / stack_images are exercised against synthetic
xarray DataArrays with a real CRS, reprojected/clipped for real via
rioxarray -- these functions don't touch the network themselves, only
their upstream callers (STAC search / actual Sentinel-1 reads) do.
"""

from unittest.mock import MagicMock

import geopandas as gpd
import numpy as np
import pytest
import rioxarray  # noqa: F401 -- registers the .rio accessor
import xarray as xr
import xrspatial
from shapely.geometry import box

from autofloods import preprocessing


def _make_grid_file(tmp_path, zone='45R', bounds=(85.0, 25.0, 85.5, 25.5), filename='grid.gpkg'):
    """A small synthetic tile grid geopackage, matching the schema
    clip_xarray_using_id/reproject_clip_stac read (ID, zone, geometry)."""
    gdf = gpd.GeoDataFrame(
        {
            'ID': ['tile1'],
            'zone': [zone],
            'geometry': [box(*bounds)],
        },
        crs='EPSG:4326',
    )
    path = tmp_path / filename
    gdf.to_file(path)
    return str(path)


def _make_source_dataarray(bounds=(84.9, 24.9, 85.6, 25.6), size=40, fill=None):
    """A synthetic EPSG:4326 DataArray, comfortably covering `bounds`
    (bigger than the grid tile from _make_grid_file), for reprojection
    into the tile's UTM zone.

    Carries a real, incidental leading 'band' dim of size 1, matching
    what rioxarray.open_rasterio(masked=True) actually returns for a
    single-band GeoTIFF/VRT (the shape autofloods.utils.
    open_rasterio_with_retry() -- used by both OPERASource and
    MPCSource -- produces in production). An earlier version of this
    fixture used a (y, x)-only array, which every test built on it
    passed against despite a real 'Dimension band already exists' crash
    in compute_dry_baseline_stats() on real data (grid_ref's
    expand_dims(band=[0]) assumed no 'band' dim existed yet) -- 139
    tests green, real bug shipped. `fill`, if given, is still a plain
    (size, size) 2D array; the leading band dim is added here so every
    call site doesn't need updating.
    """
    x_min, y_min, x_max, y_max = bounds
    xs = np.linspace(x_min, x_max, size)
    ys = np.linspace(y_max, y_min, size)  # descending, north -> south
    if fill is None:
        data = np.ones((size, size), dtype='float64')
    else:
        data = fill
    da = xr.DataArray(
        data[np.newaxis, :, :], dims=('band', 'y', 'x'),
        coords={'band': [1], 'y': ys, 'x': xs},
    )
    da.rio.write_crs('EPSG:4326', inplace=True)
    return da


class TestClipXarrayUsingId:
    def test_reprojects_to_tile_utm_zone(self, tmp_path):
        grid_path = _make_grid_file(tmp_path)
        data = _make_source_dataarray()
        ref = data  # not used unless slope=True

        result = preprocessing.clip_xarray_using_id(
            data_xarray=data,
            grid_shapefile_path=grid_path,
            aoi_id='tile1',
            ref_xarray=ref,
            cell_size=30,
        )

        assert result.rio.crs.to_epsg() == 32645  # zone '45R' -> EPSG:32645

    def test_output_grid_matches_cell_size(self, tmp_path):
        grid_path = _make_grid_file(tmp_path)
        data = _make_source_dataarray()

        cell_size = 100
        result = preprocessing.clip_xarray_using_id(
            data_xarray=data,
            grid_shapefile_path=grid_path,
            aoi_id='tile1',
            ref_xarray=data,
            cell_size=cell_size,
        )

        xs = result['x'].values
        ys = result['y'].values
        # explicit forced grid: spacing == cell_size, y descending
        np.testing.assert_allclose(np.diff(xs), cell_size, rtol=1e-6)
        np.testing.assert_allclose(np.diff(ys), -cell_size, rtol=1e-6)

    def test_buffer_expands_bounds(self, tmp_path):
        grid_path = _make_grid_file(tmp_path)
        data = _make_source_dataarray()

        no_buffer = preprocessing.clip_xarray_using_id(
            data_xarray=data, grid_shapefile_path=grid_path,
            aoi_id='tile1', ref_xarray=data, cell_size=100,
        )
        with_buffer = preprocessing.clip_xarray_using_id(
            data_xarray=data, grid_shapefile_path=grid_path,
            aoi_id='tile1', ref_xarray=data, buffer=1000, cell_size=100,
        )

        # a buffered tile polygon covers strictly more area, so its
        # regridded output must span more pixels in each direction
        assert with_buffer.sizes['x'] > no_buffer.sizes['x']
        assert with_buffer.sizes['y'] > no_buffer.sizes['y']

    def test_resolves_southern_hemisphere_epsg_from_band_letter(self, tmp_path):
        """
        clip_xarray_using_id resolves the UTM EPSG prefix via
        utils.zone_to_epsg, which reads the MGRS latitude band letter
        to pick EPSG:326xx (Northern, bands N-X) vs EPSG:327xx
        (Southern, bands C-M) -- it no longer hardcodes 326 regardless
        of hemisphere (see CLAUDE.md's Future To-Dos, fixed 2026-09-03).

        Zone '45C' uses band 'C', the southernmost MGRS band, so this
        must resolve to EPSG:32745, not EPSG:32645.
        """
        grid_path = _make_grid_file(tmp_path, zone='45C')
        data = _make_source_dataarray()

        result = preprocessing.clip_xarray_using_id(
            data_xarray=data,
            grid_shapefile_path=grid_path,
            aoi_id='tile1',
            ref_xarray=data,
            cell_size=100,
        )

        assert result.rio.crs.to_epsg() == 32745

    def test_slope_true_uses_ref_xarray_bounds(self, tmp_path):
        grid_path = _make_grid_file(tmp_path)
        data = _make_source_dataarray()
        # a smaller reference extent than the tile itself
        ref = _make_source_dataarray(bounds=(85.05, 25.05, 85.45, 25.45), size=10)

        result = preprocessing.clip_xarray_using_id(
            data_xarray=data,
            grid_shapefile_path=grid_path,
            aoi_id='tile1',
            ref_xarray=ref,
            slope=True,
            cell_size=100,
        )
        assert result.rio.crs.to_epsg() == 32645
        assert result.sizes['x'] > 0 and result.sizes['y'] > 0


class TestSmoothenSlope:
    """
    smoothen_slope's own kernel-smoothing math (mean filter over a
    (buffer*2 // cell_size)-sized, reflect-padded, odd-cell window) is
    tested by patching out xrspatial.slope with a small, fully known
    array -- this isolates the smoothing math from real DEM/slope
    computation (which needs an actual DEM and is exercised via
    flood_mapper.prepare_slope in integration, not here) while still
    exercising the real clip_xarray_using_id reprojection step that
    feeds it.
    """

    def _expected_mean_filter(self, arr, size):
        """Reference implementation of the same reflect-padded mean
        filter smoothen_slope implements by hand via
        sklearn.feature_extraction.image.extract_patches_2d."""
        pad = size // 2
        padded = np.pad(arr, pad, mode='reflect')
        out = np.zeros_like(arr, dtype='float64')
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                out[i, j] = padded[i:i + size, j:j + size].mean()
        return out

    def test_kernel_smoothing_matches_manual_mean_filter(self, tmp_path, monkeypatch):
        grid_path = _make_grid_file(tmp_path)
        dem = _make_source_dataarray()

        # buffer=90, cell_size=30 -> y_size = x_size = 180 // 30 = 6 -> odd -> 5
        buffer = 90
        cell_size = 30
        kernel_size = 5

        fake_slope_values = np.arange(25, dtype='float64').reshape(5, 5)
        fake_slope = xr.DataArray(
            fake_slope_values,
            dims=('y', 'x'),
            coords={'y': np.arange(5), 'x': np.arange(5)},
        )

        def fake_slope_fn(_arr):
            return fake_slope

        monkeypatch.setattr(preprocessing.xrspatial, 'slope', fake_slope_fn)

        result = preprocessing.smoothen_slope(
            dem_xarray=dem,
            grid_shapefile_path=grid_path,
            aoi_id='tile1',
            ref_xarray=dem,
            buffer=buffer,
            cell_size=cell_size,
        )

        expected = self._expected_mean_filter(fake_slope_values, kernel_size)
        np.testing.assert_allclose(result.values, expected, rtol=1e-8)
        assert result.dims == fake_slope.dims
        np.testing.assert_array_equal(result['x'].values, fake_slope['x'].values)
        np.testing.assert_array_equal(result['y'].values, fake_slope['y'].values)

    def test_nodata_fill_replaces_nan_before_smoothing(self, tmp_path, monkeypatch):
        grid_path = _make_grid_file(tmp_path)
        dem = _make_source_dataarray()

        buffer = 90
        cell_size = 30
        kernel_size = 5

        values = np.ones((5, 5), dtype='float64')
        values[2, 2] = np.nan
        fake_slope = xr.DataArray(
            values, dims=('y', 'x'),
            coords={'y': np.arange(5), 'x': np.arange(5)},
        )

        monkeypatch.setattr(preprocessing.xrspatial, 'slope', lambda _arr: fake_slope)

        result = preprocessing.smoothen_slope(
            dem_xarray=dem, grid_shapefile_path=grid_path, aoi_id='tile1',
            ref_xarray=dem, buffer=buffer, cell_size=cell_size, nodata=0,
        )

        # NaN at (2,2) is filled with `nodata`=0 before the kernel runs,
        # so no NaN should propagate into the smoothed output at all.
        assert not np.isnan(result.values).any()
        expected = self._expected_mean_filter(
            np.where(np.isnan(values), 0, values), kernel_size
        )
        np.testing.assert_allclose(result.values, expected, rtol=1e-8)

    def test_real_xrspatial_slope_end_to_end(self, tmp_path):
        """
        Not-mocked sanity check: a synthetic DEM with a real, gentle
        constant east-west gradient run through the real xrspatial.slope
        + real smoothing kernel should produce a small, finite, roughly
        uniform slope field (no NaNs, no wild pixel-to-pixel jumps).
        """
        grid_path = _make_grid_file(tmp_path)
        x_min, y_min, x_max, y_max = (84.9, 24.9, 85.6, 25.6)
        size = 40
        xs = np.linspace(x_min, x_max, size)
        ys = np.linspace(y_max, y_min, size)
        xx, _ = np.meshgrid(xs, ys)
        dem_values = (xx - x_min) * 1000.0  # gentle linear ramp, meters-ish
        dem = xr.DataArray(dem_values, dims=('y', 'x'), coords={'y': ys, 'x': xs})
        dem.rio.write_crs('EPSG:4326', inplace=True)

        result = preprocessing.smoothen_slope(
            dem_xarray=dem,
            grid_shapefile_path=grid_path,
            aoi_id='tile1',
            ref_xarray=dem,
            buffer=90,
            cell_size=30,
        )

        assert np.isfinite(result.values).all()
        assert (result.values >= 0).all()  # xrspatial.slope is in degrees, non-negative


class TestReadSentinel1Stac:
    def test_converts_decibel_to_linear_and_preserves_item_id(self):
        vv_db = xr.DataArray(np.array([[0.0, 10.0], [20.0, 0.0]]))
        vh_db = xr.DataArray(np.array([[10.0, 0.0], [0.0, 10.0]]))

        source = MagicMock()
        source.read_vv_vh.return_value = (vv_db, vh_db)

        stac_item = MagicMock()
        stac_item.id = 'scene123'

        item_id, out = preprocessing.read_sentinel1_stac(stac_item, source, overview_level=2)

        assert item_id == 'scene123'
        np.testing.assert_allclose(out['vv_ds'].values, 10 ** (vv_db.values / 10))
        np.testing.assert_allclose(out['vh_ds'].values, 10 ** (vh_db.values / 10))
        source.read_vv_vh.assert_called_once_with(stac_item, overview_level=2, bbox=None)

    def test_passes_bbox_through_to_source(self):
        source = MagicMock()
        source.read_vv_vh.return_value = (
            xr.DataArray(np.array([[0.0]])), xr.DataArray(np.array([[0.0]])),
        )
        stac_item = MagicMock()
        stac_item.id = 'scene123'
        bbox = (84.9, 24.9, 85.6, 25.6)

        preprocessing.read_sentinel1_stac(stac_item, source, overview_level=2, bbox=bbox)

        source.read_vv_vh.assert_called_once_with(stac_item, overview_level=2, bbox=bbox)


class TestReprojectClipStac:
    def test_reprojects_and_clips_each_scene(self, tmp_path):
        grid_path = _make_grid_file(tmp_path)
        vv = _make_source_dataarray()
        vh = _make_source_dataarray()

        reprojected_dict = {'scene1': {'vv_ds': vv, 'vh_ds': vh}}
        aoi_scene_dict = {'tile1': ['scene1']}

        out = preprocessing.reproject_clip_stac(
            reprojected_dict, aoi_scene_dict, grid_path, 'tile1', max_workers=1,
        )

        assert set(out.keys()) == {'scene1'}
        assert out['scene1']['vv_ds'].rio.crs.to_epsg() == 32645
        assert out['scene1']['vh_ds'].rio.crs.to_epsg() == 32645
        # clipped output should be smaller than the (larger) source extent
        assert out['scene1']['vv_ds'].sizes['x'] < vv.sizes['x']

    def test_multiple_scenes_all_processed(self, tmp_path):
        grid_path = _make_grid_file(tmp_path)
        reprojected_dict = {
            'scene1': {'vv_ds': _make_source_dataarray(), 'vh_ds': _make_source_dataarray()},
            'scene2': {'vv_ds': _make_source_dataarray(), 'vh_ds': _make_source_dataarray()},
        }
        aoi_scene_dict = {'tile1': ['scene1', 'scene2']}

        out = preprocessing.reproject_clip_stac(
            reprojected_dict, aoi_scene_dict, grid_path, 'tile1', max_workers=2,
        )
        assert set(out.keys()) == {'scene1', 'scene2'}


class TestStackImages:
    def test_stacks_scenes_along_band_dimension(self, tmp_path):
        grid_path = _make_grid_file(tmp_path)
        clipped_dict = {
            'scene1': {'vv_ds': _make_source_dataarray(fill=np.full((40, 40), 1.0)),
                       'vh_ds': _make_source_dataarray(fill=np.full((40, 40), 2.0))},
            'scene2': {'vv_ds': _make_source_dataarray(fill=np.full((40, 40), 3.0)),
                       'vh_ds': _make_source_dataarray(fill=np.full((40, 40), 4.0))},
        }

        result = preprocessing.stack_images(
            clipped_dict, grid_path, 'tile1', max_workers=1, cell_size=100,
        )

        assert 'vv_stack' in result and 'vh_stack' in result
        assert result['vv_stack'].sizes['band'] == 2
        assert result['vh_stack'].sizes['band'] == 2
        # constant-fill inputs -> constant-value scenes after regridding
        np.testing.assert_allclose(result['vv_stack'].isel(band=0).values, 1.0, rtol=1e-6)
        np.testing.assert_allclose(result['vh_stack'].isel(band=1).values, 4.0, rtol=1e-6)


class TestComputeDryBaselineStats:
    """
    compute_dry_baseline_stats() replaces stack_images()-then-.mean()/
    .std() for generate_mean_std_by_aoi()'s actual baseline fit, folding
    each scene into a running Welford accumulator instead of holding
    every scene in memory as one concatenated stack (see its docstring
    for why). These tests confirm the two are numerically equivalent
    before relying on the incremental path in production.
    """

    def test_matches_stack_then_reduce(self, tmp_path):
        grid_path = _make_grid_file(tmp_path)
        rng = np.random.default_rng(0)
        n_scenes = 5
        clipped_dict = {}
        for i in range(n_scenes):
            vv_fill = rng.normal(1.0, 0.1, size=(40, 40))
            vh_fill = rng.normal(2.0, 0.2, size=(40, 40))
            # A couple of scenes carry stray large sentinel values (an
            # upstream nodata artifact), same as compute_dry_baseline_stats()
            # is documented to mask to NaN before folding -- exercises the
            # NaN-aware (skipna=True-equivalent) accumulation path, not
            # just the plain-mean/std case.
            if i in (1, 3):
                vv_fill[0:3, 0:3] = 99.0
            clipped_dict[f'scene{i}'] = {
                'vv_ds': _make_source_dataarray(fill=vv_fill),
                'vh_ds': _make_source_dataarray(fill=vh_fill),
            }

        stacked = preprocessing.stack_images(
            clipped_dict, grid_path, 'tile1', max_workers=1, cell_size=100,
        )
        vv_stack = stacked['vv_stack'].where(stacked['vv_stack'] < 50, np.nan)
        vh_stack = stacked['vh_stack'].where(stacked['vh_stack'] < 50, np.nan)
        expected_vv_mean = vv_stack.mean(axis=0)
        expected_vv_std = vv_stack.std(axis=0)
        expected_vh_mean = vh_stack.mean(axis=0)
        expected_vh_std = vh_stack.std(axis=0)

        result = preprocessing.compute_dry_baseline_stats(
            clipped_dict, grid_path, 'tile1', max_workers=1, cell_size=100,
        )

        # Explicit shape checks, not just assert_allclose: a (1, y, x)
        # vs. (y, x) mismatch broadcasts silently in assert_allclose and
        # would have hidden the real "Dimension band already exists"
        # regression (grid_ref's expand_dims assumed the incidental
        # 'band' dim clip_xarray_using_id() passes through was already
        # squeezed off mean/std) -- confirm the dims explicitly so a
        # shape regression here fails loudly, not silently.
        assert result['vv']['mean'].dims == ('y', 'x')
        assert result['vv']['std'].dims == ('y', 'x')
        assert result['vh']['mean'].dims == ('y', 'x')
        assert result['vh']['std'].dims == ('y', 'x')

        np.testing.assert_allclose(result['vv']['mean'].values, expected_vv_mean.values, rtol=1e-8, atol=1e-10)
        np.testing.assert_allclose(result['vv']['std'].values, expected_vv_std.values, rtol=1e-8, atol=1e-10)
        np.testing.assert_allclose(result['vh']['mean'].values, expected_vh_mean.values, rtol=1e-8, atol=1e-10)
        np.testing.assert_allclose(result['vh']['std'].values, expected_vh_std.values, rtol=1e-8, atol=1e-10)

    def test_grid_ref_has_leading_band_dim_and_crs(self, tmp_path):
        grid_path = _make_grid_file(tmp_path)
        clipped_dict = {
            'scene1': {'vv_ds': _make_source_dataarray(fill=np.full((40, 40), 1.0)),
                       'vh_ds': _make_source_dataarray(fill=np.full((40, 40), 2.0))},
        }

        result = preprocessing.compute_dry_baseline_stats(
            clipped_dict, grid_path, 'tile1', max_workers=1, cell_size=100,
        )

        assert result['grid_ref'].sizes['band'] == 1
        assert result['grid_ref'].rio.crs is not None
        assert 'y' in result['grid_ref'].coords and 'x' in result['grid_ref'].coords

    def test_pixel_nan_in_every_scene_stays_nan(self, tmp_path):
        grid_path = _make_grid_file(tmp_path)
        vv_fill = np.full((40, 40), 99.0)  # sentinel in every scene, every pixel
        vh_fill = np.full((40, 40), 2.0)
        clipped_dict = {
            'scene1': {'vv_ds': _make_source_dataarray(fill=vv_fill), 'vh_ds': _make_source_dataarray(fill=vh_fill)},
            'scene2': {'vv_ds': _make_source_dataarray(fill=vv_fill), 'vh_ds': _make_source_dataarray(fill=vh_fill)},
        }

        result = preprocessing.compute_dry_baseline_stats(
            clipped_dict, grid_path, 'tile1', max_workers=1, cell_size=100,
        )

        assert np.isnan(result['vv']['mean'].values).all()
        assert np.isnan(result['vv']['std'].values).all()
        # VH had no sentinel values -- confirms the NaN masking is per-band
        np.testing.assert_allclose(result['vh']['mean'].values, 2.0, rtol=1e-6)
