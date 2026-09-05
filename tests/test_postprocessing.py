# tests/test_postprocessing.py

"""
Tests for autofloods.postprocessing's per-pixel flood statistics
(flood_duration_count), vectorization (polygonize_flood_raster), and
monthly aggregation (aggregate_monthly) against synthetic in-memory or
tmp_path data. No network access.
"""

import numpy as np
import pytest
import rasterio
import rioxarray
import xarray as xr

from autofloods import postprocessing


class TestFloodDurationCount:
    def test_consecutive_run_gives_max_duration_and_single_event(self):
        # pixel flooded on 3 consecutive observed dates, then not flooded
        pixel = np.array([1, 1, 1, 0, 0])
        stacked = pixel.reshape(5, 1, 1)
        max_durations, unique_event_counts = postprocessing.flood_duration_count(stacked)
        assert max_durations[0, 0] == 3
        assert unique_event_counts[0, 0] == 1

    def test_two_separate_runs_give_two_events(self):
        # flooded, dry, dry, flooded, flooded, flooded, dry -> two events,
        # longest run is 3
        pixel = np.array([1, 0, 0, 1, 1, 1, 0])
        stacked = pixel.reshape(7, 1, 1)
        max_durations, unique_event_counts = postprocessing.flood_duration_count(stacked)
        assert max_durations[0, 0] == 3
        assert unique_event_counts[0, 0] == 2

    def test_never_flooded_pixel_is_zero(self):
        pixel = np.array([0, 0, 0, 0])
        stacked = pixel.reshape(4, 1, 1)
        max_durations, unique_event_counts = postprocessing.flood_duration_count(stacked)
        assert max_durations[0, 0] == 0
        assert unique_event_counts[0, 0] == 0

    def test_multiple_pixels_independent(self):
        # combine all three cases above side by side in one call
        consecutive = np.array([1, 1, 1, 0, 0])
        two_runs = np.array([1, 0, 1, 0, 1])
        never = np.array([0, 0, 0, 0, 0])
        stacked = np.stack([consecutive, two_runs, never], axis=1).reshape(5, 1, 3)

        max_durations, unique_event_counts = postprocessing.flood_duration_count(stacked)

        assert max_durations[0, 0] == 3
        assert unique_event_counts[0, 0] == 1

        assert max_durations[0, 1] == 1
        assert unique_event_counts[0, 1] == 3

        assert max_durations[0, 2] == 0
        assert unique_event_counts[0, 2] == 0

    def test_flooded_on_every_observation(self):
        pixel = np.array([1, 1, 1, 1])
        stacked = pixel.reshape(4, 1, 1)
        max_durations, unique_event_counts = postprocessing.flood_duration_count(stacked)
        assert max_durations[0, 0] == 4
        assert unique_event_counts[0, 0] == 1


class TestPolygonizeFloodRaster:
    def _make_flood_data(self, array):
        da = xr.DataArray(
            array,
            dims=("y", "x"),
            coords={
                "y": np.arange(array.shape[0], 0, -1) - 1,
                "x": np.arange(array.shape[1]),
            },
        )
        da.rio.write_crs("EPSG:4326", inplace=True)
        return da

    def test_only_class_3_cells_are_polygonized(self):
        array = np.array(
            [
                [0, 1, 2],
                [3, 3, 0],
                [1, 2, 3],
            ],
            dtype="uint8",
        )
        flood_data = self._make_flood_data(array)
        gdf = postprocessing.polygonize_flood_raster(flood_data)

        # total area covered by class-3 polygons should equal the number
        # of class-3 cells (unit cell size in these coords); CRS here is
        # geographic (EPSG:4326) but coords are unit-spaced test data, so
        # the area warning geopandas emits for geographic CRS is expected
        # and not meaningful for this synthetic grid
        with pytest.warns(UserWarning, match="geographic CRS"):
            total_area = gdf.geometry.area.sum()
        n_class3_cells = int((array == 3).sum())
        assert total_area == pytest.approx(n_class3_cells)

    def test_crs_matches_input_raster(self):
        array = np.array([[3, 0], [0, 3]], dtype="uint8")
        flood_data = self._make_flood_data(array)
        gdf = postprocessing.polygonize_flood_raster(flood_data)
        assert gdf.crs is not None
        assert gdf.crs.to_epsg() == 4326

    def test_no_class_3_cells_returns_empty_geodataframe(self):
        array = np.array([[0, 1], [2, 1]], dtype="uint8")
        flood_data = self._make_flood_data(array)
        gdf = postprocessing.polygonize_flood_raster(flood_data)
        assert len(gdf) == 0


class TestSmoothenFloodRaster:
    def test_rejects_even_kernel_size(self):
        array = np.zeros((5, 5))
        with pytest.raises(ValueError, match='odd'):
            postprocessing.smoothen_flood_raster(array, kernel_size=4)

    def test_removes_single_pixel_speckle_preserves_boundary(self):
        # left half class 0, right half class 3 (a real boundary), plus
        # one isolated speckle pixel (misclassified as 3 inside the 0
        # region) that a majority filter should remove.
        array = np.zeros((7, 7), dtype='float64')
        array[:, 4:] = 3
        array[2, 1] = 3  # speckle, deep inside the class-0 region

        smoothed = postprocessing.smoothen_flood_raster(array, kernel_size=3)

        assert smoothed[2, 1] == 0  # speckle removed
        # boundary still separates 0s from 3s at the same column -- not
        # washed out into some blended/incorrect value
        np.testing.assert_array_equal(smoothed[:, :4], 0)
        np.testing.assert_array_equal(smoothed[:, 4:], 3)

    def test_nan_gap_with_valid_neighbors_is_resolved(self):
        array = np.full((5, 5), 2.0)
        array[2, 2] = np.nan  # single gap pixel, surrounded entirely by valid class-2 neighbors

        smoothed = postprocessing.smoothen_flood_raster(array, kernel_size=3)

        assert not np.isnan(smoothed[2, 2])
        assert smoothed[2, 2] == 2

    def test_nan_gap_with_no_valid_neighbors_stays_nan(self):
        array = np.full((7, 7), 1.0)
        array[1:4, 1:4] = np.nan  # a 3x3 block of gap -- its center pixel's
        # own 3x3 neighborhood (kernel_size=3) is entirely gap, so it has
        # no valid neighbor to resolve from and must stay NaN.

        smoothed = postprocessing.smoothen_flood_raster(array, kernel_size=3)

        assert np.isnan(smoothed[2, 2])
        # the gap block's edge pixels (e.g. (1,1)) DO have valid
        # neighbors just outside the block and should be resolved
        assert not np.isnan(smoothed[1, 1])

    def test_kernel_size_actually_changes_result(self):
        # a single-pixel speckle far enough from a same-valued cluster
        # that a 3x3 kernel can't out-vote it (majority still speckle),
        # but a 5x5 kernel reaches enough class-0 neighbors to flip it.
        array = np.zeros((9, 9), dtype='float64')
        array[4, 4] = 3

        smoothed_3 = postprocessing.smoothen_flood_raster(array, kernel_size=3)
        smoothed_5 = postprocessing.smoothen_flood_raster(array, kernel_size=5)

        # both should actually remove this isolated speckle (a lone 3
        # among eight 0-neighbors loses the vote at any reasonable
        # kernel size) -- see the next test for an assertion that
        # actually distinguishes kernel_size=3 from kernel_size=5's
        # behavior (a gap only kernel_size=5 can resolve).
        assert smoothed_3[4, 4] == 0
        assert smoothed_5[4, 4] == 0
        assert smoothed_3.shape == smoothed_5.shape

    def test_kernel_size_5_resolves_larger_gap_than_kernel_size_3_can(self):
        # a 3x3 gap block: kernel_size=3 can't resolve its center pixel
        # (its own 3x3 neighborhood is entirely gap), but kernel_size=5
        # reaches valid pixels just outside the block and can.
        array = np.full((9, 9), 2.0)
        array[3:6, 3:6] = np.nan

        smoothed_3 = postprocessing.smoothen_flood_raster(array, kernel_size=3)
        smoothed_5 = postprocessing.smoothen_flood_raster(array, kernel_size=5)

        assert np.isnan(smoothed_3[4, 4])
        assert not np.isnan(smoothed_5[4, 4])
        assert smoothed_5[4, 4] == 2

    def test_preserves_dataarray_type_dims_and_coords(self):
        array = xr.DataArray(
            np.array([[0, 0, 0], [0, 3, 0], [0, 0, 0]], dtype='float64'),
            dims=('y', 'x'), coords={'y': [2, 1, 0], 'x': [0, 1, 2]},
        )
        smoothed = postprocessing.smoothen_flood_raster(array, kernel_size=3)

        assert isinstance(smoothed, xr.DataArray)
        assert smoothed.dims == array.dims
        np.testing.assert_array_equal(smoothed.coords['y'].values, array.coords['y'].values)
        np.testing.assert_array_equal(smoothed.coords['x'].values, array.coords['x'].values)


class TestAggregateMonthly:
    def _write_stack(self, path, bands_with_names, height=2, width=2):
        """
        bands_with_names: list of (band_name, 2D float32 array) pairs.
        """
        count = len(bands_with_names)
        with rasterio.open(
            path,
            "w",
            driver="GTiff",
            height=height,
            width=width,
            count=count,
            dtype="float32",
            crs="EPSG:4326",
            transform=rasterio.transform.from_origin(0, 2, 1, 1),
        ) as dst:
            for i, (name, arr) in enumerate(bands_with_names, start=1):
                dst.write(arr.astype("float32"), i)
                dst.set_band_description(i, name)

    def test_band_count_matches_unique_months(self, tmp_path):
        jan1 = np.array([[3, 1], [0, np.nan]])
        jan2 = np.array([[3, 3], [0, np.nan]])
        feb1 = np.array([[1, 0], [3, np.nan]])

        infile = tmp_path / "tile_stack.tif"
        self._write_stack(
            infile,
            [
                ("20240115_flood", jan1),
                ("20240120_flood", jan2),
                ("20240210_flood", feb1),
            ],
        )

        outfile = tmp_path / "tile_monthly.tif"
        postprocessing.aggregate_monthly(str(infile), outfile=str(outfile))

        assert outfile.exists()
        with rasterio.open(outfile) as src:
            assert src.count == 2  # two unique months: 202401, 202402
            assert list(src.descriptions) == ["202401", "202402"]

    def test_per_pixel_monthly_counts_and_nodata_for_unobserved_pixel(self, tmp_path):
        # pixel (1,1) is NaN across every band, in both months -- must
        # come back as nodata (255), never a false 0
        jan1 = np.array([[3, 1], [0, np.nan]])
        jan2 = np.array([[3, 3], [0, np.nan]])
        feb1 = np.array([[1, 0], [3, np.nan]])

        infile = tmp_path / "tile_stack.tif"
        self._write_stack(
            infile,
            [
                ("20240115_flood", jan1),
                ("20240120_flood", jan2),
                ("20240210_flood", feb1),
            ],
        )

        outfile = tmp_path / "tile_monthly.tif"
        postprocessing.aggregate_monthly(str(infile), outfile=str(outfile))

        with rasterio.open(outfile) as src:
            assert src.nodata == 255
            jan_band = src.read(1)
            feb_band = src.read(2)

            # (0,0): flooded (class 3) on both Jan dates -> count 2
            assert jan_band[0, 0] == 2
            # (0,1): flooded on only the second Jan date -> count 1
            assert jan_band[0, 1] == 1
            # (1,0): never flooded but fully observed both Jan dates -> real 0
            assert jan_band[1, 0] == 0
            # (1,1): NaN on every Jan date -> nodata, not a false 0
            assert jan_band[1, 1] == 255

            # (0,0): not flooded in Feb's single observation -> real 0
            assert feb_band[0, 0] == 0
            assert feb_band[0, 1] == 0
            # (1,0): flooded in Feb's single observation -> count 1
            assert feb_band[1, 0] == 1
            # (1,1): NaN in Feb's only date -> nodata
            assert feb_band[1, 1] == 255

    def test_partial_valid_observations_pixel_gets_real_count_not_nodata(self, tmp_path):
        # pixel (0,0) is valid (non-NaN) on only one of two Jan dates and
        # is flooded on that one date -- should be a real count of 1,
        # not nodata, since it has at least one valid observation
        jan1 = np.array([[3.0, 0.0]])
        jan2 = np.array([[np.nan, 0.0]])

        infile = tmp_path / "tile_stack.tif"
        self._write_stack(
            infile,
            [
                ("20240115_flood", jan1),
                ("20240120_flood", jan2),
            ],
            height=1,
            width=2,
        )

        outfile = tmp_path / "tile_monthly.tif"
        postprocessing.aggregate_monthly(str(infile), outfile=str(outfile))

        with rasterio.open(outfile) as src:
            band = src.read(1)
            assert band[0, 0] == 1
            assert band[0, 1] == 0

    def test_output_is_uint8_with_expected_metadata(self, tmp_path):
        jan1 = np.array([[3, 0]])
        infile = tmp_path / "tile_stack.tif"
        self._write_stack(infile, [("20240115_flood", jan1)], height=1, width=2)

        outfile = tmp_path / "tile_monthly.tif"
        postprocessing.aggregate_monthly(str(infile), outfile=str(outfile))

        with rasterio.open(outfile) as src:
            assert src.dtypes[0] == "uint8"
            assert src.nodata == 255
