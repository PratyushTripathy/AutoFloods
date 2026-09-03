# tests/test_utils.py

"""
Tests for autofloods.utils's deterministic logic (date/string parsing,
dB conversion, date-token extraction, flood-date combination) and the
retry/IO helpers (open_rasterio_with_retry, gpd_to_json, export_xarray,
seggregate_sentinel_search) against synthetic in-memory or tmp_path
data. No network access.
"""

import datetime

import geopandas as gpd
import numpy as np
import pytest
import rasterio
import rioxarray
import xarray as xr
from shapely.geometry import Polygon, box

from autofloods import utils


class TestDateHelpers:
    def test_date_range_inclusive(self):
        start, end = utils.date_range('01/08/2023', 10)
        assert start == datetime.date(2023, 8, 1)
        assert end == datetime.date(2023, 8, 11)

    def test_string_to_date_range_basic_month(self):
        start, end = utils.string_to_date_range('2024/07', '2024/07')
        assert start == datetime.date(2024, 7, 1)
        assert end == datetime.date(2024, 7, 31)

    def test_string_to_date_range_spans_months(self):
        start, end = utils.string_to_date_range('2024/07', '2024/10')
        assert start == datetime.date(2024, 7, 1)
        assert end == datetime.date(2024, 10, 31)

    def test_string_to_date_range_december_wraps_year(self):
        start, end = utils.string_to_date_range('2024/12', '2024/12')
        assert start == datetime.date(2024, 12, 1)
        assert end == datetime.date(2024, 12, 31)

    def test_string_to_date_range_february_leap_year(self):
        start, end = utils.string_to_date_range('2024/02', '2024/02')
        assert end == datetime.date(2024, 2, 29)  # 2024 is a leap year

    def test_string_to_date_range_february_non_leap_year(self):
        start, end = utils.string_to_date_range('2023/02', '2023/02')
        assert end == datetime.date(2023, 2, 28)


class TestDecibelConversion:
    def test_round_trip(self):
        original = np.array([0.001, 0.01, 0.1, 1.0, 10.0])
        db = utils.linear_to_decibel(original)
        back = utils.decibel_to_linear(db)
        np.testing.assert_allclose(back, original, rtol=1e-10)

    def test_decibel_to_linear_known_value(self):
        # 0 dB == linear power 1
        assert utils.decibel_to_linear(0) == pytest.approx(1.0)
        # 10 dB == linear power 10
        assert utils.decibel_to_linear(10) == pytest.approx(10.0)

    def test_linear_to_decibel_known_value(self):
        assert utils.linear_to_decibel(1.0) == pytest.approx(0.0)
        assert utils.linear_to_decibel(10.0) == pytest.approx(10.0)


class TestExtractDateToken:
    def test_mpc_style_token_at_default_index(self):
        key = "S1A_IW_GRDH_1SDV_20240115T120000_20240115T120025_MPC_30_v1.0"
        assert utils._extract_date_token(key) == "20240115"

    def test_opera_style_falls_back_to_scan(self):
        key = "OPERA_PASS_20240115"
        assert utils._extract_date_token(key) == "20240115"

    def test_raises_when_no_date_token_present(self):
        with pytest.raises(ValueError):
            utils._extract_date_token("no_date_here_at_all")


class TestCombineFloodDates:
    def test_combines_dict_of_xarrays_with_max(self):
        a = xr.DataArray(np.array([[0, 1], [3, 0]]))
        b = xr.DataArray(np.array([[3, 0], [0, 0]]))
        flood_data = {
            "OPERA_PASS_20240115": a,
            "OPERA_PASS_20240115_dup": b,  # same date, different scene
        }
        result = utils.combine_flood_dates(flood_data, date_index=-5)
        assert list(result.keys()) == ["20240115"]
        np.testing.assert_array_equal(result["20240115"], np.array([[3, 1], [3, 0]]))

    def test_separates_different_dates(self):
        a = xr.DataArray(np.array([[1]]))
        b = xr.DataArray(np.array([[2]]))
        flood_data = {"OPERA_PASS_20240115": a, "OPERA_PASS_20240120": b}
        result = utils.combine_flood_dates(flood_data, date_index=-5)
        assert set(result.keys()) == {"20240115", "20240120"}

    def test_rejects_unsupported_type(self):
        with pytest.raises(TypeError):
            utils.combine_flood_dates(42)


class TestFloodData3DStack:
    def test_stack_shape_and_date_order(self):
        flood_data = {
            "OPERA_PASS_20240120": xr.DataArray(np.array([[1, 1]])),
            "OPERA_PASS_20240115": xr.DataArray(np.array([[0, 0]])),
        }
        dates, stack = utils.flood_data_3dstack(flood_data, date_index=-5)
        assert dates == ["20240115", "20240120"]  # sorted chronologically
        assert stack.shape == (2, 1, 2)


class TestDefaultMaxWorkers:
    def test_never_returns_less_than_one(self, monkeypatch):
        monkeypatch.setenv("SLURM_CPUS_PER_TASK", "1")
        assert utils.default_max_workers() >= 1

    def test_prefers_slurm_env_over_cpu_count(self, monkeypatch):
        monkeypatch.setenv("SLURM_CPUS_PER_TASK", "8")
        assert utils.default_max_workers() == 7


class TestOpenRasterioWithRetry:
    def test_succeeds_on_first_attempt(self, monkeypatch):
        fake_da = xr.DataArray(np.zeros((2, 2)))

        def fake_open_rasterio(href, **kwargs):
            return fake_da

        monkeypatch.setattr(utils.rioxarray, "open_rasterio", fake_open_rasterio)
        result = utils.open_rasterio_with_retry("fake_href", max_attempts=3, backoff_seconds=0)
        assert result is fake_da

    def test_retries_then_succeeds(self, monkeypatch):
        fake_da = xr.DataArray(np.zeros((2, 2)))
        calls = {"n": 0}

        def flaky_open_rasterio(href, **kwargs):
            calls["n"] += 1
            if calls["n"] < 3:
                raise IOError("transient failure")
            return fake_da

        monkeypatch.setattr(utils.rioxarray, "open_rasterio", flaky_open_rasterio)
        monkeypatch.setattr(utils.time, "sleep", lambda s: None)
        result = utils.open_rasterio_with_retry("fake_href", max_attempts=5, backoff_seconds=0)
        assert result is fake_da
        assert calls["n"] == 3

    def test_raises_last_exception_after_max_attempts(self, monkeypatch):
        def always_fails(href, **kwargs):
            raise IOError("permanent failure")

        monkeypatch.setattr(utils.rioxarray, "open_rasterio", always_fails)
        monkeypatch.setattr(utils.time, "sleep", lambda s: None)
        with pytest.raises(IOError, match="permanent failure"):
            utils.open_rasterio_with_retry("fake_href", max_attempts=2, backoff_seconds=0)

    def test_callable_href_reevaluated_every_attempt(self, monkeypatch):
        """
        A callable href (e.g. a re-signing lambda) must be called again
        before every attempt, not just once -- this is the exact
        mechanism the SAS-token-expiry fix (see mpc.py's read_vv_vh
        docstring) depends on.
        """
        fake_da = xr.DataArray(np.zeros((2, 2)))
        href_calls = {"n": 0}
        open_calls = {"n": 0}

        def href_factory():
            href_calls["n"] += 1
            return f"href-attempt-{href_calls['n']}"

        seen_hrefs = []

        def fake_open_rasterio(href, **kwargs):
            open_calls["n"] += 1
            seen_hrefs.append(href)
            if open_calls["n"] < 2:
                raise IOError("stale token")
            return fake_da

        monkeypatch.setattr(utils.rioxarray, "open_rasterio", fake_open_rasterio)
        monkeypatch.setattr(utils.time, "sleep", lambda s: None)
        utils.open_rasterio_with_retry(href_factory, max_attempts=3, backoff_seconds=0)

        assert href_calls["n"] == 2
        assert seen_hrefs == ["href-attempt-1", "href-attempt-2"]


class TestGpdToJson:
    def _make_grid_file(self, tmp_path):
        gdf = gpd.GeoDataFrame(
            {
                "ID": [1, 2],
                "zone": ["45R", "45R"],
                "dry_month": ["04,05", "04,05"],
                "geometry": [box(85.0, 25.0, 86.0, 26.0), box(86.0, 25.0, 87.0, 26.0)],
            },
            crs="EPSG:4326",
        )
        path = tmp_path / "grid.gpkg"
        gdf.to_file(path)
        return str(path)

    def test_separate_returns_one_bbox_per_id_with_properties(self, tmp_path):
        path = self._make_grid_file(tmp_path)
        result = utils.gpd_to_json([1, 2], path, separate=True)
        assert len(result) == 2
        ids = {r["properties"]["ID"] for r in result}
        assert ids == {1, 2}
        for r in result:
            assert r["type"] == "Polygon"
            assert len(r["coordinates"][0]) == 5  # closed ring

    def test_separate_false_returns_single_combined_bbox(self, tmp_path):
        path = self._make_grid_file(tmp_path)
        result = utils.gpd_to_json([1, 2], path, separate=False)
        assert len(result) == 1
        assert result[0]["properties"]["ID"] == 1  # placeholder id

    def test_filters_to_requested_ids_only(self, tmp_path):
        path = self._make_grid_file(tmp_path)
        result = utils.gpd_to_json([1], path, separate=True)
        assert len(result) == 1
        assert result[0]["properties"]["ID"] == 1


class TestS1ItemFootprint:
    def test_builds_geodataframe_from_item_geometry(self):
        from unittest.mock import MagicMock
        item = MagicMock()
        item.id = "scene1"
        item.geometry = {"coordinates": [[(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)]]}

        gdf = utils.s1item_footprint(item)
        assert list(gdf.columns) == ["ID", "geometry"]
        assert gdf.crs.to_epsg() == 4326
        assert gdf["ID"].iloc[0] == "scene1"


class TestSeggregateSentinelSearch:
    def _aoi(self, id_, geom):
        return {
            "type": "Polygon",
            "coordinates": [list(geom.exterior.coords)],
            "properties": {"ID": id_},
        }

    def _item(self, id_, geom):
        from unittest.mock import MagicMock
        item = MagicMock()
        item.id = id_
        item.geometry = {"coordinates": [list(geom.exterior.coords)]}
        return item

    def test_empty_search_results_map_every_aoi_to_empty_list(self):
        aoi_list = [self._aoi(1, box(0, 0, 1, 1)), self._aoi(2, box(2, 2, 3, 3))]
        aoi_scene_dict, scene_aoi_dict = utils.seggregate_sentinel_search(aoi_list, (None, []))
        assert aoi_scene_dict == {1: [], 2: []}
        assert scene_aoi_dict == {}

    def test_intersecting_scene_maps_both_directions(self):
        aoi_list = [self._aoi(1, box(0, 0, 1, 1))]
        item = self._item("scene1", box(0.5, 0.5, 1.5, 1.5))  # overlaps AOI 1
        aoi_scene_dict, scene_aoi_dict = utils.seggregate_sentinel_search(aoi_list, (None, [item]))
        assert aoi_scene_dict[1] == ["scene1"]
        assert scene_aoi_dict["scene1"] == [1]

    def test_non_intersecting_scene_excluded(self):
        aoi_list = [self._aoi(1, box(0, 0, 1, 1))]
        item = self._item("scene1", box(10, 10, 11, 11))  # nowhere near AOI 1
        aoi_scene_dict, scene_aoi_dict = utils.seggregate_sentinel_search(aoi_list, (None, [item]))
        assert aoi_scene_dict[1] == []
        assert scene_aoi_dict["scene1"] == []


class TestExportXarray:
    def test_writes_readable_2d_geotiff(self, tmp_path):
        data = np.random.rand(4, 4).astype("float32")
        da = xr.DataArray(
            data,
            dims=("y", "x"),
            coords={"y": np.linspace(10, 7, 4), "x": np.linspace(0, 3, 4)},
        )
        da.rio.write_crs("EPSG:4326", inplace=True)

        outfile = tmp_path / "out.tif"
        utils.export_xarray(da, str(outfile))

        assert outfile.exists()
        with rasterio.open(outfile) as src:
            assert src.count == 1
            assert src.width == 4
            assert src.height == 4

    def test_creates_missing_output_directory(self, tmp_path):
        data = np.zeros((2, 2), dtype="float32")
        da = xr.DataArray(
            data, dims=("y", "x"),
            coords={"y": [1, 0], "x": [0, 1]},
        )
        da.rio.write_crs("EPSG:4326", inplace=True)

        outfile = tmp_path / "nested" / "dir" / "out.tif"
        utils.export_xarray(da, str(outfile))
        assert outfile.exists()


class TestNumpyToXarray:
    def test_2d_roundtrip_preserves_georeferencing(self):
        ref = xr.DataArray(
            np.zeros((3, 3)), dims=("y", "x"),
            coords={"y": [2, 1, 0], "x": [0, 1, 2]},
        )
        ref.rio.write_crs("EPSG:4326", inplace=True)

        new_data = np.ones((3, 3))
        result = utils.numpy_to_xarray(new_data, ref)
        assert result.rio.crs == ref.rio.crs
        np.testing.assert_array_equal(result.values, new_data)

    def test_rejects_1d_input(self):
        ref = xr.DataArray(np.zeros((3, 3)), dims=("y", "x"))
        with pytest.raises(ValueError):
            utils.numpy_to_xarray(np.zeros(3), ref)


class TestZoneToEpsg:
    """
    zone_to_epsg resolves a grid_shapefile `zone` string to the
    correct UTM EPSG code from its MGRS latitude-band letter --
    EPSG:326xx (Northern, bands N-X) or EPSG:327xx (Southern, bands
    C-M). Centralizes what used to be four independent hardcoded
    EPSG:326xx call sites (autofloods.__init__.project_flood_raster,
    preprocessing.reproject_clip_stac, preprocessing.clip_xarray_using_id,
    utils.gpd_to_json), all of which silently assumed the Northern
    Hemisphere regardless of the zone's actual band letter (fixed
    2026-09-03, see CLAUDE.md's Future To-Dos).
    """

    def test_northern_band_resolves_to_326(self):
        assert utils.zone_to_epsg('45R') == 'EPSG:32645'

    def test_northern_boundary_band_n(self):
        # 'N' is the first Northern band (0-8N)
        assert utils.zone_to_epsg('45N') == 'EPSG:32645'

    def test_southern_band_resolves_to_327(self):
        assert utils.zone_to_epsg('45C') == 'EPSG:32745'

    def test_southern_boundary_band_m(self):
        # 'M' is the last Southern band (8S-0)
        assert utils.zone_to_epsg('45M') == 'EPSG:32745'

    def test_multi_digit_zone_number(self):
        assert utils.zone_to_epsg('7Q') == 'EPSG:32607'

    def test_lowercase_band_letter_accepted(self):
        assert utils.zone_to_epsg('45r') == 'EPSG:32645'

    def test_unrecognized_band_letter_raises(self):
        with pytest.raises(ValueError):
            utils.zone_to_epsg('45I')  # 'I' is never a valid MGRS band

    def test_all_valid_bands_resolve_without_error(self):
        for band in 'CDEFGHJKLM':  # Southern bands
            assert utils.zone_to_epsg(f'31{band}') == 'EPSG:32731'
        for band in 'NPQRSTUVWX':  # Northern bands
            assert utils.zone_to_epsg(f'31{band}') == 'EPSG:32631'
