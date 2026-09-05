# tests/test_mapfloods.py

"""
Tests for autofloods.mapfloods.

map_anomaly_cells() and map_floods() are deprecated (kept importable for
backward compat, no longer called internally -- see the module docstring;
use autofloods.detectors.ZScoreDetector instead) but map_anomaly_cells()'s
classification logic was previously untested, so it is exercised directly
here. flood_images() is NOT deprecated and is still actively used by
flood_mapper.map_floods() when export_maps=True.

map_floods() itself is not tested here: exercising it would require a real
GDAL-readable slope raster at a `..._{id}.nc` path opened via
xr.load_dataarray(..., engine='rasterio') plus nested mean_std_by_aoi /
wet_scenes_by_aoi xarray dicts -- a lot of scaffolding for a deprecated,
internally-unused function whose core logic (map_anomaly_cells) is already
covered below plus a simple slope threshold `.where()` call.

No network access, no real matplotlib display (module calls plt.ioff() at
import time; we also force the Agg backend before importing pyplot here in
case this file is ever collected before that import happens).
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import pytest
import xarray as xr

from autofloods import mapfloods


class TestMapAnomalyCells:
    """
    Classification encoding per the docstring:
      0 = no flood
      1 = VH only
      2 = VV only
      3 = both bands agree (high-confidence class)

    floods_vv = 1 when (vv_ds - vv_mean) / vv_std < vv_thd
    floods_vh = 1 when (vh_ds - vh_mean) / vh_std < vh_thd

    Build a 4-cell synthetic grid, one cell per class, with mean=0,
    std=1 everywhere so the anomaly value equals the post-scene value
    directly, and thresholds of -2 for both bands.
    """

    def _build(self):
        pre = xr.DataArray(
            np.array([
                [0.0, 0.0, 0.0, 0.0],  # vv_mean
                [1.0, 1.0, 1.0, 1.0],  # vv_std
                [0.0, 0.0, 0.0, 0.0],  # vh_mean
                [1.0, 1.0, 1.0, 1.0],  # vh_std
            ]),
            dims=["band", "cell"],
            coords={"band": ["vv_mean", "vv_std", "vh_mean", "vh_std"]},
        )
        post = xr.DataArray(
            np.array([
                [0.0, 0.0, -3.0, -3.0],  # vv_ds: no, no, drop, drop
                [0.0, -3.0, 0.0, -3.0],  # vh_ds: no, drop, no, drop
            ]),
            dims=["band", "cell"],
            coords={"band": ["vv_ds", "vh_ds"]},
        )
        return pre, post

    def test_classification_matches_expected_encoding(self):
        pre, post = self._build()
        result = mapfloods.map_anomaly_cells(pre, post, vv_thd=-2, vh_thd=-2)
        # cell order: [no flood, vh only, vv only, both]
        np.testing.assert_array_equal(result.values.astype(int), [0, 1, 2, 3])

    def test_threshold_boundary_is_exclusive_at_equal_value(self):
        # anomaly exactly equal to threshold should NOT flag as flooded
        # (comparison is strictly `<`, not `<=`)
        pre, post = self._build()
        post2 = post.copy(deep=True)
        post2.values[0, 0] = -2.0  # vv anomaly == -2 == vv_thd, cell 0
        result = mapfloods.map_anomaly_cells(pre, post2, vv_thd=-2, vh_thd=-2)
        assert int(result.values[0]) == 0

    def test_only_vh_flag_gives_class_one(self):
        pre, post = self._build()
        result = mapfloods.map_anomaly_cells(pre, post, vv_thd=-2, vh_thd=-2)
        assert int(result.values[1]) == 1

    def test_only_vv_flag_gives_class_two(self):
        pre, post = self._build()
        result = mapfloods.map_anomaly_cells(pre, post, vv_thd=-2, vh_thd=-2)
        assert int(result.values[2]) == 2

    def test_both_flags_give_class_three(self):
        pre, post = self._build()
        result = mapfloods.map_anomaly_cells(pre, post, vv_thd=-2, vh_thd=-2)
        assert int(result.values[3]) == 3


class TestFloodImages:
    def _make_flood_xarray(self):
        data = np.array([[0, 1], [2, 3]], dtype="int32")
        return xr.DataArray(
            data,
            dims=("y", "x"),
            coords={"y": [1.0, 0.0], "x": [0.0, 1.0]},
        )

    def test_saves_nonempty_png(self, tmp_path):
        flood_xarray = self._make_flood_xarray()
        outfile = tmp_path / "flood_map.png"

        mapfloods.flood_images(flood_xarray, str(outfile))

        assert outfile.exists()
        assert outfile.stat().st_size > 0

    def test_no_crs_falls_back_to_native_coordinates(self, tmp_path, caplog):
        # flood_xarray here has no CRS (matches the no-CRS array above) --
        # to_wgs84=True (the default) must not raise, just warn and plot
        # in whatever coordinates it already has.
        flood_xarray = self._make_flood_xarray()
        outfile = tmp_path / "flood_map.png"

        mapfloods.flood_images(flood_xarray, str(outfile))

        assert outfile.exists()
        assert "no CRS" in caplog.text

    def test_reprojects_to_wgs84_when_crs_present(self, tmp_path):
        # a real UTM-45N raster (arbitrary tile-sized coords) should get
        # reprojected to EPSG:4326 -- resulting x/y (lon/lat) must land
        # in plausible geographic ranges, not raw UTM easting/northing.
        data = np.array([[0, 1, 3], [3, 0, 1], [0, 0, 3]], dtype="uint8")
        flood_xarray = xr.DataArray(
            data, dims=("y", "x"),
            coords={"y": [3300000.0, 3299970.0, 3299940.0], "x": [500000.0, 500030.0, 500060.0]},
        ).rio.write_crs("EPSG:32645")
        outfile = tmp_path / "flood_map.png"

        mapfloods.flood_images(flood_xarray, str(outfile))

        assert outfile.exists()
        assert outfile.stat().st_size > 0

    def test_title_and_label_fontsize_applied(self, tmp_path):
        flood_xarray = self._make_flood_xarray()
        outfile = tmp_path / "flood_map.png"

        captured = {}
        real_set_title = plt.Axes.set_title

        def spy_set_title(self, *args, **kwargs):
            captured["title_kwargs"] = kwargs
            return real_set_title(self, *args, **kwargs)

        import unittest.mock as mock
        with mock.patch.object(plt.Axes, "set_title", spy_set_title):
            mapfloods.flood_images(flood_xarray, str(outfile), title_fontsize=22, label_fontsize=16)

        assert captured["title_kwargs"]["fontsize"] == 22
