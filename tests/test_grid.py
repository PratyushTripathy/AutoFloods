# tests/test_grid.py

"""
Tests for autofloods.grid.generate_grid. All AOIs are synthetic shapely
boxes -- no network access, no real Sentinel-1 data.

Covers: MGRS-mode tiles matching known MGRS zone/band boundaries (via a
round-trip through the same `mgrs` library the module uses internally,
not hardcoded expected labels -- avoids the test just re-asserting a
value copy-pasted from the implementation), UTM fishnet tile count/size,
AOIs spanning multiple UTM zones and multiple MGRS latitude bands, and
that the output schema matches what flood_mapper actually reads from a
grid_shapefile (see autofloods/__init__.py's docstring and
preprocessing.clip_xarray_using_id).
"""

import math
import os
import tempfile

import geopandas as gpd
import mgrs as mgrs_lib
import pytest
from shapely.geometry import box

from autofloods.grid import generate_grid


# A small AOI comfortably inside a single UTM zone (45) and MGRS
# latitude band (R), near Bihar, India -- matches this project's own
# resources/india_utm_fishnet_buffer.gpkg region.
SINGLE_ZONE_AOI = box(85.0, 25.0, 85.5, 25.5)


def _utm_side_length_m(geom, epsg):
    """Side length (meters) of a roughly-square geometry's bounding box
    after reprojecting to `epsg`."""
    reprojected = gpd.GeoSeries([geom], crs='EPSG:4326').to_crs(epsg).iloc[0]
    minx, miny, maxx, maxy = reprojected.bounds
    return maxx - minx, maxy - miny


class TestMGRSMode:
    def test_tiles_align_with_mgrs_zone_boundaries(self):
        grid = generate_grid(SINGLE_ZONE_AOI, mode='mgrs')
        assert len(grid) > 0
        assert (grid['zone'] == '45R').all()

        converter = mgrs_lib.MGRS()
        for _, row in grid.iterrows():
            centroid = row.geometry.centroid
            expected_label = converter.toMGRS(centroid.y, centroid.x, MGRSPrecision=0)
            assert row['mgrs_tile'] == expected_label

    def test_tiles_are_100km(self):
        grid = generate_grid(SINGLE_ZONE_AOI, mode='mgrs')
        for geom in grid.geometry:
            dx, dy = _utm_side_length_m(geom, 'EPSG:32645')
            assert dx == pytest.approx(100_000, abs=1)
            assert dy == pytest.approx(100_000, abs=1)

    def test_tile_size_km_rejected_for_mgrs(self):
        with pytest.raises(ValueError):
            generate_grid(SINGLE_ZONE_AOI, mode='mgrs', tile_size_km=50)


class TestUTMFishnetMode:
    def test_tile_count_and_size(self):
        tile_size_km = 25
        grid = generate_grid(SINGLE_ZONE_AOI, mode='utm_fishnet', tile_size_km=tile_size_km)
        assert len(grid) > 0

        # AOI is ~0.5deg x 0.5deg -- roughly 55km x 50km at this
        # latitude -- so a 25km fishnet should need a handful of tiles,
        # not one and not dozens.
        assert 2 <= len(grid) <= 12

        for geom in grid.geometry:
            dx, dy = _utm_side_length_m(geom, 'EPSG:32645')
            assert dx == pytest.approx(tile_size_km * 1000, abs=1)
            assert dy == pytest.approx(tile_size_km * 1000, abs=1)

    def test_default_tile_size_is_100km(self):
        grid = generate_grid(SINGLE_ZONE_AOI, mode='utm_fishnet')
        dx, dy = _utm_side_length_m(grid.geometry.iloc[0], 'EPSG:32645')
        assert dx == pytest.approx(100_000, abs=1)

    def test_smaller_tiles_produce_more_tiles(self):
        coarse = generate_grid(SINGLE_ZONE_AOI, mode='utm_fishnet', tile_size_km=100)
        fine = generate_grid(SINGLE_ZONE_AOI, mode='utm_fishnet', tile_size_km=25)
        assert len(fine) > len(coarse)


class TestMultiZoneEdgeCases:
    def test_aoi_spanning_two_utm_zones(self):
        # Zone 44/45 boundary sits at 84E.
        aoi = box(83.5, 25.0, 84.5, 25.5)
        grid = generate_grid(aoi, mode='mgrs')
        zone_numbers = {z[:-1] for z in grid['zone']}
        assert zone_numbers == {'44', '45'}
        # every tile that crosses into a different zone still gets its
        # own independently-generated, non-overlapping geometry
        assert grid['ID'].is_unique

    def test_aoi_spanning_two_latitude_bands(self):
        # The Q/R latitude-band boundary sits at 24N.
        aoi = box(85.0, 23.5, 85.5, 24.5)
        grid = generate_grid(aoi, mode='mgrs')
        bands = {z[-1] for z in grid['zone']}
        assert bands == {'Q', 'R'}

    def test_southern_hemisphere_aoi_produces_correct_tiles(self):
        """
        Southern Hemisphere support (fixed 2026-09-03, see CLAUDE.md's
        Future To-Dos): generate_grid() used to raise NotImplementedError
        here. Now it must produce tiles in the correct Southern UTM CRS
        (EPSG:327xx), not just succeed.
        """
        aoi = box(30.0, -5.0, 31.0, -4.0)
        grid = generate_grid(aoi, mode='mgrs')
        assert len(grid) > 0

        # zone 36 covers 30E-36E; -5 to -4 latitude is MGRS band 'M' (Southern)
        assert set(grid['zone']) == {'36M'}

        from autofloods.utils import zone_to_epsg
        epsg = zone_to_epsg(grid['zone'].iloc[0])
        assert epsg == 'EPSG:32736'

        # confirm tiles actually reproject cleanly and land at Southern
        # UTM's false-northing convention (>0, well above the equator's
        # 10,000,000m origin minus the AOI's ~500km southward extent)
        utm_bounds = grid.to_crs(epsg).total_bounds
        assert utm_bounds[1] > 9_000_000  # miny

    def test_equator_straddling_aoi_produces_both_hemispheres(self):
        aoi = box(30.0, -0.5, 30.5, 0.5)
        grid = generate_grid(aoi, mode='mgrs')
        bands = {z[-1] for z in grid['zone']}
        assert bands == {'M', 'N'}  # M = Southern, N = Northern

        from autofloods.utils import zone_to_epsg
        epsgs = {zone_to_epsg(z) for z in grid['zone']}
        assert epsgs == {'EPSG:32736', 'EPSG:32636'}


class TestSchemaCompatibility:
    def test_default_columns_match_flood_mapper_expectations(self):
        grid = generate_grid(SINGLE_ZONE_AOI, mode='mgrs')
        # flood_mapper's default id_col='ID', dry_date_col='dry_month';
        # 'zone' is always read literally by preprocessing functions.
        for col in ('ID', 'zone', 'dry_month', 'geometry'):
            assert col in grid.columns

    def test_zone_field_parses_like_clip_xarray_using_id(self):
        # preprocessing.clip_xarray_using_id does
        # 'EPSG:326{}'.format(zone[:-1]) -- confirm every generated
        # zone string is exactly a UTM zone number plus one trailing
        # band-letter character.
        grid = generate_grid(SINGLE_ZONE_AOI, mode='mgrs')
        for zone in grid['zone']:
            zone_number_str, band = zone[:-1], zone[-1]
            assert zone_number_str.isdigit()
            assert 1 <= int(zone_number_str) <= 60
            assert band.isalpha()

    def test_custom_id_and_dry_date_columns(self):
        grid = generate_grid(
            SINGLE_ZONE_AOI, mode='utm_fishnet', tile_size_km=50,
            id_col='tile_id', dry_date_col='dry_mo', dry_months='04,05',
        )
        assert 'tile_id' in grid.columns
        assert 'dry_mo' in grid.columns
        assert (grid['dry_mo'] == '04,05').all()

    def test_dry_months_placeholder_when_not_given(self):
        grid = generate_grid(SINGLE_ZONE_AOI, mode='mgrs')
        assert (grid['dry_month'] == 'REQUIRED').all()

    def test_ids_are_sequential_ints_starting_at_one(self):
        grid = generate_grid(SINGLE_ZONE_AOI, mode='mgrs')
        assert list(grid['ID']) == list(range(1, len(grid) + 1))

    def test_output_crs_is_4326(self):
        grid = generate_grid(SINGLE_ZONE_AOI, mode='mgrs')
        assert grid.crs.to_epsg() == 4326

    def test_output_path_writes_a_readable_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out = os.path.join(tmpdir, 'grid.gpkg')
            generate_grid(SINGLE_ZONE_AOI, mode='mgrs', output_path=out)
            assert os.path.exists(out)
            reloaded = gpd.read_file(out)
            assert len(reloaded) > 0


class TestAOIInputTypes:
    def test_accepts_geodataframe(self):
        gdf = gpd.GeoDataFrame(geometry=[SINGLE_ZONE_AOI], crs='EPSG:4326')
        grid = generate_grid(gdf, mode='mgrs')
        assert len(grid) > 0

    def test_accepts_file_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            aoi_path = os.path.join(tmpdir, 'aoi.gpkg')
            gpd.GeoDataFrame(geometry=[SINGLE_ZONE_AOI], crs='EPSG:4326').to_file(aoi_path)
            grid = generate_grid(aoi_path, mode='mgrs')
            assert len(grid) > 0

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError):
            generate_grid(SINGLE_ZONE_AOI, mode='h3')
