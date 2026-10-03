# tests/test_flood_mapper_aoi_route.py

"""
Covers flood_mapper's `aoi=` constructor route -- the one the SoftwareX
paper's worked example (Section 2.5) uses -- which generates the tiling
grid on the fly through autofloods.grid.generate_grid instead of taking a
pre-made grid_shapefile.

generate_grid itself is covered in tests/test_grid.py. What was NOT
covered anywhere before this file is the constructor-side glue around it
(flood_mapper.__init__, the `if aoi is not None` branch): resolving
grid_mode from the source type, requiring grid_dry_months, stamping it
into every tile, writing generated_grid.gpkg under output_dir, and
defaulting grid_id_list to every generated tile.

Scoped to the constructor only. Grid generation is pure geometry and runs
for real (no mocking); nothing downstream of the constructor -- no STAC
query, no scene read -- is called, so no network is needed. The keyword
arguments mirror the published listing exactly, except output_dir and
slope_dir, which point at pytest's tmp_path.
"""
import os

import geopandas as gpd
import pytest
from shapely.geometry import box

import autofloods
from autofloods.detectors import ZScoreDetector
from autofloods.sources import MPCSource, OPERASource

# The published listing's AOI: a box inside MGRS 100 km square 45RVJ, the
# square that covers most of the Figure 2 (Kosi basin) footprint.
LISTING_AOI = box(86.1, 25.4, 86.9, 26.1)


def _listing_kwargs(tmp_path, **overrides):
    kwargs = dict(
        aoi=LISTING_AOI,
        grid_dry_months='04,05',
        dry_years=[2024, 2024],
        wet_duration=['2024/07', '2024/10'],
        slope_dir=str(tmp_path / 'resources' / 'slope'),
        source=OPERASource(),
        detector=ZScoreDetector(vv_thd=-2.5, vh_thd=-2.5),
        output_dir=str(tmp_path / 'output' / 'bihar_2024'),
        cell_size=30,
    )
    kwargs.update(overrides)
    return kwargs


class TestPublishedListingAOIRoute:
    def test_constructor_generates_one_mgrs_tile_like_the_paper(self, tmp_path):
        fm = autofloods.flood_mapper(**_listing_kwargs(tmp_path))

        grid_path = os.path.join(str(tmp_path / 'output' / 'bihar_2024'), 'generated_grid.gpkg')
        assert os.path.exists(grid_path)
        assert fm.grid_shapefile_path == grid_path

        grid = gpd.read_file(grid_path)
        assert len(grid) == 1
        assert grid.loc[0, 'mgrs_tile'] == '45RVJ'
        assert grid.loc[0, 'dry_month'] == '04,05'
        assert fm.selected_grid_id == [1]

    def test_grid_dry_months_is_required_with_aoi(self, tmp_path):
        with pytest.raises(ValueError, match='grid_dry_months'):
            autofloods.flood_mapper(**_listing_kwargs(tmp_path, grid_dry_months=None))


class TestGridModeDefaulting:
    """grid_mode=None resolves from the source type: 'mgrs' for an
    OPERASource (OPERA RTC-S1 is natively MGRS-tiled), 'utm_fishnet'
    otherwise. The two modes leave different fingerprints in the written
    grid -- only MGRS mode adds an 'mgrs_tile' column -- which is what
    these assertions read."""

    def test_opera_source_defaults_to_mgrs(self, tmp_path):
        fm = autofloods.flood_mapper(**_listing_kwargs(tmp_path, source=OPERASource()))
        grid = gpd.read_file(fm.grid_shapefile_path)
        assert 'mgrs_tile' in grid.columns
        assert grid['mgrs_tile'].tolist() == ['45RVJ']

    def test_non_opera_source_defaults_to_utm_fishnet(self, tmp_path):
        fm = autofloods.flood_mapper(**_listing_kwargs(tmp_path, source=MPCSource()))
        grid = gpd.read_file(fm.grid_shapefile_path)
        assert 'mgrs_tile' not in grid.columns
        # a 100 km UTM fishnet over the same 0.8 x 0.7 degree box: still a
        # single zone-45R tile, just not MGRS-aligned
        assert len(grid) >= 1
        assert set(grid['zone']) == {'45R'}
        assert grid['dry_month'].tolist() == ['04,05'] * len(grid)

    def test_explicit_grid_mode_overrides_the_default(self, tmp_path):
        fm = autofloods.flood_mapper(
            **_listing_kwargs(tmp_path, source=OPERASource(), grid_mode='utm_fishnet'))
        grid = gpd.read_file(fm.grid_shapefile_path)
        assert 'mgrs_tile' not in grid.columns
