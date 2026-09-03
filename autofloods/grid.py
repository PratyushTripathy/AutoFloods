# autofloods/grid.py

"""
Grid generation for AOIs that don't already have a pre-made tiling
shapefile.

Two modes:
- mode='mgrs': tiles aligned to MGRS 100km grid squares, one fishnet
  per UTM zone the AOI intersects. Default when the source is
  OPERASource, since OPERA RTC-S1 is natively delivered in MGRS tiles
  -- using MGRS-aligned tiles avoids extra clipping/reprojection
  against the source data's own grid. This is a geometric
  approximation (see generate_grid's docstring), not NASA's
  authoritative MGRS tile database.
- mode='utm_fishnet': a fixed-size fishnet grid (tile_size_km per
  side, default 100) in the AOI's UTM zone(s), independent of MGRS.

Output matches the schema flood_mapper expects from a grid_shapefile:
an id_col (default 'ID', sequential int), a 'zone' column (UTM zone
number + MGRS latitude-band letter, e.g. "43R" or "43C" for a
Southern-Hemisphere tile -- see utils.zone_to_epsg, which resolves
this to EPSG:326<zone> or EPSG:327<zone> from the band letter), a
dry_date_col (default 'dry_month'), and polygon geometry in EPSG:4326.
"""

import math

import geopandas as gpd
import numpy as np
import shapely.geometry
from shapely.geometry import box
from shapely.ops import unary_union

try:
    import mgrs as _mgrs
except ImportError:  # pragma: no cover
    _mgrs = None


# Standard MGRS latitude band letters, south to north, 8 degrees each
# (the X band, 72N-84N, is 12 degrees -- an MGRS-standard
# irregularity). I and O are skipped per the MGRS spec (avoid
# confusion with 1/0).
_LAT_BANDS = 'CDEFGHJKLMNPQRSTUVWX'


def _latitude_band(lat):
    """MGRS latitude band letter for `lat` (degrees, -80 to 84)."""
    if lat < -80 or lat > 84:
        raise ValueError(
            f"Latitude {lat} is outside MGRS's defined range (80S-84N)."
        )
    if lat == 84:
        return 'X'
    band_index = min(int((lat + 80) // 8), len(_LAT_BANDS) - 1)
    return _LAT_BANDS[band_index]


def _utm_zone_number(lon):
    """
    UTM zone number (1-60) for `lon` (degrees). Ignores the Norway/
    Svalbard irregular-zone exceptions (31V, 32V, 31X, 33X, 35X, 37X)
    -- rare for flood-mapping AOIs; not handled in this first cut.
    """
    return int((lon + 180) // 6) + 1


def _resolve_aoi(aoi):
    """Normalize `aoi` (file path, GeoDataFrame, or shapely geometry)
    into a single shapely geometry in EPSG:4326."""
    if isinstance(aoi, str):
        gdf = gpd.read_file(aoi)
    elif isinstance(aoi, gpd.GeoDataFrame):
        gdf = aoi
    elif isinstance(aoi, shapely.geometry.base.BaseGeometry):
        return aoi
    else:
        raise TypeError(
            "aoi must be a file path (str), a GeoDataFrame, or a "
            f"shapely geometry, got {type(aoi)!r}."
        )
    if gdf.crs is not None and gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(epsg=4326)
    return unary_union(gdf.geometry.values)


def _tiles_for_zone(zone_number, hemisphere, aoi_geom_4326, tile_size_m):
    """
    Fishnet of tile_size_m x tile_size_m squares, aligned to multiples
    of tile_size_m in the UTM CRS for (zone_number, hemisphere),
    covering the part of aoi_geom_4326 that falls inside this UTM
    zone's own longitude strip AND this hemisphere's latitude range
    (0 to 84N, or 80S to 0). Northern and Southern UTM for the same
    zone number are different EPSG codes (different false-northing
    conventions), so the AOI must be split by hemisphere before
    reprojecting, not just relabeled afterward.

    Parameters
    ----------
    hemisphere : {'N', 'S'}

    Returns
    -------
    (tiles, epsg) : list of shapely boxes in `epsg` (empty if the AOI
        doesn't reach into this zone/hemisphere), and the EPSG code
        used.
    """
    zone_west = -180 + (zone_number - 1) * 6
    zone_east = zone_west + 6
    if hemisphere == 'N':
        lat_min, lat_max = 0, 84
        epsg = f"EPSG:326{zone_number}"
    else:
        lat_min, lat_max = -80, 0
        epsg = f"EPSG:327{zone_number}"

    zone_strip = box(zone_west, lat_min, zone_east, lat_max)
    aoi_in_zone = aoi_geom_4326.intersection(zone_strip)
    if aoi_in_zone.is_empty:
        return [], epsg

    aoi_utm = gpd.GeoSeries([aoi_in_zone], crs='EPSG:4326').to_crs(epsg).iloc[0]

    minx, miny, maxx, maxy = aoi_utm.bounds
    x0 = math.floor(minx / tile_size_m) * tile_size_m
    y0 = math.floor(miny / tile_size_m) * tile_size_m
    x1 = math.ceil(maxx / tile_size_m) * tile_size_m
    y1 = math.ceil(maxy / tile_size_m) * tile_size_m

    tiles = []
    for x in np.arange(x0, x1, tile_size_m):
        for y in np.arange(y0, y1, tile_size_m):
            cell = box(x, y, x + tile_size_m, y + tile_size_m)
            if cell.intersects(aoi_utm):
                tiles.append(cell)
    return tiles, epsg


def generate_grid(aoi, mode='mgrs', tile_size_km=None, output_path=None,
                   id_col='ID', dry_date_col='dry_month', dry_months=None):
    """
    Generate a tiling grid for an area of interest, in the schema
    flood_mapper expects from a grid_shapefile (see this module's
    docstring), so users don't have to supply a pre-made fishnet.

    mode='mgrs' produces a geometric approximation of true MGRS
    tiling: a 100km-aligned fishnet computed independently per UTM
    zone, labeled via the `mgrs` package's point-to-MGRS conversion.
    It is NOT NASA's authoritative MGRS tile database -- interior
    tiles match real MGRS grid squares exactly (both are defined as
    100km-truncated UTM easting/northing), but OPERA's own tile
    database has irregular boundary tiles right at UTM zone edges that
    this approximation does not replicate.

    Parameters
    ----------
    aoi : str, geopandas.GeoDataFrame, or shapely geometry
        Area of interest boundary. A str is read as a file path
        (shapefile/geopackage); a GeoDataFrame or geometry is used
        directly. Any input CRS is reprojected to EPSG:4326 internally.
    mode : {'mgrs', 'utm_fishnet'}
        'mgrs' (default): tiles aligned to MGRS 100km grid squares, one
        fishnet per UTM zone/latitude band the AOI intersects.
        Recommended (and the default) when the source is OPERASource,
        since OPERA RTC-S1 is natively delivered in MGRS tiles,
        avoiding extra clipping/reprojection against the source data's
        own grid. See the geometric-approximation note above.
        'utm_fishnet': a fixed-size fishnet (tile_size_km per side) in
        the AOI's UTM zone(s), independent of MGRS.
    tile_size_km : float, optional
        Tile side length in kilometers. Only used for
        mode='utm_fishnet' (mode='mgrs' is always 100km, the MGRS grid
        square size -- passing a value with mode='mgrs' raises).
        Defaults to 100 for 'utm_fishnet' if not given.
    output_path : str, optional
        If given, the generated grid is also written here (any format
        geopandas.GeoDataFrame.to_file supports, inferred from the
        extension, e.g. .gpkg).
    id_col : str, optional
        Name of the tile ID column (default 'ID', matching
        flood_mapper's default). Values are sequential ints.
    dry_date_col : str, optional
        Name of the dry-season-month column (default 'dry_month',
        matching flood_mapper's default).
    dry_months : str, optional
        If given, stamped into every tile's `dry_date_col` (e.g.
        "04,05"). Dry season is climate knowledge, not derivable from
        AOI geometry -- if omitted, the column is left as the
        placeholder "REQUIRED" and must be filled in before the grid
        is usable with flood_mapper (every pipeline run does a
        dry-season search as a fixed step, even for OtsuDetector,
        which only skips fitting a statistical baseline from it -- see
        flood_mapper.generate_mean_std_by_aoi's docstring).

    Returns
    -------
    geopandas.GeoDataFrame
        Columns: id_col (int), 'zone' (UTM zone number + MGRS latitude
        band letter, e.g. "43R"), dry_date_col, and (mode='mgrs' only)
        'mgrs_tile' (the true MGRS 100km-square label, e.g. "43RGM" --
        informational; id_col is still the sequential int flood_mapper
        filters on). CRS EPSG:4326, matching the existing
        resources/india_utm_fishnet_buffer.gpkg convention. An AOI
        that straddles the equator produces tiles in both hemispheres
        for the zone(s) it spans -- Northern and Southern UTM for the
        same zone number are different EPSG codes (see
        utils.zone_to_epsg), so each tile is generated in the correct
        one for its own centroid, not just relabeled afterward.
    """
    if mode not in ('mgrs', 'utm_fishnet'):
        raise ValueError(f"mode must be 'mgrs' or 'utm_fishnet', got {mode!r}")
    if mode == 'mgrs':
        if tile_size_km is not None:
            raise ValueError(
                "tile_size_km is not used for mode='mgrs' (MGRS grid "
                "squares are always 100km); pass None or omit it."
            )
        if _mgrs is None:
            raise ImportError(
                "mode='mgrs' requires the 'mgrs' package (pip install mgrs)."
            )
        tile_size_m = 100_000
    else:
        tile_size_km = 100 if tile_size_km is None else tile_size_km
        tile_size_m = tile_size_km * 1000

    aoi_geom = _resolve_aoi(aoi)
    minx, miny, maxx, maxy = aoi_geom.bounds

    zone_min = _utm_zone_number(minx)
    zone_max = _utm_zone_number(maxx)
    hemispheres = []
    if maxy >= 0:
        hemispheres.append('N')
    if miny < 0:
        hemispheres.append('S')

    mgrs_converter = _mgrs.MGRS() if mode == 'mgrs' else None

    rows = []
    tid = 1
    for zone_number in range(zone_min, zone_max + 1):
        for hemisphere in hemispheres:
            tiles_utm, epsg = _tiles_for_zone(zone_number, hemisphere, aoi_geom, tile_size_m)
            if not tiles_utm:
                continue

            tiles_4326 = gpd.GeoSeries(tiles_utm, crs=epsg).to_crs(epsg=4326)

            for geom in tiles_4326:
                centroid = geom.centroid
                band = _latitude_band(centroid.y)
                row = {
                    id_col: tid,
                    'zone': f"{zone_number}{band}",
                    'geometry': geom,
                }
                if mode == 'mgrs':
                    row['mgrs_tile'] = mgrs_converter.toMGRS(
                        centroid.y, centroid.x, MGRSPrecision=0
                    )
                rows.append(row)
                tid += 1

    if not rows:
        raise ValueError(
            "No tiles generated -- the AOI may be empty or invalid."
        )

    grid_gdf = gpd.GeoDataFrame(rows, geometry='geometry', crs='EPSG:4326')
    grid_gdf[dry_date_col] = dry_months if dry_months is not None else 'REQUIRED'

    if output_path is not None:
        grid_gdf.to_file(output_path)

    return grid_gdf
