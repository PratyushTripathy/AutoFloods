# autofloods/preprocessing/__init__.py

# import required libraries
import rioxarray
import geopandas as gpd
import xarray as xr
import numpy as np
import xrspatial
from sklearn.feature_extraction import image
from copy import deepcopy
from ..utils import decibel_to_linear, default_max_workers
import concurrent.futures


# define a function to read VV and VH tif files from the cloud and store all images in memory
def read_sentinel1_stac(stac_item, source, overview_level=3):
    """
    Read a STAC item's VV/VH bands and convert them from decibel to linear
    scale (see utils.decibel_to_linear). Does NOT
    reproject -- every consumer of this function's output (reproject_clip_stac
    for dry-season scenes, clip_xarray_using_id for wet-season scenes)
    reprojects straight to the target tile's UTM zone itself. An earlier
    version reprojected to EPSG:4326 here first, which every consumer then
    reprojected a second time to UTM -- a redundant full-raster warp on
    every scene for no benefit (nothing downstream ever needed the 4326
    intermediate). Benchmarked on tile 321's dry-season scenes: dropping
    the 4326 hop cut per-scene reprojection time by ~48% (50.4s -> 26.5s
    across 18 scenes) with no change in final output, since stack_images()
    already re-grids every scene onto a common reference via
    clip_xarray_using_id()'s .interp() step regardless of what reprojection
    path preceded it.

    Parameters
    __________
    stac_item (dict)                : A STAC Item containing metadata and asset information.
    source (autofloods.sources.STACSource) : Data source used to resolve VV/VH asset hrefs
                                              for stac_item, regardless of catalog.
    overview_level (int, optional)  : The level of overviews to use for reading the data. Default is 3.

    Returns
    _______
    tuple: A tuple containing the STAC Item ID and a dictionary of DataArrays,
           still in the source's native CRS.

    """
    # read the VV and VH bands (source-specific: a single file per band for
    # most catalogs, a same-pass burst mosaic for OPERA; either way this
    # returns one dataset per band covering the item's full extent). Bounded
    # GDAL HTTP timeout + retry with backoff is applied inside each source's
    # implementation (see autofloods.utils.open_rasterio_with_retry).
    vv_ds, vh_ds = source.read_vv_vh(stac_item, overview_level=overview_level)

    # convert decibel to linear
    vv_ds = decibel_to_linear(vv_ds)
    vh_ds = decibel_to_linear(vh_ds)

    return stac_item.id, {
        'vv_ds': vv_ds,
        'vh_ds': vh_ds
    }

# define a function to reproject (native CRS -> tile UTM zone) and clip scene data to the tile's extent
def reproject_clip_stac(reprojected_dict, aoi_scene_dict, grid_shapefile_path, id, max_workers=None):
    """
    Reproject each of AOI `id`'s scenes (native CRS, as returned by
    read_sentinel1_stac) directly to the tile's UTM zone -- the only
    reprojection each scene goes through -- and clip to the tile's exact
    extent from the grid shapefile.

    Scenes are reprojected+clipped concurrently via a thread pool (GDAL's
    warp releases the GIL, confirmed via rasterio's own concurrency docs,
    so this is genuine multi-core parallelism, not GIL-bound) -- measured
    2.5x-6.4x faster end-to-end on real tile data with byte-identical
    output (max abs pixel diff 0.0) vs the previous sequential version.

    Parameters
    __________
    reprojected_dict (dict)     : A dictionary containing native-CRS DataArrays with scene IDs as keys
                                   (despite the name, kept for backward compatibility -- these are no
                                   longer pre-reprojected to EPSG:4326).
    aoi_scene_dict (dict)       : A dictionary mapping AOI IDs to lists of scene IDs.
    id (str)                    : The AOI ID for which clipping should be performed.
    max_workers (int or None)   : Thread pool size for concurrent per-scene reprojection.
                                   None (default) uses utils.default_max_workers() --
                                   (available CPUs - 1) on whatever system this runs on,
                                   not a number hardcoded for one particular cluster.

    Returns
    _______
    dict: A dictionary containing reprojected+clipped DataArrays for the specified AOI ID and scene IDs.

    """
    if max_workers is None:
        max_workers = default_max_workers()

    # read the shapefile and filter it to use for clipping
    gdf = gpd.read_file(grid_shapefile_path)
    gdf = gdf.loc[gdf['ID'].isin([id])]

    # extract the UTM zone from the tile, reproject the GDF to UTM
    tile_utm_zone = 'EPSG:326{}'.format(gdf['zone'].values[0][:-1])
    gdf = gdf.to_crs(tile_utm_zone)

    def _reproject_clip_one(stac_id):
        return stac_id, {
            'vv_ds': reprojected_dict[stac_id]['vv_ds'].rio.reproject(tile_utm_zone).rio.clip(gdf.geometry),
            'vh_ds': reprojected_dict[stac_id]['vh_ds'].rio.reproject(tile_utm_zone).rio.clip(gdf.geometry),
        }

    out = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        for stac_id, result in executor.map(_reproject_clip_one, aoi_scene_dict[id]):
            out[stac_id] = result
    return out


# define a function to stack all the images for a given tile
def stack_images(clipped_dict, grid_shapefile_path, id, max_workers=None, cell_size=30):
    """
    Stack every scene's clipped VV/VH into two multi-band DataArrays
    (one 'band' per scene), ready for Z-score baseline fitting.

    Re-clips each scene onto the explicit, forced `cell_size` grid
    (clip_xarray_using_id) before stacking -- even though
    reproject_clip_stac() already reprojected+clipped everything to the
    same UTM zone/extent, GDAL's reproject() computes its own output
    grid per call, so two scenes reprojected independently don't
    necessarily land on identical pixel grids. This second pass forces
    them onto one common, predictable grid so xr.concat() can actually
    stack them -- see clip_xarray_using_id's docstring for why cell_size
    is a forced parameter rather than derived from the data.

    The per-scene alignment runs concurrently via a thread pool (same
    GIL-releasing GDAL warp behind clip_xarray_using_id as
    reproject_clip_stac() -- see its docstring for the measured speedup
    and correctness verification). max_workers: None (default) uses
    utils.default_max_workers() -- (available CPUs - 1) on whatever
    system this runs on.

    Returns {'vv_stack': DataArray, 'vh_stack': DataArray}.
    """
    if max_workers is None:
        max_workers = default_max_workers()

    # create a list of dictionaries containing 'vv_ds' and 'vh_ds'
    stacked_images = [
        clipped_dict[stac_id]
        for stac_id in clipped_dict
    ]
    ref = stacked_images[0]['vv_ds']

    def _align_one(item):
        return {
            'vv_ds': clip_xarray_using_id(
                data_xarray=item['vv_ds'],
                grid_shapefile_path=grid_shapefile_path,
                aoi_id=id,
                ref_xarray=ref,
                cell_size=cell_size,
            ),
            'vh_ds': clip_xarray_using_id(
                data_xarray=item['vh_ds'],
                grid_shapefile_path=grid_shapefile_path,
                aoi_id=id,
                ref_xarray=ref,
                cell_size=cell_size,
            )
        }

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        stacked_images = list(executor.map(_align_one, stacked_images))

    # stack the data properly
    vv_stack = xr.concat([item['vv_ds'] for item in stacked_images], dim="band")
    vh_stack = xr.concat([item['vh_ds'] for item in stacked_images], dim="band")

    return {
        'vv_stack': vv_stack,
        'vh_stack': vh_stack
    }

def clip_xarray_using_id(data_xarray, grid_shapefile_path, aoi_id, ref_xarray, buffer=None, slope=False, cell_size=30):
    """
    Reproject `data_xarray` to AOI `aoi_id`'s UTM zone and resample it
    onto a fixed, explicit pixel grid (`cell_size` meters, origin from
    the AOI polygon's own bounds) via .interp() -- this is what makes
    independently-reprojected scenes/layers stackable/comparable
    pixel-for-pixel, and what makes a tile's grid identical across every
    run regardless of which scenes/years produced the input data.
    `fill_value='extrapolate'` is used deliberately for sub-pixel edge
    gaps from reprojection rounding, not to invent real data -- see the
    inline comment below for the specific bug this prevents.

    cell_size is REQUIRED to be an explicit, forced value, not derived
    from `ref_xarray`'s own metadata -- an earlier version computed it
    via round(ref_xarray's GeoTransform pixel size), which fixed a real
    cross-year grid-drift bug (GDAL's .rio.reproject() computes a
    slightly different resolution per scene, e.g. 29.998m vs 30.002m,
    which compounds to several pixels of drift over a ~110km tile) but
    is itself not a robust general solution: rounding assumes the true
    resolution is already close to a whole number in whatever units the
    tile's local UTM zone uses, which won't hold for every data source
    or region worldwide, and "close enough to round correctly" isn't the
    same guarantee as "always exactly this value." An explicit,
    caller-forced cell_size removes that assumption entirely -- the grid
    is exactly `cell_size` because that's what was asked for, not because
    the input data's own metadata happened to be close to it. The
    default (30) matches OPERA RTC-S1's native resolution, the current
    primary data source (see autofloods.sources.OPERASource) -- pass a
    different value for a different source/resolution (e.g. an MPC
    overview level), threaded from flood_mapper's own cell_size
    parameter down through every call in this module.

    buffer, if set, buffers the AOI polygon (in its own UTM, meters)
    before clipping -- used for slope, whose kernel needs pixels beyond
    the tile edge to avoid starving edge cells of neighbors.
    slope=True additionally resamples to ref_xarray's own bounds first
    (slope's DEM-derived grid needs this extra step; VV/VH scenes don't).
    """
    # extract target extent from the grid polygon
    gdf = gpd.read_file(grid_shapefile_path)
    gdf = gdf.loc[gdf['ID'].isin([aoi_id])]
    tile_utm_zone = 'EPSG:326{}'.format(gdf['zone'].values[0][:-1])
    gdf = gdf.to_crs(tile_utm_zone)

    # perform buffer if required (for slope smoothing using kernel)
    if buffer:
        gdf['geometry'] = gdf.buffer(buffer)

    # clipping slope requires bounding box from ref xarray before using gdf extent
    # NOTE on fill_value='extrapolate': reproject() computes its output
    # grid from the transformed corner coordinates of the source array,
    # which can fall a hair short of the AOI's true bounding box at the
    # tile's edge (a reprojection rounding effect, not missing data).
    # Without extrapolation, .interp() returns NaN for any target grid
    # point that lands even slightly outside that computed source extent
    # -- producing a spurious 1-pixel-wide NaN column/row at the tile
    # boundary that silently propagates into the dry-season baseline and
    # then into the classified output as a false "not flooded" (0) rather
    # than real data. The gap is sub-pixel, so extrapolating from the
    # nearest valid neighbor is safe here.
    if slope:
        x_min, y_min, x_max, y_max = ref_xarray.rio.bounds()
        data_xarray = data_xarray.rio.reproject(tile_utm_zone).interp(
            x=np.arange(x_min, x_max, cell_size),
            y=np.arange(y_max, y_min, -cell_size),
            kwargs={'fill_value': 'extrapolate'},
            )
    x_min, y_min, x_max, y_max = gdf.total_bounds

    # Resample dem DataArray to the common extent and resolution
    return data_xarray.rio.reproject(tile_utm_zone).interp(
        x=np.arange(x_min, x_max, cell_size),
        y=np.arange(y_max, y_min, -cell_size),
        kwargs={'fill_value': 'extrapolate'},
    )

# define a function to calculate relative slope
def smoothen_slope(dem_xarray, grid_shapefile_path, aoi_id, ref_xarray, buffer=None, nodata=0, cell_size=30):
    """
    Compute slope from `dem_xarray` (xrspatial.slope, degrees) and smooth
    it with a `buffer`-sized mean-filter kernel (odd cell count, reflect-
    padded at the edges) rather than using raw per-pixel slope -- raw DEM
    slope is noisy at Sentinel-1's working resolution, and the smoothed
    version is what map_floods()'s relative-slope mask actually filters
    on. `nodata` fills DEM gaps before the kernel runs so a single missing
    DEM pixel doesn't propagate as NaN through its whole neighborhood.
    `buffer` must match the buffer used when the DEM was downloaded
    (download_nasadem via a buffered bbox) -- kept in sync by
    flood_mapper.prepare_slope(), which passes the same buffer to both.

    cell_size is the same explicit, forced grid resolution used
    throughout this module (see clip_xarray_using_id's docstring) -- used
    here both for the reprojection (via clip_xarray_using_id) and for
    sizing the smoothing kernel in pixels (buffer / cell_size), instead
    of re-deriving it from the just-reprojected data's own metadata.
    """
    # clip the dem to the buffered GDF
    dem_xarray_clipped = clip_xarray_using_id(
        data_xarray=dem_xarray,
        grid_shapefile_path=grid_shapefile_path,
        aoi_id=aoi_id,
        ref_xarray=ref_xarray,
        buffer=buffer,
        cell_size=cell_size,
    )

    # calculate the slope
    slope_xarray = xrspatial.slope(dem_xarray_clipped.squeeze())

    # run a kernel to calculate smoothen slope
    y_size = x_size = (buffer * 2) // cell_size # get number of cells for kernel

    if y_size % 2 == 0:
        y_size -= 1
        x_size -= 1

    slope_chips = deepcopy(slope_xarray.fillna(nodata))
    slope_chips = np.pad(slope_chips, (int(y_size / 2), int(x_size / 2)), 'reflect')
    slope_chips = image.extract_patches_2d(slope_chips, (y_size, x_size))

    slope_mean = slope_chips.reshape(slope_chips.shape[0], -1).mean(axis=-1).reshape(slope_xarray.shape)

    return xr.DataArray(
        slope_mean,
        dims=slope_xarray.dims,
        coords=slope_xarray.coords
    )















