# autofloods/preprocessing/__init__.py

# import required libraries
import rioxarray
import geopandas as gpd
import xarray as xr
import numpy as np
import xrspatial
from sklearn.feature_extraction import image
from copy import deepcopy
from ..utils import decibel_to_linear, default_max_workers, zone_to_epsg
import concurrent.futures
import gc
import itertools


# define a function to read VV and VH tif files from the cloud and store all images in memory
def read_sentinel1_stac(stac_item, source, overview_level=3):
    """
    Read a STAC item's VV/VH bands and convert them from decibel to linear
    scale (see utils.decibel_to_linear). Does NOT reproject -- every
    consumer of this function's output (reproject_clip_stac for
    dry-season scenes, clip_xarray_using_id for wet-season scenes)
    reprojects directly to the target tile's UTM zone itself.

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
    warp releases the GIL, so this is genuine multi-core parallelism, not
    GIL-bound).

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
    tile_utm_zone = zone_to_epsg(gdf['zone'].values[0])
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

    The per-scene alignment runs concurrently via a thread pool for the
    same reason as reproject_clip_stac(). max_workers: None (default)
    uses utils.default_max_workers() -- (available CPUs - 1) on whatever
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


# Stray large sentinel values (e.g. from an upstream nodata convention)
# masked to NaN before folding into the running baseline stats -- same
# threshold generate_mean_std_by_aoi() used to apply post-hoc, over the
# full in-memory stack, before this function existed.
_NODATA_SENTINEL_THRESHOLD = 50


def _welford_init(x):
    """Zeroed (count, mean, M2) accumulator state shaped like `x`."""
    zeros = xr.zeros_like(x, dtype=np.float64)
    return {'count': zeros.copy(), 'mean': zeros.copy(), 'm2': zeros.copy()}


def _welford_update(state, x):
    """
    One Welford's-algorithm update, folding one new scene `x` into
    `state` and returning the new state. NaN-aware per pixel (a NaN in
    `x` leaves that pixel's count/mean/M2 untouched), matching the
    skipna=True default of xarray's .mean()/.std() -- the old
    stack-then-reduce behavior this replaces.
    """
    mask = x.notnull()
    count = state['count'] + mask.astype(np.float64)
    # Divisor for masked-off pixels is never actually used (their delta
    # is forced to 0 below), so `1.0` here is just a safe placeholder to
    # avoid a NaN/inf from dividing by the unchanged (possibly still
    # zero) count.
    safe_count = count.where(mask, other=1.0)
    delta = (x - state['mean']).where(mask, other=0.0)
    mean = state['mean'] + delta / safe_count
    delta2 = (x - mean).where(mask, other=0.0)
    m2 = state['m2'] + delta * delta2
    return {'count': count, 'mean': mean, 'm2': m2}


def _welford_finalize(state):
    """
    (mean, std) DataArrays from a Welford accumulator state -- population
    std (ddof=0), matching xr.DataArray.std()'s default, which is what
    the old stack-then-.std(axis=0) path used. Pixels with zero valid
    (non-NaN) observations across every scene are NaN in both outputs,
    matching what an all-NaN slice's nanmean/nanstd would produce.
    """
    valid = state['count'] > 0
    mean = state['mean'].where(valid)
    variance = (state['m2'] / state['count']).where(valid)
    std = np.sqrt(variance)
    return mean, std


def compute_dry_baseline_stats(clipped_dict, grid_shapefile_path, id, max_workers=None, cell_size=30):
    """
    Per-pixel dry-season mean and standard deviation for VV and VH,
    computed via Welford's online algorithm one aligned scene at a time,
    instead of stack_images()'s xr.concat-then-.mean()/.std() over every
    scene held in memory simultaneously. Only the running (count, mean,
    M2) accumulators plus whichever scenes are concurrently in flight in
    the thread pool (bounded by max_workers) are ever resident at once,
    not the full dry-season stack -- this is what
    generate_mean_std_by_aoi() now calls instead of stack_images() for
    the actual baseline fit. stack_images() itself is unchanged and
    still used elsewhere (e.g. scripts/verification/
    benchmark_full_pipeline_comparison.py).

    The alignment step (clip_xarray_using_id, onto stack_images()'s same
    forced common grid -- see its docstring for why) still runs
    concurrently via a thread pool, since reprojection/read, not the
    stack+reduce, was the actual memory bottleneck; only the fold into
    the running accumulators is strictly one-scene-at-a-time.

    Verified to reproduce stack_images()-then-.mean()/.std() exactly
    (within floating-point tolerance) in
    tests/test_preprocessing.py::TestComputeDryBaselineStats.

    Returns
    -------
    dict with keys:
        'vv': {'mean': DataArray, 'std': DataArray}
        'vh': {'mean': DataArray, 'std': DataArray}
        'grid_ref': DataArray
            The first aligned VV scene (after its raster-artifact 'band'
            dim is squeezed off, see above) with a size-1 leading 'band'
            dim added back, to match stack_images()'s (band, y, x) shape
            -- kept only as a coords/CRS/grid reference for detectors
            that don't fit a real baseline (requires_baseline_fitting=
            False; see autofloods.generate_mean_std_by_aoi's docstring).
            Its pixel values are never read as statistics.
    """
    if max_workers is None:
        max_workers = default_max_workers()

    items = list(clipped_dict.values())
    ref = items[0]['vv_ds']

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

    vv_state = None
    vh_state = None
    grid_ref = None

    # A bounded sliding window, not executor.map(): map() submits every
    # task up front, so worker threads can (and in practice do) race
    # ahead of the main thread's fold loop and finish far more scenes
    # than have been consumed yet -- each finished-but-unconsumed result
    # sits fully in memory until read, which silently reproduces the
    # exact "everything in memory at once" problem this function exists
    # to avoid. Submitting only one replacement task per consumed result
    # caps in-flight + completed-but-unconsumed scenes at max_workers,
    # regardless of how many total scenes there are.
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        remaining = iter(items)
        in_flight = set()
        for item in itertools.islice(remaining, max_workers):
            in_flight.add(executor.submit(_align_one, item))

        while in_flight:
            done, in_flight = concurrent.futures.wait(
                in_flight, return_when=concurrent.futures.FIRST_COMPLETED
            )
            for future in done:
                aligned = future.result()
                vv = aligned['vv_ds'].where(aligned['vv_ds'] < _NODATA_SENTINEL_THRESHOLD, np.nan)
                vh = aligned['vh_ds'].where(aligned['vh_ds'] < _NODATA_SENTINEL_THRESHOLD, np.nan)

                # Every scene read via utils.open_rasterio_with_retry()
                # (both OPERASource and MPCSource) carries a real, incidental
                # 'band' dim of size 1 -- an artifact of opening a
                # single-band GeoTIFF/VRT via rioxarray, not a "which scene"
                # axis -- and clip_xarray_using_id()'s .interp() regrid
                # passes it through unchanged. Squeeze it off here so the
                # Welford accumulators end up as plain (y, x), matching the
                # old vv_stack.mean(axis=0)/.std(axis=0) contract exactly
                # (verified in tests/test_preprocessing.py::
                # TestComputeDryBaselineStats). Without this, grid_ref's
                # expand_dims(band=[0]) below raises "Dimension band
                # already exists" -- caught on a real Colab run, since
                # every synthetic test fixture used a (y, x)-only array
                # that never had this dim to begin with.
                if 'band' in vv.dims:
                    vv = vv.squeeze('band', drop=True)
                if 'band' in vh.dims:
                    vh = vh.squeeze('band', drop=True)

                if grid_ref is None:
                    grid_ref = vv.expand_dims(band=[0])

                vv_state = _welford_init(vv) if vv_state is None else vv_state
                vh_state = _welford_init(vh) if vh_state is None else vh_state
                vv_state = _welford_update(vv_state, vv)
                vh_state = _welford_update(vh_state, vh)
                del aligned, vv, vh
                # rioxarray/xarray objects (CRS, index caches) hold
                # internal reference cycles that plain refcounting can't
                # free -- without an explicit collect, discarded aligned
                # scenes pile up as cyclic garbage until Python's gen0
                # threshold happens to trigger, defeating the bounded
                # window above (confirmed via a real peak-memory
                # regression test: gc.collect() found hundreds of
                # collectable-but-uncollected objects after a 60-scene
                # run without this call).
                gc.collect()

                next_item = next(remaining, None)
                if next_item is not None:
                    in_flight.add(executor.submit(_align_one, next_item))

    vv_mean, vv_std = _welford_finalize(vv_state)
    vh_mean, vh_std = _welford_finalize(vh_state)

    return {
        'vv': {'mean': vv_mean, 'std': vv_std},
        'vh': {'mean': vh_mean, 'std': vh_std},
        'grid_ref': grid_ref,
    }

def clip_xarray_using_id(data_xarray, grid_shapefile_path, aoi_id, ref_xarray, buffer=None, slope=False, cell_size=30):
    """
    Reproject `data_xarray` to AOI `aoi_id`'s UTM zone and resample it
    onto a fixed, explicit pixel grid (`cell_size` meters, origin from
    the AOI polygon's own bounds) via .interp() -- this is what makes
    independently-reprojected scenes/layers stackable pixel-for-pixel,
    and what keeps a tile's grid identical across every run regardless
    of which scenes/years produced the input data.

    cell_size is always caller-supplied rather than derived from
    `ref_xarray`'s own metadata, since GDAL's reproject() computes a
    slightly different resolution per scene call, which would otherwise
    compound into cross-scene/cross-year grid drift. Default (30) matches
    OPERA RTC-S1's native resolution, the current primary data source;
    pass a different value for another source/resolution.

    buffer, if set, buffers the AOI polygon (in its own UTM, meters)
    before clipping -- used for slope, whose kernel needs pixels beyond
    the tile edge to avoid starving edge cells of neighbors.

    slope=True additionally resamples to ref_xarray's own bounds first
    (slope's DEM-derived grid needs this extra step; VV/VH scenes don't).

    Parameters
    ----------
    data_xarray : xarray.DataArray
        Array to reproject and resample; any input CRS.
    grid_shapefile_path : str
        Path to the tile grid shapefile/geopackage used to resolve aoi_id's UTM zone and bounds.
    aoi_id : str
        AOI/tile ID to clip to.
    ref_xarray : xarray.DataArray
        Reference array providing bounds when slope=True.
    buffer : float, optional
        Buffer distance in meters (AOI's own UTM), for slope's kernel padding.
    slope : bool, optional
        If True, resample to ref_xarray's own bounds before the final regrid.
    cell_size : float, optional
        Output pixel size in meters. Default 30 (OPERA RTC-S1 native resolution).

    Returns
    -------
    xarray.DataArray
        Reprojected array resampled onto the tile's fixed cell_size grid.
    """
    # extract target extent from the grid polygon
    gdf = gpd.read_file(grid_shapefile_path)
    gdf = gdf.loc[gdf['ID'].isin([aoi_id])]
    tile_utm_zone = zone_to_epsg(gdf['zone'].values[0])
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















