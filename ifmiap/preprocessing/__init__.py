# ifmiap/preprocessing/__init__.py

# import required libraries
import rioxarray
import geopandas as gpd
import xarray as xr
import numpy as np
import xrspatial
from sklearn.feature_extraction import image
from copy import deepcopy
from ..utils import decibel_to_linear
import math


# define a function to reproject VV and VH tif files from the cloud and store all images in memory
def read_sentinel1_stac(stac_item, overview_level=3):
    """
    Read and Reproject STAC Item's VV and VH Bands

    This function reads Very High Resolution (VV) and Very High Resolution (VH) bands from a
    SpatioTemporal Asset Catalog (STAC) Item and reprojects them to EPSG:4326 coordinate system.

    Parameters
    __________
    stac_item (dict)                : A STAC Item containing metadata and asset information.
    overview_level (int, optional)  : The level of overviews to use for reading the data. Default is 3.

    Returns
    _______
    tuple: A tuple containing the STAC Item ID and a dictionary of reprojected DataArrays.

    """
    # get the URL to the VV and VH bands
    vv_href = stac_item.assets["vv"].href
    vh_href = stac_item.assets["vh"].href

    # read the VV and VH bands
    vv_ds = rioxarray.open_rasterio(vv_href, overview_level=overview_level, masked=True)
    vh_ds = rioxarray.open_rasterio(vh_href, overview_level=overview_level, masked=True)

    # convert decibel ot linear
    vv_ds = decibel_to_linear(vv_ds)
    vh_ds = decibel_to_linear(vh_ds)

    # reproject data
    vv_ds = vv_ds.rio.reproject('EPSG:4326')
    vh_ds = vh_ds.rio.reproject('EPSG:4326')

    return stac_item.id, {
        'vv_ds': vv_ds,
        'vh_ds': vh_ds
    }

# define a function to clip the reprojected data to the given polygon extent
def reproject_clip_stac(reprojected_dict, aoi_scene_dict, grid_shapefile_path, id):
    """
    Clip Reprojected DataArrays Using a Shapefile

    This function takes a dictionary of reprojected DataArrays, a dictionary mapping scene IDs
    to area of interest (AOI) IDs, and an AOI ID. It uses a shapefile to perform clipping on
    the reprojected DataArrays corresponding to the given AOI ID.

    Parameters
    __________
    reprojected_dict (dict)     : A dictionary containing reprojected DataArrays with scene IDs as keys.
    aoi_scene_dict (dict)       : A dictionary mapping AOI IDs to lists of scene IDs.
    id (str)                    : The AOI ID for which clipping should be performed.

    Returns
    _______
    dict: A dictionary containing clipped DataArrays for the specified AOI ID and scene IDs.

    """
    # read the shapefile and filter it to use for clipping
    gdf = gpd.read_file(grid_shapefile_path)
    gdf = gdf.loc[gdf['ID'].isin([id])]

    # extract the UTM zone from the tile, reproject the GDF to UTM
    tile_utm_zone = 'EPSG:326{}'.format(gdf['zone'].values[0][:-1])
    gdf = gdf.to_crs(tile_utm_zone)

    # clip VV and VH bands for the given grid (reproject GDF to utm before clipping)
    return {
        stac_id: {
            'vv_ds': reprojected_dict[stac_id]['vv_ds'].rio.reproject(tile_utm_zone).rio.clip(gdf.geometry),
            'vh_ds': reprojected_dict[stac_id]['vh_ds'].rio.reproject(tile_utm_zone).rio.clip(gdf.geometry)
        }
        for stac_id in aoi_scene_dict[id]
    }


# define a function to stack all the images for a given tile
def stack_images(clipped_dict, grid_shapefile_path, id):
    """
    Stack Clipped DataArrays into Multi-Band Stacks

    This function takes a dictionary of clipped DataArrays, each containing 'vv_ds' and 'vh_ds'
    bands for different scenes, and stacks them into multi-band stacks. The function ensures that
    the input DataArrays are resampled to a common extent and resolution before stacking.

    Parameters
    __________
    clipped_dict (dict)     : A dictionary containing clipped DataArrays for different scenes.
    id (str)                : The AOI ID for which stacking should be performed.

    Returns
    _______
    dict                    : A dictionary containing multi-band stacked DataArrays for both VV and VH bands.

    """
    # create a list of dictionaries containing 'vv_ds' and 'vh_ds'
    stacked_images = [
        clipped_dict[stac_id]
        for stac_id in clipped_dict
    ]

    # loop through each of the grid shapefile and process their respective images separately
    # Resample each DataArray to the common extent and resolution
    stacked_images = [{
        'vv_ds': clip_xarray_using_id(
            data_xarray=item['vv_ds'],
            grid_shapefile_path=grid_shapefile_path,
            aoi_id=id,
            ref_xarray=stacked_images[0]['vv_ds']
        ),
        'vh_ds': clip_xarray_using_id(
            data_xarray=item['vh_ds'],
            grid_shapefile_path=grid_shapefile_path,
            aoi_id=id,
            ref_xarray=stacked_images[0]['vv_ds']
        )
    }
        for item in stacked_images
    ]

    # stack the data properly
    vv_stack = xr.concat([item['vv_ds'] for item in stacked_images], dim="band")
    vh_stack = xr.concat([item['vh_ds'] for item in stacked_images], dim="band")

    return {
        'vv_stack': vv_stack,
        'vh_stack': vh_stack
    }

def clip_xarray_using_id(data_xarray, grid_shapefile_path, aoi_id, ref_xarray, buffer=None, slope=False):
    cell_size = float(ref_xarray.spatial_ref.GeoTransform.split(' ')[1])

    # extract target extent from the grid polygon
    gdf = gpd.read_file(grid_shapefile_path)
    gdf = gdf.loc[gdf['ID'].isin([aoi_id])]
    tile_utm_zone = 'EPSG:326{}'.format(gdf['zone'].values[0][:-1])
    gdf = gdf.to_crs(tile_utm_zone)

    # perform buffer if required (for slope smoothing using kernel)
    if buffer:
        gdf['geometry'] = gdf.buffer(buffer)

    # clipping slope requires bounding box from ref xarray before using gdf extent
    if slope:
        x_min, y_min, x_max, y_max = ref_xarray.rio.bounds()
        data_xarray = data_xarray.rio.reproject(tile_utm_zone).interp(
            x=np.arange(x_min, x_max, cell_size),
            y=np.arange(y_max, y_min, -cell_size)
            )
    x_min, y_min, x_max, y_max = gdf.total_bounds

    # Resample dem DataArray to the common extent and resolution
    return data_xarray.rio.reproject(tile_utm_zone).interp(
        x=np.arange(x_min, x_max, cell_size),
        y=np.arange(y_max, y_min, -cell_size)
    )

# define a function to calculate relative slope
def smoothen_slope(dem_xarray, grid_shapefile_path, aoi_id, ref_xarray, buffer=None, nodata=0):
    # clip the dem to the buffered GDF
    dem_xarray_clipped = clip_xarray_using_id(
        data_xarray=dem_xarray,
        grid_shapefile_path=grid_shapefile_path,
        aoi_id=aoi_id,
        ref_xarray=ref_xarray,
        buffer=buffer
    )

    # calculate the slope
    slope_xarray = xrspatial.slope(dem_xarray_clipped.squeeze())

    # run a kernel to calculate smoothen slope
    cell_size = math.ceil(float(slope_xarray.spatial_ref.GeoTransform.split(' ')[1]))
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















