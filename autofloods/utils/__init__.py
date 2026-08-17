# autofloods/utils/__init__.py

# import required libraries
import datetime, glob, os
import pandas as pd
import geopandas as gpd
from ..authenticate import sign_in
import rasterio
from shapely.geometry import Polygon
import rioxarray
from rioxarray import merge as rioxarray_merge
import numpy as np
import xarray as xr

# DEFINE constants
#INDIA_GRID_SHAPEFILE_PATH = r'resources/india_utm_fishnet.gpkg'
CATALOG = sign_in()


# this function creates python datetime objects using a given date and number of days to advance
def date_range(start, days):
    """
    Generate a Date Range

      This function takes a start date and a number of days, and generates a date range
      that includes the start date and the specified number of days.

    Parameters:
          start (str): The start date in the format 'dd/mm/yyyy'.
          days (int): The number of days to generate in the date range.

    Returns:
          tuple: A tuple containing the minimum and maximum dates in the generated range.

    Example:
          >>> start_date = '01/08/2023'
          >>> num_days = 10
          >>> date_min, date_max = date_range(start_date, num_days)

    """
    temp_date = pd.date_range(datetime.datetime.strptime(start, '%d/%m/%Y'), periods=days+1, freq='D')

    return (temp_date.min().date(), temp_date.max().date())


# define a function to format string correctly and create date range
def string_to_date_range(start, end):
    """
    Convert Start and End Strings to Date Range

    Converts start and end date strings to a date range.

    :param start: Start date (format 'dd/mm/yyyy').
    :param end: End date (format 'dd/mm/yyyy').
    :return: Tuple (start date, end date).

    """
    start_year, start_month = start.split('/')
    end_year, end_month = end.split('/')

    delta_days_end = (datetime.date(int(end_year), int(end_month) % 12 + 1, 1) - datetime.timedelta(days=1)).day

    return (
        datetime.datetime.strptime(f'01/{start_month}/{start_year}', '%d/%m/%Y').date(),
        datetime.datetime.strptime(f'{delta_days_end:02d}/{end_month}/{end_year}', '%d/%m/%Y').date()
    )

# define a function to get bounding box as json from a shapefile using GeoPandas
def gpd_to_json(id_list, infile, separate=True, id_key='ID', zone_key='zone', buffer=None):
    """
    Converts selected polygons from a GeoDataFrame to GeoJSON format.

    This function reads a shapefile using GeoPandas, filters the polygons based on the provided ID list,
    and then generates GeoJSON representations of the filtered polygons' bounding boxes.
    :param id_list: A list of IDs to filter the polygons in the GeoDataFrame.
    :param infile: The path to the input shapefile. Default is INDIA_GRID_SHAPEFILE_PATH.
    :param separate: If True, each polygon's bounding box will be generated separately in GeoJSON.
                                  If False, a single bounding box covering all filtered polygons will be generated.
                                  Default is True.
    :param id_key: The key representing the ID field in the GeoDataFrame. Default is 'ID'.
    :return:  list: A list of dictionaries representing GeoJSON polygons.

    """
    # create GeoJSON using the bounds
    def get_bbox(bbox):
        lon_min, lat_min, lon_max, lat_max = bbox

        return [
            [lon_min, lat_min],
            [lon_max, lat_min],
            [lon_max, lat_max],
            [lon_min, lat_max],
            [lon_min, lat_min]
        ]


    # read the shapefile
    gdf = gpd.read_file(infile)
    
    # filter using the given ID list
    gdf = gdf.loc[gdf[id_key].isin(id_list)]

    # if buffer parameter is passed, reproject to local, then buffer and preproject back to wgs84
    # this is the case only for downloading DEM because we need slope for a slightly bigger region to run kernel
    if buffer:
        tile_utm_zone = 'EPSG:326{}'.format(gdf['zone'].values[0][:-1])
        gdf = gdf.to_crs(tile_utm_zone)
        gdf['geometry'] = gdf.buffer(buffer).to_crs('EPSG:4326')
    
    # extract bounding box of each of the filtered polygons
    if separate == True:
        gdf_bbox = [
            (row[id_key], row[zone_key], row.geometry.bounds)
            for idx, row in gdf.iterrows()
                   ]

        return [{
            "type": "Polygon",
            "coordinates": [
                get_bbox(bounds_item)
            ],
            "properties": {
                id_key: id,
                zone_key: zone
            }
        }
            for id, zone, bounds_item in gdf_bbox
        ]

    else:
        gdf_bbox = [(1, gdf.total_bounds)]

    
        return [{
            "type": "Polygon",
            "coordinates": [
                get_bbox(bounds_item)
            ],
            "properties": {
                id_key: id
            }
        }
        for id, bounds_item in gdf_bbox
        ]

# define a function to search for Sentinel-1 data
def search_sentinel_data(bbox, start_date=None, end_date=None):
    """
    Search for Sentinel-1 data within a specified time range and bounding box of interest.

    Parameters:
    -----------
    bbox              : Dictionary
                        The bounding box of interest in GeoJSON format, specified as a dictionary.
                        
    start_date        : Datetime object
                        The start date of the time range to search for data.
                        
    end_date          : Datetime object
                        The end date of the time range to search for data.

    Returns:
    --------
    results           : list
                        A list of pystac.Item objects containing the searched Sentinel-1 data.

    """
    all_results = []

    # Define the date range using start and end datetime objects
    date_range = f'{start_date.strftime("%Y-%m-%dT00:00:00Z")}/{end_date.strftime("%Y-%m-%dT23:59:59Z")}'

    # search for Sentinel-1 scenes
    results = CATALOG.search(
        collections = ["sentinel-1-rtc"],
        intersects = bbox,
        datetime = date_range
    )
    
    # select scenes that have VV and VH bands in them
    for item in results.get_items():
        if ('vh' in item.assets) and ('vv' in item.assets):
            all_results.append(item)

    return all_results

## define a function to get footprint of a given S1 item
def s1item_footprint(item):
    """
    Create a GeoDataFrame containing the footprint polygon of a Sentinel-1 item.

    Parameters:
    -----------
    item: The Sentinel-1 item containing geometry information

    Returns
    _______
    A GeoDataFrame containing the footprint polygon with the 'ID' and 'geometry' columns.
    """
    polygon = Polygon(item.geometry['coordinates'][0])

    return gpd.GeoDataFrame({'ID':item.id, 'geometry': [polygon]}, crs='EPSG:4326')

# define a function that calls the above function for a given ID list but separates the search items for each given ID
def seggregate_sentinel_search(aoi_list, search_items):
    """
    Segregate Sentinel search results based on intersecting Area of Interest (AOI) polygons.

    This function takes a list of AOI polygons and a list of search items (such as Sentinel-1 scenes).
    It identifies which search items intersect with each AOI and vice versa, creating dictionaries to store
    this information.

    Parameters
    -----------
    aoi_list        : A list of AOI polygons in GeoJSON-like format.
    search_items    : A tuple containing two lists of search items.
                              The first list is not used in this function.
                              The second list contains the search items (e.g., Sentinel-1 scenes) to process.

    Returns
    -------
    tuple           : A tuple containing two dictionaries.
                      The first dictionary maps AOI IDs to lists of intersecting search item IDs.
                      The second dictionary maps search item IDs to lists of intersecting AOI IDs.
    """
    # create GDF from the AOI list
    aoi_footprints = gpd.GeoDataFrame([
        {'ID': aoi['properties']['ID'], 'geometry': Polygon(aoi['coordinates'][0])}
        for aoi in aoi_list],
        crs='EPSG:4326'
    )

    # create GDF from the stac items list
    s1_footprints = pd.concat([s1item_footprint(item) for item in search_items[1]], axis=0)

    # create two dictionary to store two way information
    aoi_scene_dict = {}
    scene_aoi_dict = {}

    # loop through AOI polygons to find intersecting scenes IDs
    for idx, row in aoi_footprints.iterrows():
        intersecting_s1scene_ids = s1_footprints[row.geometry.intersects(s1_footprints.geometry)]['ID'].values

        # Converting key to int because JSON export causes problems
        # with numpy integer data types while exporting json.
        aoi_scene_dict[int(row['ID'])] = [item.id for item in search_items[1] if item.id in intersecting_s1scene_ids]

    # loop through scenes IDs to find intersecting AOI polygons
    for idx, row in s1_footprints.iterrows():
        intersecting_aoi_ids = aoi_footprints[row.geometry.intersects(aoi_footprints.geometry)]['ID'].values

        # Converting items to int inside list comprehension because JSON export causes problems
        # with numpy integer data types while exporting json.
        scene_aoi_dict[row['ID']] = [int(item) for item in list(intersecting_aoi_ids)]

    return aoi_scene_dict, scene_aoi_dict

def query_nasadem(aoi_union_bbox):
    # search for Sentinel-1 scenes
    results = CATALOG.search(
        collections=["nasadem"],
        intersects=aoi_union_bbox
    )

    # select scenes that have VV and VH bands in them
    return [
        item
        for item in results.get_items()
        if 'elevation' in item.assets
           ]

def download_nasadem(bbox, overview_level=1, nodata=0.0):
    dem_item_list = query_nasadem(bbox)
    dem_xarray_list = [
        rioxarray.open_rasterio(item.assets['elevation'].href, overview_level=overview_level, masked=True)
        for item in dem_item_list
    ]

    mosaic_xarray = rioxarray_merge.merge_arrays(dem_xarray_list, nodata=nodata)

    return mosaic_xarray

# define a function to export xarray.DataArray object to TIFF file
def export_xarray(xarray_data, filename, bandnames=None):
    """
    Export Xarray data as a GeoTIFF file.

    This function takes an Xarray dataset or data array and exports it as a GeoTIFF file. It calculates
    the bounding box and other necessary metadata from the Xarray data and writes it to the specified file.

    Parameters
    ----------
    xarray_data                 : The Xarray dataset or data array to be exported.
    filename                    : The path and filename of the output GeoTIFF file.

    Raises
    ______
    InputDataDimensionError     : Raised when the input data has an unexpected number of dimensions.

    """
    with rasterio.Env():
        xmin, ymin, xmax, ymax = [
            xarray_data.x.min().values,
            xarray_data.y.min().values,
            xarray_data.x.max().values,
            xarray_data.y.max().values
        ]

        if len(xarray_data.dims) == 3:
            bands, rows, cols = xarray_data.shape
        elif len(xarray_data.dims) == 2:
            rows, cols = xarray_data.shape
        else:
            print('Number of dimensions in input data exceed three. Please check.')

        # Define the metadata for the output file
        profile = {
            'driver': 'GTiff',
            'dtype': rasterio.float32,
            'nodata': np.nan,
            'width': cols,
            'height': rows,
            'count': xarray_data.shape[0] if len(xarray_data.dims) == 3 else 1,
            'transform': rasterio.transform.from_bounds(xmin, ymin, xmax, ymax, cols, rows)
        }

        with rasterio.open(
                filename,
                'w', **profile) as dst:
            # if input array is 2D
            if len(xarray_data.dims) == 2:
                dst.write(xarray_data.data, 1)
            # if input data is 3D
            elif len(xarray_data.dims) == 3:
                for band in range(xarray_data.shape[0]):
                    dst.write(xarray_data.data[band, :, :], band+1)
            else:
                raise('InputDataDimensionError: Unexpected number of dimensions found in the input data.')

            if bandnames:
                for n, item in enumerate(bandnames):
                    dst.set_band_description(n+1, item)


# define a function to convert numpy array to xarray object using a reference xarray object
def numpy_to_xarray(numpy_data, ref_xarray):
    """
    Create an xarray DataArray from a NumPy array using coordinates and geospatial information from a reference xarray object.
    If the input NumPy array is 3D, bands are added to the xarray object with names 'Band1', 'Band2', and so on.

    Parameters
    ----------
    numpy_data : numpy.ndarray
        The NumPy array to be converted to an xarray DataArray.
    ref_xarray : xarray.DataArray
        The reference xarray object containing geospatial information.

    Returns
    -------
    xarray.DataArray
        An xarray DataArray created from the NumPy array with coordinates and geospatial information aligned with the reference xarray.

    Raises
    ------
    ValueError
        If the spatial dimensions of the input NumPy array do not match the dimensions of the reference xarray object.

    Examples
    --------
    >>> import numpy as np
    >>> import xarray as xr
    >>> import rioxarray
    >>> # Assuming numpy_data is your NumPy array and ref_xarray is your reference xarray object
    >>> xarray_data = numpy_to_xarray(numpy_data, ref_xarray)
    """

    # Validate spatial dimensions
    if len(numpy_data.shape) not in [2, 3]:
        raise ValueError("Input NumPy array must be 2D or 3D.")

    # Get the geospatial information from the reference xarray object
    transform = ref_xarray.rio.transform()
    crs = ref_xarray.rio.crs

    # Create band names for 3D NumPy array
    band_names = [f'Band{i+1}' for i in range(numpy_data.shape[0])] if len(numpy_data.shape) == 3 else None

    # Create an xarray DataArray with geospatial information and bands using rioxarray
    if band_names:
        xarray_data = xr.DataArray(data=numpy_data, dims=['band', *ref_xarray.dims], coords={'band': band_names, **ref_xarray.coords}, attrs=ref_xarray.attrs)
    else:
        xarray_data = xr.DataArray(data=numpy_data, dims=ref_xarray.dims, coords=ref_xarray.coords, attrs=ref_xarray.attrs)

    xarray_data.rio.write_crs(crs)
    xarray_data.rio.write_transform(transform)

    return xarray_data


# define a function to merge the exported flood extent file using date
def merge_flood_gdfs(flood_dir, date_index=5, delimiter='_'):
    # get the list of files present in the directory
    files_list = glob.glob(f'{flood_dir}/*.gpkg')

    # extract unique dates from the flood vector file names
    unique_dates = [os.path.split(file)[-1].split(delimiter)[date_index-1][:8] for file in files_list]

    # merge
    gdf_union = pd.concat([
        gpd.read_file(file).assign(id=[file] * gpd.read_file(file).shape[0])
        for file in files_list
    ], axis=0)

    # dissolve flood extent shapefile
    gdf_union['date'] = gdf_union.id.apply(lambda x: x.split('_')[5][:8])
    gdf_union['tile_id'] = gdf_union['id'].apply(lambda x: x.split('_')[4])
    gdf_union['uid'] = gdf_union['tile_id'] + '_' + gdf_union['date']
    dry_years = gdf_union.id.apply(lambda x: '_'.join(x.split('_')[2:4])).values[0]

    # dissolve
    gdf_union = gdf_union.dissolve('uid')[['geometry', 'date', 'tile_id']]

    # export tile wise flood extent
    for tile_id in gdf_union['tile_id'].unique():
        temp_gdf = gdf_union.loc[gdf_union['tile_id'] == tile_id]

        outfile = os.path.join(
            flood_dir.replace('flood_vector', 'final_output'),
            f'{dry_years}_{tile_id}_FloodExtent.gpkg'
        )

        temp_gdf.to_file(outfile)



def decibel_to_linear(decibels):
    return 10 ** (decibels / 10)


def linear_to_decibel(linear):
    return 10 * np.log10(linear)


def combine_flood_dates(flood_data, date_index=-5):
    data_dict = None

    # the input flood data can either be a dictionary of xarray objects
    if type(flood_data) == type(dict()):
        unique_dates = sorted(set(
            [
                key.split('_')[date_index][:8]
                for key in flood_data
                ]
        ))

        data_dict = {
            u_date: np.maximum.reduce([
                flood_data[key].to_numpy()
                for key in flood_data
                if key.split('_')[date_index][:8] == u_date
            ])
            for u_date in unique_dates
        }

        return data_dict

    # or it can be a list of paths to geotiff files
    elif type(flood_data) == type(list()):
        unique_dates = sorted(set(
            [
                file.split('_')[date_index][:8]
                for file in flood_data
                ]
        ))

        data_dict = {
            key: np.maximum.reduce([
                rasterio.open(filepath).read(1)
                for filepath in flood_data
                if filepath.split('_')[date_index][:8] == key
            ])
            for key in unique_dates
        }

        return data_dict

    else:
        raise TypeError("Only dictionary containing xarray objects or list containing .tif file paths are supported.")


def flood_data_3dstack(flood_data, date_index=-5):
    """
    This function works under the assumption that input flood data belongs to the same tile.

    Args:
        flood_data: dict or list

        date_index: integer

    Returns:

    """
    # call the function that neatly combines flood maps for different dates
    data_dict = combine_flood_dates(flood_data, date_index=date_index)


    # for now let's say every non-zero cell is flood
    # stack all of them in a 3D array with cell value as date
    return list(data_dict.keys()), np.stack([
        data_dict[key]
        for key in data_dict
        ])
