# ifmiap/utils/__init__.py

# import required libraries
from datetime import datetime
import pandas as pd
import geopandas as gpd
from ..authenticate import sign_in
import rasterio
import numpy as np

# DEFINE constants
INDIA_GRID_SHAPEFILE_PATH = r'resources/india_fishnet_4326.shp'
CATALOG = sign_in()


# this function creates python datetime objects using a given date and number of days to advance
def date_range(start, days):
    temp_date = pd.date_range(datetime.strptime(start, '%d/%m/%Y'), periods=days+1, freq='D')

    return (temp_date.min().date(), temp_date.max().date())


# define a function to get bounding box as json from a shapefile using GeoPandas
def gpd_to_json(id_list, infile=INDIA_GRID_SHAPEFILE_PATH):
    # read the shapefile
    gdf = gpd.read_file(infile)
    
    # filter using the given ID list
    gdf = gdf.loc[gdf['ID'].isin(id_list)]
    
    # extract bounding box of each of the filtered polygons
    gdf_bbox = [
        row.geometry.bounds
        for idx, row in gdf.iterrows()
               ]
    
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
    
    return [{
        "type": "Polygon",
        "coordinates": [
            get_bbox(bounds_item)
        ],
    }
    for bounds_item in gdf_bbox
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
        collections = ["sentinel-1-grd"],
        intersects = bbox,
        datetime = date_range
    )
    
    # select scenes that have VV and VH bands in them
    for item in results.get_items():
        if ('vh' in item.assets) and ('vv' in item.assets):
            all_results.append(item)

    return all_results

# define a function to export xarray.DataArray object to TIFF file
def export_xarray(xarray_data, filename):
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
            'count': xarray_data.shape[0],
            'transform': rasterio.transform.from_bounds(xmin, ymin, xmax, ymax, cols, rows)
        }

        with rasterio.open(
                filename,
                'w', **profile) as dst:
            for band in range(xarray_data.shape[0]):
                dst.write(xarray_data.data[band, :, :], band+1)

###################################################################


from shapely.geometry import box
import geopandas as gpd
shapefile_path = "resources/Grid_shapefile/shp_4326.shp"


def load_grid_shapefile(shapefile_path):
    """
    Load a grid shapefile as a GeoDataFrame.

    Parameters:
    -----------
    shapefile_path    : string
                        The file path to the grid shapefile.

    Returns:
    --------
    GeoDataFrame      : GeoPandas.GeoDataFrame
                        A GeoDataFrame representing the grid polygons loaded from the shapefile.
    """
    # Load the grid shapefile using GeoPandas' read_file function
    grid_gdf = gpd.read_file(shapefile_path)
    # Return the loaded GeoDataFrame
    return grid_gdf



def filter_items_dryPeriod(all_results):
    """
        Filter items from the provided list based on their intersection with the grid shapefile.

        Parameters:
            all_results (list): A list of pystac.Item objects containing the Sentinel-1 data.

        Returns:
            list: A filtered list of pystac.Item objects that intersect with the grid shapefile.

        Raises:
            None.
    """
    gdf_shapefile = load_grid_shapefile(shapefile_path)
    filtered_items = []
    for item in all_results:
        coordinates = item.geometry['coordinates']
        if len(coordinates) > 0:
            coords = coordinates[0]
            minx, miny = coords[0][0], coords[0][1]
            maxx, maxy = coords[2][0], coords[2][1]
            bbox = box(minx, miny, maxx, maxy)
            if gdf_shapefile.intersects(bbox).any():
                filtered_items.append(item)
    return filtered_items


def filter_items_floodPeriod(all_results):
    """
        Filter items from the provided list based on their intersection with the grid shapefile.

        Parameters:
            all_results (list): A list of pystac.Item objects containing the Sentinel-1 data.

        Returns:
            tuple: A tuple containing:
                - A filtered list of pystac.Item objects that intersect with the grid shapefile.
                - A list of intersecting grid IDs.
                - A list of intersecting grid geometries.

        Raises:
            None.
    """
    # Specify the path to the Grid_shapefile
    grid_gdf = load_grid_shapefile(shapefile_path)
    intersecting_ids = []
    intersecting_geometries = []
    filtered_items = []
    for item in all_results:
        coordinates = item.geometry['coordinates']
        if len(coordinates) > 0:
            coords = coordinates[0]
            minx, miny = coords[0][0], coords[0][1]
            maxx, maxy = coords[2][0], coords[2][1]
            bbox = box(minx, miny, maxx, maxy)
            if grid_gdf.intersects(bbox).any():
                intersecting_ids.extend(grid_gdf.loc[grid_gdf.intersects(bbox), 'id'])
                intersecting_geometries.extend(grid_gdf.loc[grid_gdf.intersects(bbox), 'geometry'])
                filtered_items.append(item)

    return filtered_items, intersecting_ids, intersecting_geometries


def grid_bounds():
    """
        Extracts the bounding box coordinates from the grid shapefile and creates a bounding box of interest.

        Parameters:
            None.

        Returns:
            dict: A dictionary representing the bounding box of interest in the desired format.

        Raises:
            None.
    """
    grid = load_grid_shapefile(shapefile_path)
    # Extract the bounding box coordinates
    minx, miny, maxx, maxy = grid.total_bounds

    # Create the bounding box coordinates in the desired format
    bbox_coordinates = [
        [minx, miny],
        [maxx, miny],
        [maxx, maxy],
        [minx, maxy],
        [minx, miny]
    ]
    bbox_of_interest = {

        "coordinates": [
            [
                bbox_coordinates
            ]
        ],
    }
    return bbox_of_interest
