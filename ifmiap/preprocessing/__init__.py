# ifmiap/preprocessing/__init__.py

# import required libraries
import rioxarray
import geopandas as gpd
from ..utils import INDIA_GRID_SHAPEFILE_PATH
import xarray as xr

# define a function to read and reproject VV and VH tif files from the cloud
def read_reproj_clip(stac_item, id_list):
    # get the URL to the VV and VH bands
    vv_href = stac_item.assets["vv"].href
    vh_href = stac_item.assets["vh"].href

    # read the VV and VH bands
    vv_ds = rioxarray.open_rasterio(vv_href, overview_level=3, masked=True)
    vh_ds = rioxarray.open_rasterio(vh_href, overview_level=3, masked=True)

    # reproject the VV and VH bands
    vv_ds = vv_ds.rio.reproject("EPSG:4326")
    vh_ds = vh_ds.rio.reproject("EPSG:4326")

    # read the shapefile and filter it to use for clipping
    gdf = gpd.read_file(INDIA_GRID_SHAPEFILE_PATH)
    gdf = gdf.loc[gdf['ID'].isin(id_list)]

    # clip VV and VH bands for the given grid
    vv_ds = vv_ds.rio.clip(gdf.geometry)
    vh_ds = vh_ds.rio.clip(gdf.geometry)

    return {
        'vv_ds': vv_ds,
        'vh_ds': vh_ds
    }

# define a function to stack all the images for a given tile
def stack_images(stac_list, id_list):
    # call the previous function
    stacked_images = [
        read_reproj_clip(stac_item, id_list)
        for stac_item in stac_list
    ]

    # stack the data properly
    vv_stack = xr.concat([item['vv_ds'] for item in stacked_images], dim="band")
    vh_stack = xr.concat([item['vh_ds'] for item in stacked_images], dim="band")

    return {
        'vv_stack': vv_stack,
        'vh_stack': vh_stack
    }

###############################################################

import numpy as np
from rioxarray.exceptions import NoDataInBounds
from rasterio.errors import RasterioIOError
from rasterio.enums import Resampling
import rioxarray
import multiprocessing
import datetime
from functools import partial
from ifmiap.utils import load_grid_shapefile, shapefile_path, grid_bounds, search_sentinel_data,filter_items_dryPeriod



def get_grid_data_dry_period(catalog, n):
    """
        Retrieve grid data for dry periods from the Sentinel satellite imagery catalog.

        Parameters:
            catalog (Sentinel API catalog object): The catalog object used to search for data in the Sentinel API.
            n (int): The number of years (integer) for which the function will fetch data.

        Returns:
            list: A list containing filtered grid data for the specified dry periods.
    """
    # Initialize an empty list to store dry period date ranges
    dry_period_ranges = []
    # Get the grid bounds for India using the grid_bounds() function
    grid_data_india = grid_bounds()
    # Iterate over the years from 2016 to 2017 + n
    for year in range(2016, 2016 + n):  # Iterate from 2016 to 2017
        # Define start and end dates for the dry period in each year (March 1 to June 15)
        start_date_dry = datetime.datetime(year, 3, 1)
        end_date_dry = datetime.datetime(year, 6, 15)
        # Append the date range in ISO format (start_date_dry/end_date_dry) to the dry_period_ranges list
        dry_period_ranges.append(f"{start_date_dry.isoformat()}/{end_date_dry.isoformat()}")
    # Call the search_sentinel_data function with the provided catalog, grid bounds, and dry period ranges
    grid_data_results = search_sentinel_data(catalog, grid_data_india, date_ranges=dry_period_ranges)
    # Call the filter_items_dryPeriod function to filter the results based on certain criteria
    filtered_grid_data_results = filter_items_dryPeriod(grid_data_results)
    # Return the filtered grid data for dry periods
    return filtered_grid_data_results

def reproject_and_resample(vv_clipped, vh_clipped, target_resolution):
    """
       Reproject and resample the clipped raster data to the target resolution.

       Parameters:
           vv_clipped (rasterio.DatasetReader): Clipped VV band raster data.
           vh_clipped (rasterio.DatasetReader): Clipped VH band raster data.
           target_resolution (float): The target resolution (in meters) for resampling.

       Returns:
           tuple: A tuple containing the resampled VV and VH raster data as rasterio.DatasetReader objects.
    """
    # Resample the clipped rasters to the target resolution using nearest-neighbor interpolation
    vv_resampled = vv_clipped.rio.reproject(vv_clipped.rio.estimate_utm_crs(), target_resolution,
                                            resampling=Resampling.nearest)
    vh_resampled = vh_clipped.rio.reproject(vh_clipped.rio.estimate_utm_crs(), target_resolution,
                                            resampling=Resampling.nearest)
    # Return the resampled VV and VH raster data
    return vv_resampled, vh_resampled


def process_raster_data(filtered_results, intersecting_ids, intersecting_geometries):
    """
        Process raster data by clipping, reprojecting, and stacking the VV and VH bands for each grid polygon.

        Parameters:
            filtered_results (list): A list of filtered items from the Sentinel API search results.
            intersecting_ids (list): A list of grid IDs that intersect with the geometries of filtered items.
            intersecting_geometries (list): A list of geometries corresponding to intersecting grid polygons.

        Returns:
            tuple: A tuple containing dictionaries with stacked VV and VH raster data for each grid polygon.
    """
    # Initialize dictionaries to store stacked VV and VH raster data for each grid polygon
    stacked_images_during_vv = {}
    stacked_images_during_vh = {}
    # Keep track of processed filenames to avoid duplication
    processed_filenames = set()
    # Iterate through the filtered items from the Sentinel API search results
    for item in filtered_results:
        vv = (
            rioxarray.open_rasterio(item.assets["vv"].href, overview_level=3, masked=True)

        )
        vh = (
            rioxarray.open_rasterio(item.assets["vh"].href, overview_level=3, masked=True)

        )
        # Reproject VV and VH bands to EPSG:4326 coordinate reference system (CRS)
        vv = vv.rio.reproject("EPSG:4326")
        vh = vh.rio.reproject("EPSG:4326")
        # Iterate through intersecting grid IDs and geometries
        for intersecting_id, intersecting_geometry in zip(intersecting_ids, intersecting_geometries):
            # Get the grid ID and form filenames for VV and VH images
            grid_id = intersecting_id
            vv_filename = f"{item.id}_vv_{grid_id}.tif"
            vh_filename = f"{item.id}_vh_{grid_id}.tif"
            # Clip the VV and VH bands to the current grid polygon
            try:
                # Clip the raster to the grid polygon
                vv_clipped = vv.rio.clip([intersecting_geometry])

                vh_clipped = vh.rio.clip([intersecting_geometry])
                # Set the desired resolution for resampling (10 meters x 10 meters)
                target_resolution = (10, 10)
                # Uncomment the following lines if reprojecting and resampling is required
                # vv_resampled, vh_resampled = reproject_and_resample(vv_clipped, vh_clipped, target_resolution)
                # Check if the VV image filename has been processed before
                if vv_filename not in processed_filenames:
                    # Stack the VV clipped images for the current grid polygon
                    if grid_id not in stacked_images_during_vv:
                        stacked_images_during_vv[grid_id] = []
                    vv_clipped.attrs['Image_ID'] = item.id
                    vv_clipped.attrs['Grid_ID'] = grid_id
                    vv_clipped.attrs['Geometry'] = intersecting_geometry
                    stacked_images_during_vv[grid_id].append(vv_clipped)
                    processed_filenames.add(vv_filename)
                # Check if the VH image filename has been processed before
                if vh_filename not in processed_filenames:
                    # Stack the VH clipped images for the current grid polygon
                    if grid_id not in stacked_images_during_vh:
                        stacked_images_during_vh[grid_id] = []
                    vh_clipped.attrs['Image_ID'] = item.id
                    vh_clipped.attrs['Grid_ID'] = grid_id
                    vh_clipped.attrs['Geometry'] = intersecting_geometry
                    stacked_images_during_vh[grid_id].append(vh_clipped)
                    processed_filenames.add(vh_filename)


            except NoDataInBounds:
                # Skip to the next grid polygon if there is no valid data
                continue
            except Exception as e:
                # Handle the exception and continue to the next shape
                print("Invalid or empty shape will not be rasterized:", str(e))
                continue
    # Return the dictionaries with stacked VV and VH raster data for each grid polygon
    return stacked_images_during_vv, stacked_images_during_vh


def process_shapefile_item(item, grid_gdf):
    """
        Process raster data by clipping, reprojecting, and stacking the VV and VH bands for each grid polygon.

        Parameters:
            item (object): An item from the Sentinel API search results with VV and VH band raster data.
            grid_gdf (geopandas.GeoDataFrame): A GeoDataFrame containing grid polygons with associated IDs.

        Returns:
            tuple: A tuple containing dictionaries with stacked VV and VH raster data for each grid polygon.
    """
    # Initialize dictionaries to store stacked VV and VH raster data for each grid polygon
    stacked_images_before_vv = {}
    stacked_images_before_vh = {}
    # Open VV and VH bands as rioxarray.DatasetReader objects with overview_level=3 and masked=True
    vv = rioxarray.open_rasterio(item.assets["vv"].href, overview_level=3, masked=True)
    vh = rioxarray.open_rasterio(item.assets["vh"].href, overview_level=3, masked=True)
    # Reproject VV and VH bands to EPSG:4326 coordinate reference system (CRS)
    vv = vv.rio.reproject("EPSG:4326")
    vh = vh.rio.reproject("EPSG:4326")
    # Iterate through grid polygons in the GeoDataFrame
    for index, row in grid_gdf.iterrows():
        # Get the polygon and grid ID for the current row
        polygon = row.geometry
        grid_id = row['id']

        try:
            # Clip the VV and VH bands to the current grid polygon
            vv_clipped = vv.rio.clip([polygon])
            vh_clipped = vh.rio.clip([polygon])
            # Set the desired resolution for resampling (10 meters x 10 meters)
            target_resolution = (10, 10)
            # Uncomment the following lines if reprojecting and resampling is required
            # vv_resampled, vh_resampled = reproject_and_resample(vv_clipped, vh_clipped, target_resolution)
            # Stack the VV clipped images for the current grid polygon
            if grid_id not in stacked_images_before_vv:
                stacked_images_before_vv[grid_id] = []
            stacked_images_before_vv[grid_id].append(vv_clipped)
            # Stack the VH clipped images for the current grid polygon
            if grid_id not in stacked_images_before_vh:
                stacked_images_before_vh[grid_id] = []
            stacked_images_before_vh[grid_id].append(vh_clipped)
        except NoDataInBounds:
            # Skip to the next grid polygon if there is no valid data within the bounds
            continue
        except RasterioIOError as e:
            print("RasterioIOError:", str(e))
            continue
        except Exception as e:
            # Handle the exception and continue to the next shape
            print("Invalid or empty shape will not be rasterized:", str(e))
            continue
    # Return the dictionaries with stacked VV and VH raster data for each grid polygon
    return stacked_images_before_vv, stacked_images_before_vh


def process_shapefile_data(filtered_items):
    """
        Process raster data for each item in the filtered_items list by clipping and stacking VV and VH bands for
        each grid polygon.

        Parameters:
            filtered_items (list): A list of items containing VV and VH band raster data to be processed.

        Returns:
            tuple: A tuple containing dictionaries with stacked VV and VH raster data for each grid polygon.
    """
    # Load the grid GeoDataFrame from the provided shapefile path
    grid_gdf = load_grid_shapefile(shapefile_path)
    # Get the number of available CPU cores for parallel processing
    num_processes = multiprocessing.cpu_count()
    # Create a multiprocessing pool with the number of available CPU cores
    pool = multiprocessing.Pool(processes=num_processes)
    # Define a partial function for processing each shapefile item with the grid_gdf as a fixed argument
    partial_process_item = partial(process_shapefile_item, grid_gdf=grid_gdf)
    # Use multiprocessing to process shapefile items in parallel
    results = pool.map(partial_process_item, filtered_items)
    # Close the pool of processes after all items are processed
    pool.close()
    pool.join()
    # Initialize dictionaries to store stacked VV and VH raster data for each grid polygon
    stacked_images_before_vv = {}
    stacked_images_before_vh = {}
    # Iterate through the results, which contain stacked VV and VH raster data for each grid polygon
    for vv, vh in results:
        # Update the stacked_images_before_vv dictionary with the processed VV images for each grid polygon
        for grid_id, vv_images in vv.items():
            if grid_id not in stacked_images_before_vv:
                stacked_images_before_vv[grid_id] = []
            stacked_images_before_vv[grid_id].extend(vv_images)
        # Update the stacked_images_before_vh dictionary with the processed VH images for each grid polygon
        for grid_id, vh_images in vh.items():
            if grid_id not in stacked_images_before_vh:
                stacked_images_before_vh[grid_id] = []
            stacked_images_before_vh[grid_id].extend(vh_images)
    # Return the dictionaries with stacked VV and VH raster data for each grid polygon
    return stacked_images_before_vv, stacked_images_before_vh


def calculate_mean_and_std(array):
    """
        Calculate the mean and standard deviation of the given array.

        Parameters:
            array (list or numpy.ndarray or pandas.Series): An input array for which to calculate the mean and standard deviation.

        Returns:
            tuple: A tuple containing two numpy arrays representing the mean and standard deviation of the input array.
    """
    # Check if the input is a list, and convert it to a numpy array if necessary
    if isinstance(array, list):
        dataarray_value = np.array(array)
    else:
        dataarray_value = array.values
    # Calculate the mean and standard deviation along the first axis
    mean = np.squeeze(np.mean(dataarray_value, axis=0))
    std = np.squeeze(np.std(dataarray_value, axis=0))
    # Return the calculated mean and standard deviation as a tuple
    return mean, std
