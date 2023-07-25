# ifmiap/utils/__init__.py

from shapely.geometry import box
import geopandas as gpd

shapefile_path = "inputs/Grid_shapefile/shp_4326.shp"


def load_grid_shapefile(shapefile_path):
    """
        Load a grid shapefile as a GeoDataFrame.

        Parameters:
            shapefile_path (str): The file path to the grid shapefile.

        Returns:
            gpd.GeoDataFrame: A GeoDataFrame representing the grid polygons loaded from the shapefile.
    """
    # Load the grid shapefile using GeoPandas' read_file function
    grid_gdf = gpd.read_file(shapefile_path)
    # Return the loaded GeoDataFrame
    return grid_gdf


# def search_sentinel_data(catalog, date_ranges, bbox_of_interest):
#     all_results = []
#     for date_range in date_ranges:
#         # Search for Sentinel-1 data
#         results = catalog.search(
#             collections=["sentinel-1-grd"],
#             intersects=bbox_of_interest,
#             datetime=date_range,
#         )
#         for item in results.get_items():
#             if 'vh' in item.assets:
#                 all_results.append(item)
#     return all_results



def search_sentinel_data(catalog, bbox_of_interest, date_ranges=None, start_date=None, end_date=None, time_of_interest_date=None):
    """
       Search for Sentinel-1 data within a specified time range and bounding box of interest.

       Parameters:
           catalog (pystac.Catalog): The pystac Catalog object representing the STAC catalog to search.
           bbox_of_interest (dict): The bounding box of interest in GeoJSON format, specified as a dictionary.
           date_ranges (list, optional): A list of date ranges in ISO 8601 format (start_date/end_date) to search for data.
                                         If provided, this parameter takes precedence over start_date and end_date.
           start_date (datetime.date, optional): The start date of the time range to search for data.
           end_date (datetime.date, optional): The end date of the time range to search for data.
           time_of_interest_date (datetime.date, optional): The date of interest (a single date) to search for data.

       Returns:
           list: A list of pystac.Item objects containing the searched Sentinel-1 data.

       Raises:
           ValueError: If both date_ranges and start_date/end_date are None or if both are provided simultaneously.
    """
    all_results = []

    if date_ranges is not None:
        # If date_ranges is provided, use it to perform the search
        for date_range in date_ranges:
            # Search for Sentinel-1 data
            results = catalog.search(
                collections=["sentinel-1-grd"],
                intersects=bbox_of_interest,
                datetime=date_range,
            )
            for item in results.get_items():
                if 'vh' in item.assets:
                    all_results.append(item)
    else:
        if start_date is not None and end_date is not None:
            # If date_ranges is not provided, construct the date range using start_date and end_date
            # Convert start_date and end_date to ISO 8601 format
            start_date_str = start_date.strftime("%Y-%m-%dT00:00:00Z")
            end_date_str = end_date.strftime("%Y-%m-%dT23:59:59Z")

            # If time_of_interest_date is not None, construct the list of date ranges
            date_ranges = []
            if time_of_interest_date is not None:
                date_ranges.append(f"{start_date_str}/{end_date_str}")
            else:
                # Search for the single date
                date_ranges.append(start_date_str)

            for date_range in date_ranges:
                # Search for Sentinel-1 data
                results = catalog.search(
                    collections=["sentinel-1-grd"],
                    intersects=bbox_of_interest,
                    datetime=date_range,
                )
                for item in results.get_items():
                    if 'vh' in item.assets:
                        all_results.append(item)
        else:
            raise ValueError("Invalid input: either date_ranges or start_date and end_date should be provided.")

    return all_results

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
