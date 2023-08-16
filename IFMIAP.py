import datetime
import warnings
import time
import sys
import os
from rasterio.errors import NotGeoreferencedWarning

from ifmiap.authenticate import sign_in
from ifmiap.utils import search_sentinel_data, filter_items_floodPeriod, grid_bounds, filter_items_dryPeriod
from ifmiap.preprocessing import process_raster_data, process_shapefile_data, get_grid_data_dry_period
from ifmiap.postprocessing import process_vv_stacked_images, process_vh_stacked_images
from ifmiap.mapfloods import map_floods

# Ignore the NotGeoreferencedWarning
warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)
# Authenticate and get the STAC catalog
catalog = sign_in()


def run_script(time_of_interest_date, stacked_images_before_vv, stacked_images_before_vh):
    # Define the area of interest as a polygon for the STAC API search
    bbox_of_interest = {
        "type": "Polygon",
        "coordinates": [
            [
                [85.7755, 25.6779],
                [86.5281, 25.6779],
                [86.5281, 26.1128],
                [85.7755, 26.1128],
                [85.7755, 25.6779]
            ]
        ],
    }
    # Calculate the start_date and end_date for data retrieval
    start_date = time_of_interest_date - datetime.timedelta(days=0)
    end_date = time_of_interest_date + datetime.timedelta(days=7)
    # Call the STAC API to search for satellite imagery meeting the specified criteria
    results = search_sentinel_data(catalog, bbox_of_interest, start_date=start_date, end_date=end_date,
                                   time_of_interest_date=time_of_interest_date)

    # print(f"Returned {len(results)} Images")
    # Filter the retrieved data based on criteria for flood period items
    filtered_results, intersecting_ids, intersecting_geometries = filter_items_floodPeriod(results)
    # print(f"Returned {len(filtered_results)} filtered_items")
    # Process the raster data for VV and VH bands during the flood period
    stacked_images_during_vv, stacked_images_during_vh = process_raster_data(filtered_results, intersecting_ids,
                                                                             intersecting_geometries)

    # POSTPROCESS
    # Initialize lists to store VV flood extent data
    vv_flood_extent = []
    image_ids = []
    grid_ids = []
    geometry_ids = []
    # Call the function to process stacked images for VV band
    vv_flood_extent, image_ids, grid_ids, geometry_ids = process_vv_stacked_images(
        stacked_images_during_vv, stacked_images_before_vv, vv_flood_extent, image_ids, grid_ids, geometry_ids
    )
    # Initialize lists to store VH flood extent data
    vh_flood_extent = []
    image_ids_vh = []
    grid_ids_vh = []
    geometry_ids_vh = []
    # Call the function to process stacked images for VH band
    vh_flood_extent, image_ids_vh, grid_ids_vh, geometry_ids_vh = process_vh_stacked_images(
        stacked_images_during_vh, stacked_images_before_vh, vh_flood_extent, image_ids_vh, grid_ids_vh, geometry_ids_vh
    )
    # Map the flood extents for both VV and VH bands
    map_floods(vv_flood_extent, vh_flood_extent, image_ids, grid_ids, geometry_ids)




if __name__ == '__main__':
    nearest_dates = []
    # Date of interest for the flood map
    time_of_interest_date = datetime.date(2023, 7, 1)
    time_of_interest_time = datetime.time(18, 0, 0)  # 10:50 AM
    # Combine the target date and time
    time_of_interest_datetime = datetime.datetime.combine(time_of_interest_date, time_of_interest_time)
    n = 1
    # Call process_shapefile_data function to obtain stacked_images_before_vv and stacked_images_before_vh
    filtered_grid_data_results = get_grid_data_dry_period(catalog, n)
    stacked_images_before_vv, stacked_images_before_vh = process_shapefile_data(filtered_grid_data_results)
    # Start the continuous execution until the target date
    while True:
        # Get the current date and time
        current_datetime = datetime.datetime.now()
        if time_of_interest_date >= datetime.date(2023, 7, 10):
            time_of_interest_date += datetime.timedelta(days=11)
        if time_of_interest_datetime >= current_datetime:
            # Calculate the time difference until the next day
            next_day = current_datetime + datetime.timedelta(days=1)
            time_difference = datetime.datetime.combine(next_day.date(), time_of_interest_time) - current_datetime
            # Check if time_difference is positive (not in the past)
            if time_difference.total_seconds() > 0:
                while time_difference.total_seconds() > 0:
                    time_left = str(time_difference).split(".")[0]  # Extract the time portion
                    sys.stdout.write(f"\rProgram is sleeping. Time left until waking up: {time_left}")
                    sys.stdout.flush()
                    time.sleep(1)  # Update the time every 1 second
                    time_difference = datetime.datetime.combine(next_day.date(),
                                                                time_of_interest_time) - datetime.datetime.now()

                # Clear the line
                os.system('cls' if os.name == 'nt' else 'clear')

                print("Waking up!")
                continue
        # Execute the main script
        run_script(time_of_interest_date, stacked_images_before_vv, stacked_images_before_vh)

        # Increment the target date by one day
        time_of_interest_date += datetime.timedelta(days=1)

        # Combine the updated target date and time
        time_of_interest_datetime = datetime.datetime.combine(time_of_interest_date, time_of_interest_time)
    # Calculate the time difference until the next day
    next_day = current_datetime + datetime.timedelta(days=1)
    time_difference = datetime.datetime.combine(next_day.date(), time_of_interest_time) - current_datetime
    # Sleep until the next day
    # time.sleep(time_difference.total_seconds())
