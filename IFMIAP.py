import numpy as np
import datetime
from ifmiap.authenticate import sign_in
from ifmiap.preprocessing import preprocess, process_nearest_date
from ifmiap.postprocessing import postprocess
from ifmiap.mapfloods import map_floods
import sys
import warnings
from rasterio.errors import NotGeoreferencedWarning

# Ignore the NotGeoreferencedWarning
warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)
nearest_dates = []
# Date of interest for the flood map
time_of_interest_date = datetime.date(2022, 7, 1)
time_of_interest_time = datetime.time(10, 50, 0)  # 10:50 AM
# Combine the target date and time
time_of_interest_datetime = datetime.datetime.combine(time_of_interest_date, time_of_interest_time)
# Authenticate and get the STAC catalog
catalog = sign_in()


def run_script():
    """
        This function executes the main processing steps for generating flood maps.

        It follows the following steps:
        1. Define the bounding box and date range.
        2. Preprocess the data by sorting collections and obtaining relevant data for the time of interest.
        3. Process the nearest date to the time of interest and retrieve necessary data arrays.
        4. Check if the data arrays are valid. If not, skip the rest of the code and move to the next iteration.
        5. Load preprocessed data arrays for the dry period.
        6. Calculate mean and standard deviation of VV and VH data for the dry period.
        7. Perform postprocessing using the VV and VH data during the time of interest and the calculated statistics.
        8. Generate a flood map using the postprocessed data.
        9. Repeat the above steps for subsequent target dates until the script is stopped.

        Note: The function assumes the availability of specific files and functions used in the processing steps.

        Returns:
            None
        """
    # Define the bounding box and date range
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

    # Start and end date period
    start_date = time_of_interest_date - datetime.timedelta(days=7)
    end_date = time_of_interest_date + datetime.timedelta(days=7)

    # Preprocess the data
    sorted_collections, df_before = preprocess(catalog, bbox_of_interest, start_date, end_date, time_of_interest_date)
    # Finding Nearest date
    vh_during_stack, vv_during_stack, data_array = process_nearest_date(catalog, sorted_collections,
                                                                        time_of_interest_date,
                                                                        bbox_of_interest)
    if vh_during_stack is None or vv_during_stack is None or data_array is None:
        return  # Skip the rest of the code and move to the next iteration
    # Load Dry period VV and VH Numpy array file
    vv_before_stack = np.load('ifmiap/postprocessing/Dry_period/vv_before_stack.npy')
    vh_before_stack = np.load('ifmiap/postprocessing/Dry_period/vh_before_stack.npy')
    # Calculate mean and standard deviation of VV and VH data for the dry period
    vv_mean = np.mean(vv_before_stack, axis=0)
    vv_std = np.std(vv_before_stack, axis=0)

    vh_mean = np.mean(vh_before_stack, axis=0)
    vh_std = np.std(vh_before_stack, axis=0)

    # Perform postprocessing using VV and VH data during the time of interest and the calculated statistics
    vv_flood_extent, vh_flood_extent = postprocess(vv_during_stack, vh_during_stack, vv_mean, vv_std, vh_mean,
                                                   vh_std)

    # Generate flood map
    nearest_date = time_of_interest_date.strftime("%Y-%m-%d")
    map_floods(vv_flood_extent, vh_flood_extent, vv_during_stack, nearest_date, data_array)

# The code below continuously executes the script until a target date is reached
while True:
    # Get the current date and time
    current_datetime = datetime.datetime.now()

    # Check if it's the target date and time to run the script
    if current_datetime >= time_of_interest_datetime:
        if time_of_interest_date >= datetime.date(2022, 8, 1):
            # Increment the target date by 12 days
            # time_of_interest_date += datetime.timedelta(days=12)
            # print(time_of_interest_date,"target")

            # # Combine the updated target date and time
            # time_of_interest_datetime = datetime.datetime.combine(time_of_interest_date, time_of_interest_time)
            sys.exit("Script stopped as the target date has been reached.")
        # Execute the main script
        run_script()

        # Increment the target date by one day
        time_of_interest_date += datetime.timedelta(days=1)
        print(time_of_interest_date, "target")

        # Combine the updated target date and time
        time_of_interest_datetime = datetime.datetime.combine(time_of_interest_date, time_of_interest_time)
    # Calculate the time difference until the next day
    next_day = current_datetime + datetime.timedelta(days=1)
    time_difference = datetime.datetime.combine(next_day.date(), time_of_interest_time) - current_datetime

#     # # Sleep until the next day
#     # time.sleep(time_difference.total_seconds())
