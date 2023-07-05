import numpy as np
import datetime
from ifmiap.authenticate import sign_in
from ifmiap.preprocessing import preprocess, process_nearest_date
from ifmiap.postprocessing import postprocess
from ifmiap.mapfloods import map_floods
import sys
import warnings
from rasterio.errors import NotGeoreferencedWarning
import multiprocessing

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

def process_target_date(target_date):
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
    start_date = target_date - datetime.timedelta(days=7)
    end_date = target_date + datetime.timedelta(days=7)

    # Preprocess the data
    sorted_collections, df_before = preprocess(catalog, bbox_of_interest, start_date, end_date, target_date)
    # Finding Nearest date
    vh_during_stack, vv_during_stack, data_array = process_nearest_date(catalog, sorted_collections,
                                                                        target_date,
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
    nearest_date = target_date.strftime("%Y-%m-%d")
    map_floods(vv_flood_extent, vh_flood_extent, vv_during_stack, nearest_date, data_array)


def run_script_parallel():
    num_processes = multiprocessing.cpu_count()  # Number of available CPU cores
    print(f"Number of CPUs used: {num_processes}")
    pool = multiprocessing.Pool(processes=num_processes)
    target_dates = []

    while True:
        current_datetime = datetime.datetime.now()

        if current_datetime >= time_of_interest_datetime:
            if time_of_interest_date >= datetime.date(2022, 8, 1):
                sys.exit("Script stopped as the target date has been reached.")
            if time_of_interest_date == current_datetime:
                next_day = current_datetime + datetime.timedelta(days=1)
                time_difference = datetime.datetime.combine(next_day, time_of_interest_time) - datetime.datetime.now()

                if time_difference.total_seconds() > 0:
                    time.sleep(time_difference.total_seconds())
                    continue

            target_dates.append(time_of_interest_date)
            time_of_interest_date += datetime.timedelta(days=1)
            time_of_interest_datetime = datetime.datetime.combine(time_of_interest_date, time_of_interest_time)

        next_day = current_datetime + datetime.timedelta(days=1)
        time_difference = datetime.datetime.combine(next_day.date(), time_of_interest_time) - current_datetime

        if len(target_dates) > 0:
            pool.map(process_target_date, target_dates)
            target_dates = []

        # time.sleep(time_difference.total_seconds())

if __name__ == '__main__':
    run_script_parallel()
