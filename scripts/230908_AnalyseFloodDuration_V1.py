import numpy as np
from pyrsgis import raster
import glob


# create a dictionary that contains input file names by date
files_list = glob.glob(r'../../stacked_flood_sample/*.tif')
files_dict = {
    int(file.split('_')[-1][:8]): file
    for file in files_list
}

# read the data sorted by date
def read_array(file):
    ds, arr = raster.read(file)
    return arr

data_dict = {
    key: read_array(files_dict[key])
    for key in sorted(files_dict)
}

# for now let's say every non-zero cell is flood
# stack all of them in a 3D array with cell value as date
flood_data_3d = np.stack([
    (data_dict[key] > 0).astype(int)
    #(data_dict[key] > 0) * key
    for key in data_dict
    ])


# analyse the max duration of floods and the number of times it flooded
import numpy as np

def analyze_flood_data_3d(flood_data_3d):
    max_durations = np.zeros_like(flood_data_3d[0, :, :], dtype=int)
    unique_event_counts = np.zeros_like(flood_data_3d[0, :, :], dtype=int)

    for x in range(flood_data_3d.shape[1]):
        for y in range(flood_data_3d.shape[2]):
            flood_data = flood_data_3d[:, x, y]
            max_duration = 0
            current_duration = 0
            current_event = 0
            unique_event_count = 0

            for is_flooding in flood_data:
                is_flooding = int(is_flooding)  # Ensure it's treated as an integer
                current_duration = current_duration + 1 if is_flooding else 0
                if current_duration > max_duration:
                    max_duration = current_duration

                if is_flooding:
                    if current_event == 0:
                        unique_event_count += 1
                    current_event = 1
                else:
                    current_event = 0

            max_durations[x, y] = max_duration
            unique_event_counts[x, y] = unique_event_count

    return max_durations, unique_event_counts


max_duration, unique_event_count = analyze_flood_data_3d(flood_data_3d)

# read dummy data for template
ds, _ = raster.read(files_list[0])
raster.export(max_duration, ds, r'../../stacked_flood_sample/max_duration.tif', compress='deflate')
raster.export(unique_event_count, ds, r'../../stacked_flood_sample/unique_event_count.tif', compress='deflate')