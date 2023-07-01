# ifmiap/mapfloods/__init__.py

import os



import matplotlib.pyplot as plt
import numpy as np
import rasterio


def map_floods(vv_flood_extent, vh_flood_extent, vv_during_stack, nearest_date,data_array):
    """
        Generates a flood map based on the flood extent arrays for VV and VH data.

        Args:
            vv_flood_extent (list): List of flood extent arrays for VV data.
            vh_flood_extent (list): List of flood extent arrays for VH data.
            vv_during_stack (ndarray): Stack of VV data during the flood period.
            nearest_date (str): Nearest date to the target date.
            data_array: The data array associated with the flood extent arrays.

        Returns:
            None

        """
    for i, j in zip(vv_flood_extent, vh_flood_extent):
        combined_flood_extent = i.astype(np.uint8) + j.astype(np.uint8)
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        ax.set_title(f'Flood Map (Dry Period - 01 February to 15 May 2020) (Wet Date - {nearest_date})')
        ax.set_axis_off()
        ax.imshow(np.flip(combined_flood_extent, axis=1))
        # plt.show()
        total_pixels = combined_flood_extent.size
        flood_pixels = (combined_flood_extent == 1).sum()
        flood_percentage = (flood_pixels / total_pixels) * 100
        threshold = 2
        is_flood = flood_percentage > threshold
        # Print the classification result
        if is_flood:
            print("The image represents a flood scene.")
            export_flood_map(vv_during_stack, combined_flood_extent, data_array)
        else:
            print("The image does not represent a flood scene.")


def export_flood_map(vv_during_stack, combined_flood_extent, data_array):
    """
        Exports the flood map as a GeoTIFF file.

        Args:
            vv_during_stack (ndarray): Stack of VV data during the flood period.
            combined_flood_extent (ndarray): Combined flood extent array.
            data_array: The data array associated with the flood extent arrays.

        Returns:
            None

        """
    for i, np_array in enumerate(vv_during_stack):
        attribute_value = data_array[i].attrs["Image_ID"]

        with rasterio.Env():
            data = combined_flood_extent

            # Define the metadata for the output file
            profile = {
                'driver': 'GTiff',
                'dtype': rasterio.float32,
                'nodata': None,
                'width': data.shape[1],
                'height': data.shape[0],
                'count': 1,
            }

            with rasterio.open(
                    "ifmiap/mapfloods/Flood_images_output/" + attribute_value + ".tif",
                    'w', **profile) as dst:
                dst.write(data, 1)
