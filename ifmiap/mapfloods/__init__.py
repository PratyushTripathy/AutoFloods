# ifmiap/mapfloods/__init__.py


# define a function to identify anomaly cells (flood cells)
def anomaly_cells(pre_stack, post_stack, vv_thd, vh_thd):
    """
    Detect Anomaly and Flood Cells in Multi-Band Stacks

    This function takes multi-band stacks of pre- and post-flood images for VV and VH bands,
    along with specified thresholds, and detects anomaly and flood cells based on statistical
    analysis of the image data.

    Parameters
    __________
    pre_stack (dict)        : A dictionary containing multi-band stacked pre-flood DataArrays for VV and VH bands.
    post_stack (dict)       : A dictionary containing multi-band stacked post-flood DataArrays for VV and VH bands.
    vv_thd (float)          : Threshold for anomaly detection in the VV band.
    vh_thd (float)          : Threshold for anomaly detection in the VH band.

    Returns
    _______
    xarray.DataArray        : A DataArray indicating combined anomaly and flood cells.

    """
    # calculate anomaly and flood cells for VV band
    pre_mean_vv = pre_stack['vv_stack'].mean(axis=0)
    pre_std_vv = pre_stack['vv_stack'].std(axis=0)
    anomaly_vv = (post_stack['vv_stack'].mean(axis=0) - pre_mean_vv) / pre_std_vv
    floods_vv = (abs(anomaly_vv) > vv_thd).astype(int)

    # calculate anomaly and flood cells for VH band
    pre_mean_vh = pre_stack['vh_stack'].mean(axis=0)
    pre_std_vh = pre_stack['vh_stack'].std(axis=0)
    anomaly_vh = (post_stack['vh_stack'].mean(axis=0) - pre_mean_vh) / pre_std_vh
    floods_vh = (abs(anomaly_vh) > vh_thd).astype(int)
    floods_vh.values[floods_vv.values == 1] = 2

    # combine flood extent for VV and VH bands
    combined_floods = floods_vv + floods_vh

    return combined_floods








############################################################


import os
import matplotlib.pyplot as plt
import numpy as np
import rasterio
import xarray as xr


def map_floods(vv_flood_extent, vh_flood_extent, image_ids, grid_ids, geometry_ids):
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
    # Loop through vv_flood_extent and vh_flood_extent lists
    for i, j, image_id, grid_id, geometry_id in zip(vv_flood_extent, vh_flood_extent, image_ids, grid_ids,
                                                    geometry_ids):
        combined_flood_extent = i.astype(np.uint8) + j.astype(np.uint8)

        total_pixels = combined_flood_extent.size
        flood_pixels = (combined_flood_extent == 1).sum()
        flood_percentage = (flood_pixels / total_pixels) * 100
        threshold = 2
        is_flood = flood_percentage > threshold
        # Print the classification result
        if is_flood:
            print("The image represents a flood scene.")
            export_flood_map(combined_flood_extent, image_id, grid_id, geometry_id)
        else:
            print("The image does not represent a flood scene.")


def export_flood_map(combined_flood_extent, image_id, grid_id, geometry_id):
    """
        Exports the flood map as a GeoTIFF file.

        Args:
            combined_flood_extent (ndarray): Combined flood extent array.
            data_array: The data array associated with the flood extent arrays.

        Returns:
            None

        """
    image_id = image_id
    grid_id = grid_id
    with rasterio.Env():
        # Create a xarray DataArray from the combined flood extent array
        data = xr.DataArray(combined_flood_extent)
        xmin, ymin, xmax, ymax = geometry_id.bounds
        # Define the metadata for the output file
        profile = {
            'driver': 'GTiff',
            'dtype': rasterio.float32,
            'nodata': None,
            'width': data.shape[1],
            'height': data.shape[0],
            'count': 1,
            'transform': rasterio.transform.from_bounds(xmin, ymin, xmax, ymax, data.shape[1], data.shape[0])
        }

        with rasterio.open(
                "Flood_images_output/" + image_id + "_" + str(grid_id) + ".tif",
                'w', **profile) as dst:
            dst.write(data, 1)
