# ifmiap/mapfloods/__init__.py
import os
import xarray as xr
import matplotlib.pyplot as plt
import xrspatial

# switch off displaying maps
plt.ioff()

# define a function to identify anomaly cells (flood cells)
def map_anomaly_cells(pre_xarray, post_xarray, vv_thd, vh_thd):
    """
    Detect Anomaly and Flood Cells in Multi-Band Stacks

    This function takes multi-band stacks of pre- and post-flood images for VV and VH bands,
    along with specified thresholds, and detects anomaly and flood cells based on statistical
    analysis of the image data.

    Parameters
    __________
    pre_stack               : xarray.Dataset
                              A dictionary containing multi-band stacked pre-flood DataArrays for VV and VH bands.
    post_stack              : xarray.Dataset
                              A dictionary containing multi-band stacked post-flood DataArrays for VV and VH bands.
    vv_thd                  : float
                              Threshold for anomaly detection in the VV band.
    vh_thd                  : float
                              Threshold for anomaly detection in the VH band.

    Returns
    _______
    xarray.DataArray        : xarray.DataArray
                              A DataArray indicating combined anomaly and flood cells.

    """
    # calculate anomaly and flood cells for VV band
    anomaly_vv = (post_xarray.loc['vv_ds'] - pre_xarray.loc['vv_mean']) / pre_xarray.loc['vv_std']
    floods_vv = (abs(anomaly_vv) > vv_thd).astype(int)

    # calculate anomaly and flood cells for VH band
    anomaly_vh = (post_xarray.loc['vh_ds'] - pre_xarray.loc['vh_mean']) / pre_xarray.loc['vh_std']
    floods_vh = (abs(anomaly_vh) > vh_thd).astype(int)

    # here's what numbers in the flood map mean
    # 1. flood cells identified in the VH band
    # 2. flood cells identified in the VV band
    # 3. flood cells identified in both VV and VH bands
    # handling conditional update was real pain, be cautious in future
    combined_floods = floods_vv + floods_vh
    combined_floods = combined_floods.where(floods_vh.values != 1, 1)
    combined_floods = combined_floods.where(floods_vv.values != 1, 2)
    combined_floods = combined_floods.where((floods_vv + floods_vh).values != 2, 3)

    # combine flood extent for VV and VH bands

    return combined_floods


# define a function to map floods (identify anomaly cells and perform slope and elevaation mask)
def map_floods(mean_std_by_aoi, wet_scenes_by_aoi, dem_path, vv_thd, vh_thd, dem_thd, slp_thd):
    """
    Map Flood Anomalies within Areas of Interest (AOIs)

    This function takes statistical data and multi-scene wetness information within Areas of Interest (AOIs),
    along with elevation and slope thresholds, to map flood anomalies. It generates a dictionary containing
    flood anomaly maps for each AOI and scene.

    Parameters
    __________
    mean_std_by_aoi         : A dictionary mapping AOI IDs to mean and standard deviation data.
    wet_scenes_by_aoi       : A dictionary mapping AOI IDs to scenes with wetness information.
    dem_path                : string
                              Path to the digital elevation model (DEM) data.
    vv_thd                  : float
                              Threshold value for VV (Vertical-Vertical) polarization.
    vh_thd                  : float
                              Threshold value for VH (Vertical-Horizontal) polarization.
    dem_thd                 : float
                              Threshold value for the DEM.
    slp_thd                 : float
                              Threshold value for the slope.

    Returns
    _______
    dict                    : A nested dictionary containing flood anomaly maps for each AOI and scene.

    """
    # generate id and scene wise anomaly cells
    anomaly_dict = {
        id: {
            scene_id: map_anomaly_cells(
                mean_std_by_aoi[id], wet_scenes_by_aoi[id][scene_id], vv_thd=vv_thd, vh_thd=vh_thd
            )
            for scene_id in wet_scenes_by_aoi[id]
        }
        for id in mean_std_by_aoi
    }

    # apply elevation and slope mask
    for id in anomaly_dict:
        dem_xarray = xr.load_dataarray(dem_path.replace('_id.nc', f'_{id}.nc'), engine='rasterio')
        slope_xarray = xrspatial.slope(dem_xarray.squeeze())
        for scene_id in anomaly_dict[id]:
            anomaly_dict[id][scene_id] = anomaly_dict[id][scene_id].where(dem_xarray.values[0, :, :] < dem_thd, 0)
            anomaly_dict[id][scene_id] = anomaly_dict[id][scene_id].where(slope_xarray.values < slp_thd, 0)

    return anomaly_dict


# define a function to export flood maps as images
def flood_images(flood_xarray, outfile_flood):
    """
    Export Flood Raster Image

    This function takes a flood xarray dataset and exports an image representing the flood raster data. It calculates
    the image's extent, creates a plot, and saves the image to a file.

    Parameters
    __________
    flood_xarray              : An xarray dataset containing flood raster data.
    outfile_flood             : string
                                The path and filename for the exported flood image.

    Returns
    _______
    None

    """
    x_min = flood_xarray.x.min()
    y_min = flood_xarray.y.min()
    x_max = flood_xarray.x.max()
    y_max = flood_xarray.y.max()

    # export an image of the combined flood rasters
    height, width = flood_xarray.shape
    fig, ax = plt.subplots(1, 1, figsize=(width / 100, height / 100))

    img = flood_xarray.plot(ax=ax, cmap='Blues')
    img.colorbar.remove()

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_title(os.path.split(os.path.splitext(outfile_flood)[0])[-1])

    plt.savefig(outfile_flood, bbox_inches='tight', dpi=100)












