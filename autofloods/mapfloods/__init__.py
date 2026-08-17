# autofloods/mapfloods/__init__.py
import os
import xarray as xr
import matplotlib.pyplot as plt

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
    pre_stack (dict)        : A dictionary containing multi-band stacked pre-flood DataArrays for VV and VH bands.
    post_stack (dict)       : A dictionary containing multi-band stacked post-flood DataArrays for VV and VH bands.
    vv_thd (float)          : Threshold for anomaly detection in the VV band.
    vh_thd (float)          : Threshold for anomaly detection in the VH band.

    Returns
    _______
    xarray.DataArray        : A DataArray indicating combined anomaly and flood cells.

    """
    # calculate anomaly and flood cells for VV band
    anomaly_vv = (post_xarray.loc['vv_ds'] - pre_xarray.loc['vv_mean']) / pre_xarray.loc['vv_std']
    #floods_vv = (abs(anomaly_vv) > vv_thd).astype(int)
    floods_vv = (anomaly_vv < vv_thd).astype(int)

    # calculate anomaly and flood cells for VH band
    anomaly_vh = (post_xarray.loc['vh_ds'] - pre_xarray.loc['vh_mean']) / pre_xarray.loc['vh_std']
    #floods_vh = (abs(anomaly_vh) > vh_thd).astype(int)
    floods_vh = (anomaly_vh < vh_thd).astype(int)

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
def map_floods(mean_std_by_aoi, wet_scenes_by_aoi, slope_path, vv_thd, vh_thd, rel_slope_thd):
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

    # apply relative slope mask
    for id in anomaly_dict:
        slope_xarray = xr.load_dataarray(slope_path.replace('_id.nc', f'_{id}.nc'), engine='rasterio')
        for scene_id in anomaly_dict[id]:
            anomaly_dict[id][scene_id] = anomaly_dict[id][scene_id].where(slope_xarray.values[0, :, :] < rel_slope_thd, 0)

    return anomaly_dict


# define a function to export flood maps as images
def flood_images(flood_xarray, outfile_flood):
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












