# ifmiap/__init__.py
import numpy as np

def postprocess(vv_during_stack, vh_during_stack, vv_mean, vv_std, vh_mean, vh_std):
    """
       Post-processes the VV and VH stacks during flood using the provided mean and standard deviation arrays.

       Args:
           vv_during_stack (ndarray): Stack of VV data during the flood period.
           vh_during_stack (ndarray): Stack of VH data during the flood period.
           vv_mean (ndarray): Mean array of VV data.
           vv_std (ndarray): Standard deviation array of VV data.
           vh_mean (ndarray): Mean array of VH data.
           vh_std (ndarray): Standard deviation array of VH data.

       Returns:
           vv_flood_extent (list): List of flood extent arrays for VV data.
           vh_flood_extent (list): List of flood extent arrays for VH data.

       """
    vv_flood_extent = []
    for i in range(vv_during_stack.shape[0]):
        layer = vv_during_stack[i, :, :]
        if layer.shape == vv_mean.shape:
            vv_flood_extent.append(np.nan_to_num(((layer - vv_mean) / vv_std) > -3, 0))
        else:
            min_shape = np.minimum(layer.shape, vv_mean.shape)
            # Resize the larger array to match the minimum shape
            layer = np.resize(layer, min_shape)
            vv_mean = np.resize(vv_mean, min_shape)
            vv_std = np.resize(vv_std, min_shape)
            vv_flood_extent.append(np.nan_to_num(((layer - vv_mean) / vv_std) > -3, 0))

    vh_flood_extent = []
    for i in range(vh_during_stack.shape[0]):
        layer = vh_during_stack[i, :, :]
        if layer.shape == vh_mean.shape:
            vh_flood_extent.append(np.nan_to_num(((layer - vh_mean) / vh_std) > -3, 0))
        else:
            min_shape = np.minimum(layer.shape, vh_mean.shape)
            layer = np.resize(layer, min_shape)
            vh_mean = np.resize(vh_mean, min_shape)
            vh_std = np.resize(vh_std, min_shape)
            vh_flood_extent.append(np.nan_to_num(((layer - vh_mean) / vh_std) > -3, 0))

    return vv_flood_extent, vh_flood_extent
