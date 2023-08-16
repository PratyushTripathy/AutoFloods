# ifmiap/postprocessing/__init__.py
import numpy as np

from ifmiap.preprocessing import calculate_mean_and_std

def process_vv_stacked_images(stacked_images_during_vv, stacked_images_before_vv, vv_flood_extent, image_ids, grid_ids,
                              geometry_ids):
    """
      Process the stacked VV images for each grid polygon during a specific period by comparing them with the
      stacked VV images from the dry period. Calculate the flood extent based on the difference in mean and
      standard deviation.

      Parameters:
          stacked_images_during_vv (dict): A dictionary containing stacked VV images for each grid polygon during a
                                           specific period.
          stacked_images_before_vv (dict): A dictionary containing stacked VV images for each grid polygon from the
                                           dry period.
          vv_flood_extent (list): A list to store the calculated flood extent for each grid polygon.
          image_ids (list): A list to store the image IDs corresponding to the flood extent for each grid polygon.
          grid_ids (list): A list to store the grid IDs for each grid polygon.
          geometry_ids (list): A list to store the geometry IDs for each grid polygon.

      Returns:
          tuple: A tuple containing the updated vv_flood_extent list, image IDs, grid IDs, and geometry IDs.
    """
    for grid_id in stacked_images_during_vv:
        # Retrieve the stacked images for the current grid ID
        images_for_grid = stacked_images_during_vv[grid_id]
        # Process the stacked images as needed
        for images_tuple in images_for_grid:
            # Access the individual stacked images
            vv_stack_during = images_tuple[0]
            # iterate through each layer
            layer = vv_stack_during
            try:
                # Retrieve the image_id for the current stacked image
                image_id = vv_stack_during.attrs.get('Image_ID', None)
                # Retrieve the Geometry
                geometry_id = vv_stack_during.attrs.get('Geometry', None)
                # Retrieve vv_mean and vv_std from stacked_images_before_vv
                vv_mean, vv_std = calculate_mean_and_std(stacked_images_before_vv[grid_id][0])
                if layer.shape == vv_mean.shape:
                    vv_flood_extent.append(np.nan_to_num(((layer - vv_mean) / vv_std) > -3, 0))
                else:
                    min_shape = np.minimum(layer.shape, vv_mean.shape)
                    # Resize the larger array to match the minimum shape
                    layer = np.resize(layer, min_shape)
                    vv_mean = np.resize(vv_mean, min_shape)
                    vv_std = np.resize(vv_std, min_shape)
                    vv_flood_extent.append(np.nan_to_num(((layer - vv_mean) / vv_std) > -3, 0))

                # Append the image_id, grid_id, and geometry_id to their respective lists
                image_ids.append(image_id)
                grid_ids.append(grid_id)
                geometry_ids.append(geometry_id)
            except KeyError:
                print(f"grid_id {grid_id} not found in Dry period!")
                continue
    # Return the updated vv_flood_extent list, image IDs, grid IDs, and geometry IDs
    return vv_flood_extent, image_ids, grid_ids, geometry_ids


def process_vh_stacked_images(stacked_images_during_vh, stacked_images_before_vh, vh_flood_extent, image_ids_vh,
                              grid_ids_vh,
                              geometry_ids_vh):
    """
        Process the stacked VH images for each grid polygon during the flood period and calculate the flood extent.

        Parameters:
            stacked_images_during_vh (dict): A dictionary containing stacked VH images for each grid polygon during the flood period.
            stacked_images_before_vh (dict): A dictionary containing stacked VH images for each grid polygon before the flood period.
            vh_flood_extent (list): An empty list to store the calculated flood extent for each grid polygon.
            image_ids_vh (list): A list to store the image IDs for each processed stacked image.
            grid_ids_vh (list): A list to store the grid IDs for each processed grid polygon.
            geometry_ids_vh (list): A list to store the geometry IDs for each processed grid polygon.

        Returns:
            tuple: A tuple containing the updated vh_flood_extent list, image IDs, grid IDs, and geometry IDs.
    """
    for grid_id in stacked_images_during_vh:
        # Retrieve the stacked images for the current grid ID
        images_for_grid = stacked_images_during_vh[grid_id]
        # Process the stacked images as needed
        for images_tuple in images_for_grid:
            # Access the individual stacked images
            vh_stack_during = images_tuple[0]
            # iterate through each layer
            layer = vh_stack_during
            try:
                # Retrieve the image_id for the current stacked image
                image_id = vh_stack_during.attrs.get('Image_ID', None)
                # Retrieve the Geometry
                geometry_id = vh_stack_during.attrs.get('Geometry', None)
                # Retrieve vv_mean and vv_std from stacked_images_before_vh
                vh_mean, vh_std = calculate_mean_and_std(stacked_images_before_vh[grid_id][0])
                if layer.shape == vh_mean.shape:
                    vh_flood_extent.append(np.nan_to_num(((layer - vh_mean) / vh_std) > -3, 0))
                else:
                    min_shape = np.minimum(layer.shape, vh_mean.shape)
                    # Resize the larger array to match the minimum shape
                    layer = np.resize(layer, min_shape)
                    vh_mean = np.resize(vh_mean, min_shape)
                    vh_std = np.resize(vh_std, min_shape)
                    vh_flood_extent.append(np.nan_to_num(((layer - vh_mean) / vh_std) > -3, 0))

                # Append the image_id, grid_id, and geometry_id to their respective lists
                image_ids_vh.append(image_id)
                grid_ids_vh.append(grid_id)
                geometry_ids_vh.append(geometry_id)
            except KeyError:
                print(f"grid_id {grid_id} not found in Dry period!")
                continue
    # Return the updated vh_flood_extent list, image IDs, grid IDs, and geometry IDs
    return vh_flood_extent, image_ids_vh, grid_ids_vh, geometry_ids_vh
