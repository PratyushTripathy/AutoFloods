# ifmiap/__init__.py


import rioxarray
import numpy as np

import geopandas as gpd

nearest_dates = []


def preprocess(catalog, bbox_of_interest, start_date, end_date, time_of_interest_date):
    """
        This function performs the preprocessing steps to obtain relevant data for the time of interest.

        Args:
            catalog (Catalog): A catalog object providing access to the data collections.
            bbox_of_interest (dict): A dictionary defining the bounding box of interest.
            start_date (datetime.date): The start date of the date range for data retrieval.
            end_date (datetime.date): The end date of the date range for data retrieval.
            time_of_interest_date (datetime.date): The specific date of interest.

        Returns:
            sorted_collections (list): A list of data items sorted based on their proximity to the time of interest.
            df_before (GeoDataFrame): A GeoDataFrame containing the features of the retrieved data items.

        """
    # Search for data items within the specified date range and bounding box
    search_before = catalog.search(
        collections=["sentinel-1-grd"],
        intersects=bbox_of_interest,
        datetime=(str(start_date), str(end_date)),
    )
    # Get the item collection from the search results
    items_after = search_before.item_collection()
    # Convert the item collection to a GeoDataFrame
    df_before = gpd.GeoDataFrame.from_features(items_after.to_dict())
    # Sort the data items based on their proximity to the time of interest
    sorted_collections = sorted(items_after, key=lambda x: abs(x.datetime.date() - time_of_interest_date))
    return sorted_collections,df_before



def process_nearest_date(catalog, sorted_collections, time_of_interest_date, bbox_of_interest):
    """
        This function processes the data from the nearest date to the time of interest.

        Args:
            catalog (Catalog): A catalog object providing access to the data collections.
            sorted_collections (list): A list of data items sorted based on their proximity to the time of interest.
            time_of_interest_date (datetime.date): The specific date of interest.
            bbox_of_interest (dict): A dictionary defining the bounding box of interest.

        Returns:
            vh_during_stack (ndarray): Stacked Vh data during the flood.
            vv_during_stack (ndarray): Stacked Vv data during the flood.
            data_array (DataArray): A DataArray object representing one of the Vh data arrays.

        """
    # Check if any datasets are available
    if len(sorted_collections) > 0:
        nearest_collection = sorted_collections[0]
        nearest_date = nearest_collection.datetime.date()
        print(f"Data is not available for {time_of_interest_date}. Using data from the nearest date: {nearest_date}")

        # Check if the nearest date is the same as the previous nearest date
        if nearest_date in nearest_dates:
            # New nearest date is equal to the previous nearest date, skip processing
            print("New nearest date is equal to the previous nearest date, skip processing")
            return None, None, None    # Skip the rest of the code and move to the next iteration
        print(nearest_date, "nearest_date")
        # Add the new nearest date to the list
        nearest_dates.append(nearest_date)

        # Access the nearest_collection to retrieve the data
        search_data = catalog.search(
            collections=["sentinel-1-grd"],
            intersects=bbox_of_interest,
            datetime=str(nearest_date),
        )
        items_during = search_data.item_collection()
        print(f"Returned {len(items_during)} Items")

        # Retrieve Vv data during flood
        vv_during = []
        for item in items_during:
            item_id = item.id
            vv_during.append(
                rioxarray.open_rasterio(item.assets["vv"].href, overview_level=2)
                .astype(float)
                .squeeze()
            )
            for data_array in vv_during:
                data_array.attrs['Image_ID'] = item_id

        # Retrieve Vh data during flood
        vh_during = []
        for item in items_during:
            item_vh_id = item.id
            vh_during.append(
                rioxarray.open_rasterio(item.assets["vh"].href, overview_level=2)
                .astype(float)
                .squeeze()
            )
            for data_array in vh_during:
                data_array.attrs['Image_ID'] = item_vh_id

        # Print the attributes of each DataArray in the list during Flood
        for data_array in vh_during:
            print(data_array.Image_ID)

        # Find the smallest shape in the list during Flood
        smallest_shape_vv_during = min([arr.shape for arr in vv_during])
        resized_array_list = [np.resize(arr, smallest_shape_vv_during) for arr in vv_during]

        # Stacked VV during flood
        vv_during_stack = np.stack(resized_array_list)
        # print(vv_during_stack, "vv during flood stacked")

        # Find the smallest shape in the list
        smallest_shape_vh_during = min([arr.shape for arr in vh_during])
        resized_array_list = [np.resize(arr, smallest_shape_vh_during) for arr in vh_during]

        # Stacked Vh during flood
        vh_during_stack = np.stack(resized_array_list)
        # print(vh_during_stack, "vh during flood stacked")
        return vh_during_stack, vv_during_stack, data_array
