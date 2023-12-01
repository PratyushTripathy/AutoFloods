# ifmiap/postprocessing/__init__.py

# import required libraries
import os
import rasterio
from shapely.geometry import shape
import geopandas as gpd
import numpy as np

# define a function to polygonize the flood raster
def polygonize_flood_raster(flood_data):
    """
    Polygonize Flood Raster Data

    This function takes a raster data representing flood extents and polygonizes it to create a
    GeoDataFrame containing polygons representing flooded areas.

    Parameters
    __________
    flood_data (xarray.DataArray)       : A DataArray containing raster flood data.

    Returns
    _______
    geopandas.GeoDataFrame              : A GeoDataFrame containing polygons representing flooded areas.

    """
    data = flood_data.astype('uint8')
    shapes_gen = rasterio.features.shapes(data.values, mask=data.values != 0, transform=data.rio.transform())
    polygons = [shape(geom) for geom, value in shapes_gen if value == 3]  # Adjust the condition as needed

    # create a GPD GDF from the polygons
    gdf = gpd.GeoDataFrame({'geometry': polygons})

    # assign a CRS (Coordinate Reference System) to the GDF
    gdf.crs = data.rio.crs.to_string()

    """
    extend the polygons at the edges because there is gap between tiles
    """
    """
    # get the cell size from the raster layer
    cell_size = float(flood_data.spatial_ref.GeoTransform.split(' ')[1])

    # buffer the polygons
    gdf_buffer = gdf.buffer(cell_size * 1.5)

    # extract the total extent of GDF as a GDF
    gdf_extent = box(*gdf.total_bounds)

    # create buffer of the polygon layer and dissolve it
    gdf_buffer = pd.concat([
        gpd.GeoDataFrame(geometry=gdf_buffer, crs=gdf.crs),
        gpd.GeoDataFrame(geometry=[gdf_extent])
    ], axis=0).dissolve()

    # perform spatial difference of the buffer and the total extent
    gdf_extra = gdf_buffer.symmetric_difference(gdf_extent)

    # merge the extra part with the original GDF
    gdf = pd.concat([
        gdf, gpd.GeoDataFrame(geometry=gdf_extra, crs=gdf.crs)
    ], axis=0)
    """

    """ 
    The below part was to fix the vertical flip of polygons. 
    That is now fixed at the raster level. This piece of code 
    is not needed any more.
    """
    # for some reason, the polygonised layer is vertically flipped with y min as the origin, correct that
    #center_top_point = ((gdf.total_bounds[0] + gdf.total_bounds[2]) / 2, gdf.total_bounds[3])
    #gdf['geometry'] = gdf['geometry'].scale(xfact=1, yfact=-1, origin=center_top_point)

    return gdf

def flood_duration_count(stacked_flood_data):
    max_durations = np.zeros_like(stacked_flood_data[0, :, :], dtype=int)
    unique_event_counts = np.zeros_like(stacked_flood_data[0, :, :], dtype=int)

    for x in range(stacked_flood_data.shape[1]):
        for y in range(stacked_flood_data.shape[2]):
            flood_data = stacked_flood_data[:, x, y]
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

def aggregate_monthly(infile, outfile=None):
    # extract year and month information from the file name
    yearmonthtag = '_'.join(infile.split('_')[1:-1])
    
    # Output GeoTIFF file path for the stacked bands
    if outfile == None:
        outfile = infile.replace('/flood_raster/', f'/flood_raster/monthlyadded_{yearmonthtag}/').replace('.tif', '_monthly.tif')

    # create monthly folder for that month and year
    folder_to_create = os.path.split(outfile)[0]
    if not os.path.exists(folder_to_create):
        os.mkdir(folder_to_create)

    # Open the input GeoTIFF file to get width, height, and metadata
    with rasterio.open(infile) as src:
        width = src.width
        height = src.height
        metadata = src.meta.copy()
        bandnames = src.descriptions

        # Extract unique months from band names
        band_name_patterns = list(set([bandname[:6] for bandname in bandnames]))

        if bandnames[0] != None:
            # Initialize an empty array to store the stacked bands
            stacked_bands = []

            # Process each wildcard pattern
            for band_n, band_name_pattern in enumerate(band_name_patterns):
                # Get band names that match the current wildcard pattern
                selected_band_indices = [i for i, name in enumerate(bandnames, start=1) if band_name_pattern in name]

                # Read selected bands and create a sum for the current pattern
                combined_band = np.zeros((height, width), dtype=np.uint8)  # Assuming dtype is uint8
                for band_index in selected_band_indices:
                    band_arr = src.read(band_index)
                    combined_band += (band_arr == 3).astype(np.uint8)  # Assuming condition for summing is (band_arr == 3)

                # Append the combined band to the list
                stacked_bands.append(combined_band)

            # Stack the bands along the third axis to create a 3-band array
            stacked_array = np.stack(stacked_bands, axis=2)

            # Update metadata for the output file (number of bands and data type)
            metadata.update({
                'count': len(band_name_patterns),  # Number of bands in the output file
                'dtype': np.uint8,  # Data type of the stacked bands (uint8)
                'nodata': 255  # Set the nodata value within the valid range of uint8 (0 to 255)
            })

            # Write the stacked bands to the output GeoTIFF file
            with rasterio.open(outfile, 'w', **metadata) as dst:
                # Write the stacked array as bands in the output file
                for i, band_name in enumerate(band_name_patterns):
                    dst.write(stacked_array[:, :, i], i+1)  # Write each band separately
                    dst.set_band_description(i+1, band_name)  # Set band names

