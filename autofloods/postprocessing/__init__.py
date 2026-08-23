# autofloods/postprocessing/__init__.py

# import required libraries
import os
import rasterio
from shapely.geometry import shape
import geopandas as gpd
import numpy as np

# define a function to polygonize the flood raster
def polygonize_flood_raster(flood_data):
    """
    Vectorize `flood_data`'s high-confidence flood cells (class 3 only --
    see detectors.ZScoreDetector -- not the VV- or VH-only partial-confidence
    classes 1/2) into polygons, CRS matching the input raster. Called from
    map_floods() when export_vector=True (off by default; see
    utils.merge_flood_gdfs for a known OPERA-naming gap in the downstream
    merge step for this vector output).
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
    """
    Per-pixel flood statistics across a (date, y, x) stack of binary/
    truthy flood layers (e.g. flood_mapper.flood_by_date, one boolean
    layer per observed date -- NOT resampled to a fixed daily cadence, so
    "duration" here means consecutive OBSERVED flooded dates, not
    consecutive calendar days). Returns (max_durations, unique_event_counts):
    the longest consecutive-flooded-observation run, and how many separate
    flood events (a run bounded by non-flooded observations on both sides)
    occurred, per pixel.
    """
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
    """
    Collapse a per-date flood-classification stack (from
    flood_mapper.merge_floods_by_date(), one band per observed date) into
    one band per calendar month: the COUNT of dates within that month each
    pixel was high-confidence flooded (class 3). Not a binary flag -- a
    pixel flooded on 3 of a month's 5 observed dates gets value 3, not 1.
    `infile`'s band descriptions must be YYYYMMDD-prefixed (as
    merge_floods_by_date() writes them); months are inferred from the
    first 6 characters of each band name. Pixels with zero valid (non-NaN)
    observations all month get nodata (255), not a false 0 -- see the
    inline comment below for why that distinction matters. Default
    `outfile` mirrors `infile`'s path under .../flood_raster/monthlyadded/.
    """
    # extract year and month information from the file name
    yearmonthtag = '_'.join(os.path.split(infile)[-1].split('_')[1:-1])
    
    # Output GeoTIFF file path for the stacked bands
    if outfile == None:
        outfile = infile.replace('/flood_raster/floodextentstacked', f'/flood_raster/monthlyadded').replace('.tif', '_monthly.tif')

    # create monthly folder for that month and year
    folder_to_create = os.path.split(outfile)[0]
    if not os.path.exists(folder_to_create):
        os.makedirs(folder_to_create, exist_ok=True)

    # Open the input GeoTIFF file to get width, height, and metadata
    with rasterio.open(infile) as src:
        width = src.width
        height = src.height
        metadata = src.meta.copy()
        bandnames = src.descriptions

        # Extract unique months from band names, sorted chronologically.
        # set() iteration order is hash-based and varies per process
        # (PYTHONHASHSEED), so without sorting, output band order was
        # arbitrary and inconsistent from run to run -- different tiles
        # in the same batch ended up with different month orderings in
        # their monthly output, which silently breaks any downstream
        # step (e.g. mosaicking) that assumes band position means the
        # same month across files. YYYYMM strings sort chronologically.
        band_name_patterns = sorted(set([bandname[:6] for bandname in bandnames]))

        if bandnames[0] != None:
            # Initialize an empty array to store the stacked bands
            stacked_bands = []

            # Process each wildcard pattern
            for band_n, band_name_pattern in enumerate(band_name_patterns):
                # Get band names that match the current wildcard pattern
                selected_band_indices = [i for i, name in enumerate(bandnames, start=1) if band_name_pattern in name]

                # Read selected bands and create a sum for the current pattern
                combined_band = np.zeros((height, width), dtype=np.uint8)  # Assuming dtype is uint8
                valid_count = np.zeros((height, width), dtype=np.uint16)
                for band_index in selected_band_indices:
                    band_arr = src.read(band_index)
                    combined_band += (band_arr == 3).astype(np.uint8)  # Assuming condition for summing is (band_arr == 3)
                    valid_count += (~np.isnan(band_arr)).astype(np.uint16)

                # A pixel with zero valid (non-NaN) observations across
                # every date in the month has no real information --
                # e.g. a tile-edge interpolation artifact (see
                # preprocessing.clip_xarray_using_id) or a genuine
                # coverage gap. `band_arr == 3` is False for NaN too, so
                # without this check such a pixel would silently read as
                # "flooded on zero dates" (0), indistinguishable from a
                # real, fully-observed non-flood pixel. Write nodata
                # (255) instead wherever nothing valid was ever observed.
                combined_band = np.where(valid_count == 0, 255, combined_band).astype(np.uint8)

                # Append the combined band to the list
                stacked_bands.append(combined_band)

            # Stack the bands along the third axis to create a 3-band array
            stacked_array = np.stack(stacked_bands, axis=2)

            # Update metadata for the output file (number of bands and data type)
            metadata.update({
                'count': len(band_name_patterns),  # Number of bands in the output file
                'dtype': np.uint8,  # Data type of the stacked bands (uint8)
                'nodata': 255,  # Set the nodata value within the valid range of uint8 (0 to 255)
                # Write as a COG rather than plain GeoTIFF -- tiled,
                # compressed, overviews auto-built by GDAL. `driver` from
                # src.meta.copy() above reads back as 'GTiff' even for a
                # COG-written input (that's just how GDAL reports it), so
                # it must be set explicitly here, not inherited. predictor=2
                # (horizontal differencing) is for integer data -- this
                # output is uint8, unlike export_xarray's float32 output
                # which needs predictor=3.
                'driver': 'COG',
                'compress': 'DEFLATE',
                'predictor': 2,
                'overview_resampling': 'nearest',
            })

            # Write the stacked bands to the output GeoTIFF file
            with rasterio.open(outfile, 'w', **metadata) as dst:
                # Write the stacked array as bands in the output file
                for i, band_name in enumerate(band_name_patterns):
                    dst.write(stacked_array[:, :, i], i+1)  # Write each band separately
                    dst.set_band_description(i+1, band_name)  # Set band names

