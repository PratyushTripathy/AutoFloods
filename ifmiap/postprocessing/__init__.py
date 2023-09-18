# ifmiap/postprocessing/__init__.py

# import required libraries
import rioxarray
import rasterio.features
from shapely.geometry import shape
import geopandas as gpd
from shapely.geometry import box
import pandas as pd

# define a function to polygonize the flood raster
def polygonize_flood_raster(flood_data):
    """
    Polygonize Flood Raster Data

    This function takes a raster data representing flood extents and polygonizes it to create a
    GeoDataFrame containing polygons representing flooded areas.

    Parameters
    __________
    flood_data                  : xarray.DataArray
                                  A DataArray containing raster flood data.

    Returns
    _______
    geopandas.GeoDataFrame      : A GeoDataFrame containing polygons representing flooded areas.

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



