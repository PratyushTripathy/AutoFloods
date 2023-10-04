# ifmiap/__init__.py

import geopandas as gpd
import ifmiap.utils
import ifmiap.preprocessing
import ifmiap.postprocessing
import ifmiap.mapfloods
from datetime import datetime
import os, json, shutil
import xarray as xr
from matplotlib import pyplot as plt

__version__ = '2023.9.1'
DEM_OUTFILE = r'nasadem_aoi_id.nc'
JSON_OUTFILE = r'../resources/.json'
NC_OUTFILE = f'../output/mean_std/aoi_id_vv_vh_mean_std.nc'
FLOOD_RASTER_OUTFILE = f'../output/flood_raster/floodextent_id.tif'
FLOOD_VECTOR_OUTFILE = f'../output/flood_vector/floodextent_id.gpkg'
FLOOD_MAP_OUTFILE = f'../output/flood_image/floodmap_id.png'
FLOOD_DURATION_RASTER_OUTFILE = f'../output/flood_duration_raster/floodduration_id.tif'
FLOOD_COUNT_RASTER_OUTFILE = f'../output/flood_count_raster/floodcount_id.tif'
FOLDERS_TO_CREATE = [
    r'../output/',
    r'../output/mean_std',
    r'../output/flood_raster',
    r'../output/flood_vector',
    r'../output/flood_image',
    r'../output/final_output',
    r'../output/flood_duration_raster',
    r'../output/flood_count_raster',
    r'../resources/dem'
]


# create a class object to bring together all the pieces
class flood_mapper():

    def __init__(self, grid_shapefile, grid_id_list, dry_date_col='dry_month', id_col='ID',
                 dry_years=list(range(2015, 2021)), wet_duration=['2020/07', '2020/09'], dem_dir=None):
        self.grid_shapefile_path = grid_shapefile
        self.selected_grid_id = grid_id_list
        self.id_key = id_col
        self.dry_date_col = dry_date_col
        self.dry_years = range(min(dry_years), max(dry_years)+1)
        self.wet_dates = [utils.string_to_date_range(wet_duration[0], wet_duration[1])]
        self.dem_dir = dem_dir
        self.create_out_dirs()
        self.generate_defaults()

    def create_out_dirs(self):
        for folder in FOLDERS_TO_CREATE:
            if not os.path.exists(folder):
                os.mkdir(folder)

    def generate_defaults(self):
        # this is for searching scenes
        self.aoi_union = utils.gpd_to_json(id_list=self.selected_grid_id, infile=self.grid_shapefile_path,
                                           separate=False, id_key=self.id_key)

        # this is to seggregate the search
        self.aoi_list = utils.gpd_to_json(id_list=self.selected_grid_id, infile=self.grid_shapefile_path,
                                          separate=True, id_key=self.id_key)

        # create timestamp to use later
        self.timestamp = datetime.now().strftime('%y%m%d_%H%M')

        # define export filenames
        dry_year_begin = min(self.dry_years)
        dry_year_end = max(self.dry_years)
        self.dry_aoi_scene_json_file = JSON_OUTFILE.replace('.json', f'{dry_year_begin}_{dry_year_end}_dry_aoi_scene.json')
        self.dry_scene_aoi_json_file = JSON_OUTFILE.replace('.json', f'{dry_year_begin}_{dry_year_end}_dry_scene_aoi.json')
        self.wet_aoi_scene_json_file = JSON_OUTFILE.replace('.json', f'{dry_year_begin}_{dry_year_end}_wet_aoi_scene.json')
        self.wet_scene_aoi_json_file = JSON_OUTFILE.replace('.json', f'{dry_year_begin}_{dry_year_end}_wet_scene_aoi.json')
        self.nc_outfile = NC_OUTFILE.replace('aoi_', f'{dry_year_begin}_{dry_year_end}_aoi_')

        # read previously exported json and get IDs of already processed polygons
        self.old_dry_aoi_scene_dict = dict()
        self.already_processed_aoi_ids = []
        if os.path.exists(self.dry_aoi_scene_json_file):
            with open(self.dry_aoi_scene_json_file) as f:
                self.old_dry_aoi_scene_dict = json.load(f)
            self.already_processed_aoi_ids = [
                int(item)
                for item in list(self.old_dry_aoi_scene_dict.keys())
                if int(item) in self.selected_grid_id
            ]

            print(
                'Following previously processed AOI IDs found. Will be skipped. If you are running with a new year range, consider modifying or deleting the json file.\n',
                self.already_processed_aoi_ids
            )

    def get_dry_dates(self):
        """
        This method extract dry months from the attributes of the shapefile.
        Returns:

        """

        gdf = gpd.read_file(self.grid_shapefile_path)
        gdf = gdf.loc[gdf[self.id_key].isin(self.selected_grid_id)]

        # remove polygons that have already been processed to save time
        if not self.already_processed_aoi_ids == None:
            gdf = gdf.loc[~gdf[self.id_key].isin(self.already_processed_aoi_ids)]

        self.aoi_ids_to_process = gdf[self.id_key].tolist()

        # extract dry months from the grid shapefile
        self.dry_months = gdf[
            [self.id_key, self.dry_date_col]
        ].groupby(self.id_key).first()[self.dry_date_col].to_dict()

        # convert months to number
        self.dry_months = {
            key: sorted([
                int(item)
                for item in self.dry_months[key].split(',')
                ])
            for key in self.dry_months.keys()
        }

    def generate_dry_date_ranges(self):
        """
        This method generates a list of date ranges in a well formatted way for next steps.

        This method runs under the assumption that the dry date across the given
        list of IDs of the grid polygons are the same. If the months are different,
        the earlier and latest month of all grid polygons will be used as the dry
        duration.
        Args:
            years:

        Returns:

        """

        month_start = min([
            item
            for list_item in self.dry_months.values()
            for item in list_item
        ])
        month_end = max([
            item
            for list_item in self.dry_months.values()
            for item in list_item
        ])

        self.dry_dates = [
            utils.string_to_date_range(
                f'{year}/{month_start:02d}',
                f'{year}/{month_end:02d}'
            )
            for year in self.dry_years
            ]

    def get_s1_items(self, dry_wet='dry', verbose=False):
        if dry_wet == 'dry':
            dates = self.dry_dates
        elif dry_wet == 'wet':
            dates = self.wet_dates

        s1_scenes = {
            aoi['properties'][self.id_key]:
            [
                item2
                for item1 in[
                utils.search_sentinel_data(
                    bbox=aoi,
                    start_date=date_start,
                    end_date=date_end
                )
                for (date_start, date_end) in dates
                ]
                for item2 in item1
                ]
            for aoi in self.aoi_union
            }

        if dry_wet == 'dry':
            self.dry_s1_scenes = s1_scenes
        elif dry_wet == 'wet':
            self.wet_s1_scenes = s1_scenes

        # generate scene_aoi dictionaries
        self.generate_scene_aoi_dict(dry_wet=dry_wet, verbose=verbose)

    def generate_scene_aoi_dict(self, dry_wet='dry', verbose=False):
        if dry_wet == 'dry':
            aoi_scene_dict, scene_aoi_dict = utils.seggregate_sentinel_search(
                self.aoi_list, self.dry_s1_scenes
            )
            self.dry_s1_scenes = [item for item in list(self.dry_s1_scenes.values())[0]]
            aoi_scene_json_file = self.dry_aoi_scene_json_file
            scene_aoi_json_file = self.dry_scene_aoi_json_file
        elif dry_wet == 'wet':
            aoi_scene_dict, scene_aoi_dict = utils.seggregate_sentinel_search(
                self.aoi_list, self.wet_s1_scenes
            )
            self.wet_s1_scenes = [item for item in list(self.wet_s1_scenes.values())[0]]
            aoi_scene_json_file = self.wet_aoi_scene_json_file
            scene_aoi_json_file = self.wet_scene_aoi_json_file

        # export both aoi_scene and scene_aoi dictionaries
        ## neatly hand and export aoi_scene_json first
        skipped_ids = list()
        final_json_data = aoi_scene_dict
        if os.path.exists(aoi_scene_json_file):
            with open(aoi_scene_json_file) as f:
                old_content = json.load(f)

                # remove IDs in the old_json that are in the current search (deleting duplicates)
                for key in aoi_scene_dict.keys():
                    if str(key) in old_content.keys():
                        del old_content[str(key)]
                        skipped_ids.append(key)

                # merge new info with old content
                final_json_data = old_content | aoi_scene_dict

        # sort the dictionaries and the json files
        final_json_data = {int(key): final_json_data[key] for key in final_json_data.keys()}
        final_json_data = {key: sorted(final_json_data[key]) for key in sorted(final_json_data)}

        with open(aoi_scene_json_file, 'w') as f:
            f.write(
                json.dumps(
                    final_json_data, indent=4)
            )
            f.write('\n')

        # repeat the same process for scene_aoi_json
        final_json_data = scene_aoi_dict
        if os.path.exists(scene_aoi_json_file):
            with open(scene_aoi_json_file) as f:
                old_content = json.load(f)

            # remove IDs in the old_json that are in the current search
            for key in scene_aoi_dict.keys():
                if key in old_content.keys():
                    scene_aoi_dict[key] = list(set(
                        scene_aoi_dict[key] + old_content[key]
                    ))
                    del old_content[key]

            final_json_data = old_content | scene_aoi_dict

        # sort the dictionaries and the json files
        final_json_data = {key: sorted(final_json_data[key]) for key in sorted(final_json_data)}

        with open(scene_aoi_json_file, 'w') as f:
            f.write(
                json.dumps(
                    final_json_data, indent=4)
            )
            f.write('\n')


        if verbose == True:
            for id in self.selected_grid_id:
                print(
                    f'ID: {id}, {len(aoi_scene_dict[id])} scenes found.'
                )

        if dry_wet == 'dry':
            self.dry_aoi_scene_dict, self.dry_scene_aoi_dict = aoi_scene_dict, scene_aoi_dict
            self.dry_skipped_ids = skipped_ids
        elif dry_wet == 'wet':
            self.wet_aoi_scene_dict, self.wet_scene_aoi_dict = aoi_scene_dict, scene_aoi_dict
            self.wet_skipped_ids = skipped_ids

    def read_scenes(self, dry_wet='dry', overview_level=3):
        if dry_wet == 'dry':
            s1_scenes = self.dry_s1_scenes
        elif dry_wet == 'wet':
            s1_scenes = self.wet_s1_scenes

        # read and reproject the images all at once, not separately for each tile
        s1_combined = {
            stac_id: stac_ds
            for item in s1_scenes
            for (stac_id, stac_ds) in [preprocessing.read_sentinel1_stac(item, overview_level)]
        }

        if dry_wet == 'dry':
            self.s1_dry_dict = s1_combined
        elif dry_wet == 'wet':
            self.s1_wet_dict = s1_combined

    def generate_mean_std_by_aoi(self):
        # separate and clip the reprojected image for each tile
        reprojected_clipped_dry = {
            id: preprocessing.reproject_clip_stac(self.s1_dry_dict, self.dry_aoi_scene_dict,
                                                  self.grid_shapefile_path, id)
            for id in self.selected_grid_id
            if not id in self.already_processed_aoi_ids
        }

        # stack the reprojected and clipped images
        self.stacked_dry = {
            id: preprocessing.stack_images(reprojected_clipped_dry[id], self.grid_shapefile_path, id)
            for id in reprojected_clipped_dry
            }

        # calculate cell level mean and std for the dry scenes
        self.mean_std_by_aoi = {
            id: xr.concat(
                [
                    self.stacked_dry[id]['vv_stack'].mean(axis=0),
                    self.stacked_dry[id]['vv_stack'].std(axis=0),
                    self.stacked_dry[id]['vh_stack'].mean(axis=0),
                    self.stacked_dry[id]['vh_stack'].std(axis=0)
                ], dim='band').assign_coords(
                band =['vv_mean', 'vv_std', 'vh_mean', 'vh_std']
                )
            for id in self.stacked_dry.keys()
        }
        self.stacked_dry = None


        # save the cell level mean and std to different files (separate for each ID)
        for id in self.mean_std_by_aoi:
            outfile = self.nc_outfile.replace('_id_', f'_{id}_')
            self.mean_std_by_aoi[id].to_netcdf(outfile)

        self.load_mean_std_by_aoi()

    def load_mean_std_by_aoi(self):
        if not hasattr(self, 'mean_std_by_aoi'):
            self.mean_std_by_aoi = dict()

        # Read the mean and std stacked raster if already processed
        # and save in the existing self.mean_std_by_aoi dictionary
        for id in self.already_processed_aoi_ids:
            if not int(id) in self.mean_std_by_aoi.keys():
                infile = self.nc_outfile.replace('_id_', f'_{id}_')
                try:
                    self.mean_std_by_aoi[int(id)] = xr.load_dataarray(infile)
                    print(f'Previously processed {infile} read successfully!')
                except:
                    print(f'{infile} present in JSON as processed but .nc file is missing.')

    def prepare_dem(self, dem_overview=1, nodata=0.0):
        dem_id_to_process = [
            id
            for id in self.selected_grid_id
            if not os.path.exists(os.path.join(
                self.dem_dir, DEM_OUTFILE.replace('_id.nc', f'_{id}.nc')
            ))
        ]

        if len(dem_id_to_process) > 0:
            # download DEM for select IDs
            bbox_for_dem_download = utils.gpd_to_json(id_list=dem_id_to_process, infile=self.grid_shapefile_path,
                                                      separate=False, id_key=self.id_key)
            dem_merged_xarray = ifmiap.utils.download_nasadem(bbox_for_dem_download[0], overview_level=dem_overview, nodata=nodata)

            # for each of the ids, clip dem and export to the .nc file
            self.dem = dict()
            for id in dem_id_to_process:
                print(f'DEM for tile ID {id} not found. Downloading...')
                self.dem[id] = ifmiap.preprocessing.clip_xarray_using_id(
                    data_xarray=dem_merged_xarray,
                    grid_shapefile_path=self.grid_shapefile_path,
                    aoi_id=id,
                    ref_xarray=self.mean_std_by_aoi[id]
                )

                outfile = os.path.join(self.dem_dir, DEM_OUTFILE.replace('_id.nc', f'_{id}.nc'))
                ifmiap.utils.export_xarray(self.dem[id], outfile)

        else: # load the dem if already present
            for id in self.selected_grid_id:
                print(f'DEM for tile ID {id} found, will not be downloaded.')
                self.dem = {
                    id: xr.load_dataarray(os.path.join(self.dem_dir, DEM_OUTFILE.replace('_id.nc', f'_{id}.nc')), engine='rasterio')
                    for id in self.selected_grid_id
                }

    def prepare_wet_scenes(self, overview_level=3):
        # call the previouisly defined method to get S1 scenes for the wet period
        self.get_s1_items(dry_wet='wet')
        self.read_scenes(dry_wet='wet', overview_level=overview_level)

        # loop through each id and clip every S1 wet scene
        self.wet_scenes_by_aoi = {
            id: {
                scene_id: xr.concat(
                    [
                        ifmiap.preprocessing.clip_xarray_using_id(
                            data_xarray=self.s1_wet_dict[scene_id]['vv_ds'],
                            grid_shapefile_path=self.grid_shapefile_path,
                            aoi_id=id,
                            ref_xarray=self.mean_std_by_aoi[id]
                        ),
                        ifmiap.preprocessing.clip_xarray_using_id(
                            data_xarray=self.s1_wet_dict[scene_id]['vh_ds'],
                            grid_shapefile_path=self.grid_shapefile_path,
                            aoi_id=id,
                            ref_xarray=self.mean_std_by_aoi[id]
                        )
                    ], dim='band').assign_coords(
                    band =['vv_ds', 'vh_ds']
                    )
                for scene_id in self.wet_aoi_scene_dict[id]
            }
            for id in self.wet_aoi_scene_dict
        }

    def map_floods(self, vv_thd=2.5, vh_thd=2.5, dem_thd=600, slp_thd=25, export_raster=False, export_vector=False, export_maps=False):
        self.flood_dict = mapfloods.map_floods(
            mean_std_by_aoi=self.mean_std_by_aoi,
            wet_scenes_by_aoi=self.wet_scenes_by_aoi,
            dem_path=os.path.join(self.dem_dir, DEM_OUTFILE),
            vv_thd=vv_thd,
            vh_thd=vh_thd,
            dem_thd=dem_thd,
            slp_thd=slp_thd
        )

        dry_year_begin = min(self.dry_years)
        dry_year_end = max(self.dry_years)

        # export the flood rasters
        if export_raster == True:
            for id in self.flood_dict:
                for scene_id in self.flood_dict[id]:
                    outfile_flood = FLOOD_RASTER_OUTFILE.replace('_id.tif', f'_{dry_year_begin}_{dry_year_end}_{id}_{"_".join(scene_id.split("_")[4:])}.tif')
                    ifmiap.utils.export_xarray(self.flood_dict[id][scene_id], outfile_flood)

        # polygonize the flood rasters
        if export_vector == True:
            self.flood_gdf_dict = {
                id: {
                    scene_id: ifmiap.postprocessing.polygonize_flood_raster(self.flood_dict[id][scene_id])
                    for scene_id in self.flood_dict[id]
                }
                for id in self.flood_dict
            }

            for id in self.flood_dict:
                for scene_id in self.flood_dict[id]:
                    outfile_flood = FLOOD_VECTOR_OUTFILE.replace('_id.gpkg', f'_{dry_year_begin}_{dry_year_end}_{id}_{"_".join(scene_id.split("_")[4:])}.gpkg')
                    # export only if the GDF has any flood cells
                    if self.flood_gdf_dict[id][scene_id].shape[0] > 0:
                        self.flood_gdf_dict[id][scene_id].to_crs("EPSG:4326").to_file(outfile_flood, index=False)
                    else:
                        # remove that scene_id from the dictionary to avoid exporting map later
                        del self.flood_gdf_dict[id][scene_id]
                        print(f'Flood cells not found in {id}_{scene_id}.')

        # export flood maps as images
        if export_maps == True:
            for id in self.flood_dict:
                for scene_id in self.flood_dict[id]:
                    outfile_flood = FLOOD_MAP_OUTFILE.replace('_id.png', f'_{dry_year_begin}_{dry_year_end}_{id}_{"_".join(scene_id.split("_")[4:])}.png')

                    mapfloods.flood_images(
                        flood_xarray=self.flood_dict[id][scene_id],
                        outfile_flood=outfile_flood
                    )

    def get_duration_count(self, export_raster=False):
        self.flood_max_duration_dict = dict()
        self.unique_flood_events_count_dict = dict()

        for id in self.flood_dict:
            flood_3d = ifmiap.utils.flood_data_3dstack(self.flood_dict[id])
            max_durations, unique_event_counts = ifmiap.postprocessing.flood_duration_count(flood_3d)

            self.flood_max_duration_dict[id] = utils.numpy_to_xarray(max_durations, list(self.flood_dict[id].values())[0])
            self.unique_flood_events_count_dict[id] = utils.numpy_to_xarray(unique_event_counts, list(self.flood_dict[id].values())[0])

        # export the flood rasters
        dry_year_begin = min(self.dry_years)
        dry_year_end = max(self.dry_years)
        if export_raster == True:
            for id in self.flood_max_duration_dict:
                outfile_duration = FLOOD_DURATION_RASTER_OUTFILE.replace('_id.tif',
                                                                         f'_{dry_year_begin}_{dry_year_end}_{id}.tif')
                outfile_count = FLOOD_COUNT_RASTER_OUTFILE.replace('_id.tif',
                                                                         f'_{dry_year_begin}_{dry_year_end}_{id}.tif')

                ifmiap.utils.export_xarray(self.flood_max_duration_dict[id], outfile_duration)
                ifmiap.utils.export_xarray(self.unique_flood_events_count_dict[id], outfile_count)

    def flush_output(self, remove_dem=False):
        if remove_dem:
            folders_to_delete = FOLDERS_TO_CREATE[:]
        else:
            folders_to_delete = FOLDERS_TO_CREATE[:-1]

        # add json files to the delete list
        folders_to_delete = folders_to_delete + [
            self.dry_aoi_scene_json_file,
            self.dry_scene_aoi_json_file,
            self.wet_aoi_scene_json_file,
            self.wet_scene_aoi_json_file
        ]

        for folder in folders_to_delete:
            if os.path.isfile(folder):
                os.remove(folder)
            elif os.path.exists(folder):
                shutil.rmtree(folder, ignore_errors=True)





