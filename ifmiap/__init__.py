# ifmiap/__init__.py

import geopandas as gpd
import ifmiap.utils
import ifmiap.preprocessing
from datetime import datetime
import os, json
import xarray as xr

JSON_OUTFILE = r'resources/.json'


# create a class object to bring together all the pieces
class flood_mapper():

    def __init__(self, grid_shapefile, grid_id_list, dry_date_col='dry_month', id_col='ID'):
        self.grid_shapefile_path = grid_shapefile
        self.selected_grid_id = grid_id_list
        self.id_key = id_col
        self.dry_date_col = dry_date_col
        self.generate_defaults()

    def generate_defaults(self):
        # this is for searching scenes
        self.aoi_union = utils.gpd_to_json(id_list=self.selected_grid_id, separate=False, id_key=self.id_key)

        # this is to seggregate the search
        self.aoi_list = utils.gpd_to_json(id_list=self.selected_grid_id, separate=True, id_key=self.id_key)

        # create timestamp to use later
        self.timestamp = datetime.now().strftime('%y%m%d_%H%M')

        # define export filenames
        self.dry_aoi_scene_json_file = JSON_OUTFILE.replace('.json', 'dry_aoi_scene.json')
        self.dry_scene_aoi_json_file = JSON_OUTFILE.replace('.json', 'dry_scene_aoi.json')

        # read previously exported json and get IDs of already processed polygons
        self.old_dry_aoi_scene_dict = dict()
        if os.path.exists(self.dry_aoi_scene_json_file):
            with open(self.dry_aoi_scene_json_file) as f:
                self.old_dry_aoi_scene_dict = json.load(f)
            self.already_processed_aoi_ids = [int(item) for item in list(self.old_dry_aoi_scene_dict.keys())]

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
        gdf = gdf.loc[~gdf[self.id_key].isin(self.already_processed_aoi_ids)]

        self.aoi_ids_to_process = gdf.shape[0]

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

    def generate_dry_date_ranges(self, years=list(range(2015, 2021))):
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
        years = range(min(years), max(years)+1)

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
            for year in years
            ]

    def get_s1_items(self, verbose=False):
        self.dry_s1_scenes = {aoi['properties'][self.id_key]:
        [item2
         for item1 in[
        utils.search_sentinel_data(
            bbox=aoi,
            start_date=dry_date_start,
            end_date=dry_date_end
        )
            for (dry_date_start, dry_date_end) in self.dry_dates
            ]
         for item2 in item1
         ]
            for aoi in self.aoi_union
        }

        # generate scene_aoi dictionaries
        self.generate_scene_aoi_dict(verbose=verbose)

    def generate_scene_aoi_dict(self, verbose=False):
        self.dry_aoi_scene_dict, self.dry_scene_aoi_dict = utils.seggregate_sentinel_search(
            self.aoi_list, self.dry_s1_scenes
        )

        self.dry_s1_scenes = [item for item in list(self.dry_s1_scenes.values())[0]]


        # export both aoi_scene and scene_aoi dictionaries
        ## neatly hand and export aoi_scene_json first
        if os.path.exists(self.dry_aoi_scene_json_file):
            with open(self.dry_aoi_scene_json_file) as f:
                old_content = json.load(f)

            with open(self.dry_aoi_scene_json_file, 'w') as f:
                # remove IDs in the old_json that are in the current search
                for key in self.dry_aoi_scene_dict.keys():
                    if str(key) in old_content.keys():
                        del old_content[str(key)]

                f.write(
                    json.dumps(
                        old_content | self.dry_aoi_scene_dict, indent=4)
                )
                f.write('\n')
        else:
            with open(self.dry_aoi_scene_json_file, 'w') as f:
                f.write(
                    json.dumps(
                        self.dry_aoi_scene_dict, indent=4)
                )
                f.write('\n')

        # repeat the same process for scene_aoi_json
        if os.path.exists(self.dry_scene_aoi_json_file):
            with open(self.dry_scene_aoi_json_file) as f:
                old_content = json.load(f)

            with open(self.dry_scene_aoi_json_file, 'w') as f:
                # remove IDs in the old_json that are in the current search
                for key in self.dry_scene_aoi_dict.keys():
                    if key in old_content.keys():
                        self.dry_scene_aoi_dict[key] = list(set(
                            self.dry_scene_aoi_dict[key] + old_content[key]
                        ))
                        del old_content[key]

                f.write(
                    json.dumps(
                        old_content | self.dry_scene_aoi_dict, indent=4)
                )
                f.write('\n')
        else:
            with open(self.dry_scene_aoi_json_file, 'w') as f:
                f.write(
                    json.dumps(
                        self.dry_scene_aoi_dict, indent=4)
                )
                f.write('\n')


        if verbose == True:
            for id in self.selected_grid_id:
                print(
                    f'ID: {id}, {len(self.dry_aoi_scene_dict[id])} dry scenes found.'
                )

    def read_dry_scenes(self, overview_level=3):
        # read and reproject the images all at once, not separately for each tile
        self.reprojected_dry_combined = {
            stac_id: stac_ds
            for item in self.dry_s1_scenes
            for (stac_id, stac_ds) in [preprocessing.read_reproject(item, overview_level)]
        }

    def generate_mean_std_by_aoi(self):
        # separate and clip the reprojected image for each tile
        reprojected_clipped_dry = {
            id: preprocessing.clip_stac(self.reprojected_dry_combined, self.dry_aoi_scene_dict, id)
            for id in self.selected_grid_id
            if not id in self.already_processed_aoi_ids
        }

        # stack the reprojected and clipped images
        self.stacked_dry = {
            id: preprocessing.stack_images(reprojected_clipped_dry[id], id)
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
            outfile = f"output/aoi_{id}_vv_vh_mean_std.nc"
            self.mean_std_by_aoi[id].to_netcdf(outfile)

        self.load_mean_std_by_aoi()

    def load_mean_std_by_aoi(self):
        if not hasattr(self, 'mean_std_by_aoi'):
            self.mean_std_by_aoi = dict()

        # Read the mean and std stacked raster if already processed
        # and save in the existing self.mean_std_by_aoi dictionary
        for id in self.already_processed_aoi_ids:
            if not int(id) in self.mean_std_by_aoi.keys():
                infile = f"output/aoi_{id}_vv_vh_mean_std.nc"
                try:
                    self.mean_std_by_aoi[int(id)] = xr.load_dataarray(infile)
                    print(f'Previously processed {infile} read successfully!')
                except:
                    print(f'{infile} present in JSON as processed but .nc file is missing.')
