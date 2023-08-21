# ifmiap/__init__.py

import geopandas as gpd
import ifmiap.utils

# create a class object to bring together all the pieces
class flood_mapper():

    def __init__(self, grid_shapefile, grid_id_list, dry_date_col='dry_month', id_col='ID'):
        self.grid_shapefile_path = grid_shapefile
        self.selected_grid_id = grid_id_list
        self.id_key = id_col
        self.dry_date_col = dry_date_col
        self.aoi_union = utils.gpd_to_json(id_list=self.selected_grid_id, separate=False, id_key=self.id_key) # this is for searching scenes
        self.aoi_list = utils.gpd_to_json(id_list=self.selected_grid_id, separate=True, id_key=self.id_key) # this is to seggregate the search

    def get_dry_dates(self):
        """
        THis method extract dry months from the attributes of the shapefile.
        Returns:

        """

        gdf = gpd.read_file(self.grid_shapefile_path)
        gdf = gdf.loc[gdf[self.id_key].isin(self.selected_grid_id)]

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

    def get_s1_items(self):
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



