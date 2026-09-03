# autofloods/__init__.py

"""
Core orchestrator for the AutoFloods SAR flood-mapping pipeline.
See flood_mapper for the pipeline entry point and method call order.
"""

from importlib.metadata import version as _pkg_version

import geopandas as gpd
import autofloods.utils
import autofloods.preprocessing
import autofloods.postprocessing
import autofloods.mapfloods
import autofloods.sources as sources
import autofloods.detectors as detectors
import autofloods.grid as grid
from datetime import datetime
import os, json, shutil
import concurrent.futures
import xarray as xr
import numpy as np
import rasterio
from rasterio.enums import Resampling

__version__ = _pkg_version("autofloods")
#DEM_OUTFILE = r'nasadem_aoi_id.nc'
SLOPE_OUTFILE = r'slope_aoi_id.nc'

# NOTE: the path templates below (JSON, mean/std, flood raster/vector/image
# outputs, and the folder list) are no longer module-level constants -- they
# are computed per-instance in flood_mapper.generate_defaults() from
# self.output_dir (or the historical '../output' / '../resources' relative
# paths if output_dir isn't given), so that every flood_mapper instance can
# write to its own isolated location. See generate_defaults() for the
# instance attributes this produces: self.output_base, self.resources_base,
# self.folders_to_create, and the *_outfile template attributes.


# create a class object to bring together all the pieces
class flood_mapper():
    """
    Orchestrates the full Z-score SAR flood-mapping pipeline for one or
    more AOI tiles: search + read dry-season Sentinel-1 scenes, fit a
    per-pixel VV/VH baseline, compute a terrain-slope mask, search + read
    wet-season scenes, classify them against the baseline, and aggregate
    the result into per-date, per-month, and per-scene-count rasters.

    Methods are called in this order for a full run (see scripts/
    run_autofloods.py or scripts/bihar2024_tile.py for a working example):

        get_dry_dates() -> generate_dry_date_ranges() -> get_s1_items('dry')
        -> read_scenes('dry') -> generate_mean_std_by_aoi() -> prepare_slope()
        -> prepare_wet_scenes() -> map_floods() -> merge_floods_by_date()
        -> generate_number_of_scenes() -> monthly_sum()

    Each stage's output is cached to disk under `output_dir` (mean/std as
    NetCDF, everything else as GeoTIFF/COG) and reloaded rather than
    recomputed on a later run for the same tile/date range -- see
    is_fully_processed()/expected_monthly_outfile() to check a tile's
    completion status upfront, and already_processed_aoi_ids (set in
    generate_defaults()) for the per-baseline resume mechanism.

    Swappable via `source`/`detector`: any autofloods.sources.STACSource
    (MPCSource, OPERASource) and any autofloods.detectors.FloodDetector
    (currently only ZScoreDetector) plug in without changing this class.
    """

    def __init__(self, grid_shapefile=None, grid_id_list=None, dry_date_col='dry_month', id_col='ID',
                 dry_years=list(range(2015, 2021)), wet_duration=['2020/07', '2020/09'], slope_dir=None,
                 source=None, detector=None, output_dir=None, cell_size=30,
                 aoi=None, grid_mode=None, grid_tile_size_km=None, grid_dry_months=None):
        """
        Construct a flood_mapper and create its output directory tree.
        See the module-level pipeline overview in this class's docstring
        for the full method call sequence.

        Parameters
        __________
        grid_shapefile                  : str, optional
                                          Path to the shapefile containing grid information.
                                          Required unless `aoi` is given instead, in which case
                                          a grid is generated on the fly (see `aoi` below).
        grid_id_list                    : list, optional
                                          List of grid IDs to process. Required when
                                          `grid_shapefile` is given; if omitted when a grid is
                                          generated from `aoi`, defaults to every generated tile.
        dry_date_col                    : str
                                          Column name for dry month information in the shapefile (default: 'dry_month').
        id_col                          : str
                                          Column name for grid IDs in the shapefile (default: 'ID').
        dry_years                       : list
                                          List of dry years to consider (default: range from 2015 to 2020).
        wet_duration                    : list
                                          List of wet duration in the format 'YYYY/MM' (default: ['2020/07', '2020/09']).
        slope_dir                       : str
                                          Directory where slope calculated from the downloaded DEM will be stored. (default: None).
        source                          : autofloods.sources.STACSource
                                          STAC data source to search and retrieve imagery from
                                          (default: autofloods.sources.MPCSource()).
        detector                        : autofloods.detectors.FloodDetector
                                          Flood detection backend to fit a baseline and classify
                                          scenes with (default: autofloods.detectors.ZScoreDetector()).
        output_dir                      : str
                                          Root directory for all outputs and cache files (JSON
                                          scene caches, mean/std layers, slope masks, flood
                                          rasters/vectors/images). If None (default), preserves
                                          the historical behavior of writing under '../output/'
                                          and '../resources/' relative to the working directory.
                                          If set, everything is written under this single path
                                          instead -- intended for isolating concurrent SLURM jobs
                                          from each other, one output_dir per job.
        cell_size                       : int
                                          Pixel size in meters (in each tile's own UTM zone)
                                          that every reprojection in the pipeline is forced
                                          onto, regardless of source scene resolution -- keeps
                                          a tile's grid identical across runs/years (default
                                          30 matches OPERA RTC-S1's native resolution; see
                                          preprocessing.clip_xarray_using_id for details).
        aoi                              : str, geopandas.GeoDataFrame, or shapely geometry, optional
                                          Area of interest boundary to generate a tiling grid
                                          from, instead of supplying a pre-made `grid_shapefile`
                                          (see autofloods.grid.generate_grid, which this calls
                                          internally). Either `grid_shapefile` or `aoi` must be
                                          given, not neither. The generated grid is written to
                                          `resources_base/generated_grid.gpkg` and used as if it
                                          had been passed via `grid_shapefile`.
        grid_mode                       : str, optional
                                          'mgrs' or 'utm_fishnet', passed to generate_grid when
                                          `aoi` is given. Defaults to 'mgrs' if `source` is an
                                          OPERASource (OPERA RTC-S1 is natively MGRS-tiled),
                                          otherwise 'utm_fishnet'.
        grid_tile_size_km               : float, optional
                                          Tile size for grid_mode='utm_fishnet', passed to
                                          generate_grid when `aoi` is given. Ignored for
                                          grid_mode='mgrs' (always 100km).
        grid_dry_months                 : str, optional
                                          Dry-season months (e.g. "04,05") stamped into every
                                          generated tile's `dry_date_col`, passed to
                                          generate_grid when `aoi` is given. Required in that
                                          case -- dry season is climate knowledge that can't be
                                          derived from AOI geometry alone.

        Sets aoi_union/aoi_list (combined and per-AOI search bboxes), normalizes
        dry_years to a contiguous range, and populates already_processed_aoi_ids
        (AOIs whose dry-season baseline already exists on disk, set in
        generate_defaults()).

        `grid_shapefile` must have columns `id_col` (AOI ID), `dry_date_col`
        (comma-separated dry months, e.g. "04,05"), and `zone` (UTM zone number
        as a string, e.g. "45R") -- every tile's UTM target derives from `zone`.

        Examples
        --------
        >>> from autofloods import flood_mapper
        >>> fm = flood_mapper(
        ...     grid_shapefile='resources/india_utm_fishnet_buffer.gpkg',
        ...     grid_id_list=[321],
        ...     dry_years=[2024, 2024],
        ...     slope_dir='resources/slope/',
        ...     wet_duration=['2024/07', '2024/10'],
        ... )
        """
        self.id_key = id_col
        self.dry_date_col = dry_date_col
        self.source = source if source is not None else sources.MPCSource()
        self.detector = detector if detector is not None else detectors.ZScoreDetector(vv_thd=-2.5, vh_thd=-2.5)
        self.cell_size = cell_size
        self.dry_years = range(min(dry_years), max(dry_years)+1)
        self.wet_dates = [utils.string_to_date_range(wet_duration[0], wet_duration[1])]
        self.wet_yearmonths = [date_obj.strftime("%Y%m") for date_obj in self.wet_dates[0]]
        #self.dem_dir = dem_dir
        self.output_dir = output_dir
        # output_base and resources_base are historically two separate trees
        # ('../output' and '../resources'). When output_dir is set, both
        # collapse to that single root so every write for this instance
        # (cache files, mean/std, slope, flood outputs) is contained in one
        # isolated location.
        self.output_base = self.output_dir if self.output_dir else '../output'
        self.resources_base = self.output_dir if self.output_dir else '../resources'
        # slope_dir stays independently overridable, but if the caller didn't
        # set it explicitly and output_dir was given, default it inside
        # output_dir rather than the shared '../resources/slope' default.
        self.slope_dir = slope_dir if slope_dir is not None else (
            os.path.join(self.output_dir, 'slope') if self.output_dir else None
        )
        self.create_out_dirs()

        # grid_shapefile is required unless aoi is given, in which case a
        # grid is generated on the fly and used in its place -- additive
        # convenience path, doesn't change the grid_shapefile-based API.
        if grid_shapefile is None:
            if aoi is None:
                raise ValueError(
                    "Either grid_shapefile or aoi must be given. Pass an "
                    "existing grid file via grid_shapefile, or an AOI "
                    "boundary via aoi to generate one on the fly (see "
                    "autofloods.grid.generate_grid)."
                )
            if grid_dry_months is None:
                raise ValueError(
                    "grid_dry_months is required when generating a grid "
                    "from aoi -- dry season is climate knowledge that "
                    "can't be derived from AOI geometry alone (e.g. "
                    "grid_dry_months='04,05')."
                )
            resolved_mode = grid_mode if grid_mode is not None else (
                'mgrs' if isinstance(self.source, sources.OPERASource) else 'utm_fishnet'
            )
            generated_grid_path = os.path.join(self.resources_base, 'generated_grid.gpkg')
            generated_grid = grid.generate_grid(
                aoi, mode=resolved_mode, tile_size_km=grid_tile_size_km,
                output_path=generated_grid_path, id_col=id_col,
                dry_date_col=dry_date_col, dry_months=grid_dry_months,
            )
            grid_shapefile = generated_grid_path
            if grid_id_list is None:
                grid_id_list = generated_grid[id_col].tolist()
        elif grid_id_list is None:
            raise ValueError("grid_id_list is required when grid_shapefile is given.")

        self.grid_shapefile_path = grid_shapefile
        self.selected_grid_id = grid_id_list

        self.generate_defaults()

    def create_out_dirs(self):
        """
        Build self.folders_to_create (the full output/resources directory tree
        for this instance) and create each folder on disk, including parents.
        """
        self.folders_to_create = [
            self.output_base,
            os.path.join(self.output_base, 'mean_std'),
            os.path.join(self.output_base, 'flood_raster'),
            os.path.join(self.output_base, 'flood_raster', 'monthlyadded'),
            os.path.join(self.output_base, 'flood_vector'),
            os.path.join(self.output_base, 'flood_image'),
            os.path.join(self.output_base, 'final_output'),
            os.path.join(self.resources_base, 'slope'),
        ]
        for folder in self.folders_to_create:
            # makedirs (not mkdir) so a fresh output_dir with missing parent
            # directories (e.g. a new per-job scratch path) is created
            # correctly in one step, not just its immediate leaf folder.
            os.makedirs(folder, exist_ok=True)

    def generate_defaults(self):
        """
        Compute per-instance search/output state derived from __init__ args:
        aoi_union/aoi_list (search bboxes), self.timestamp, the JSON cache and
        mean/std NetCDF path templates (dry/wet aoi_scene and scene_aoi JSON
        files, self.nc_outfile), and already_processed_aoi_ids -- AOI IDs whose
        dry-season baseline .nc already exists on disk and will be skipped and
        reloaded rather than recomputed. Called once at the end of __init__.
        """
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
        json_outfile = os.path.join(self.resources_base, '.json')
        nc_outfile = os.path.join(self.output_base, 'mean_std', 'aoi_id_vv_vh_mean_std.nc')
        self.dry_aoi_scene_json_file = json_outfile.replace('.json', f'{dry_year_begin}_{dry_year_end}_dry_aoi_scene.json')
        self.dry_scene_aoi_json_file = json_outfile.replace('.json', f'{dry_year_begin}_{dry_year_end}_dry_scene_aoi.json')
        self.wet_aoi_scene_json_file = json_outfile.replace('.json', f'{dry_year_begin}_{dry_year_end}_wet_aoi_scene.json')
        self.wet_scene_aoi_json_file = json_outfile.replace('.json', f'{dry_year_begin}_{dry_year_end}_wet_scene_aoi.json')
        self.nc_outfile = nc_outfile.replace('aoi_', f'{dry_year_begin}_{dry_year_end}_aoi_')

        # An AOI counts as "already processed" (dry-season baseline can be
        # skipped and reloaded from disk) only if its mean/std NetCDF
        # actually exists -- NOT if it merely appears as a key in the
        # dry-season scene-SEARCH cache (dry_aoi_scene_json_file). That
        # search cache is written by generate_scene_aoi_dict()/get_s1_items()
        # right after the STAC search completes -- long before the actual
        # (slow, network-bound, failure-prone) scene reads and baseline
        # computation happen. Using it as the "done" signal meant a job
        # killed or crashed anywhere after the search but before
        # generate_mean_std_by_aoi() finished left a resumed run believing
        # the AOI was fully done, silently skipping it -- which then
        # crashed generate_dry_date_ranges() with "min() arg is an empty
        # sequence" once dry_months lost its entry for that "already
        # processed" AOI. Checking for the .nc file itself -- the actual
        # completion artifact generate_mean_std_by_aoi() writes only after
        # it succeeds -- fixes this: a killed/restarted run now correctly
        # re-does exactly the AOIs that never finished, and skips (reloads
        # from disk via load_mean_std_by_aoi()) exactly the ones that did.
        self.old_dry_aoi_scene_dict = dict()
        if os.path.exists(self.dry_aoi_scene_json_file):
            with open(self.dry_aoi_scene_json_file) as f:
                self.old_dry_aoi_scene_dict = json.load(f)

        self.already_processed_aoi_ids = [
            id for id in self.selected_grid_id
            if os.path.exists(self.nc_outfile.replace('_id_', f'_{id}_'))
        ]

        if self.already_processed_aoi_ids:
            print(
                'Following AOI IDs have a completed dry-season baseline (.nc) on disk already. '
                'Will be skipped and reloaded from disk.\n',
                self.already_processed_aoi_ids
            )

    def get_dry_dates(self):
        """
        Read each AOI's dry season (grid_shapefile's `dry_date_col`, e.g.
        "04,05" -> [4, 5]) into self.dry_months, after dropping any AOI in
        already_processed_aoi_ids -- so an AOI whose baseline is already
        done contributes nothing to generate_dry_date_ranges()'s combined
        date range. Sets self.aoi_ids_to_process (the AOIs actually
        being (re)computed this run) as a side effect.
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
        Turn self.dry_months (per-AOI) into self.dry_dates: one
        (start, end) date range per year in dry_years, covering the union
        of every AOI's dry months -- i.e. if different AOIs in this batch
        have different dry months, the earliest-to-latest month across
        all of them is used for every AOI's search, not each AOI's own
        narrower window. Fine when processing one AOI at a time (the
        production pattern -- see scripts/bihar2024_tile.py); worth
        knowing if you pass grid_id_list with AOIs on very different
        dry-season calendars.
        """

        if not self.dry_months:
            # Every requested AOI already has a completed dry-season
            # baseline on disk (see already_processed_aoi_ids in
            # generate_defaults()), so get_dry_dates() filtered all of
            # them out of self.dry_months -- there's no dry-season date
            # range left to generate. This is the partial-resume case: a
            # tile whose baseline finished in a prior run but crashed
            # before completing wet-season processing. Skipping here
            # (instead of crashing on min()/max() of an empty sequence)
            # lets get_s1_items()/read_scenes(dry_wet='dry') run as
            # harmless no-ops on an empty date range, and
            # generate_mean_std_by_aoi() reload the baseline from disk
            # via load_mean_std_by_aoi() as usual.
            self.dry_dates = []
            print(
                'All requested AOIs already have a completed dry-season baseline -- '
                'skipping dry-season date range generation.', flush=True
            )
            return

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
        """
        STAC-search `self.source` over the combined AOI bbox (aoi_union)
        for dry_dates or wet_dates (per `dry_wet`), then split the results
        back out per-AOI via generate_scene_aoi_dict() -- one combined
        search rather than one per AOI, since AOIs in the same batch
        typically have overlapping/adjacent footprints. Sets
        self.dry_s1_scenes/self.wet_s1_scenes (flat list, after
        generate_scene_aoi_dict() runs) for read_scenes() to consume.
        """
        if dry_wet == 'dry':
            dates = self.dry_dates
        elif dry_wet == 'wet':
            dates = self.wet_dates

        s1_scenes = {
            aoi['properties'][self.id_key]:
            [
                item2
                for item1 in[
                self.source.search_sentinel1(
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
        # It could be that the ID does not have any wet scene, this is a bug and needs to be fixed later
        self.generate_scene_aoi_dict(dry_wet=dry_wet, verbose=verbose)
        

    def generate_scene_aoi_dict(self, dry_wet='dry', verbose=False):
        """
        Cross-reference this run's search results against each AOI's
        footprint (utils.seggregate_sentinel_search), then merge the
        result into the on-disk scene-lookup cache (dry_*_json_file or
        wet_*_json_file) rather than overwriting it -- so scenes found in
        an earlier run for a different date range stay recorded. Sets
        self.dry_aoi_scene_dict/self.dry_scene_aoi_dict (or the wet
        equivalents) as the final, merged mapping read_scenes() and
        downstream steps use -- these differ from the raw dicts
        seggregate_sentinel_search() returns once an on-disk cache exists.
        """
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
                #final_json_data = old_content | aoi_scene_dict
                final_json_data = {**old_content, **aoi_scene_dict}

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

            #final_json_data = old_content | scene_aoi_dict
            final_json_data = {**old_content, **scene_aoi_dict}

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

    def read_scenes(self, dry_wet='dry', overview_level=3, max_workers=6):
        """
        Download+read every scene found by get_s1_items() (self.dry_s1_scenes
        or self.wet_s1_scenes) via `self.source`, concurrently. Sets
        self.s1_dry_dict/self.s1_wet_dict: {scene_id: {'vv_ds':..., 'vh_ds':...}},
        still in native CRS (see preprocessing.read_sentinel1_stac).
        `overview_level` is source-dependent -- MPCSource's COGs have an
        internal pyramid (lower = higher resolution); OPERASource ignores
        it (no pyramid, always native 30m). `max_workers` default (6) is
        a safe concurrency level against OPERASource -- see
        autofloods.sources's module docstring for source-specific guidance.
        """
        if dry_wet == 'dry':
            s1_scenes = self.dry_s1_scenes
        elif dry_wet == 'wet':
            s1_scenes = self.wet_s1_scenes

        # read and reproject the images concurrently. This is I/O-bound
        # (network reads of remote COGs), not CPU-bound, so a thread pool
        # is used rather than a process pool: threads release the GIL
        # during I/O, and avoid the pickling/IPC cost of shipping large
        # xarray/rasterio objects back across process boundaries. Each
        # individual read already has its own bounded timeout and retry
        # (see autofloods.utils.open_rasterio_with_retry); this adds
        # concurrency across scenes on top of that. max_workers is kept
        # modest (not unbounded) to avoid hammering the data source with
        # too many simultaneous requests.
        s1_combined = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(preprocessing.read_sentinel1_stac, item, self.source, overview_level)
                for item in s1_scenes
            ]
            for future in concurrent.futures.as_completed(futures):
                stac_id, stac_ds = future.result()
                s1_combined[stac_id] = stac_ds

        if dry_wet == 'dry':
            self.s1_dry_dict = s1_combined
        elif dry_wet == 'wet':
            self.s1_wet_dict = s1_combined

    def generate_mean_std_by_aoi(self, reproject_max_workers=None):
        """
        Fit each AOI's dry-season Z-score baseline (self.detector.fit_baseline)
        from its stacked/clipped dry scenes, write it to NetCDF, and load
        it into self.mean_std_by_aoi -- alongside any already-processed
        AOIs reloaded from disk (load_mean_std_by_aoi()). Pixels with
        stray large sentinel values (>= 50, e.g. from an upstream nodata
        convention) are masked to NaN before fitting so they don't skew
        the mean/std.

        If self.detector.requires_baseline_fitting is False, fit_baseline()
        is never called and no baseline .nc is written; mean_std_by_aoi[id]
        is set instead to the tile's own reprojected dry-season VV stack,
        used purely as a CRS/grid reference by prepare_slope(), map_floods(),
        merge_floods_by_date(), and generate_number_of_scenes() -- it is NOT
        a statistical baseline and must not be read as one.

        reproject_max_workers is passed through to
        preprocessing.reproject_clip_stac()/stack_images()'s thread pools
        (CPU-bound reprojection concurrency) -- None (default) uses
        utils.default_max_workers().
        """
        # separate and clip the reprojected image for each tile
        reprojected_clipped_dry = {
            id: preprocessing.reproject_clip_stac(self.s1_dry_dict, self.dry_aoi_scene_dict,
                                                  self.grid_shapefile_path, id,
                                                  max_workers=reproject_max_workers)
            for id in self.selected_grid_id
            if not id in self.already_processed_aoi_ids
        }

        # stack the reprojected and clipped images
        self.stacked_dry = {
            id: preprocessing.stack_images(reprojected_clipped_dry[id], self.grid_shapefile_path, id,
                                           max_workers=reproject_max_workers, cell_size=self.cell_size)
            for id in reprojected_clipped_dry
            }

        # handle newly introduced nodata values
        for id in self.stacked_dry:
            for n in range(len(self.stacked_dry[id]['vv_stack'])):
                self.stacked_dry[id]['vv_stack'][n] = self.stacked_dry[id]['vv_stack'][n].where(self.stacked_dry[id]['vv_stack'][n] < 50, np.nan)
                self.stacked_dry[id]['vh_stack'][n] = self.stacked_dry[id]['vh_stack'][n].where(self.stacked_dry[id]['vh_stack'][n] < 50, np.nan)

        # calculate cell level baseline (Z-score: mean and std) for the dry scenes.
        # Skipped for detectors that don't fit a per-tile baseline at all
        # (e.g. a pretrained model that loads weights once, globally).
        if self.detector.requires_baseline_fitting:
            self.mean_std_by_aoi = {
                id: self.detector.fit_baseline(
                    self.stacked_dry[id]['vv_stack'], self.stacked_dry[id]['vh_stack']
                )
                for id in self.stacked_dry.keys()
            }
        else:
            # No real baseline is fit -- fit_baseline() is never called --
            # but downstream steps still need a CRS/grid reference for
            # this tile-year (see this method's docstring). The tile's
            # own reprojected dry-season VV stack already carries that
            # grid at zero extra cost (same forced cell_size, same AOI
            # bounds every dry scene was clipped to), so reuse it as a
            # marker rather than inventing a second grid-reference path.
            self.mean_std_by_aoi = {
                id: self.stacked_dry[id]['vv_stack'] for id in self.stacked_dry.keys()
            }
        self.stacked_dry = None


        # save the cell level mean and std to different files (separate for each ID)
        if self.detector.requires_baseline_fitting:
            for id in self.mean_std_by_aoi:
                outfile = self.nc_outfile.replace('_id_', f'_{id}_')
                self.mean_std_by_aoi[id].to_netcdf(outfile)

        self.load_mean_std_by_aoi()

    def load_mean_std_by_aoi(self):
        """
        Load each already_processed_aoi_ids AOI's baseline NetCDF from
        disk into self.mean_std_by_aoi, skipping any already present
        (e.g. just fit in this same run). A load failure (file present
        per already_processed_aoi_ids but corrupt/unreadable) is caught
        and only printed, not raised -- that AOI is silently left out of
        mean_std_by_aoi, which will surface as a KeyError in a later
        step rather than here.
        """
        if not hasattr(self, 'mean_std_by_aoi'):
            self.mean_std_by_aoi = dict()

        # Read the mean and std stacked raster if already processed
        # and save in the existing self.mean_std_by_aoi dictionary
        for id in self.already_processed_aoi_ids:
            if not int(id) in self.mean_std_by_aoi.keys():
                infile = self.nc_outfile.replace('_id_', f'_{id}_')
                try:
                    self.mean_std_by_aoi[int(id)] = xr.load_dataarray(infile)
                    self.mean_std_by_aoi[int(id)] = self.mean_std_by_aoi[int(id)].where(self.mean_std_by_aoi[int(id)] != 3.4028234663852886e+38, np.nan)
                    print(f'Previously processed {infile} read successfully!')
                except:
                    print(f'{infile} present in JSON as processed but .nc file is missing.')

    def prepare_slope(self, dem_overview=1, nodata=0.0, buffer=500, max_workers=6):
        """
        Compute (or reload from `slope_dir` if already cached) a smoothed
        relative-slope raster per AOI, used by the detector's slope mask
        to suppress false-positive floods on steep terrain. Downloads DEM
        tiles via `self.source` only for AOIs missing a cached slope file
        -- already-cached AOIs are skipped entirely, no DEM download.
        `buffer` (meters, in the AOI's own UTM zone) must match what
        smoothen_slope()'s kernel expects; passed through unchanged.
        `max_workers` is DEM download concurrency (network-bound, passed
        to utils.download_nasadem) -- separate from the CPU-bound
        reprojection thread pools elsewhere in this class.
        """
        slope_id_to_process = [
            id
            for id in self.selected_grid_id
            if not os.path.exists(os.path.join(
                self.slope_dir, SLOPE_OUTFILE.replace('_id.nc', f'_{id}.nc')
            ))
        ]

        if len(slope_id_to_process) > 0:
            # download DEM for select IDs
            bbox_for_dem_download = utils.gpd_to_json(id_list=slope_id_to_process, infile=self.grid_shapefile_path,
                                                      separate=False, id_key=self.id_key, buffer=buffer)
            dem_merged_xarray = autofloods.utils.download_nasadem(
                bbox_for_dem_download[0], self.source, overview_level=dem_overview,
                nodata=nodata, max_workers=max_workers,
            )

            # for each ID, clip using the buffered GDF, calculate relative slope
            # and then clip to the actual tile extent
            self.slope = dict()
            for id in slope_id_to_process:
                print(f'Slope for tile ID {id} not found. Downloading DEM...')
                self.slope[id] = autofloods.preprocessing.smoothen_slope(
                    dem_xarray=dem_merged_xarray,
                    grid_shapefile_path=self.grid_shapefile_path,
                    aoi_id=id,
                    ref_xarray=self.mean_std_by_aoi[id],
                    buffer=buffer,
                    nodata=nodata,
                    cell_size=self.cell_size,
                )

            # for each of the ids, clip slope and export to the .nc file
            for id in slope_id_to_process:
                self.slope[id] = autofloods.preprocessing.clip_xarray_using_id(
                    data_xarray=self.slope[id],
                    grid_shapefile_path=self.grid_shapefile_path,
                    aoi_id=id,
                    ref_xarray=self.mean_std_by_aoi[id],
                    buffer=None,
                    slope=True,
                    cell_size=self.cell_size,
                )

                outfile = os.path.join(self.slope_dir, SLOPE_OUTFILE.replace('_id.nc', f'_{id}.nc'))
                autofloods.utils.export_xarray(self.slope[id], outfile)

        else: # load the slope if already present
            for id in self.selected_grid_id:
                print(f'Slope for tile ID {id} found, will not be downloaded.')
                self.slope = {
                    id: xr.load_dataarray(os.path.join(self.slope_dir, SLOPE_OUTFILE.replace('_id.nc', f'_{id}.nc')), engine='rasterio')
                    for id in self.selected_grid_id
                }

    def prepare_wet_scenes(self, overview_level=3, max_workers=6, reproject_max_workers=None):
        """
        Search, read, reproject, and clip every wet-season scene for each
        AOI onto that AOI's baseline grid (ref_xarray=mean_std_by_aoi[id]) --
        must run after generate_mean_std_by_aoi(), which that reference
        grid comes from. Sets self.wet_scenes_by_aoi:
        {aoi_id: {scene_id: DataArray with band coord ['vv_ds', 'vh_ds']}},
        ready for map_floods() to classify directly against mean_std_by_aoi.

        max_workers controls read_scenes()'s download concurrency (network-
        bound); reproject_max_workers is a separate thread pool for the
        per-scene reproject+clip loop below (CPU-bound) -- same GIL-
        releasing-GDAL-warp pattern as preprocessing.reproject_clip_stac().
        None (default) uses utils.default_max_workers() -- (available
        CPUs - 1) on whatever system this runs on.
        """
        if reproject_max_workers is None:
            reproject_max_workers = autofloods.utils.default_max_workers()
        # call the previouisly defined method to get S1 scenes for the wet period
        self.get_s1_items(dry_wet='wet')
        self.read_scenes(dry_wet='wet', overview_level=overview_level, max_workers=max_workers)

        # loop through each id and clip every S1 wet scene, concurrently
        def _clip_one_scene(id, scene_id):
            return scene_id, xr.concat(
                [
                    autofloods.preprocessing.clip_xarray_using_id(
                        data_xarray=self.s1_wet_dict[scene_id]['vv_ds'],
                        grid_shapefile_path=self.grid_shapefile_path,
                        aoi_id=id,
                        ref_xarray=self.mean_std_by_aoi[id],
                        cell_size=self.cell_size,
                    ),
                    autofloods.preprocessing.clip_xarray_using_id(
                        data_xarray=self.s1_wet_dict[scene_id]['vh_ds'],
                        grid_shapefile_path=self.grid_shapefile_path,
                        aoi_id=id,
                        ref_xarray=self.mean_std_by_aoi[id],
                        cell_size=self.cell_size,
                    )
                ], dim='band').assign_coords(band=['vv_ds', 'vh_ds'])

        self.wet_scenes_by_aoi = {}
        for id in self.wet_aoi_scene_dict:
            scene_out = {}
            with concurrent.futures.ThreadPoolExecutor(max_workers=reproject_max_workers) as executor:
                futures = [
                    executor.submit(_clip_one_scene, id, scene_id)
                    for scene_id in self.wet_aoi_scene_dict[id]
                ]
                for future in concurrent.futures.as_completed(futures):
                    scene_id, result = future.result()
                    scene_out[scene_id] = result
            self.wet_scenes_by_aoi[id] = scene_out

        # handle no data
        for id in self.wet_scenes_by_aoi:
            for scene_id in self.wet_scenes_by_aoi[id]:
                self.wet_scenes_by_aoi[id][scene_id] = self.wet_scenes_by_aoi[id][scene_id].where(
                    self.wet_scenes_by_aoi[id][scene_id] < 50, np.nan)

    def map_floods(self, vv_thd=-3, vh_thd=-3, rel_slope_thd=20, export_raster=True, export_vector=False, export_maps=False):
        """
        Classify every wet-season scene against its AOI's baseline
        (self.detector.detect()), apply the slope mask if the detector
        needs one, and set self.flood_dict: {aoi_id: {scene_id: classified
        DataArray}} (0/1/2/3 -- see detectors.ZScoreDetector). Per-scene
        exports (export_raster/export_vector/export_maps, all default
        False except export_raster) are one file per scene -- for the
        per-date/per-month aggregates actually used downstream, see
        merge_floods_by_date()/monthly_sum() instead. NOTE: export_vector's
        and the raster export's output filenames derive the date/track
        suffix from scene_id.split('_')[4:], which assumes MPC-style scene
        IDs -- an OPERA_PASS_{date} scene_id (3 tokens) yields an empty
        suffix there, not a crash, just a less-informative filename.
        """
        # keep backward-compatible per-call threshold overrides for detectors
        # that expose them (e.g. ZScoreDetector); a detector without these
        # attributes ignores the override and uses its own configuration.
        if hasattr(self.detector, 'vv_thd'):
            self.detector.vv_thd = vv_thd
        if hasattr(self.detector, 'vh_thd'):
            self.detector.vh_thd = vh_thd

        # generate id and scene wise anomaly cells
        self.flood_dict = {
            id: {
                scene_id: self.detector.detect(
                    self.mean_std_by_aoi[id], self.wet_scenes_by_aoi[id][scene_id]
                )
                for scene_id in self.wet_scenes_by_aoi[id]
            }
            for id in self.mean_std_by_aoi
        }

        # apply relative slope mask, if this detector needs it
        if self.detector.requires_slope_mask:
            slope_path = os.path.join(self.slope_dir, SLOPE_OUTFILE)
            for id in self.flood_dict:
                slope_xarray = xr.load_dataarray(slope_path.replace('_id.nc', f'_{id}.nc'), engine='rasterio')
                # xr.load_dataarray(..., engine='rasterio') does not wire
                # up rioxarray's CRS accessor the way rioxarray.open_rasterio()
                # does, even though the file has a spatial_ref coordinate --
                # slope_xarray.rio.crs is None here despite being
                # georeferenced, which reproject_match() below requires.
                # Every existing use of this loading pattern only ever read
                # .values (plain numpy, no CRS needed), so this was never
                # hit before reproject_match() was added. Slope and the
                # dry-season baseline are always in the same tile UTM zone,
                # so borrowing mean_std_by_aoi[id]'s (properly populated,
                # since it comes from a native rioxarray reproject chain)
                # CRS is correct, not a workaround.
                slope_xarray = slope_xarray.rio.write_crs(self.mean_std_by_aoi[id].rio.crs)
                # Slope is DEM-derived and cached once per tile (terrain
                # doesn't change), reused across every year that tile is
                # processed -- but each year's dry-season baseline
                # (mean_std_by_aoi[id]) can land on a slightly different
                # pixel grid, since GDAL's .rio.reproject() isn't
                # perfectly deterministic across different scene sets
                # (observed: off by 1-4 pixels at the edge between years,
                # first surfaced running a multi-year batch against the
                # same slope cache for the first time). Align slope onto
                # THIS year's actual flood-raster grid before masking --
                # reproject_match handles CRS/transform/shape together;
                # 'nearest' is appropriate since this is a threshold
                # comparison, not a value needing smoothing.
                slope_xarray = slope_xarray.rio.reproject_match(
                    self.mean_std_by_aoi[id], resampling=Resampling.nearest
                )
                for scene_id in self.flood_dict[id]:
                    self.flood_dict[id][scene_id] = self.flood_dict[id][scene_id].where(
                        slope_xarray.values[0, :, :] < rel_slope_thd, 0
                    )

        dry_year_begin = min(self.dry_years)
        dry_year_end = max(self.dry_years)
        wet_yearmonth_begin = self.wet_yearmonths[0]
        wet_yearmonth_end = self.wet_yearmonths[-1]

        # export the flood rasters
        flood_raster_outfile = os.path.join(self.output_base, 'flood_raster', 'floodextent_id.tif')
        if export_raster == True:
            for id in self.flood_dict:
                for scene_id in self.flood_dict[id]:
                    outfile_flood = flood_raster_outfile.replace('_id.tif', f'_DRY_{dry_year_begin}_{dry_year_end}_WET_{wet_yearmonth_begin}_{wet_yearmonth_end}_{id}_{"_".join(scene_id.split("_")[4:])}.tif')
                    autofloods.utils.export_xarray(self.flood_dict[id][scene_id], outfile_flood)

        # polygonize the flood rasters
        if export_vector == True:
            self.flood_gdf_dict = {
                id: {
                    scene_id: autofloods.postprocessing.polygonize_flood_raster(self.flood_dict[id][scene_id])
                    for scene_id in self.flood_dict[id]
                }
                for id in self.flood_dict
            }

            flood_vector_outfile = os.path.join(self.output_base, 'flood_vector', 'floodextent_id.gpkg')
            for id in self.flood_dict:
                for scene_id in self.flood_dict[id]:
                    outfile_flood = flood_vector_outfile.replace('_id.gpkg', f'_DRY_{dry_year_begin}_{dry_year_end}_WET_{wet_yearmonth_begin}_{wet_yearmonth_end}_{id}_{"_".join(scene_id.split("_")[4:])}.gpkg')
                    # export only if the GDF has any flood cells
                    if self.flood_gdf_dict[id][scene_id].shape[0] > 0:
                        self.flood_gdf_dict[id][scene_id].to_crs("EPSG:4326").to_file(outfile_flood, index=False)
                    else:
                        # remove that scene_id from the dictionary to avoid exporting map later
                        del self.flood_gdf_dict[id][scene_id]
                        print(f'Flood cells not found in {id}_{scene_id}.')

        # export flood maps as images
        if export_maps == True:
            flood_map_outfile = os.path.join(self.output_base, 'flood_image', 'floodmap_id.png')
            for id in self.flood_dict:
                for scene_id in self.flood_dict[id]:
                    outfile_flood = flood_map_outfile.replace('_id.png', f'_DRY_{dry_year_begin}_{dry_year_end}_WET_{wet_yearmonth_begin}_{wet_yearmonth_end}_{id}_{"_".join(scene_id.split("_")[4:])}.png')

                    mapfloods.flood_images(
                        flood_xarray=self.flood_dict[id][scene_id],
                        outfile_flood=outfile_flood
                    )

    def _yearmonthtag(self):
        """Shared dry/wet date-range tag used to name every downstream
        output file (flood_raster stacks, monthly aggregates). Factored
        out so expected_monthly_outfile()/is_fully_processed() compute
        the exact same tag merge_floods_by_date() uses, rather than a
        second, driftable copy of this string logic."""
        dry_year_begin = min(self.dry_years)
        dry_year_end = max(self.dry_years)
        wet_yearmonth_begin = self.wet_yearmonths[0]
        wet_yearmonth_end = self.wet_yearmonths[-1]
        return f'DRY_{dry_year_begin}_{dry_year_end}_WET_{wet_yearmonth_begin}_{wet_yearmonth_end}'

    def expected_monthly_outfile(self, id):
        """
        Deterministic path to AOI `id`'s final monthly-aggregated flood
        raster -- computed the same way merge_floods_by_date()/
        monthly_sum()/postprocessing.aggregate_monthly() derive it, but
        WITHOUT running any of the actual pipeline. Only depends on
        __init__-time state (output_base, dry_years, wet_yearmonths), so
        it's safe to call immediately after construction, before any
        search/read/compute has happened -- see is_fully_processed().
        """
        yearmonthtag = self._yearmonthtag()
        flood_raster_stacked_outfile = os.path.join(self.output_base, 'flood_raster', 'floodextentstacked_id.tif')
        folder_to_create = os.path.split(flood_raster_stacked_outfile)[0].replace(
            '/flood_raster', f'/flood_raster/floodextentstacked_{yearmonthtag}/'
        )
        stacked_outfile = os.path.join(
            folder_to_create,
            os.path.split(flood_raster_stacked_outfile)[-1].replace('_id.tif', f'{yearmonthtag}_{id}.tif')
        )
        # matches aggregate_monthly()'s own default outfile derivation
        return stacked_outfile.replace('/flood_raster/floodextentstacked', '/flood_raster/monthlyadded').replace('.tif', '_monthly.tif')

    def is_fully_processed(self, id):
        """
        True if AOI `id`'s entire pipeline (through monthly aggregation)
        already completed in a prior run for these exact dry_years/
        wet_duration settings -- i.e. it's safe to skip the whole tile
        rather than resuming mid-pipeline. Intended to be checked right
        after constructing flood_mapper, before calling any other method,
        so a production script can skip already-finished tiles cheaply
        (recovering from a killed/restarted batch without redoing tiles
        that already finished). Mid-pipeline resumption for a tile that's partially done (e.g.
        dry-season baseline computed but wet-season detection not yet
        run) is handled separately -- see already_processed_aoi_ids in
        generate_defaults(), which skips the (expensive, network-bound)
        dry-season baseline recomputation specifically once its .nc file
        exists, independent of whether the rest of the tile is done.
        """
        return os.path.exists(self.expected_monthly_outfile(id))

    def merge_floods_by_date(self, export_raster=False):
        """
        Collapse map_floods()'s per-scene classifications into one band
        per observed DATE per AOI (utils.flood_data_3dstack -- same-date
        scenes take the per-pixel max, so any scene flagging a pixel
        flooded wins for that date). Sets self.flood_by_date and, if
        export_raster, writes 'floodextentstacked_<DRY_..._WET_...>_<id>.tif'
        (band descriptions are the sorted date strings) and populates
        self.flood_raster_dict -- the input aggregate_monthly()/monthly_sum()
        reads. This is the file expected_monthly_outfile()/is_fully_processed()
        check for (via its monthly-aggregated form) to decide whether an
        AOI needs (re)processing at all.
        """
        self.flood_by_date = dict()

        for id in self.flood_dict:
            if len(self.flood_dict[id]) > 0: # process only if there is a wet scene (throws error otherwise)
                dates_list, floods_stacked = autofloods.utils.flood_data_3dstack(self.flood_dict[id])
                self.flood_by_date[id] = xr.DataArray(
                    floods_stacked,
                    dims=['date', 'y', 'x'],
                    coords = {'date': dates_list,
                              **{
                                  'y':self.mean_std_by_aoi[id].coords['y'],
                                  'x':self.mean_std_by_aoi[id].coords['x']
                              }}
                )

        # export if the export parameter is true
        yearmonthtag = self._yearmonthtag()

        if export_raster:
            flood_raster_stacked_outfile = os.path.join(self.output_base, 'flood_raster', 'floodextentstacked_id.tif')
            folder_to_create = os.path.split(flood_raster_stacked_outfile)[0].replace('/flood_raster', f'/flood_raster/floodextentstacked_{yearmonthtag}/')
            if not os.path.exists(folder_to_create):
                os.makedirs(folder_to_create, exist_ok=True)

            self.flood_raster_dict = dict()
            for id in self.flood_by_date:
                outfile_flood = os.path.join(
                    folder_to_create,
                    os.path.split(flood_raster_stacked_outfile)[-1].replace('_id.tif', f'{yearmonthtag}_{id}.tif')
                )
                autofloods.utils.export_xarray(self.flood_by_date[id], outfile_flood,
                                           sorted(self.flood_by_date[id].date.to_dict()['data'])
                                           )
                self.project_flood_raster(outfile_flood, id)
                self.flood_raster_dict[id] = outfile_flood

    def generate_number_of_scenes(self, export_raster=False):
        """
        Per-pixel count, across all of an AOI's wet-season scenes, of how
        many scenes had a NaN (missing/invalid) observation at that pixel
        -- i.e. a per-pixel data-GAP count, not a count of usable/valid
        observations despite the method/output-file name ("scenes count").
        Worth double-checking against intent before relying on this for
        anything beyond a rough QA signal for coverage gaps. Writes
        'floodscenescount_<DRY_..._WET_...>_<id>.tif' if export_raster.
        """
        self.scene_count = dict()

        for id in self.wet_scenes_by_aoi:
            if len(self.wet_scenes_by_aoi[id]) > 0: # process only if there is a wet scene (throws error otherwise)
                self.scene_count[id] = xr.DataArray(np.stack([
                    np.any(np.isnan(
                        self.wet_scenes_by_aoi[id][key]
                    ), axis=0)
                    for key in self.wet_scenes_by_aoi[id]
                    ]).sum(axis=0),
                                                    dims=self.mean_std_by_aoi[id].dims[1:],
                                                    coords={
                                                        'y':self.mean_std_by_aoi[id].coords['y'],
                                                        'x':self.mean_std_by_aoi[id].coords['x']
                                                    }
                                                    )

        # export if the export parameter is true
        dry_year_begin = min(self.dry_years)
        dry_year_end = max(self.dry_years)
        wet_yearmonth_begin = self.wet_yearmonths[0]
        wet_yearmonth_end = self.wet_yearmonths[-1]

        if export_raster:
            flood_scenes_count_outfile = os.path.join(self.output_base, 'flood_raster', 'floodscenescount_id.tif')
            yearmonthtag = f'DRY_{dry_year_begin}_{dry_year_end}_WET_{wet_yearmonth_begin}_{wet_yearmonth_end}'
            folder_to_create = os.path.split(flood_scenes_count_outfile)[0].replace('/flood_raster', f'/flood_raster/floodscenescount_{yearmonthtag}/')
            if not os.path.exists(folder_to_create):
                os.makedirs(folder_to_create, exist_ok=True)

            for id in self.scene_count:
                outfile_flood = os.path.join(
                    folder_to_create,
                    os.path.split(flood_scenes_count_outfile)[-1].replace('_id.tif', f'{yearmonthtag}_{id}.tif')
                )
                autofloods.utils.export_xarray(self.scene_count[id], outfile_flood)
                self.project_flood_raster(outfile_flood, id)


    def project_flood_raster(self, infile, tile_id):
        """
        Stamp `infile`'s CRS tag as `tile_id`'s UTM zone (in place --
        pixel data untouched). export_xarray() writes rasters without
        embedding a CRS, since the reprojected/clipped input DataArray's
        own CRS metadata isn't always reliably attached by that point in
        the pipeline; this fixes that up afterward from the grid
        shapefile's authoritative `zone` column instead.
        """
        # Define the correct projection
        gdf = gpd.read_file(self.grid_shapefile_path)
        new_crs = utils.zone_to_epsg(gdf.loc[gdf.ID == tile_id].zone.values[0])

        # Open the input GeoTIFF file and get its metadata. Since
        # export_xarray() started writing COGs (see autofloods.utils),
        # GDAL refuses in-place edits of a COG-layout file by default --
        # even a metadata-only one like this CRS-tag assignment -- to
        # protect the optimized tiling/overview layout from a rewrite
        # that could break it. A CRS tag update alone never touches
        # pixel data, tiling, or overviews, so IGNORE_COG_LAYOUT_BREAK is
        # genuinely safe here, not a real layout compromise.
        with rasterio.open(infile, 'r+', IGNORE_COG_LAYOUT_BREAK='YES') as src:
            src.crs = new_crs


    def monthly_sum(self):
        """
        Aggregate every AOI's per-date flood stack (merge_floods_by_date()'s
        output, self.flood_raster_dict) into a per-month flood-day-count
        raster via postprocessing.aggregate_monthly() -- the final output
        expected_monthly_outfile()/is_fully_processed() check for.
        """
        for id in self.flood_raster_dict:
            autofloods.postprocessing.aggregate_monthly(self.flood_raster_dict[id])


    def flush_output(self, remove_slope=False):
        """
        Delete this run's intermediate output folders and scene-lookup
        JSON caches (NOT the final monthly/flood_raster outputs -- see
        create_out_dirs() for folder order; folders_to_create[-1] is
        the DEM/slope cache, excluded unless remove_slope=True since
        slope is expensive to recompute and reusable across years for
        the same AOI). Destructive and immediate -- no confirmation, no
        recycle bin. Not currently called anywhere in this package;
        provided for callers who want to reclaim disk space between runs.
        """
        if remove_slope:
            folders_to_delete = self.folders_to_create[1:]
        else:
            folders_to_delete = self.folders_to_create[1:-1]

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





