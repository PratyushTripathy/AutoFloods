![AutoFloods](https://raw.githubusercontent.com/PratyushTripathy/AutoFloods/main/autofloods_logo.png)

AutoFloods is a Python package for automated flood mapping at scale from Sentinel-1 SAR imagery, with pluggable data sources (Microsoft Planetary Computer, NASA OPERA) and detection methods (Z-score, Otsu).

## Quickstart

OPERASource requires a free NASA Earthdata Login; MPCSource works with no credentials at all (a free subscription key just raises rate limits) — see [Authentication](https://autofloods.readthedocs.io/en/latest/getting-started.html#authentication).

<!-- TODO: drop --pre once a stable (non-alpha/beta) 0.1.0 release exists -->
Pre-release: `pip install --pre autofloods`

Basic usage:

```python
from autofloods import flood_mapper
from autofloods.sources import OPERASource

fm = flood_mapper(
    grid_shapefile='resources/india_utm_fishnet_buffer.gpkg',  # AOI grid; needs id_col, dry_date_col, zone columns
    grid_id_list=[321],                   # which AOI IDs from the grid to process
    dry_years=[2024, 2024],               # dry-season years to build the baseline from
    slope_dir='resources/slope/',         # where the terrain-slope mask is cached
    wet_duration=['2024/07', '2024/10'],  # wet-season date range to classify
    source=OPERASource(),                 # or MPCSource() (the default)
    output_dir='output/my_run',           # root dir for all outputs and caches
)

fm.get_dry_dates()                    # read each AOI's dry season into self.dry_months
fm.generate_dry_date_ranges()         # turn dry_months into per-year search date ranges
fm.get_s1_items(dry_wet='dry')        # STAC-search the source for dry-season scenes
fm.read_scenes(dry_wet='dry', overview_level=None, max_workers=6)  # download + read those scenes
fm.generate_mean_std_by_aoi()         # fit the dry-season Z-score baseline per AOI

fm.prepare_slope(dem_overview=0, buffer=500)  # compute/cache the terrain-slope mask

fm.prepare_wet_scenes(overview_level=None, max_workers=6)  # search, read, reproject wet-season scenes
fm.map_floods(vv_thd=-2.5, vh_thd=-2.5, rel_slope_thd=20,
              export_raster=False, export_vector=False, export_maps=False)  # classify each scene against the baseline
fm.merge_floods_by_date(export_raster=True)       # collapse per-scene results into one band per date
fm.generate_number_of_scenes(export_raster=True)  # per-pixel count of scenes with data gaps
fm.monthly_sum()                      # aggregate per-date results into per-month flood-day counts
```

See the documentation for full usage, API reference, and citation details: https://autofloods.readthedocs.io/en/latest/
