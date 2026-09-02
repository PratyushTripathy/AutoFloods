# AutoFloods

AutoFloods is a Python package for automated flood mapping at scale from Sentinel-1 SAR imagery, with pluggable data sources (Microsoft Planetary Computer, NASA OPERA) and detection methods (Z-score, Otsu).

## Quickstart

OPERASource requires a free NASA Earthdata Login; MPCSource works with no credentials at all (a free subscription key just raises rate limits) — see [Authentication](https://autofloods.readthedocs.io/en/latest/getting-started.html#authentication).

<!-- TODO: drop --pre once a stable (non-alpha/beta) 0.1.0 release exists -->
This is a pre-release — install with: `pip install --pre autofloods`

Basic usage:

```python
from autofloods import flood_mapper
from autofloods.sources import OPERASource

fm = flood_mapper(
    grid_shapefile='resources/india_utm_fishnet_buffer.gpkg',
    grid_id_list=[321],
    dry_years=[2024, 2024],
    slope_dir='resources/slope/',
    wet_duration=['2024/07', '2024/10'],
    source=OPERASource(),          # or MPCSource() (the default)
    output_dir='output/my_run',
)

fm.get_dry_dates()
fm.generate_dry_date_ranges()
fm.get_s1_items(dry_wet='dry')
fm.read_scenes(dry_wet='dry', overview_level=None, max_workers=6)
fm.generate_mean_std_by_aoi()

fm.prepare_slope(dem_overview=0, buffer=500)

fm.prepare_wet_scenes(overview_level=None, max_workers=6)
fm.map_floods(vv_thd=-2.5, vh_thd=-2.5, rel_slope_thd=20,
              export_raster=False, export_vector=False, export_maps=False)
fm.merge_floods_by_date(export_raster=True)
fm.generate_number_of_scenes(export_raster=True)
fm.monthly_sum()
```

See the documentation for full usage, API reference, and citation details: https://autofloods.readthedocs.io/en/latest/
