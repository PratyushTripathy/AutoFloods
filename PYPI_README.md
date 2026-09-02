# AutoFloods

AutoFloods is a Python package for automated flood mapping at scale from Sentinel-1 SAR imagery, with pluggable data sources and detection methods.

## Quickstart

Install:

<!-- TODO: drop --pre once a stable (non-alpha/beta) 0.1.0 release exists -->
```bash
pip install --pre autofloods
```

Basic usage:

```python
from autofloods import flood_mapper
from autofloods.sources import OPERASource

fm = flood_mapper(
    grid_shapefile='path/to/grid.gpkg',
    grid_id_list=[321],
    dry_years=[2024, 2024],
    slope_dir='resources/slope/',
    wet_duration=['2024/07', '2024/10'],
    source=OPERASource(),
    output_dir='output/my_run',
)

fm.get_dry_dates()
fm.generate_dry_date_ranges()
fm.get_s1_items(dry_wet='dry')
fm.read_scenes(dry_wet='dry')
fm.generate_mean_std_by_aoi()
fm.prepare_slope()
fm.prepare_wet_scenes()
fm.map_floods()
fm.merge_floods_by_date(export_raster=True)
```

See the documentation for full usage, API reference, and citation details: [link — TBD, Read the Docs not live yet]
