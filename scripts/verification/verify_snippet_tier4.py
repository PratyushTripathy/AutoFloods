"""
Tier 4.1 verification: the exact code snippet going into the manuscript's
Section 2.3, run as-is (not through run_autofloods.py's wrapper) to confirm
it's real and copy-pasteable, not illustrative pseudocode.
"""
import sys
import pathlib as _pathlib
BASE = str(_pathlib.Path(__file__).resolve().parents[2])  # repo root (scripts/verification/<this file>)
sys.path.append(BASE)

from autofloods import flood_mapper
from autofloods.sources import OPERASource, MPCSource
from autofloods.detectors import ZScoreDetector

fm = flood_mapper(
    grid_shapefile='resources/india_utm_fishnet_buffer.gpkg',
    grid_id_list=[318],
    dry_years=[2024, 2024],
    wet_duration=['2024/07', '2024/10'],
    slope_dir='resources/slope',
    source=OPERASource(),           # swap for MPCSource() to change provider
    detector=ZScoreDetector(vv_thd=-2.5, vh_thd=-2.5),
    output_dir='output/_tier4_snippet_check',
    cell_size=30,
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
fm.generate_number_of_scenes(export_raster=True)
fm.monthly_sum()

print('SNIPPET_OK', flush=True)
