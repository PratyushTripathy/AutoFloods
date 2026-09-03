import sys, time
import pathlib as _pathlib
BASE = str(_pathlib.Path(__file__).resolve().parents[2])  # repo root (scripts/verification/<this file>)
sys.path.append(BASE)

from autofloods import flood_mapper
from autofloods.sources import OPERASource
from autofloods.detectors import ZScoreDetector

t0 = time.monotonic()
fm = flood_mapper(
    grid_shapefile='resources/india_utm_fishnet_buffer.gpkg',
    grid_id_list=[318],
    dry_years=[2024, 2024],
    wet_duration=['2024/08', '2024/08'],   # narrowed to 1 month for tutorial speed
    slope_dir='resources/slope',
    source=OPERASource(),
    detector=ZScoreDetector(vv_thd=-2.5, vh_thd=-2.5),
    output_dir='output/_tutorial_notebook_check',
    cell_size=30,
)
fm.get_dry_dates()
fm.generate_dry_date_ranges()
fm.get_s1_items(dry_wet='dry')
print(f'dry scenes: {len(fm.dry_aoi_scene_dict.get(318, []))}', flush=True)
fm.read_scenes(dry_wet='dry')
fm.generate_mean_std_by_aoi()
t1 = time.monotonic()
print(f'dry baseline done: {t1-t0:.1f}s', flush=True)
fm.prepare_slope()
t2 = time.monotonic()
print(f'slope done: {t2-t1:.1f}s', flush=True)
fm.prepare_wet_scenes()
print(f'wet scenes: {sum(len(v) for v in fm.wet_scenes_by_aoi.values())}', flush=True)
fm.map_floods()
fm.merge_floods_by_date(export_raster=True)
fm.generate_number_of_scenes(export_raster=True)
fm.monthly_sum()
t3 = time.monotonic()
print(f'wet+detect+aggregate done: {t3-t2:.1f}s', flush=True)
print(f'TOTAL: {t3-t0:.1f}s', flush=True)
