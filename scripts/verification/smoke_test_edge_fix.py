import sys
import pathlib as _pathlib
BASE = str(_pathlib.Path(__file__).resolve().parents[2])  # repo root (scripts/verification/<this file>)
sys.path.append(BASE)

from autofloods import flood_mapper

TILE_IDS = [321, 325]
OUTPUT_DIR = f'{BASE}/scratch_edge_fix_test'

fm = flood_mapper(
    grid_shapefile=f'{BASE}/resources/india_utm_fishnet_buffer.gpkg',
    grid_id_list=TILE_IDS,
    dry_date_col='dry_month',
    id_col='ID',
    dry_years=[2024, 2024],
    slope_dir=f'{BASE}/resources/slope/',
    wet_duration=['2024/08', '2024/08'],
    output_dir=OUTPUT_DIR,
)

print(f'output_dir: {fm.output_dir}', flush=True)

fm.get_dry_dates()
fm.generate_dry_date_ranges()
fm.get_s1_items(dry_wet='dry')
for tid in TILE_IDS:
    print(f'[{tid}] dry scenes: {len(fm.dry_aoi_scene_dict.get(tid, []))}', flush=True)
fm.read_scenes(dry_wet='dry', overview_level=2)
fm.generate_mean_std_by_aoi()
print('mean/std computed', flush=True)

fm.prepare_slope(dem_overview=0, buffer=500)
print('slope computed', flush=True)

fm.prepare_wet_scenes(overview_level=2)
for tid in TILE_IDS:
    print(f'[{tid}] wet scenes: {len(fm.wet_scenes_by_aoi.get(tid, {}))}', flush=True)

fm.map_floods(vv_thd=-2.5, vh_thd=-2.5, rel_slope_thd=20,
              export_raster=False, export_vector=False, export_maps=False)
fm.merge_floods_by_date(export_raster=True)
fm.generate_number_of_scenes(export_raster=True)
fm.monthly_sum()

print('DONE', flush=True)
