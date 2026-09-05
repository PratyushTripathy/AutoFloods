import sys
import argparse

sys.path.append('/home/emlab/projects/current-projects/edge-autofloods/AutoFloods')

from autofloods import flood_mapper

BASE = '/home/emlab/projects/current-projects/edge-autofloods/AutoFloods'

parser = argparse.ArgumentParser()
parser.add_argument('tile_id', type=int)
args = parser.parse_args()

AOI_ID = args.tile_id
OUTPUT_DIR = f'{BASE}/output/bihar_2024_20m/tile{AOI_ID}'

fm = flood_mapper(
    grid_shapefile=f'{BASE}/resources/india_utm_fishnet_buffer.gpkg',
    grid_id_list=[AOI_ID],
    dry_date_col='dry_month',
    id_col='ID',
    dry_years=[2024, 2024],
    slope_dir=f'{BASE}/resources/slope/',
    wet_duration=['2024/07', '2024/10'],
    output_dir=OUTPUT_DIR,
)

print(f'[{AOI_ID}] output_dir: {fm.output_dir}', flush=True)

if fm.is_fully_processed(AOI_ID):
    print(f'[{AOI_ID}] already fully processed (found {fm.expected_monthly_outfile(AOI_ID)}) -- skipping.', flush=True)
    print(f'[{AOI_ID}] DONE', flush=True)
    sys.exit(0)

fm.get_dry_dates()
fm.generate_dry_date_ranges()
fm.get_s1_items(dry_wet='dry')
print(f'[{AOI_ID}] dry scenes: {len(fm.dry_aoi_scene_dict.get(AOI_ID, []))}', flush=True)
fm.read_scenes(dry_wet='dry', overview_level=0, max_workers=1)
fm.generate_mean_std_by_aoi()
print(f'[{AOI_ID}] mean/std computed', flush=True)

fm.prepare_slope(dem_overview=0, buffer=500)
print(f'[{AOI_ID}] slope computed', flush=True)

fm.prepare_wet_scenes(overview_level=0, max_workers=1)
print(f'[{AOI_ID}] wet scenes: {sum(len(v) for v in fm.wet_scene_paths.values())}', flush=True)

fm.map_floods(vv_thd=-2.5, vh_thd=-2.5, rel_slope_thd=20,
              export_vector=False, export_maps=False)
fm.merge_floods_by_date(export_raster=True)
fm.generate_number_of_scenes(export_raster=True)
fm.monthly_sum()

print(f'[{AOI_ID}] DONE', flush=True)
