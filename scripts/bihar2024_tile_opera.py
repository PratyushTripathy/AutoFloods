"""
Production per-tile Bihar 2024 flood-mapping job -- OPERA RTC-S1 (AWS)
variant. Mirrors scripts/bihar2024_tile.py (the original MPC-backed
production script, left unmodified/still supported) but reads Sentinel-1
via autofloods.sources.OPERASource instead of the default MPCSource.

Why this exists alongside bihar2024_tile.py, not instead of it: MPC
throttles sustained reads at 20m resolution (see project history), with
no fix found after testing job-level serialization, reduced in-process
concurrency, and GDAL-level read tuning. OPERASource's download-then-
local read path was soak-tested against this exact tile's full 2024
dry+wet season workload (102 real scenes, 0 failures, ~33 min) and holds
up where MPC did not. Tradeoff: OPERA RTC-S1 is native 30m, not 20m --
this script keeps that native resolution rather than forcing MPC's
20m target, since OPERA ships no internal overview pyramid to downsample
from cheaply.

Output lands in a separate directory (bihar_2024_opera_30m/) so this
never collides with the existing MPC-backed 20m/80m runs.
"""
import sys
import argparse

sys.path.append('/home/emlab/projects/current-projects/edge-autofloods/AutoFloods')

from autofloods import flood_mapper
from autofloods.sources import OPERASource

BASE = '/home/emlab/projects/current-projects/edge-autofloods/AutoFloods'

parser = argparse.ArgumentParser()
parser.add_argument('tile_id', type=int)
args = parser.parse_args()

AOI_ID = args.tile_id
OUTPUT_DIR = f'{BASE}/output/bihar_2024_opera_30m/tile{AOI_ID}'

fm = flood_mapper(
    grid_shapefile=f'{BASE}/resources/india_utm_fishnet_buffer.gpkg',
    grid_id_list=[AOI_ID],
    dry_date_col='dry_month',
    id_col='ID',
    dry_years=[2024, 2024],
    slope_dir=f'{BASE}/resources/slope/',
    wet_duration=['2024/07', '2024/10'],
    source=OPERASource(),
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
# overview_level is ignored by OPERASource (no internal pyramid; native
# 30m). max_workers=6: both a fully-sequential soak test (102/102 passes,
# 32.7 min) and a concurrent one matching this exact setting (68/68
# passes, 5.9 min, no throttling -- also faster) have now validated
# OPERA/ASF under production concurrency, tile 321, real Apr-Oct 2024
# data.
fm.read_scenes(dry_wet='dry', overview_level=None, max_workers=6)
fm.generate_mean_std_by_aoi()
print(f'[{AOI_ID}] mean/std computed', flush=True)

fm.prepare_slope(dem_overview=0, buffer=500)
print(f'[{AOI_ID}] slope computed', flush=True)

fm.prepare_wet_scenes(overview_level=None, max_workers=6)
print(f'[{AOI_ID}] wet scenes: {sum(len(v) for v in fm.wet_scene_paths.values())}', flush=True)

fm.map_floods(vv_thd=-2.5, vh_thd=-2.5, rel_slope_thd=20,
              export_vector=False, export_maps=False)
fm.merge_floods_by_date(export_raster=True)
fm.generate_number_of_scenes(export_raster=True)
fm.monthly_sum()

print(f'[{AOI_ID}] DONE', flush=True)
