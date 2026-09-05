"""
Tier 0.2 resumability demonstration for the SoftwareX manuscript. Not part
of the package -- a one-off verification script.

Simulates: (1) a run that completes the dry-season baseline + slope, then
is killed before wet-season processing; (2) a restart with an identical
command, showing it skips the completed baseline/slope steps and resumes
directly at wet-season processing. Prints wall-clock for both phases so the
manuscript can report a concrete resumed-vs-cold-run wall-clock comparison.
"""
import sys
import time

import pathlib as _pathlib
BASE = str(_pathlib.Path(__file__).resolve().parents[2])  # repo root (scripts/verification/<this file>)
sys.path.append(BASE)

from autofloods import flood_mapper
from autofloods.sources import OPERASource
from autofloods.detectors import ZScoreDetector

GRID_SHAPEFILE = f'{BASE}/resources/india_utm_fishnet_buffer.gpkg'
OUTPUT_DIR = f'{BASE}/output/_tier0_resumability/tile318'
SLOPE_DIR = f'{BASE}/resources/slope'
TILE_ID = 318


def build_fm():
    return flood_mapper(
        grid_shapefile=GRID_SHAPEFILE,
        grid_id_list=[TILE_ID],
        dry_years=[2024, 2024],
        wet_duration=['2024/07', '2024/10'],
        slope_dir=SLOPE_DIR,
        source=OPERASource(),
        detector=ZScoreDetector(),
        output_dir=OUTPUT_DIR,
        cell_size=30,
    )


print('=== PHASE 1: cold run through dry baseline + slope only (simulated pre-kill state) ===', flush=True)
t0 = time.monotonic()
fm1 = build_fm()
fm1.get_dry_dates()
fm1.generate_dry_date_ranges()
fm1.get_s1_items(dry_wet='dry')
fm1.read_scenes(dry_wet='dry', overview_level=None, max_workers=6)
fm1.generate_mean_std_by_aoi(reproject_max_workers=None)
fm1.prepare_slope(dem_overview=0, buffer=500, max_workers=6)
t1 = time.monotonic()
print(f'PHASE 1 elapsed (dry baseline + slope): {t1 - t0:.1f}s', flush=True)
print('=== SIMULATING KILL: process ends here, wet-season work never started ===', flush=True)

print('=== PHASE 2: restart with an identical command (fresh flood_mapper instance) ===', flush=True)
t2 = time.monotonic()
fm2 = build_fm()
if fm2.is_fully_processed(TILE_ID):
    print(f'[{TILE_ID}] already fully processed -- skipping (unexpected for this test)', flush=True)
else:
    fm2.get_dry_dates()
    fm2.generate_dry_date_ranges()
    fm2.get_s1_items(dry_wet='dry')
    fm2.read_scenes(dry_wet='dry', overview_level=None, max_workers=6)
    fm2.generate_mean_std_by_aoi(reproject_max_workers=None)
    fm2.prepare_slope(dem_overview=0, buffer=500, max_workers=6)
    fm2.prepare_wet_scenes(overview_level=None, max_workers=6, reproject_max_workers=None)
    fm2.map_floods(vv_thd=-2.5, vh_thd=-2.5, rel_slope_thd=20,
                    export_vector=False, export_maps=False)
    fm2.merge_floods_by_date(export_raster=True)
    fm2.generate_number_of_scenes(export_raster=True)
    fm2.monthly_sum()
t3 = time.monotonic()
print(f'PHASE 2 elapsed (restart -> finish): {t3 - t2:.1f}s', flush=True)
print(f'TOTAL (phase 1 + phase 2): {(t1 - t0) + (t3 - t2):.1f}s', flush=True)
print('DONE', flush=True)
