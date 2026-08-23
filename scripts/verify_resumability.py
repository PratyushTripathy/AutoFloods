"""One-off verification (not production): confirm resumability fix #3
end-to-end. Points at the SAME output dir used by verify_reprojfix.py,
which already has a completed dry-season baseline (.nc) for tile 321 from
a prior run (job 1080918) but nothing past that. This run should:
  1. NOT be flagged as fully processed (monthly output doesn't exist yet).
  2. Skip dry-season search/read/baseline computation entirely (reload
     the existing .nc instead) -- verified by timing: this should be much
     faster than the ~183s the original baseline computation took.
  3. Proceed through slope/wet-season/detection/monthly without crashing.
Not part of the production pipeline; safe to delete once confirmed."""
import sys
import time

sys.path.append('/home/emlab/projects/current-projects/edge-autofloods/AutoFloods')
from autofloods import flood_mapper
from autofloods.sources import OPERASource

BASE = '/home/emlab/projects/current-projects/edge-autofloods/AutoFloods'
AOI_ID = 321

fm = flood_mapper(
    grid_shapefile=f'{BASE}/resources/india_utm_fishnet_buffer.gpkg',
    grid_id_list=[AOI_ID],
    dry_date_col='dry_month',
    id_col='ID',
    dry_years=[2024, 2024],
    slope_dir=f'{BASE}/resources/slope/',
    wet_duration=['2024/07', '2024/07'],
    source=OPERASource(),
    output_dir=f'{BASE}/output/verify_reprojfix/tile{AOI_ID}',
)

print(f'[{AOI_ID}] output_dir: {fm.output_dir}', flush=True)
print(f'[{AOI_ID}] is_fully_processed: {fm.is_fully_processed(AOI_ID)} '
      f'(expect False -- monthly output not done yet)', flush=True)

t0 = time.time()
fm.get_dry_dates()
fm.generate_dry_date_ranges()
fm.get_s1_items(dry_wet='dry')
n_dry_scenes = len(fm.dry_aoi_scene_dict.get(AOI_ID, []))
print(f'[{AOI_ID}] dry scenes found this run: {n_dry_scenes} (expect 0 -- baseline already done, '
      f'nothing to search/read)', flush=True)
fm.read_scenes(dry_wet='dry', overview_level=None, max_workers=6)
fm.generate_mean_std_by_aoi()
baseline_time = time.time() - t0
print(f'[{AOI_ID}] baseline stage took {baseline_time:.1f}s (expect fast -- reloaded from disk, '
      f'not recomputed; original computation took 183.1s)', flush=True)

ms = fm.mean_std_by_aoi[AOI_ID]
print(f'[{AOI_ID}] mean_std shape: {ms.shape}, has values: {not bool(ms.isnull().all())}', flush=True)

fm.prepare_slope(dem_overview=0, buffer=500)
print(f'[{AOI_ID}] slope computed', flush=True)

fm.prepare_wet_scenes(overview_level=None, max_workers=6)
print(f'[{AOI_ID}] wet scenes: {sum(len(v) for v in fm.wet_scenes_by_aoi.values())}', flush=True)

fm.map_floods(vv_thd=-2.5, vh_thd=-2.5, rel_slope_thd=20,
              export_raster=False, export_vector=False, export_maps=False)
fm.merge_floods_by_date(export_raster=True)
fm.generate_number_of_scenes(export_raster=True)
fm.monthly_sum()

print(f'[{AOI_ID}] is_fully_processed after this run: {fm.is_fully_processed(AOI_ID)} (expect True)', flush=True)
print('VERIFY_OK')
