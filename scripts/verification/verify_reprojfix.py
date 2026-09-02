"""One-off verification (not production): confirm the double-reprojection
fix in autofloods/preprocessing/__init__.py works end-to-end through the
real flood_mapper code path for tile 321's dry season. Not part of the
production pipeline; safe to delete once the fix is confirmed."""
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

t0 = time.time()
fm.get_dry_dates()
fm.generate_dry_date_ranges()
fm.get_s1_items(dry_wet='dry')
print(f'[{AOI_ID}] dry scenes: {len(fm.dry_aoi_scene_dict.get(AOI_ID, []))}', flush=True)
fm.read_scenes(dry_wet='dry', overview_level=None, max_workers=6)
fm.generate_mean_std_by_aoi()
print(f'[{AOI_ID}] mean/std computed in {time.time()-t0:.1f}s', flush=True)

ms = fm.mean_std_by_aoi[AOI_ID]
print(f'mean_std shape: {ms.shape}, dims: {ms.dims}')
print(f'has NaN: {bool(ms.isnull().any())}, all NaN: {bool(ms.isnull().all())}')
print(f'value range: min={float(ms.min()):.4f} max={float(ms.max()):.4f}')
print('VERIFY_OK')
