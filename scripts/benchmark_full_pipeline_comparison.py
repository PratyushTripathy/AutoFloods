"""
Real apples-to-apples full-pipeline comparison: baseline (current
sequential production code) vs optimized (ThreadPoolExecutor(max_workers=8)
around the per-scene reproject/clip loops -- the winning approach from
scripts/benchmark_parallelization.py), for tile 321's ACTUAL 2024
production settings (dry_month='04,05', wet_duration=2024/07-2024/10).

Per the user's explicit instruction: scenes are downloaded ONCE (real
network cost paid a single time) and REUSED for both timed variants --
we are not testing any change to the download step, only what happens
after the data is already in memory. Slope is also reused as-is from the
existing cache (resources/slope/slope_aoi_321.nc, already computed under
the fixed deterministic-grid code) -- not a parallelization target.

Both variants call the REAL, unmodified autofloods code
(preprocessing.clip_xarray_using_id, detectors.ZScoreDetector,
postprocessing.aggregate_monthly, flood_mapper.map_floods/
merge_floods_by_date/generate_number_of_scenes/monthly_sum) for
everything except the two loop structures being compared
(reproject_clip_stac-equivalent and stack_images-equivalent and the
wet-scene clip loop) -- those are reimplemented here in sequential and
threaded variants using the EXACT same underlying calls as production,
so the comparison measures the loop structure, not different logic.

Correctness: baseline's final monthly output is compared pixel-for-pixel
against the optimized variant's.
"""
import sys
import os
import time
import concurrent.futures

sys.path.append('/home/emlab/projects/current-projects/edge-autofloods/AutoFloods')

import numpy as np
import geopandas as gpd
import xarray as xr
import rasterio

from autofloods import flood_mapper
from autofloods.sources import OPERASource
import autofloods.preprocessing as preprocessing
import autofloods.postprocessing as postprocessing
import autofloods.utils as utils

BASE = '/home/emlab/projects/current-projects/edge-autofloods/AutoFloods'
AOI_ID = 321
MAX_WORKERS = 8

# --- Stage 0: download once (dry + wet), real 2024 production settings ---

print('=== Stage 0: download tile 321 dry+wet scenes ONCE (real 2024 production settings) ===', flush=True)
fm_dl = flood_mapper(
    grid_shapefile=f'{BASE}/resources/india_utm_fishnet_buffer.gpkg',
    grid_id_list=[AOI_ID], dry_date_col='dry_month', id_col='ID',
    dry_years=[2024, 2024], slope_dir=f'{BASE}/resources/slope',
    wet_duration=['2024/07', '2024/10'], source=OPERASource(),
    output_dir=f'{BASE}/output/_bench_fullpipeline/download_scratch',
)
t0 = time.time()
fm_dl.get_dry_dates()
fm_dl.generate_dry_date_ranges()
fm_dl.get_s1_items(dry_wet='dry')
fm_dl.read_scenes(dry_wet='dry', overview_level=None, max_workers=6)
n_dry = len(fm_dl.dry_aoi_scene_dict.get(AOI_ID, []))
print(f'  dry scenes: {n_dry}', flush=True)

fm_dl.get_s1_items(dry_wet='wet')
fm_dl.read_scenes(dry_wet='wet', overview_level=None, max_workers=6)
n_wet = len(fm_dl.wet_aoi_scene_dict.get(AOI_ID, []))
print(f'  wet scenes: {n_wet}', flush=True)
print(f'  download total: {time.time()-t0:.1f}s', flush=True)

CACHED = {
    's1_dry_dict': fm_dl.s1_dry_dict,
    's1_wet_dict': fm_dl.s1_wet_dict,
}


def make_flood_mapper(output_dir):
    """Fresh flood_mapper instance, real search (cheap, not a download),
    scene DOWNLOAD reused from CACHED (skips read_scenes' network call)."""
    fm = flood_mapper(
        grid_shapefile=f'{BASE}/resources/india_utm_fishnet_buffer.gpkg',
        grid_id_list=[AOI_ID], dry_date_col='dry_month', id_col='ID',
        dry_years=[2024, 2024], slope_dir=f'{BASE}/resources/slope',
        wet_duration=['2024/07', '2024/10'], source=OPERASource(),
        output_dir=output_dir,
    )
    fm.get_dry_dates()
    fm.generate_dry_date_ranges()
    fm.get_s1_items(dry_wet='dry')          # real search, cheap, not a download
    fm.s1_dry_dict = CACHED['s1_dry_dict']  # reuse cached download
    fm.get_s1_items(dry_wet='wet')
    fm.s1_wet_dict = CACHED['s1_wet_dict']  # reuse cached download
    return fm


# --- Sequential (baseline) and threaded (optimized) stage implementations ---
# Mirror production's exact logic (autofloods/preprocessing/__init__.py,
# autofloods/__init__.py) -- only the loop structure (sequential vs
# ThreadPoolExecutor) differs; every underlying call is the real,
# unmodified production function.

def reproject_clip_stac_seq(s1_dict, aoi_scene_dict, grid_shapefile_path, id):
    return preprocessing.reproject_clip_stac(s1_dict, aoi_scene_dict, grid_shapefile_path, id)


def reproject_clip_stac_threaded(s1_dict, aoi_scene_dict, grid_shapefile_path, id, max_workers=MAX_WORKERS):
    gdf = gpd.read_file(grid_shapefile_path)
    gdf = gdf.loc[gdf['ID'].isin([id])]
    tile_utm_zone = 'EPSG:326{}'.format(gdf['zone'].values[0][:-1])
    gdf = gdf.to_crs(tile_utm_zone)

    def _one(stac_id):
        return stac_id, {
            'vv_ds': s1_dict[stac_id]['vv_ds'].rio.reproject(tile_utm_zone).rio.clip(gdf.geometry),
            'vh_ds': s1_dict[stac_id]['vh_ds'].rio.reproject(tile_utm_zone).rio.clip(gdf.geometry),
        }
    out = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
        for stac_id, result in ex.map(_one, aoi_scene_dict[id]):
            out[stac_id] = result
    return out


def stack_images_seq(clipped_dict, grid_shapefile_path, id):
    return preprocessing.stack_images(clipped_dict, grid_shapefile_path, id)


def stack_images_threaded(clipped_dict, grid_shapefile_path, id, max_workers=MAX_WORKERS):
    stacked_images = [clipped_dict[stac_id] for stac_id in clipped_dict]
    ref = stacked_images[0]['vv_ds']

    def _one(item):
        return {
            'vv_ds': preprocessing.clip_xarray_using_id(item['vv_ds'], grid_shapefile_path, id, ref),
            'vh_ds': preprocessing.clip_xarray_using_id(item['vh_ds'], grid_shapefile_path, id, ref),
        }
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
        stacked_images = list(ex.map(_one, stacked_images))

    vv_stack = xr.concat([item['vv_ds'] for item in stacked_images], dim="band")
    vh_stack = xr.concat([item['vh_ds'] for item in stacked_images], dim="band")
    return {'vv_stack': vv_stack, 'vh_stack': vh_stack}


def wet_scenes_seq(fm):
    return {
        id: {
            scene_id: xr.concat([
                preprocessing.clip_xarray_using_id(fm.s1_wet_dict[scene_id]['vv_ds'], fm.grid_shapefile_path, id, fm.mean_std_by_aoi[id]),
                preprocessing.clip_xarray_using_id(fm.s1_wet_dict[scene_id]['vh_ds'], fm.grid_shapefile_path, id, fm.mean_std_by_aoi[id]),
            ], dim='band').assign_coords(band=['vv_ds', 'vh_ds'])
            for scene_id in fm.wet_aoi_scene_dict[id]
        }
        for id in fm.wet_aoi_scene_dict
    }


def wet_scenes_threaded(fm, max_workers=MAX_WORKERS):
    out = {}
    for id in fm.wet_aoi_scene_dict:
        def _one(scene_id):
            return scene_id, xr.concat([
                preprocessing.clip_xarray_using_id(fm.s1_wet_dict[scene_id]['vv_ds'], fm.grid_shapefile_path, id, fm.mean_std_by_aoi[id]),
                preprocessing.clip_xarray_using_id(fm.s1_wet_dict[scene_id]['vh_ds'], fm.grid_shapefile_path, id, fm.mean_std_by_aoi[id]),
            ], dim='band').assign_coords(band=['vv_ds', 'vh_ds'])
        scene_out = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
            for scene_id, result in ex.map(_one, fm.wet_aoi_scene_dict[id]):
                scene_out[scene_id] = result
        out[id] = scene_out
    return out


def run_variant(name, threaded):
    print(f'\n=== Variant: {name} ===', flush=True)
    out_dir = f'{BASE}/output/_bench_fullpipeline/{name}'
    fm = make_flood_mapper(out_dir)

    t0 = time.time()

    # dry-season reproject+clip+stack (generate_mean_std_by_aoi's body,
    # with the loop implementation swapped in)
    reproject_fn = reproject_clip_stac_threaded if threaded else reproject_clip_stac_seq
    stack_fn = stack_images_threaded if threaded else stack_images_seq

    reprojected_clipped_dry = {
        id: reproject_fn(fm.s1_dry_dict, fm.dry_aoi_scene_dict, fm.grid_shapefile_path, id)
        for id in fm.selected_grid_id
    }
    stacked_dry = {id: stack_fn(reprojected_clipped_dry[id], fm.grid_shapefile_path, id) for id in reprojected_clipped_dry}
    for id in stacked_dry:
        for n in range(len(stacked_dry[id]['vv_stack'])):
            stacked_dry[id]['vv_stack'][n] = stacked_dry[id]['vv_stack'][n].where(stacked_dry[id]['vv_stack'][n] < 50, np.nan)
            stacked_dry[id]['vh_stack'][n] = stacked_dry[id]['vh_stack'][n].where(stacked_dry[id]['vh_stack'][n] < 50, np.nan)
    fm.mean_std_by_aoi = {
        id: fm.detector.fit_baseline(stacked_dry[id]['vv_stack'], stacked_dry[id]['vh_stack'])
        for id in stacked_dry
    }
    t_dry = time.time() - t0
    print(f'  dry-season baseline (reproject+stack): {t_dry:.1f}s', flush=True)

    # slope: reuse cache as-is, not a parallelization target
    t0 = time.time()
    fm.prepare_slope(dem_overview=0, buffer=500)
    t_slope = time.time() - t0
    print(f'  slope (cached, not parallelized): {t_slope:.1f}s', flush=True)

    # wet-season reproject+clip
    t0 = time.time()
    wet_fn = wet_scenes_threaded if threaded else wet_scenes_seq
    fm.wet_scenes_by_aoi = wet_fn(fm)
    for id in fm.wet_scenes_by_aoi:
        for scene_id in fm.wet_scenes_by_aoi[id]:
            fm.wet_scenes_by_aoi[id][scene_id] = fm.wet_scenes_by_aoi[id][scene_id].where(
                fm.wet_scenes_by_aoi[id][scene_id] < 50, np.nan)
    t_wet = time.time() - t0
    print(f'  wet-season reproject+clip: {t_wet:.1f}s', flush=True)

    # detection, merge, monthly -- REAL production code, unmodified, same for both variants
    t0 = time.time()
    fm.map_floods(vv_thd=-2.5, vh_thd=-2.5, rel_slope_thd=20,
                  export_raster=False, export_vector=False, export_maps=False)
    fm.merge_floods_by_date(export_raster=True)
    fm.generate_number_of_scenes(export_raster=True)
    fm.monthly_sum()
    t_rest = time.time() - t0
    print(f'  detect+merge+monthly (unmodified, same both variants): {t_rest:.1f}s', flush=True)

    total = t_dry + t_slope + t_wet + t_rest
    print(f'  TOTAL post-download time: {total:.1f}s', flush=True)
    return {
        'name': name, 't_dry': t_dry, 't_slope': t_slope, 't_wet': t_wet,
        't_rest': t_rest, 'total': total,
        'monthly_outfile': fm.expected_monthly_outfile(AOI_ID),
    }


if __name__ == '__main__':
    baseline = run_variant('baseline_sequential', threaded=False)
    optimized = run_variant('optimized_threadpool8', threaded=True)

    print('\n=== CORRECTNESS CHECK (final monthly output) ===', flush=True)
    with rasterio.open(baseline['monthly_outfile']) as src_b, rasterio.open(optimized['monthly_outfile']) as src_o:
        arr_b = src_b.read()
        arr_o = src_o.read()
        same_shape = arr_b.shape == arr_o.shape
        max_diff = float(np.nanmax(np.abs(arr_b.astype(float) - arr_o.astype(float)))) if same_shape else None
    print(f'shapes match: {same_shape}, max_abs_diff: {max_diff}', flush=True)

    print('\n=== SUMMARY ===', flush=True)
    print(f"baseline  : dry={baseline['t_dry']:.1f}s slope={baseline['t_slope']:.1f}s "
          f"wet={baseline['t_wet']:.1f}s rest={baseline['t_rest']:.1f}s TOTAL={baseline['total']:.1f}s", flush=True)
    print(f"optimized : dry={optimized['t_dry']:.1f}s slope={optimized['t_slope']:.1f}s "
          f"wet={optimized['t_wet']:.1f}s rest={optimized['t_rest']:.1f}s TOTAL={optimized['total']:.1f}s", flush=True)
    saved = baseline['total'] - optimized['total']
    pct = saved / baseline['total'] * 100
    print(f"SAVED: {saved:.1f}s ({pct:.1f}% of post-download total)", flush=True)
    print('DONE', flush=True)
