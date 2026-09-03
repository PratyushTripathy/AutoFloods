"""
Benchmark: post-download processing optimizations, tile 321, real OPERA
dry-season scenes downloaded ONCE and reused across every subsequent
stage -- so network cost is identical and constant across all
comparisons, and any timing difference is attributable only to the code
change under test, not to network variance. Stages run strictly
sequentially (no stage overlaps another, nothing here uses more than one
concurrency mechanism at a time except where that IS the thing being
measured, e.g. stage 4).

Stages:
  0. Download + cache tile 321's dry-season scenes (native CRS, post
     decibel_to_linear, pre-reproject) -- once.
  1. Reprojection: current double-reproject (native -> EPSG:4326 -> tile
     UTM) vs a direct single-reproject (native -> tile UTM), on the same
     cached scenes. Includes a correctness check (max abs pixel diff).
  2. Export: current uncompressed float32 GeoTIFF write
     (autofloods.utils.export_xarray) vs the same write with
     COMPRESS=DEFLATE, PREDICTOR=2, TILED=YES.
  3. GDAL warp multithreading: rioxarray .rio.reproject() with default
     (single-threaded) vs num_threads=<ncpu>.
  4. Parallel scene processing: current ThreadPoolExecutor(max_workers=6)
     vs a dask.distributed LocalCluster (6 worker processes), reprojecting
     the full cached scene set both ways.

Nothing here modifies autofloods/ source or writes into production
output directories -- this is a measurement script only. Results are
printed as they complete and collected into a final summary table.
"""
import sys
import os
import time
import shutil
import tempfile
import multiprocessing
import concurrent.futures
import datetime

import pathlib as _pathlib
BASE = str(_pathlib.Path(__file__).resolve().parents[2])  # repo root (scripts/verification/<this file>)
sys.path.append(BASE)

import numpy as np
import geopandas as gpd
import rasterio
import xarray as xr

from autofloods.sources import OPERASource
from autofloods.utils import decibel_to_linear, export_xarray

AOI_ID = 321
GRID = f'{BASE}/resources/india_utm_fishnet_buffer.gpkg'
BBOX = {
    'type': 'Polygon',
    'coordinates': [[
        [85.998, 24.998], [87.002, 24.998], [87.002, 26.002], [85.998, 26.002], [85.998, 24.998]
    ]],
}
START = datetime.date(2024, 4, 1)
END = datetime.date(2024, 5, 31)

results = []


def log_stage(name, seconds, extra=None):
    entry = {'stage': name, 'seconds': round(seconds, 2) if seconds is not None else None}
    if extra:
        entry.update(extra)
    results.append(entry)
    tail = f' ({extra})' if extra else ''
    print(f'[RESULT] {name}: {seconds:.2f}s{tail}' if seconds is not None else f'[RESULT] {name}: FAILED{tail}',
          flush=True)


def export_xarray_compressed(xarray_data, filename):
    """Same shape/logic as autofloods.utils.export_xarray, but with
    COMPRESS=DEFLATE, PREDICTOR=2, TILED=YES added to the GTiff profile."""
    with rasterio.Env():
        xmin, ymin, xmax, ymax = [
            xarray_data.x.min().values, xarray_data.y.min().values,
            xarray_data.x.max().values, xarray_data.y.max().values,
        ]
        if len(xarray_data.dims) == 3:
            bands, rows, cols = xarray_data.shape
        elif len(xarray_data.dims) == 2:
            rows, cols = xarray_data.shape
        else:
            raise ValueError('unexpected ndim')

        profile = {
            'driver': 'GTiff', 'dtype': rasterio.float32, 'nodata': np.nan,
            'width': cols, 'height': rows,
            'count': xarray_data.shape[0] if len(xarray_data.dims) == 3 else 1,
            'transform': rasterio.transform.from_bounds(xmin, ymin, xmax, ymax, cols, rows),
            'compress': 'DEFLATE', 'predictor': 2, 'tiled': True,
        }
        with rasterio.open(filename, 'w', **profile) as dst:
            if len(xarray_data.dims) == 2:
                dst.write(xarray_data.data, 1)
            else:
                for band in range(xarray_data.shape[0]):
                    dst.write(xarray_data.data[band, :, :], band + 1)


if __name__ == '__main__':
    print('=== Stage 0: download + cache tile 321 dry-season scenes (once) ===', flush=True)
    source = OPERASource()
    t0 = time.time()
    passes = source.search_sentinel1(BBOX, START, END)
    print(f'Found {len(passes)} passes', flush=True)

    native_scenes = {}
    for p in passes:
        vv_ds, vh_ds = source.read_vv_vh(p)
        native_scenes[p.id] = {
            'vv_ds': decibel_to_linear(vv_ds),
            'vh_ds': decibel_to_linear(vh_ds),
        }
    download_time = time.time() - t0
    log_stage('stage0_download_and_cache', download_time, {'n_scenes': len(native_scenes)})

    gdf_all = gpd.read_file(GRID)
    gdf = gdf_all.loc[gdf_all['ID'] == AOI_ID]
    tile_utm_zone = 'EPSG:326{}'.format(gdf['zone'].values[0][:-1])
    gdf_utm = gdf.to_crs(tile_utm_zone)

    print('\n=== Stage 1: reprojection (current double-reproject vs direct) ===', flush=True)
    t0 = time.time()
    double_results = {}
    for sid, ds in native_scenes.items():
        vv_4326 = ds['vv_ds'].rio.reproject('EPSG:4326')
        vh_4326 = ds['vh_ds'].rio.reproject('EPSG:4326')
        double_results[sid] = {
            'vv': vv_4326.rio.reproject(tile_utm_zone).rio.clip(gdf_utm.geometry),
            'vh': vh_4326.rio.reproject(tile_utm_zone).rio.clip(gdf_utm.geometry),
        }
    double_time = time.time() - t0
    log_stage('stage1_double_reproject_native_4326_utm', double_time)

    t0 = time.time()
    direct_results = {}
    for sid, ds in native_scenes.items():
        direct_results[sid] = {
            'vv': ds['vv_ds'].rio.reproject(tile_utm_zone).rio.clip(gdf_utm.geometry),
            'vh': ds['vh_ds'].rio.reproject(tile_utm_zone).rio.clip(gdf_utm.geometry),
        }
    direct_time = time.time() - t0
    log_stage('stage1_direct_reproject_native_utm', direct_time)

    try:
        any_sid = next(iter(native_scenes))
        a = double_results[any_sid]['vv'].values
        b = direct_results[any_sid]['vv'].values
        if a.shape == b.shape:
            max_diff = float(np.nanmax(np.abs(a - b)))
        else:
            max_diff = None
        log_stage('stage1_correctness_check', 0, {
            'sample_scene': any_sid, 'shapes_match': a.shape == b.shape, 'max_abs_diff': max_diff,
        })
    except Exception as exc:
        log_stage('stage1_correctness_check', 0, {'error': repr(exc)})

    print('\n=== Stage 2: export (uncompressed vs DEFLATE+PREDICTOR+TILED) ===', flush=True)
    outdir = tempfile.mkdtemp(prefix='bench_export_')
    sample = direct_results[any_sid]['vv']
    try:
        t0 = time.time()
        p1 = os.path.join(outdir, 'uncompressed.tif')
        export_xarray(sample, p1)
        uncompressed_time = time.time() - t0
        log_stage('stage2_export_uncompressed', uncompressed_time, {'size_bytes': os.path.getsize(p1)})

        t0 = time.time()
        p2 = os.path.join(outdir, 'compressed.tif')
        export_xarray_compressed(sample, p2)
        compressed_time = time.time() - t0
        log_stage('stage2_export_compressed_deflate', compressed_time, {'size_bytes': os.path.getsize(p2)})
    finally:
        shutil.rmtree(outdir, ignore_errors=True)

    print('\n=== Stage 3: GDAL warp multithreading ===', flush=True)
    ncpu = multiprocessing.cpu_count()
    t0 = time.time()
    for sid, ds in native_scenes.items():
        _ = ds['vv_ds'].rio.reproject(tile_utm_zone)
    default_thread_time = time.time() - t0
    log_stage('stage3_reproject_default_threads', default_thread_time)

    try:
        t0 = time.time()
        for sid, ds in native_scenes.items():
            _ = ds['vv_ds'].rio.reproject(tile_utm_zone, num_threads=ncpu)
        multi_thread_time = time.time() - t0
        log_stage(f'stage3_reproject_num_threads_{ncpu}', multi_thread_time)
    except TypeError:
        with rasterio.Env(GDAL_NUM_THREADS=str(ncpu)):
            t0 = time.time()
            for sid, ds in native_scenes.items():
                _ = ds['vv_ds'].rio.reproject(tile_utm_zone)
            multi_thread_time = time.time() - t0
        log_stage(f'stage3_reproject_GDAL_NUM_THREADS_{ncpu}_env', multi_thread_time)

    print('\n=== Stage 4: parallel scene processing (ThreadPoolExecutor vs Dask) ===', flush=True)

    def reproject_one(item):
        sid, ds = item
        _ = ds['vv_ds'].rio.reproject(tile_utm_zone).rio.clip(gdf_utm.geometry)
        _ = ds['vh_ds'].rio.reproject(tile_utm_zone).rio.clip(gdf_utm.geometry)
        return sid

    t0 = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
        list(executor.map(reproject_one, native_scenes.items()))
    threadpool_time = time.time() - t0
    log_stage('stage4_threadpoolexecutor_max_workers_6', threadpool_time)

    try:
        from dask.distributed import Client, LocalCluster
        t0 = time.time()
        cluster = LocalCluster(n_workers=6, threads_per_worker=1, processes=True)
        client = Client(cluster)
        futures = client.map(reproject_one, list(native_scenes.items()))
        client.gather(futures)
        client.close()
        cluster.close()
        dask_time = time.time() - t0
        log_stage('stage4_dask_localcluster_6_workers', dask_time)
    except Exception as exc:
        log_stage('stage4_dask_localcluster_6_workers', None, {'error': repr(exc)})

    print('\n=== SUMMARY ===', flush=True)
    for r in results:
        print(r, flush=True)
    print('DONE', flush=True)
