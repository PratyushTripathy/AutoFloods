"""
Side test (not part of the production pipeline): compare the current
eager, per-scene-download approach against a lazy Dask-graph approach for
computing dry-season VV mean/std at 20m (overview_level=0).

Both approaches read the SAME scenes for the SAME tile, so the timings are
directly comparable. Nothing here is imported by or changes the real
pipeline in autofloods/.
"""
import sys
import time
import datetime

import pathlib as _pathlib
BASE = str(_pathlib.Path(__file__).resolve().parents[2])  # repo root (scripts/verification/<this file>)
sys.path.append(BASE)

import numpy as np
import rioxarray
import xarray as xr
import rasterio

from autofloods.sources import MPCSource
from autofloods.utils import GDAL_HTTP_ENV

TILE_BBOX = {
    'type': 'Polygon',
    'coordinates': [[
        [85.998, 24.998], [87.002, 24.998], [87.002, 26.002], [85.998, 26.002], [85.998, 24.998]
    ]],
}  # tile 321's bbox
MAX_SCENES = 6  # cap for a fast, apples-to-apples comparison test
OVERVIEW_LEVEL = 0  # 20m


def get_vv_hrefs(source, n):
    items = source.search_sentinel1(TILE_BBOX, datetime.date(2024, 7, 1), datetime.date(2024, 7, 31))
    hrefs = []
    for item in items[:n]:
        vv_href, _ = source.vv_vh_hrefs(item)
        hrefs.append(vv_href)
    return hrefs


def eager_approach(hrefs):
    print(f'\n=== EAGER (current pipeline behavior): {len(hrefs)} scenes ===', flush=True)
    t0 = time.time()
    arrays = []
    with rasterio.Env(**GDAL_HTTP_ENV):
        for i, href in enumerate(hrefs):
            t_scene = time.time()
            da = rioxarray.open_rasterio(href, overview_level=OVERVIEW_LEVEL, masked=True).load()
            arrays.append(da)
            print(f'  scene {i+1}/{len(hrefs)}: {time.time()-t_scene:.1f}s, shape={da.shape}', flush=True)
    t_read = time.time() - t0

    t0 = time.time()
    stack = xr.concat(arrays, dim='scene')
    mean = stack.mean(dim='scene').compute() if hasattr(stack.mean(dim='scene'), 'compute') else stack.mean(dim='scene')
    std = stack.std(dim='scene').compute() if hasattr(stack.std(dim='scene'), 'compute') else stack.std(dim='scene')
    t_reduce = time.time() - t0

    print(f'  total read time: {t_read:.1f}s, reduction time: {t_reduce:.1f}s, TOTAL: {t_read+t_reduce:.1f}s', flush=True)
    return t_read + t_reduce, mean, std


def lazy_dask_approach(hrefs):
    print(f'\n=== LAZY DASK: {len(hrefs)} scenes ===', flush=True)
    t0 = time.time()
    arrays = []
    with rasterio.Env(**GDAL_HTTP_ENV):
        for i, href in enumerate(hrefs):
            # chunks='auto' (or a dict) makes rioxarray back the array with
            # a dask array instead of eagerly reading into a numpy array --
            # no bytes are transferred yet at this point.
            da = rioxarray.open_rasterio(href, overview_level=OVERVIEW_LEVEL, masked=True, chunks=True)
            arrays.append(da)
    t_graph_build = time.time() - t0
    print(f'  graph build (lazy open, no data transferred yet): {t_graph_build:.1f}s', flush=True)
    print(f'  backing array type: {type(arrays[0].data)}', flush=True)

    t0 = time.time()
    stack = xr.concat(arrays, dim='scene')
    mean_lazy = stack.mean(dim='scene')
    std_lazy = stack.std(dim='scene')
    t_graph_construct = time.time() - t0
    print(f'  reduction graph construction (still lazy): {t_graph_construct:.1f}s', flush=True)

    # single .compute() call executes the whole graph in one pass --
    # this is where all the actual data transfer + computation happens
    t0 = time.time()
    mean, std = xr.compute(mean_lazy, std_lazy)
    t_compute = time.time() - t0
    print(f'  .compute() (actual transfer + reduction): {t_compute:.1f}s', flush=True)

    total = t_graph_build + t_graph_construct + t_compute
    print(f'  TOTAL: {total:.1f}s', flush=True)
    return total, mean, std


if __name__ == '__main__':
    source = MPCSource()
    print(f'Searching for up to {MAX_SCENES} dry-season scenes over tile 321 bbox, July 2024...', flush=True)
    hrefs = get_vv_hrefs(source, MAX_SCENES)
    print(f'Found {len(hrefs)} scenes', flush=True)

    eager_time, eager_mean, eager_std = eager_approach(hrefs)
    lazy_time, lazy_mean, lazy_std = lazy_dask_approach(hrefs)

    print('\n=== COMPARISON ===', flush=True)
    print(f'Eager (current pipeline): {eager_time:.1f}s', flush=True)
    print(f'Lazy Dask:                {lazy_time:.1f}s', flush=True)
    print(f'Speedup: {eager_time/lazy_time:.2f}x', flush=True)

    # sanity check: results should match (within floating point tolerance)
    mean_diff = np.nanmax(np.abs(eager_mean.values - lazy_mean.values))
    std_diff = np.nanmax(np.abs(eager_std.values - lazy_std.values))
    print(f'\nmax mean difference: {mean_diff:.6f}, max std difference: {std_diff:.6f} (should be ~0)', flush=True)
