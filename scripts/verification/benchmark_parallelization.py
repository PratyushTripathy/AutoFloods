"""
Parallelization benchmark for post-download compute, per the approved plan
at ~/.claude/plans/curious-drifting-lake.md. Tile 321's real dry-season
scenes are downloaded ONCE (Stage 0) and reused for every arm below, so
network cost is identical across all comparisons and any timing
difference is attributable only to the parallelization strategy under
test -- same methodology as scripts/benchmark_post_download.py.

Arms tested on Stage A (dry-season reproject+clip -- the same mechanism
as Stage B/C, and the biggest sequential chunk of post-download compute):
  0. Baseline -- sequential scene loop, GDAL num_threads=default(1).
  1. GDAL num_threads only -- N=2,4,8, sequential scene loop.
  2. ThreadPoolExecutor across scenes only -- workers=2,4,6,8, GDAL threads=1.
  3. Combined ThreadPoolExecutor x GDAL num_threads -- splits totaling <=8
     total (4x2, 2x4).
  4. ProcessPoolExecutor across scenes -- workers=4.
  5. dask.array chunked reproject, in-process threaded scheduler (NOT
     distributed/LocalCluster, which was already tested and ruled out in
     the prior benchmark -- 3.2x slower than ThreadPoolExecutor).

Every arm's output is checked against the baseline's output (max abs
pixel diff) before its timing is trusted -- a fast-but-wrong result is
reported as FAILED, not fast. CPU utilization is sampled via psutil
during each arm's run (process-tree-wide, to capture worker
threads/processes). Each arm runs REPS times; median is reported.

This is a MEASUREMENT script only -- no autofloods/ source is modified.
"""
import sys
import os
import time
import threading
import concurrent.futures
import multiprocessing
import datetime

import pathlib as _pathlib
BASE = str(_pathlib.Path(__file__).resolve().parents[2])  # repo root (scripts/verification/<this file>)
sys.path.append(BASE)

import numpy as np
import psutil
import geopandas as gpd
import xarray as xr
import rioxarray  # noqa: F401 -- registers .rio accessor
import rasterio

from autofloods.sources import OPERASource
from autofloods.utils import decibel_to_linear

# GDAL warp memory: research (rasterio/GDAL docs) explicitly warns that
# raising num_threads without also raising warp memory just makes
# threads contend for a too-small shared buffer, understating any real
# speedup. 512MB is a generous bump for ~4000x4000 float32 arrays
# (~64MB/band) without over-committing a shared 8-CPU node.
os.environ.setdefault('GDAL_CACHEMAX', '512')

# multiprocessing.cpu_count() reports the WHOLE NODE's CPU count on a
# shared cluster node, not this job's actual SLURM allocation -- e.g. 72
# on hpc-06 even for an 8-CPU job. SLURM_CPUS_PER_TASK is the correct
# source for "how many CPUs does THIS job actually have," matching
# production jobs' --cpus-per-task=8, and is what "% of allocated CPUs
# used" should be normalized against.
ALLOCATED_CPUS = int(os.environ.get('SLURM_CPUS_PER_TASK', multiprocessing.cpu_count()))

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
REPS = 3

results = []
_proc = psutil.Process(os.getpid())


def _cpu_sampler(stop_event, samples, interval=0.2):
    """Sample this process + all children's CPU% every `interval` seconds
    until stop_event is set. Captures worker threads (counted under the
    parent process by psutil) and worker processes (children) alike."""
    _proc.cpu_percent(interval=None)  # prime the counter
    while not stop_event.is_set():
        time.sleep(interval)
        try:
            total = _proc.cpu_percent(interval=None)
            for child in _proc.children(recursive=True):
                try:
                    total += child.cpu_percent(interval=None)
                except (psutil.NoSuchProcess, psutil.ZombieProcess):
                    pass
            samples.append(total)
        except Exception:
            pass


def timed_run(fn, *args, **kwargs):
    """Run fn, sampling CPU% throughout. Returns (result, elapsed, avg_cpu_pct)."""
    samples = []
    stop_event = threading.Event()
    sampler = threading.Thread(target=_cpu_sampler, args=(stop_event, samples), daemon=True)
    sampler.start()
    t0 = time.time()
    result = fn(*args, **kwargs)
    elapsed = time.time() - t0
    stop_event.set()
    sampler.join(timeout=2)
    avg_cpu = float(np.mean(samples)) if samples else 0.0
    return result, elapsed, avg_cpu


def log_result(arm, elapsed_list, cpu_list, correctness_ok, extra=None):
    med_elapsed = float(np.median(elapsed_list))
    med_cpu = float(np.median(cpu_list))
    ncpu = ALLOCATED_CPUS
    entry = {
        'arm': arm, 'median_seconds': round(med_elapsed, 2),
        'median_cpu_pct': round(med_cpu, 1),
        'median_cpu_pct_of_allocated': round(med_cpu / (ncpu * 100) * 100, 1),
        'correctness_ok': correctness_ok,
        'all_seconds': [round(e, 2) for e in elapsed_list],
    }
    if extra:
        entry.update(extra)
    results.append(entry)
    print(f"[RESULT] {arm}: median={med_elapsed:.2f}s, cpu={med_cpu:.0f}% "
          f"({med_cpu/(ncpu*100)*100:.0f}% of {ncpu} CPUs), "
          f"correct={correctness_ok}, all_runs={entry['all_seconds']}", flush=True)


def reproject_one_scene(vv_ds, vh_ds, tile_utm_zone, gdf_geom, num_threads=None):
    kwargs = {} if num_threads is None else {'num_threads': num_threads}
    vv_out = vv_ds.rio.reproject(tile_utm_zone, **kwargs).rio.clip(gdf_geom)
    vh_out = vh_ds.rio.reproject(tile_utm_zone, **kwargs).rio.clip(gdf_geom)
    return vv_out, vh_out


def arm_sequential(native_scenes, tile_utm_zone, gdf_geom, num_threads=None):
    out = {}
    for sid, ds in native_scenes.items():
        out[sid] = reproject_one_scene(ds['vv_ds'], ds['vh_ds'], tile_utm_zone, gdf_geom, num_threads)
    return out


def arm_threadpool(native_scenes, tile_utm_zone, gdf_geom, max_workers, num_threads=None):
    out = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {
            ex.submit(reproject_one_scene, ds['vv_ds'], ds['vh_ds'], tile_utm_zone, gdf_geom, num_threads): sid
            for sid, ds in native_scenes.items()
        }
        for fut in concurrent.futures.as_completed(futures):
            out[futures[fut]] = fut.result()
    return out


def _process_worker(args):
    sid, vv_vals, vv_coords, vv_dims, vv_crs, vh_vals, vh_coords, vh_dims, vh_crs, tile_utm_zone, gdf_wkb = args
    import shapely.wkb
    vv_ds = xr.DataArray(vv_vals, dims=vv_dims, coords=vv_coords).rio.write_crs(vv_crs)
    vh_ds = xr.DataArray(vh_vals, dims=vh_dims, coords=vh_coords).rio.write_crs(vh_crs)
    geom = shapely.wkb.loads(gdf_wkb)
    vv_out = vv_ds.rio.reproject(tile_utm_zone).rio.clip([geom])
    vh_out = vh_ds.rio.reproject(tile_utm_zone).rio.clip([geom])
    return sid, vv_out, vh_out


def arm_processpool(native_scenes, tile_utm_zone, gdf_geom, max_workers):
    import shapely.wkb
    geom_wkb = shapely.wkb.dumps(gdf_geom.iloc[0] if hasattr(gdf_geom, 'iloc') else list(gdf_geom)[0])
    tasks = []
    for sid, ds in native_scenes.items():
        vv, vh = ds['vv_ds'], ds['vh_ds']
        tasks.append((
            sid, vv.values, vv.coords, vv.dims, vv.rio.crs,
            vh.values, vh.coords, vh.dims, vh.rio.crs, tile_utm_zone, geom_wkb,
        ))
    out = {}
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as ex:
        for sid, vv_out, vh_out in ex.map(_process_worker, tasks):
            out[sid] = (vv_out, vh_out)
    return out


def arm_dask_chunked(native_scenes, tile_utm_zone, gdf_geom, chunk_size=1024):
    out = {}
    for sid, ds in native_scenes.items():
        vv_c = ds['vv_ds'].chunk({'x': chunk_size, 'y': chunk_size})
        vh_c = ds['vh_ds'].chunk({'x': chunk_size, 'y': chunk_size})
        vv_out = vv_c.rio.reproject(tile_utm_zone).rio.clip(gdf_geom).compute()
        vh_out = vh_c.rio.reproject(tile_utm_zone).rio.clip(gdf_geom).compute()
        out[sid] = (vv_out, vh_out)
    return out


def max_abs_diff(baseline_out, candidate_out, sample_key):
    a = baseline_out[sample_key][0].values
    b = candidate_out[sample_key][0].values
    if a.shape != b.shape:
        return None, False
    diff = float(np.nanmax(np.abs(a - b)))
    return diff, diff < 1e-3


if __name__ == '__main__':
    print(f'=== Benchmark environment: node has {multiprocessing.cpu_count()} CPUs total, '
          f'this job is allocated {ALLOCATED_CPUS} (SLURM_CPUS_PER_TASK) ===', flush=True)

    print('=== Stage 0: download + cache tile 321 dry-season scenes (once) ===', flush=True)
    source = OPERASource()
    t0 = time.time()
    passes = source.search_sentinel1(BBOX, START, END)
    native_scenes = {}
    for p in passes:
        vv_ds, vh_ds = source.read_vv_vh(p)
        native_scenes[p.id] = {'vv_ds': decibel_to_linear(vv_ds), 'vh_ds': decibel_to_linear(vh_ds)}
    print(f'Downloaded {len(native_scenes)} scenes in {time.time()-t0:.1f}s', flush=True)

    gdf_all = gpd.read_file(GRID)
    gdf = gdf_all.loc[gdf_all['ID'] == AOI_ID]
    tile_utm_zone = 'EPSG:326{}'.format(gdf['zone'].values[0][:-1])
    gdf_utm = gdf.to_crs(tile_utm_zone)
    sample_key = next(iter(native_scenes))

    print('\n=== Arm 0: baseline (sequential, num_threads=default) ===', flush=True)
    baseline_out = None
    elapsed_list, cpu_list = [], []
    for r in range(REPS):
        out, elapsed, cpu = timed_run(arm_sequential, native_scenes, tile_utm_zone, gdf_utm.geometry)
        elapsed_list.append(elapsed)
        cpu_list.append(cpu)
        if baseline_out is None:
            baseline_out = out
    log_result('arm0_baseline_sequential', elapsed_list, cpu_list, True)

    print('\n=== Arm 1: GDAL num_threads only ===', flush=True)
    for n in [2, 4, 8]:
        elapsed_list, cpu_list = [], []
        out = None
        for r in range(REPS):
            out, elapsed, cpu = timed_run(arm_sequential, native_scenes, tile_utm_zone, gdf_utm.geometry, num_threads=n)
            elapsed_list.append(elapsed)
            cpu_list.append(cpu)
        diff, ok = max_abs_diff(baseline_out, out, sample_key)
        log_result(f'arm1_gdal_num_threads_{n}', elapsed_list, cpu_list, ok, {'max_abs_diff': diff})

    print('\n=== Arm 2: ThreadPoolExecutor across scenes only ===', flush=True)
    for w in [2, 4, 6, 8]:
        elapsed_list, cpu_list = [], []
        out = None
        for r in range(REPS):
            out, elapsed, cpu = timed_run(arm_threadpool, native_scenes, tile_utm_zone, gdf_utm.geometry, w)
            elapsed_list.append(elapsed)
            cpu_list.append(cpu)
        diff, ok = max_abs_diff(baseline_out, out, sample_key)
        log_result(f'arm2_threadpool_workers_{w}', elapsed_list, cpu_list, ok, {'max_abs_diff': diff})

    print('\n=== Arm 3: combined ThreadPoolExecutor x GDAL num_threads ===', flush=True)
    for w, n in [(4, 2), (2, 4)]:
        elapsed_list, cpu_list = [], []
        out = None
        for r in range(REPS):
            out, elapsed, cpu = timed_run(arm_threadpool, native_scenes, tile_utm_zone, gdf_utm.geometry, w, num_threads=n)
            elapsed_list.append(elapsed)
            cpu_list.append(cpu)
        diff, ok = max_abs_diff(baseline_out, out, sample_key)
        log_result(f'arm3_combined_{w}workers_x_{n}threads', elapsed_list, cpu_list, ok, {'max_abs_diff': diff})

    print('\n=== Arm 4: ProcessPoolExecutor across scenes ===', flush=True)
    try:
        elapsed_list, cpu_list = [], []
        out = None
        for r in range(REPS):
            out, elapsed, cpu = timed_run(arm_processpool, native_scenes, tile_utm_zone, gdf_utm.geometry, 4)
            elapsed_list.append(elapsed)
            cpu_list.append(cpu)
        diff, ok = max_abs_diff(baseline_out, out, sample_key)
        log_result('arm4_processpool_workers_4', elapsed_list, cpu_list, ok, {'max_abs_diff': diff})
    except Exception as exc:
        log_result('arm4_processpool_workers_4', [0], [0], False, {'error': repr(exc)})

    print('\n=== Arm 5: dask.array chunked reproject (in-process threaded scheduler) ===', flush=True)
    try:
        elapsed_list, cpu_list = [], []
        out = None
        for r in range(REPS):
            out, elapsed, cpu = timed_run(arm_dask_chunked, native_scenes, tile_utm_zone, gdf_utm.geometry)
            elapsed_list.append(elapsed)
            cpu_list.append(cpu)
        diff, ok = max_abs_diff(baseline_out, out, sample_key)
        log_result('arm5_dask_chunked_threaded', elapsed_list, cpu_list, ok, {'max_abs_diff': diff})
    except Exception as exc:
        log_result('arm5_dask_chunked_threaded', [0], [0], False, {'error': repr(exc)})

    print('\n=== SUMMARY (sorted by median wall-clock) ===', flush=True)
    for r in sorted(results, key=lambda x: x['median_seconds']):
        print(r, flush=True)
    print('DONE', flush=True)
