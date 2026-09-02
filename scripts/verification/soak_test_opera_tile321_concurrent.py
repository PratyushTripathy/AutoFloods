"""
Soak test (not part of the production pipeline): same tile, same bbox,
same date window as soak_test_opera_tile321.py (which ran fully
sequentially, max_workers=1: 102/102 passes OK, 32.7 min) -- but this
time read via a ThreadPoolExecutor(max_workers=6), matching production's
actual concurrency (autofloods.flood_mapper.read_scenes's default).

Purpose: the sequential soak test never exercised concurrent OPERA/ASF
reads at all. MPC's throttling showed up under BOTH job-level and
thread-level concurrency; OPERA/CloudFront's behavior under concurrent
load from this HPC is untested until now. Same tile/bbox/dates as the
sequential run, so results are directly comparable (apples to apples).

This does NOT write any production output -- it only exercises the
search + read path and reports timing/failures. Nothing here is imported
by or changes the real pipeline in autofloods/.
"""
import datetime
import sys
import time
import concurrent.futures

sys.path.append('/home/emlab/projects/current-projects/edge-autofloods/AutoFloods')

from autofloods.sources import OPERASource
from autofloods.preprocessing import read_sentinel1_stac

TILE_BBOX = {
    'type': 'Polygon',
    'coordinates': [[
        [85.998, 24.998], [87.002, 24.998], [87.002, 26.002], [85.998, 26.002], [85.998, 24.998]
    ]],
}  # tile 321's bbox -- same as the sequential soak test

START = datetime.date(2024, 4, 1)
END = datetime.date(2024, 10, 31)
MAX_WORKERS = 6  # matches flood_mapper.read_scenes()'s default, i.e. production

if __name__ == '__main__':
    source = OPERASource()
    print(f'Searching tile 321 bbox, {START} to {END}...', flush=True)
    t0 = time.time()
    passes = source.search_sentinel1(TILE_BBOX, START, END)
    print(f'Found {len(passes)} passes in {time.time()-t0:.1f}s\n', flush=True)
    print(f'Reading with ThreadPoolExecutor(max_workers={MAX_WORKERS})...\n', flush=True)

    results = []
    overall_start = time.time()

    def _read_one(p):
        t_scene = time.time()
        try:
            stac_id, ds_dict = read_sentinel1_stac(p, source, overview_level=None)
            vv_ds = ds_dict['vv_ds']
            elapsed = time.time() - t_scene
            return {'pass': p.id, 'ok': True, 'elapsed': elapsed, 'shape': str(vv_ds.shape)}
        except Exception as exc:
            elapsed = time.time() - t_scene
            return {'pass': p.id, 'ok': False, 'elapsed': elapsed, 'error': repr(exc)}

    n_done = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(_read_one, p): p for p in passes}
        for future in concurrent.futures.as_completed(futures):
            r = future.result()
            n_done += 1
            total_elapsed = time.time() - overall_start
            if r['ok']:
                print(f"[{n_done}/{len(passes)}] {r['pass']} -> OK in {r['elapsed']:.1f}s, "
                      f"shape={r['shape']}, total_elapsed={total_elapsed:.1f}s", flush=True)
            else:
                print(f"[{n_done}/{len(passes)}] {r['pass']} -> FAILED after {r['elapsed']:.1f}s: "
                      f"{r['error']}, total_elapsed={total_elapsed:.1f}s", flush=True)
            results.append(r)

    total = time.time() - overall_start
    n_ok = sum(r['ok'] for r in results)
    print(f'\n=== SUMMARY ===', flush=True)
    print(f'{n_ok}/{len(results)} passes succeeded (max_workers={MAX_WORKERS})', flush=True)
    print(f'Total wall time: {total:.1f}s ({total/60:.1f} min)', flush=True)
    if n_ok < len(results):
        print('Failures:', flush=True)
        for r in results:
            if not r['ok']:
                print(f"  {r['pass']}: {r['error']}", flush=True)
    print('DONE', flush=True)
