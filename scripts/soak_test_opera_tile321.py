"""
Soak test (not part of the production pipeline): read tile 321's FULL
dry-season scene set via OPERASource's rewritten download-then-local read
path, sequentially, to see whether it survives the ~45-90 minute window
where MPCSource has consistently thrown sustained 403s at 20m.

This does NOT write any production output -- it only exercises the
search + read path and reports timing/failures. Nothing here is imported
by or changes the real pipeline in autofloods/.
"""
import datetime
import sys
import time

sys.path.append('/home/emlab/projects/current-projects/edge-autofloods/AutoFloods')

from autofloods.sources import OPERASource
from autofloods.preprocessing import read_sentinel1_stac

TILE_BBOX = {
    'type': 'Polygon',
    'coordinates': [[
        [85.998, 24.998], [87.002, 24.998], [87.002, 26.002], [85.998, 26.002], [85.998, 24.998]
    ]],
}  # tile 321's bbox

# Covers both the production dry-season window (dry_month '04,05') AND the
# wet-season detection window (Jul-Oct) in one run, to get enough real
# read volume to meaningfully approach the ~45-90 min window where MPC's
# /vsicurl/ path has always eventually thrown sustained 403s -- the
# original Apr-May-only run finished in under 9 minutes, too short to be
# a real test of that failure mode.
START = datetime.date(2024, 4, 1)
END = datetime.date(2024, 10, 31)

if __name__ == '__main__':
    source = OPERASource()
    print(f'Searching tile 321 bbox, {START} to {END}...', flush=True)
    t0 = time.time()
    passes = source.search_sentinel1(TILE_BBOX, START, END)
    print(f'Found {len(passes)} passes in {time.time()-t0:.1f}s\n', flush=True)

    results = []
    overall_start = time.time()
    for i, p in enumerate(passes):
        t_scene = time.time()
        try:
            stac_id, ds_dict = read_sentinel1_stac(p, source, overview_level=None)
            vv_ds = ds_dict['vv_ds']
            elapsed = time.time() - t_scene
            print(f'[{i+1}/{len(passes)}] {p.id} ({len(p.bursts)} bursts) '
                  f'-> OK in {elapsed:.1f}s, shape={vv_ds.shape}, '
                  f'total_elapsed={time.time()-overall_start:.1f}s', flush=True)
            results.append({'pass': p.id, 'ok': True, 'elapsed': elapsed})
        except Exception as exc:
            elapsed = time.time() - t_scene
            print(f'[{i+1}/{len(passes)}] {p.id} ({len(p.bursts)} bursts) '
                  f'-> FAILED after {elapsed:.1f}s: {exc!r}, '
                  f'total_elapsed={time.time()-overall_start:.1f}s', flush=True)
            results.append({'pass': p.id, 'ok': False, 'elapsed': elapsed, 'error': str(exc)})

    total = time.time() - overall_start
    n_ok = sum(r['ok'] for r in results)
    print(f'\n=== SUMMARY ===', flush=True)
    print(f'{n_ok}/{len(results)} passes succeeded', flush=True)
    print(f'Total wall time: {total:.1f}s ({total/60:.1f} min)', flush=True)
    if n_ok < len(results):
        print('Failures:', flush=True)
        for r in results:
            if not r['ok']:
                print(f"  {r['pass']}: {r['error']}", flush=True)
    print('DONE', flush=True)
