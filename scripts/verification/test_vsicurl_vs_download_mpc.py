"""
Side test (not part of the production pipeline): compare the current
/vsicurl/-streaming read approach against a download-to-local-file-then-
open approach, both against MPC, at 20m (overview_level=0), for the same
tile 321 dry-season scenes that have been failing with sustained 403s /
"not recognized as being in a supported file format" errors.

This tests a hypothesis from a sibling project (edge-india-crop-mapping's
satellite-download-aws pipeline): that GDAL's /vsicurl/ driver mishandles
throttled/redirected responses on these signed-URL blob endpoints, and
that downloading the file locally first (plain requests.get, then
rasterio.open on the local path) sidesteps that failure mode. That
sibling project's evidence is for OPERA/AWS (S3), not MPC/Azure -- this
script checks whether the same fix helps here too, before touching
autofloods/sources/mpc.py or autofloods/utils/__init__.py.

Nothing here is imported by or changes the real pipeline in autofloods/.
"""
import datetime
import os
import sys
import tempfile
import time

import pathlib as _pathlib
BASE = str(_pathlib.Path(__file__).resolve().parents[2])  # repo root (scripts/verification/<this file>)
sys.path.append(BASE)

import rasterio
import requests
import rioxarray

from autofloods.sources import MPCSource
from autofloods.utils import GDAL_HTTP_ENV

TILE_BBOX = {
    'type': 'Polygon',
    'coordinates': [[
        [85.998, 24.998], [87.002, 24.998], [87.002, 26.002], [85.998, 26.002], [85.998, 24.998]
    ]],
}  # tile 321's bbox
MAX_SCENES = 6
OVERVIEW_LEVEL = 0  # 20m
MAX_ATTEMPTS = 3
BACKOFF_SECONDS = 15


def get_vv_hrefs(source, n):
    items = source.search_sentinel1(TILE_BBOX, datetime.date(2024, 5, 1), datetime.date(2024, 5, 31))
    hrefs = []
    for item in items[:n]:
        vv_href, _ = source.vv_vh_hrefs(item)
        hrefs.append((item.id, vv_href))
    return hrefs


def open_vsicurl(source, href, max_attempts=MAX_ATTEMPTS):
    """Current pipeline behavior: GDAL /vsicurl/ streaming via rioxarray."""
    signed = source.sign(href)
    last_exc = None
    for attempt in range(1, max_attempts + 1):
        t0 = time.time()
        try:
            with rasterio.Env(**GDAL_HTTP_ENV):
                da = rioxarray.open_rasterio(signed, overview_level=OVERVIEW_LEVEL, masked=True).load()
            return True, time.time() - t0, None
        except Exception as exc:
            last_exc = exc
            print(f'    vsicurl attempt {attempt}/{max_attempts} failed ({exc!r:.150}); '
                  f'retrying in {BACKOFF_SECONDS}s', flush=True)
            if attempt < max_attempts:
                time.sleep(BACKOFF_SECONDS)
    return False, None, str(last_exc)


def open_download_then_local(source, href, max_attempts=MAX_ATTEMPTS):
    """Candidate fix: plain requests download to a local temp file, then
    rasterio.open() the local path -- no GDAL /vsicurl/ involved."""
    signed = source.sign(href)
    last_exc = None
    for attempt in range(1, max_attempts + 1):
        t0 = time.time()
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix='.tif', delete=False) as tmp:
                tmp_path = tmp.name
                with requests.get(signed, stream=True, timeout=120) as r:
                    r.raise_for_status()
                    for chunk in r.iter_content(chunk_size=4 * 1024 * 1024):
                        tmp.write(chunk)
            da = rioxarray.open_rasterio(tmp_path, overview_level=OVERVIEW_LEVEL, masked=True).load()
            return True, time.time() - t0, None
        except Exception as exc:
            last_exc = exc
            print(f'    download attempt {attempt}/{max_attempts} failed ({exc!r:.150}); '
                  f'retrying in {BACKOFF_SECONDS}s', flush=True)
            if attempt < max_attempts:
                time.sleep(BACKOFF_SECONDS)
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)
    return False, None, str(last_exc)


if __name__ == '__main__':
    source = MPCSource()
    print(f'Searching for up to {MAX_SCENES} dry-season scenes over tile 321 bbox, May 2024...', flush=True)
    scenes = get_vv_hrefs(source, MAX_SCENES)
    print(f'Found {len(scenes)} scenes\n', flush=True)

    results = []
    for scene_id, href in scenes:
        print(f'=== {scene_id} ===', flush=True)

        print('  [vsicurl]', flush=True)
        ok_v, t_v, err_v = open_vsicurl(source, href)
        print(f'    -> {"OK" if ok_v else "FAILED"}'
              + (f' in {t_v:.1f}s' if ok_v else f': {err_v[:150]}'), flush=True)

        print('  [download-then-local]', flush=True)
        ok_d, t_d, err_d = open_download_then_local(source, href)
        print(f'    -> {"OK" if ok_d else "FAILED"}'
              + (f' in {t_d:.1f}s' if ok_d else f': {err_d[:150]}'), flush=True)

        results.append({
            'scene': scene_id,
            'vsicurl_ok': ok_v, 'vsicurl_time': t_v,
            'download_ok': ok_d, 'download_time': t_d,
        })
        print(flush=True)

    print('=== SUMMARY ===', flush=True)
    vsicurl_success = sum(r['vsicurl_ok'] for r in results)
    download_success = sum(r['download_ok'] for r in results)
    print(f'vsicurl:             {vsicurl_success}/{len(results)} succeeded', flush=True)
    print(f'download-then-local: {download_success}/{len(results)} succeeded', flush=True)
    for r in results:
        print(f"  {r['scene']}: vsicurl={'OK' if r['vsicurl_ok'] else 'FAIL'} "
              f"download={'OK' if r['download_ok'] else 'FAIL'}", flush=True)
