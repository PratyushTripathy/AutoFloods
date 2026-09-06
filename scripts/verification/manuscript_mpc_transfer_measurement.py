"""
Real (not synthetic) measurement of MPCSource's windowed-read network
transfer reduction, for the manuscript. Reads ONE real Sentinel-1 RTC
VV asset from Microsoft Planetary Computer twice: once full-extent,
once windowed to a real tile's bbox -- measuring actual kernel-level
TCP bytes received (via `ss -ti`, TCP_INFO's bytes_received, summed
across every ESTABLISHED socket owned by this process) between
checkpoints, not an estimate. GDAL's CPL_DEBUG/CPL_CURL_VERBOSE debug
logging was tried first and produced no output in this environment
(likely a non-debug GDAL build) -- ss-based kernel byte counters are
the fallback and are independently reliable regardless of GDAL's build.

Usage:
    python scripts/verification/manuscript_mpc_transfer_measurement.py --tile 318
"""
import argparse
import os
import re
import subprocess
import sys
from datetime import datetime

import geopandas as gpd

import pathlib as _pathlib
BASE = str(_pathlib.Path(__file__).resolve().parents[2])  # repo root (scripts/verification/<this file>)
sys.path.append(BASE)

from autofloods.sources import MPCSource

GRID = os.path.join(BASE, 'resources', 'india_utm_fishnet_buffer.gpkg')


def total_bytes_received(pid):
    """Sum TCP_INFO's bytes_received across every ESTABLISHED socket
    this process currently owns, via `ss -tinp`."""
    out = subprocess.run(['ss', '-tinp'], capture_output=True, text=True).stdout
    total = 0
    lines = out.splitlines()
    for i, line in enumerate(lines):
        # ss prints a socket summary line (ending in users:(("proc",pid=X,fd=Y)))
        # followed by an indented TCP_INFO line carrying bytes_received --
        # pid is only on the summary line, bytes_received only on the next.
        if f'pid={pid},' in line and i + 1 < len(lines):
            m = re.search(r'bytes_received:(\d+)', lines[i + 1])
            if m:
                total += int(m.group(1))
    return total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tile', type=int, required=True)
    args = parser.parse_args()
    tile_id = args.tile
    pid = os.getpid()

    gdf = gpd.read_file(GRID)
    row = gdf[gdf['ID'] == tile_id].iloc[0]
    bounds = row.geometry.bounds  # (minx, miny, maxx, maxy) in EPSG:4326

    source = MPCSource()
    source.authenticate()

    items = source.search_sentinel1(
        bbox=row.geometry.__geo_interface__,
        start_date=datetime(2024, 8, 1), end_date=datetime(2024, 8, 31),
    )
    if not items:
        raise SystemExit(f'No MPC scenes found for tile {tile_id} in Aug 2024 -- pick a different window.')
    item = items[0]
    vv_href, _ = source.vv_vh_hrefs(item)
    print(f'Using real MPC scene: {item.id}', flush=True)
    print(f'Tile {tile_id} bbox (EPSG:4326): {bounds}', flush=True)

    checkpoint_0 = total_bytes_received(pid)
    print(f'Bytes received before any read (auth+search overhead): {checkpoint_0}', flush=True)

    # windowed read first (smaller, so its own connections are less likely
    # to still be draining buffered data when we checkpoint). Goes through
    # the real public read_vv_vh() API (not open_rasterio_with_retry
    # directly) so bbox gets properly reprojected into the item's native
    # CRS via _windowed_bbox_for_item() -- the same path map_floods()'s
    # pipeline actually uses.
    vv_win, _ = source.read_vv_vh(item, bbox=bounds)
    vv_win.load()
    checkpoint_1 = total_bytes_received(pid)
    windowed_bytes = checkpoint_1 - checkpoint_0
    print(f'Windowed read: shape={vv_win.shape}, bytes received this step: {windowed_bytes}', flush=True)

    # full-extent read second
    vv_full, _ = source.read_vv_vh(item, bbox=None)
    vv_full.load()
    checkpoint_2 = total_bytes_received(pid)
    full_bytes = checkpoint_2 - checkpoint_1
    print(f'Full-extent read: shape={vv_full.shape}, bytes received this step: {full_bytes}', flush=True)

    if windowed_bytes > 0:
        print(f'\nReduction factor (full / windowed): {full_bytes / windowed_bytes:.1f}x')
    print(f'Full scene size (bytes received): {full_bytes} ({full_bytes / 1e6:.1f} MB)')
    print(f'Windowed read size (bytes received): {windowed_bytes} ({windowed_bytes / 1e6:.1f} MB)')


if __name__ == '__main__':
    main()
