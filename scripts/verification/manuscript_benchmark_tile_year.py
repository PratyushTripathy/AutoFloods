"""
Real (not synthetic) per-stage memory + timing benchmark for the current
(post-streaming-rewrite) pipeline, against a real Bihar tile-year via
OPERASource -- for the manuscript's memory and timing figures.

Runs the exact fixed call sequence run_autofloods.py drives, wrapping
each stage in tracemalloc (peak allocated during that call) and
psutil-based RSS (overall resident memory, before/after), plus
wall-clock timing. Writes one JSON result file per tile-year to
output/manuscript_benchmark_20260906/results/.

Usage:
    python scripts/verification/manuscript_benchmark_tile_year.py --tile 318 --year 2024
"""
import argparse
import json
import os
import sys
import time
import tracemalloc

import psutil

import pathlib as _pathlib
BASE = str(_pathlib.Path(__file__).resolve().parents[2])  # repo root (scripts/verification/<this file>)
sys.path.append(BASE)

from autofloods import flood_mapper
from autofloods.sources import OPERASource
from autofloods.detectors import ZScoreDetector

OUT_ROOT = os.path.join(BASE, 'output', 'manuscript_benchmark_20260906')
GRID = os.path.join(BASE, 'resources', 'india_utm_fishnet_buffer.gpkg')
SLOPE_DIR = os.path.join(BASE, 'resources', 'slope')

proc = psutil.Process()


def rss_mb():
    return proc.memory_info().rss / 1e6


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tile', type=int, required=True)
    parser.add_argument('--year', type=int, required=True)
    args = parser.parse_args()

    tile_id = args.tile
    year = args.year
    output_dir = os.path.join(OUT_ROOT, str(year), f'tile{tile_id}')

    stages = []

    def log_stage(name, wall_start, wall_end, tm_peak, rss_before, rss_after):
        entry = {
            'stage': name,
            'wall_seconds': round(wall_end - wall_start, 2),
            'tracemalloc_peak_mb': round(tm_peak / 1e6, 1),
            'rss_before_mb': round(rss_before, 1),
            'rss_after_mb': round(rss_after, 1),
        }
        stages.append(entry)
        print(f'[{tile_id}/{year}] {name}: {entry["wall_seconds"]}s, '
              f'tracemalloc peak {entry["tracemalloc_peak_mb"]}MB, '
              f'RSS {entry["rss_before_mb"]}->{entry["rss_after_mb"]}MB', flush=True)

    def run_stage(name, fn, *fn_args, **fn_kwargs):
        rss_before = rss_mb()
        tracemalloc.start()
        wall_start = time.monotonic()
        fn(*fn_args, **fn_kwargs)
        wall_end = time.monotonic()
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        rss_after = rss_mb()
        log_stage(name, wall_start, wall_end, peak, rss_before, rss_after)

    overall_wall_start = time.monotonic()

    fm = flood_mapper(
        grid_shapefile=GRID,
        grid_id_list=[tile_id],
        dry_years=[year, year],
        slope_dir=SLOPE_DIR,
        wet_duration=[f'{year}/07', f'{year}/10'],
        source=OPERASource(),
        detector=ZScoreDetector(vv_thd=-2.5, vh_thd=-2.5),
        output_dir=output_dir,
    )

    run_stage('get_dry_dates', fm.get_dry_dates)
    run_stage('generate_dry_date_ranges', fm.generate_dry_date_ranges)
    run_stage('get_s1_items_dry', fm.get_s1_items, dry_wet='dry')
    run_stage('read_scenes_dry', fm.read_scenes, dry_wet='dry', overview_level=None, max_workers=6)
    run_stage('generate_mean_std_by_aoi', fm.generate_mean_std_by_aoi)
    run_stage('prepare_slope', fm.prepare_slope, dem_overview=0, buffer=500)
    run_stage('prepare_wet_scenes', fm.prepare_wet_scenes, overview_level=None, max_workers=6)
    run_stage('map_floods', fm.map_floods, vv_thd=-2.5, vh_thd=-2.5, slope_thd=15,
              export_vector=False, export_maps=False)
    run_stage('merge_floods_by_date', fm.merge_floods_by_date, export_raster=True)
    run_stage('generate_number_of_scenes', fm.generate_number_of_scenes, export_raster=True)
    run_stage('monthly_sum', fm.monthly_sum)

    overall_wall_end = time.monotonic()
    n_dry = len(fm.dry_aoi_scene_dict.get(tile_id, []))
    n_wet = sum(len(v) for v in fm.wet_scene_paths.values())

    result = {
        'tile_id': tile_id,
        'year': year,
        'source': 'OPERASource',
        'n_dry_scenes': n_dry,
        'n_wet_scenes': n_wet,
        'total_wall_seconds': round(overall_wall_end - overall_wall_start, 2),
        'stages': stages,
    }

    os.makedirs(os.path.join(OUT_ROOT, 'results'), exist_ok=True)
    result_path = os.path.join(OUT_ROOT, 'results', f'{year}_tile{tile_id}.json')
    with open(result_path, 'w') as f:
        json.dump(result, f, indent=2)

    print(f'[{tile_id}/{year}] TOTAL: {result["total_wall_seconds"]}s, '
          f'dry={n_dry}, wet={n_wet} -- written to {result_path}', flush=True)


if __name__ == '__main__':
    main()
