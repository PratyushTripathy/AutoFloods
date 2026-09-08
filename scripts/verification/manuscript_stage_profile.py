"""
Instrumented, real (not synthetic) per-stage timing/memory profiler for
one Bihar tile-year -- runs the exact same call sequence
scripts/run_autofloods.py drives in production, wrapping each stage in
tracemalloc + psutil RSS + wall-clock measurement, for the SoftwareX
manuscript's memory/timing numbers (2026-09-06). Post-dates this
session's wet-season streaming/windowed-read/generate_number_of_scenes
rewrites -- the point of this script is to get a citable number
reflecting the CURRENT code, since every prior real production run
(2026-08-22) predates those changes.

Usage:
    python scripts/verification/manuscript_stage_profile.py --config <yaml> --output_json <path>

Writes one JSON file per run: per-stage wall_seconds, tracemalloc peak
(MB), and RSS before/after (MB), plus overall wall time and overall
peak RSS. Real values only -- no synthetic proxy data anywhere in this
script.
"""
import argparse
import gc
import json
import sys
import time
import tracemalloc

import psutil
import yaml

import pathlib as _pathlib
BASE = str(_pathlib.Path(__file__).resolve().parents[2])  # repo root (scripts/verification/<this file>)
sys.path.append(BASE)

from autofloods import flood_mapper
from autofloods.sources import MPCSource, OPERASource
from autofloods.detectors import ZScoreDetector

SOURCES = {'mpc': MPCSource, 'opera': OPERASource}
DETECTORS = {'zscore': ZScoreDetector}

proc = psutil.Process()


def rss_mb():
    return proc.memory_info().rss / 1e6


class StageProfiler:
    def __init__(self):
        self.stages = {}

    def run(self, name, fn):
        gc.collect()
        rss_before = rss_mb()
        tracemalloc.start()
        t0 = time.monotonic()
        result = fn()
        elapsed = time.monotonic() - t0
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        rss_after = rss_mb()
        self.stages[name] = {
            'wall_seconds': elapsed,
            'tracemalloc_peak_mb': peak / 1e6,
            'rss_before_mb': rss_before,
            'rss_after_mb': rss_after,
        }
        print(
            f'[stage] {name}: {elapsed:.1f}s, tracemalloc peak {peak/1e6:.1f}MB, '
            f'RSS {rss_before:.1f}->{rss_after:.1f}MB',
            flush=True,
        )
        return result


def build_source(cfg):
    source_cfg = cfg.get('source', {'type': 'opera'})
    source_type = source_cfg.get('type', 'opera')
    kwargs = {k: v for k, v in source_cfg.items() if k != 'type'}
    return SOURCES[source_type](**kwargs)


def build_detector(cfg):
    detector_cfg = cfg.get('detector', {'type': 'zscore'})
    detector_type = detector_cfg.get('type', 'zscore')
    kwargs = {k: v for k, v in detector_cfg.items() if k != 'type'}
    return DETECTORS[detector_type](**kwargs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--output_json', required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    aoi_id = cfg['aoi']['grid_id_list'][0]
    aoi_cfg = cfg['aoi']
    dates_cfg = cfg['dates']
    detection_cfg = cfg.get('detection', {})
    read_cfg = cfg.get('read', {})
    slope_cfg = cfg.get('slope', {})
    output_dir = f"{cfg['output_dir'].rstrip('/')}/tile{aoi_id}"

    profiler = StageProfiler()
    overall_start = time.monotonic()
    overall_rss_start = rss_mb()

    fm = flood_mapper(
        grid_shapefile=aoi_cfg['grid_shapefile'],
        grid_id_list=[aoi_id],
        dry_date_col=aoi_cfg.get('dry_date_col', 'dry_month'),
        id_col=aoi_cfg.get('id_col', 'ID'),
        dry_years=dates_cfg['dry_years'],
        wet_duration=dates_cfg['wet_duration'],
        slope_dir=cfg['slope_dir'],
        source=build_source(cfg),
        detector=build_detector(cfg),
        output_dir=output_dir,
        cell_size=cfg.get('cell_size', 30),
    )
    print(f'[{aoi_id}] output_dir: {fm.output_dir}', flush=True)

    overview_level = read_cfg.get('overview_level', 3)
    # None (the default when the YAML omits either key) resolves to
    # flood_mapper.DEFAULT_MAX_WORKERS (2), not a value hardcoded here
    # -- matches scripts/run_autofloods.py's own resolution.
    max_workers = read_cfg.get('max_workers', None)
    reproject_max_workers = read_cfg.get('reproject_max_workers', None)

    profiler.run('get_dry_dates', fm.get_dry_dates)
    profiler.run('generate_dry_date_ranges', fm.generate_dry_date_ranges)
    profiler.run('get_s1_items_dry', lambda: fm.get_s1_items(dry_wet='dry'))
    print(f'[{aoi_id}] dry scenes: {len(fm.dry_aoi_scene_dict.get(aoi_id, []))}', flush=True)
    profiler.run('read_scenes_dry', lambda: fm.read_scenes(
        dry_wet='dry', overview_level=overview_level, max_workers=max_workers))
    profiler.run('generate_mean_std_by_aoi', lambda: fm.generate_mean_std_by_aoi(
        reproject_max_workers=reproject_max_workers))
    profiler.run('prepare_slope', lambda: fm.prepare_slope(
        dem_overview=slope_cfg.get('dem_overview', 1),
        buffer=slope_cfg.get('buffer', 500),
        max_workers=slope_cfg.get('max_workers', None),
    ))
    profiler.run('prepare_wet_scenes', lambda: fm.prepare_wet_scenes(
        overview_level=overview_level, max_workers=max_workers,
        reproject_max_workers=reproject_max_workers))
    n_wet = sum(len(v) for v in fm.wet_scene_paths.values())
    print(f'[{aoi_id}] wet scenes: {n_wet}', flush=True)

    slope_thd = detection_cfg.get('slope_thd', detection_cfg.get('rel_slope_thd', 15))
    profiler.run('map_floods', lambda: fm.map_floods(
        vv_thd=detection_cfg.get('vv_thd', -3),
        vh_thd=detection_cfg.get('vh_thd', -3),
        slope_thd=slope_thd,
        export_vector=False, export_maps=False,
    ))
    profiler.run('merge_floods_by_date', lambda: fm.merge_floods_by_date(export_raster=True))
    profiler.run('generate_number_of_scenes', lambda: fm.generate_number_of_scenes(export_raster=True))
    profiler.run('monthly_sum', fm.monthly_sum)

    overall_elapsed = time.monotonic() - overall_start
    overall_rss_peak = max(s['rss_after_mb'] for s in profiler.stages.values())

    output = {
        'aoi_id': aoi_id,
        'config': args.config,
        'n_dry_scenes': len(fm.dry_aoi_scene_dict.get(aoi_id, [])),
        'n_wet_scenes': n_wet,
        'overall_wall_seconds': overall_elapsed,
        'overall_rss_start_mb': overall_rss_start,
        'overall_rss_peak_mb': overall_rss_peak,
        'stages': profiler.stages,
    }
    with open(args.output_json, 'w') as f:
        json.dump(output, f, indent=2)

    print(f'[{aoi_id}] DONE in {overall_elapsed:.1f}s, peak RSS {overall_rss_peak:.1f}MB', flush=True)
    print(f'[{aoi_id}] wrote {args.output_json}', flush=True)


if __name__ == '__main__':
    main()
