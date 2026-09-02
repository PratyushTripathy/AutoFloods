"""
Config-driven pipeline runner. A plain script outside the autofloods
package (like scripts/bihar2024_tile.py and scripts/bihar2024_tile_opera.py,
both of which remain as-is, unmodified) -- imports the package and drives
a run from a YAML config instead of a hand-edited script, so a run can be
launched by editing a config file rather than a notebook or a per-tile
Python script.

Usage:
    python scripts/run_autofloods.py --config examples/bihar_2024_opera_config.yaml

Processes grid_id_list one AOI at a time, each through the full pipeline
in the same fixed step order the hand-written per-tile scripts use --
this mirrors already-tested behavior (one flood_mapper instance, one AOI,
one SLURM job is the pattern actually exercised in production) rather
than introducing new, unvalidated multi-AOI-per-instance batch semantics.
"""
import argparse
import sys

import yaml

sys.path.append('/home/emlab/projects/current-projects/edge-autofloods/AutoFloods')

from autofloods import flood_mapper
from autofloods.sources import MPCSource, OPERASource
from autofloods.detectors import ZScoreDetector

SOURCES = {
    'mpc': MPCSource,
    'opera': OPERASource,
}

DETECTORS = {
    'zscore': ZScoreDetector,
}


def build_source(cfg):
    source_cfg = cfg.get('source', {'type': 'mpc'})
    source_type = source_cfg.get('type', 'mpc')
    if source_type not in SOURCES:
        raise ValueError(f"Unknown source type {source_type!r}. Valid: {sorted(SOURCES)}")
    kwargs = {k: v for k, v in source_cfg.items() if k != 'type'}
    return SOURCES[source_type](**kwargs)


def build_detector(cfg):
    detector_cfg = cfg.get('detector', {'type': 'zscore'})
    detector_type = detector_cfg.get('type', 'zscore')
    if detector_type not in DETECTORS:
        raise ValueError(f"Unknown detector type {detector_type!r}. Valid: {sorted(DETECTORS)}")
    kwargs = {k: v for k, v in detector_cfg.items() if k != 'type'}
    return DETECTORS[detector_type](**kwargs)


def run_one_aoi(cfg, aoi_id):
    aoi_cfg = cfg['aoi']
    dates_cfg = cfg['dates']
    detection_cfg = cfg.get('detection', {})
    read_cfg = cfg.get('read', {})
    output_dir = cfg['output_dir'].rstrip('/').format(aoi_id=aoi_id) if '{aoi_id}' in cfg['output_dir'] \
        else f"{cfg['output_dir'].rstrip('/')}/tile{aoi_id}"

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

    if fm.is_fully_processed(aoi_id):
        print(f'[{aoi_id}] already fully processed (found {fm.expected_monthly_outfile(aoi_id)}) -- skipping.',
              flush=True)
        return

    overview_level = read_cfg.get('overview_level', 3)
    max_workers = read_cfg.get('max_workers', 6)  # download concurrency (network-bound)
    # reproject_max_workers: CPU-bound reprojection concurrency. None (the
    # YAML default when the key is omitted) resolves to
    # utils.default_max_workers() -- (available CPUs - 1) on whatever
    # system this runs on, not a value hardcoded for one cluster.
    reproject_max_workers = read_cfg.get('reproject_max_workers', None)
    slope_cfg = cfg.get('slope', {})

    fm.get_dry_dates()
    fm.generate_dry_date_ranges()
    fm.get_s1_items(dry_wet='dry')
    print(f'[{aoi_id}] dry scenes: {len(fm.dry_aoi_scene_dict.get(aoi_id, []))}', flush=True)
    fm.read_scenes(dry_wet='dry', overview_level=overview_level, max_workers=max_workers)
    fm.generate_mean_std_by_aoi(reproject_max_workers=reproject_max_workers)
    print(f'[{aoi_id}] mean/std computed', flush=True)

    fm.prepare_slope(
        dem_overview=slope_cfg.get('dem_overview', 1),
        buffer=slope_cfg.get('buffer', 500),
        max_workers=slope_cfg.get('max_workers', 6),
    )
    print(f'[{aoi_id}] slope computed', flush=True)

    fm.prepare_wet_scenes(overview_level=overview_level, max_workers=max_workers,
                           reproject_max_workers=reproject_max_workers)
    print(f'[{aoi_id}] wet scenes: {sum(len(v) for v in fm.wet_scenes_by_aoi.values())}', flush=True)

    fm.map_floods(
        vv_thd=detection_cfg.get('vv_thd', -3),
        vh_thd=detection_cfg.get('vh_thd', -3),
        rel_slope_thd=detection_cfg.get('rel_slope_thd', 20),
        export_raster=False, export_vector=False, export_maps=False,
    )
    fm.merge_floods_by_date(export_raster=True)
    fm.generate_number_of_scenes(export_raster=True)
    fm.monthly_sum()

    print(f'[{aoi_id}] DONE', flush=True)


def main():
    parser = argparse.ArgumentParser(description='Run the autofloods pipeline from a YAML config file.')
    parser.add_argument('--config', required=True, help='Path to a YAML config file')
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    for aoi_id in cfg['aoi']['grid_id_list']:
        run_one_aoi(cfg, aoi_id)


if __name__ == '__main__':
    main()
