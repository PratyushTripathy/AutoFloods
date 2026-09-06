"""
Real dual-backend (OPERASource vs MPCSource) agreement check for the
manuscript. Runs MPCSource fresh through monthly output for a tile/year
that already has real OPERASource production output (output/bihar_opera_30m),
holding grid/dry season/detector/thresholds/cell_size identical, then
compares the two monthly flood-count rasters pixel by pixel over pixels
valid (non-nodata) in both.

Usage:
    python scripts/verification/manuscript_dual_backend_compare.py --tile 318 --year 2024
"""
import argparse
import glob
import os
import sys

import numpy as np
import rasterio
import rioxarray  # noqa: F401 -- registers the .rio accessor
import xarray as xr

import pathlib as _pathlib
BASE = str(_pathlib.Path(__file__).resolve().parents[2])  # repo root (scripts/verification/<this file>)
sys.path.append(BASE)

from autofloods import flood_mapper
from autofloods.sources import MPCSource
from autofloods.detectors import ZScoreDetector

GRID = os.path.join(BASE, 'resources', 'india_utm_fishnet_buffer.gpkg')
SLOPE_DIR = os.path.join(BASE, 'resources', 'slope')
OPERA_OUTPUT_ROOT = os.path.join(BASE, 'output', 'bihar_opera_30m')
MPC_OUTPUT_ROOT = os.path.join(BASE, 'output', 'manuscript_dual_backend_20260906')


def run_mpc(tile_id, year):
    output_dir = os.path.join(MPC_OUTPUT_ROOT, str(year), f'tile{tile_id}')
    fm = flood_mapper(
        grid_shapefile=GRID,
        grid_id_list=[tile_id],
        dry_years=[year, year],
        slope_dir=SLOPE_DIR,
        wet_duration=[f'{year}/07', f'{year}/10'],
        source=MPCSource(),
        detector=ZScoreDetector(vv_thd=-2.5, vh_thd=-2.5),
        output_dir=output_dir,
    )
    fm.get_dry_dates()
    fm.generate_dry_date_ranges()
    fm.get_s1_items(dry_wet='dry')
    fm.read_scenes(dry_wet='dry', overview_level=None, max_workers=6)
    fm.generate_mean_std_by_aoi()
    fm.prepare_slope(dem_overview=0, buffer=500)
    fm.prepare_wet_scenes(overview_level=None, max_workers=6)
    fm.map_floods(vv_thd=-2.5, vh_thd=-2.5, slope_thd=15, export_vector=False, export_maps=False)
    fm.merge_floods_by_date(export_raster=True)
    fm.generate_number_of_scenes(export_raster=True)
    fm.monthly_sum()
    n_dry = len(fm.dry_aoi_scene_dict.get(tile_id, []))
    n_wet = sum(len(v) for v in fm.wet_scene_paths.values())
    print(f'[MPC {tile_id}/{year}] dry={n_dry}, wet={n_wet}', flush=True)
    return fm.expected_monthly_outfile(tile_id)


def find_opera_monthly(tile_id, year):
    pattern = os.path.join(
        OPERA_OUTPUT_ROOT, str(year), f'tile{tile_id}', 'flood_raster', 'monthlyadded_*', '*.tif',
    )
    matches = glob.glob(pattern)
    if not matches:
        raise SystemExit(f'No existing OPERA monthly output found for tile {tile_id}/{year} at {pattern}')
    return matches[0]


def compare(opera_path, mpc_path):
    opera_da = rioxarray.open_rasterio(opera_path, masked=False)
    mpc_da = rioxarray.open_rasterio(mpc_path, masked=False)

    with rasterio.open(opera_path) as src:
        opera_bands = list(src.descriptions)
    with rasterio.open(mpc_path) as src:
        mpc_bands = list(src.descriptions)

    common_months = sorted(set(opera_bands) & set(mpc_bands))
    print(f'OPERA months: {opera_bands}')
    print(f'MPC months: {mpc_bands}')
    print(f'Common months compared: {common_months}')
    if not common_months:
        raise SystemExit('No common months between OPERA and MPC monthly output -- cannot compare.')

    # Reproject MPC onto OPERA's exact grid (nearest -- this is
    # categorical/count integer data, not something to interpolate).
    mpc_matched = mpc_da.rio.reproject_match(opera_da, resampling=rasterio.enums.Resampling.nearest)

    total_valid_both = 0
    total_valid_either = 0
    total_exact = 0
    total_within1 = 0
    diffs = []

    for month in common_months:
        o_idx = opera_bands.index(month)
        m_idx = mpc_bands.index(month)
        o_band = opera_da.isel(band=o_idx).values.astype(float)
        m_band = mpc_matched.isel(band=m_idx).values.astype(float)

        o_valid = o_band != 255
        m_valid = m_band != 255
        both_valid = o_valid & m_valid
        either_valid = o_valid | m_valid

        n_both = int(both_valid.sum())
        n_either = int(either_valid.sum())
        total_valid_both += n_both
        total_valid_either += n_either

        if n_both > 0:
            o_vals = o_band[both_valid]
            m_vals = m_band[both_valid]
            diff = m_vals - o_vals
            diffs.append(diff)
            total_exact += int((diff == 0).sum())
            total_within1 += int((np.abs(diff) <= 1).sum())

    print(f'\nTotal valid-in-both pixels (across {len(common_months)} common months): {total_valid_both}')
    print(f'Total valid-in-either pixels: {total_valid_either}')
    print(f'Fraction with data in both vs. either: {total_valid_both / total_valid_either:.4f}' if total_valid_either else 'N/A')

    if total_valid_both > 0:
        pct_exact = 100 * total_exact / total_valid_both
        pct_within1 = 100 * total_within1 / total_valid_both
        all_diffs = np.concatenate(diffs)
        mean_bias = float(np.mean(all_diffs))  # positive => MPC reports more flood-days than OPERA
        print(f'Exact agreement: {pct_exact:.2f}%')
        print(f'Within +/-1 flood-day count: {pct_within1:.2f}%')
        print(f'Mean bias (MPC - OPERA): {mean_bias:+.3f} flood-days/pixel')
        print(f'Std of diff: {float(np.std(all_diffs)):.3f}')
    else:
        print('No pixels valid in both -- cannot compute agreement.')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tile', type=int, required=True)
    parser.add_argument('--year', type=int, required=True)
    args = parser.parse_args()

    opera_path = find_opera_monthly(args.tile, args.year)
    print(f'Existing real OPERA monthly output: {opera_path}')

    mpc_path = run_mpc(args.tile, args.year)
    print(f'Fresh real MPC monthly output: {mpc_path}')

    compare(opera_path, mpc_path)


if __name__ == '__main__':
    main()
