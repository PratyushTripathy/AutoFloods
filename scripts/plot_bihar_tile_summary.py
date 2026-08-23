import sys
import glob
import os

import numpy as np
import xarray as xr
import rioxarray
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = '/home/emlab/projects/current-projects/edge-autofloods/AutoFloods'
OUTPUT_ROOT = f'{BASE}/output/bihar_2024'
PLOT_DIR = f'{OUTPUT_ROOT}/summary_plots'
os.makedirs(PLOT_DIR, exist_ok=True)


def percentile_stretch(arr, low=2, high=98):
    valid = arr[np.isfinite(arr)]
    if valid.size == 0:
        return np.zeros_like(arr)
    lo, hi = np.percentile(valid, [low, high])
    if hi <= lo:
        return np.zeros_like(arr)
    stretched = (arr - lo) / (hi - lo)
    return np.clip(stretched, 0, 1)


def plot_tile(tile_id):
    tile_dir = f'{OUTPUT_ROOT}/tile{tile_id}'

    mean_std_path = glob.glob(f'{tile_dir}/mean_std/*_{tile_id}_vv_vh_mean_std.nc')
    flood_path = glob.glob(f'{tile_dir}/flood_raster/floodextentstacked_*/floodextentstacked*_{tile_id}.tif')

    if not mean_std_path or not flood_path:
        print(f'[{tile_id}] missing mean_std or flood_raster output, skipping', flush=True)
        return

    mean_std = xr.open_dataarray(mean_std_path[0])
    vv_mean = mean_std.sel(band='vv_mean').values
    vh_mean = mean_std.sel(band='vh_mean').values

    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = np.where(vh_mean != 0, vv_mean / vh_mean, np.nan)

    r = percentile_stretch(vv_mean)
    g = percentile_stretch(vh_mean)
    b = percentile_stretch(ratio)
    rgb = np.dstack([r, g, b])

    flood_stack = rioxarray.open_rasterio(flood_path[0])
    flood_sum = (flood_stack == 3).sum(dim='band').values
    n_dates = flood_stack.shape[0]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color('black')
            spine.set_linewidth(1.5)

    axes[0].imshow(rgb)
    axes[0].set_title(f'Tile {tile_id}: Sentinel-1 dry-season composite\n(R=VV mean, G=VH mean, B=VV/VH ratio)')

    im = axes[1].imshow(flood_sum, cmap='Blues', vmin=0, vmax=15)
    axes[1].set_title(f'Tile {tile_id}: high-confidence flood count\n(sum of {n_dates} wet-season dates, Jul-Oct 2024)')
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04, label='# dates flagged flooded (capped at 15)')

    plt.tight_layout()
    outfile = f'{PLOT_DIR}/tile{tile_id}_summary.png'
    plt.savefig(outfile, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'[{tile_id}] saved {outfile}', flush=True)


if __name__ == '__main__':
    tile_ids = [int(x) for x in sys.argv[1:]]
    for tid in tile_ids:
        plot_tile(tid)
