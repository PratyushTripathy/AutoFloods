# autofloods/visualize.py

"""
Quick-look matplotlib visualizations of flood_mapper pipeline outputs --
the dry-season baseline, terrain slope, wet-season scenes paired with
their flood classifications, and monthly_sum()'s flood-day-count output.

Every function here reads from DISK (the actual file paths the pipeline
writes, reconstructed deterministically from `fm`'s own constructor
attributes -- output_base, dry_years, wet_yearmonths, slope_dir, etc.),
not from in-memory state (fm.mean_std_by_aoi, fm.wet_scene_paths,
fm.flood_dict). This means every function here works against a freshly
constructed flood_mapper pointed at a completed run's output_dir, in a
brand new session, with no pipeline methods having been called this
session -- as long as `fm` is constructed with the same grid_shapefile/
dry_years/wet_duration/output_dir/slope_dir the original run used (the
same assumption flood_mapper.expected_monthly_outfile()/
is_fully_processed() already make).

No function here calls plt.show() -- each returns a matplotlib Figure,
which renders normally in a notebook (as the last expression in a cell)
or via plt.show()/fig.savefig() in a script.

Units: Sentinel-1 VV/VH are converted from decibel to linear power
immediately on read (see preprocessing.read_sentinel1_stac ->
utils.decibel_to_linear) and stay in linear power in every on-disk
cache these functions read (mean_std, wet_scenes_cache). Display
functions convert back to dB via utils.linear_to_decibel() -- its
docstring already says "use for display" -- since dB gives SAR
backscatter's standard, visually sane dynamic range; linear power is
heavily right-skewed and looks like a near-black image with occasional
bright specks.
"""

import glob
import math
import os

import matplotlib.pyplot as plt
import numpy as np
import rasterio
import xarray as xr
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch

from . import SLOPE_OUTFILE
from .utils import _extract_date_token, linear_to_decibel

# switch off displaying maps (matches autofloods.mapfloods's convention)
plt.ioff()


def _percentile_clip(array, low=2, high=98):
    """(vmin, vmax) from the low/high percentiles of the finite values in
    `array` -- a standard display stretch, robust to a few extreme
    outlier pixels that would otherwise wash out the whole image if a
    plain min/max were used."""
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return 0.0, 1.0
    return float(np.percentile(finite, low)), float(np.percentile(finite, high))


def _normalize_to_unit(array, vmin, vmax):
    if vmax <= vmin:
        return np.zeros_like(array, dtype=float)
    return np.clip((array - vmin) / (vmax - vmin), 0, 1)


def plot_baseline(fm, aoi_id):
    """
    2x2 grid of the dry-season Z-score baseline (fm.nc_outfile's
    per-AOI NetCDF): VV mean, VV std, VH mean, VH std. Mean panels
    (converted to dB for display) share one percentile-clipped color
    scale; std panels (native linear units -- dB of a std isn't a
    physically meaningful transform) share a separate one, since mean
    and std are on very different numeric scales.
    """
    infile = fm.nc_outfile.replace('_id_', f'_{aoi_id}_')
    baseline = xr.load_dataarray(infile)

    vv_mean_db = linear_to_decibel(baseline.sel(band='vv_mean').values)
    vh_mean_db = linear_to_decibel(baseline.sel(band='vh_mean').values)
    vv_std = baseline.sel(band='vv_std').values
    vh_std = baseline.sel(band='vh_std').values

    mean_vmin, mean_vmax = _percentile_clip(np.stack([vv_mean_db, vh_mean_db]))
    std_vmin, std_vmax = _percentile_clip(np.stack([vv_std, vh_std]))

    fig, axes = plt.subplots(2, 2, figsize=(9, 8), constrained_layout=True)
    panels = [
        (axes[0, 0], vv_mean_db, 'VV mean (dB)', mean_vmin, mean_vmax),
        (axes[0, 1], vh_mean_db, 'VH mean (dB)', mean_vmin, mean_vmax),
        (axes[1, 0], vv_std, 'VV std (linear)', std_vmin, std_vmax),
        (axes[1, 1], vh_std, 'VH std (linear)', std_vmin, std_vmax),
    ]
    for ax, data, title, vmin, vmax in panels:
        im = ax.imshow(data, cmap='viridis', vmin=vmin, vmax=vmax)
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(f'Dry-season baseline -- AOI {aoi_id}')
    return fig


def plot_terrain(fm, aoi_id):
    """
    Terrain slope (degrees) for `aoi_id`, read from
    {fm.slope_dir}/slope_aoi_{aoi_id}.nc.

    Single panel, not DEM+slope side by side: the DEM mosaic
    download_nasadem()/prepare_slope() use to derive slope is
    transient (read once, used to compute slope, discarded) -- nothing
    caches it to disk anywhere in the pipeline, so there is no DEM file
    for this function to read in a fresh session. See CLAUDE.md's
    Future To-Dos if that ever changes.
    """
    slope_path = os.path.join(fm.slope_dir, SLOPE_OUTFILE.replace('_id.nc', f'_{aoi_id}.nc'))
    slope = xr.load_dataarray(slope_path, engine='rasterio').squeeze('band', drop=True).values

    vmin, vmax = _percentile_clip(slope)

    fig, ax = plt.subplots(1, 1, figsize=(6, 5), constrained_layout=True)
    im = ax.imshow(slope, cmap='terrain', vmin=vmin, vmax=vmax)
    ax.set_title(f'Terrain slope (degrees) -- AOI {aoi_id}')
    ax.set_xticks([])
    ax.set_yticks([])
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='degrees')
    return fig


_FLOOD_CLASS_COLORS = {
    0: '#f0f0f0',   # not flooded
    1: '#74a9cf',   # VH-only (low confidence)
    2: '#2b8cbe',   # VV-only (low confidence)
    3: '#045a8d',   # high-confidence flood
}
_FLOOD_CLASS_LABELS = {
    0: 'Not flooded',
    1: 'VH-only (low confidence)',
    2: 'VV-only (low confidence)',
    3: 'High-confidence flood',
}
_FLOOD_MASKED_COLOR = 'lightgray'
_FLOOD_MASKED_LABEL = 'Masked / no data'

_flood_cmap = ListedColormap([_FLOOD_CLASS_COLORS[i] for i in range(4)])
_flood_cmap.set_bad(_FLOOD_MASKED_COLOR)
_flood_norm = BoundaryNorm(boundaries=[-0.5, 0.5, 1.5, 2.5, 3.5], ncolors=4)


def _flood_scene_raster_path(fm, aoi_id, scene_id):
    """Deterministically reconstruct map_floods()'s output path for one
    scene -- same formula map_floods() itself uses -- so this module
    never needs fm.flood_dict (in-memory, only present mid-run)."""
    dry_year_begin = min(fm.dry_years)
    dry_year_end = max(fm.dry_years)
    wet_yearmonth_begin = fm.wet_yearmonths[0]
    wet_yearmonth_end = fm.wet_yearmonths[-1]
    outfile = os.path.join(fm.output_base, 'flood_raster', 'floodextent_id.tif')
    return outfile.replace(
        '_id.tif',
        f'_DRY_{dry_year_begin}_{dry_year_end}_WET_{wet_yearmonth_begin}_{wet_yearmonth_end}_'
        f'{aoi_id}_{"_".join(scene_id.split("_")[4:])}.tif'
    )


def _discover_wet_scenes(fm, aoi_id):
    """{scene_id: cache_path} for every wet scene cached on disk for
    `aoi_id`, sorted chronologically by the scene_id's embedded date."""
    pattern = os.path.join(fm.wet_scenes_cache_dir, f'wetscene_{aoi_id}_*.nc')
    scene_ids = {}
    for path in glob.glob(pattern):
        basename = os.path.basename(path)
        scene_id = basename[len(f'wetscene_{aoi_id}_'):-len('.nc')]
        scene_ids[scene_id] = path
    return dict(sorted(scene_ids.items(), key=lambda kv: _extract_date_token(kv[0])))


def _rgb_composite(scene_path):
    """R=VV, G=VH, B=(VV/VH log-ratio), each in dB and independently
    percentile-clipped for display. B is computed as a dB DIFFERENCE
    (linear_to_decibel(vv) - linear_to_decibel(vh)) rather than
    linear_to_decibel(vv / vh) -- mathematically identical
    (10*log10(vv/vh) == 10*log10(vv) - 10*log10(vh)) but avoids a
    linear-space division that can blow up near-zero VH pixels."""
    scene = xr.load_dataarray(scene_path)
    vv_db = linear_to_decibel(scene.sel(band='vv_ds').values)
    vh_db = linear_to_decibel(scene.sel(band='vh_ds').values)
    ratio_db = vv_db - vh_db

    channels = []
    for channel in (vv_db, vh_db, ratio_db):
        vmin, vmax = _percentile_clip(channel)
        channels.append(_normalize_to_unit(channel, vmin, vmax))
    return np.dstack(channels)


def plot_scenes_and_floods(fm, aoi_id, max_scenes=None, target_cols=6):
    """
    Paired RGB-composite / flood-classification grid, one column per
    wet-season scene, arranged in row-pairs of up to `target_cols`
    columns each: the RGB composite (R=VV, G=VH, B=VV/VH log-ratio, all
    dB, percentile-clipped) directly above that scene's flood
    classification, wrapping into additional row-pairs beneath when
    there are more scenes than fit in one row. E.g. 11 scenes at
    target_cols=6 -> row-pair 1 (6 scenes), row-pair 2 (5 scenes, one
    unused column turned off, not left blank with ticks/frame).

    Reads wet scenes from fm.wet_scenes_cache_dir (prepare_wet_scenes()'s
    persistent cache) and flood rasters from map_floods()'s
    deterministic output path -- a scene with no flood raster yet
    (map_floods() never ran for it) is silently dropped, counted toward
    the "skipped" note in the figure title. `max_scenes` caps the total
    number of (chronologically earliest) scenes rendered; if any scenes
    are skipped for either reason, the figure's suptitle says how many.
    """
    all_scenes = _discover_wet_scenes(fm, aoi_id)
    scene_ids = list(all_scenes.keys())

    n_capped = 0
    if max_scenes is not None and len(scene_ids) > max_scenes:
        n_capped = len(scene_ids) - max_scenes
        scene_ids = scene_ids[:max_scenes]

    kept = []
    n_missing_flood = 0
    for scene_id in scene_ids:
        flood_path = _flood_scene_raster_path(fm, aoi_id, scene_id)
        if os.path.exists(flood_path):
            kept.append((scene_id, all_scenes[scene_id], flood_path))
        else:
            n_missing_flood += 1

    n_scenes = len(kept)
    if n_scenes == 0:
        raise ValueError(
            f'No wet scenes with a matching flood raster found for AOI {aoi_id} -- '
            f'has prepare_wet_scenes()/map_floods() run for this output_dir?'
        )

    n_cols = min(target_cols, n_scenes)
    n_row_pairs = math.ceil(n_scenes / n_cols)
    n_rows = n_row_pairs * 2

    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(n_cols * 2.2, n_rows * 2.2 + 1.0),
        constrained_layout=True, squeeze=False,
    )

    for i, (scene_id, wet_path, flood_path) in enumerate(kept):
        row_pair, col = divmod(i, n_cols)
        rgb_ax = axes[row_pair * 2, col]
        flood_ax = axes[row_pair * 2 + 1, col]

        rgb_ax.imshow(_rgb_composite(wet_path))
        date_token = _extract_date_token(scene_id)
        date_str = f'{date_token[:4]}-{date_token[4:6]}-{date_token[6:8]}'
        rgb_ax.set_title(date_str, fontsize=10)
        rgb_ax.set_xticks([])
        rgb_ax.set_yticks([])

        with rasterio.open(flood_path) as src:
            classified = src.read(1)
        flood_ax.imshow(np.ma.masked_invalid(classified), cmap=_flood_cmap, norm=_flood_norm)
        flood_ax.set_xticks([])
        flood_ax.set_yticks([])

    # turn off any unused trailing axes in the last row-pair
    for i in range(n_scenes, n_row_pairs * n_cols):
        row_pair, col = divmod(i, n_cols)
        axes[row_pair * 2, col].axis('off')
        axes[row_pair * 2 + 1, col].axis('off')

    legend_handles = [
        Patch(facecolor=_FLOOD_CLASS_COLORS[i], label=_FLOOD_CLASS_LABELS[i])
        for i in range(4)
    ] + [Patch(facecolor=_FLOOD_MASKED_COLOR, label=_FLOOD_MASKED_LABEL)]
    fig.legend(handles=legend_handles, loc='lower center', ncol=5, bbox_to_anchor=(0.5, 0.0))

    n_skipped = n_capped + n_missing_flood
    title = f'Wet-season scenes and flood classification -- AOI {aoi_id} ({n_scenes} scene(s) shown)'
    if n_skipped:
        title += f', {n_skipped} skipped ({n_capped} over max_scenes, {n_missing_flood} missing flood raster)'
    fig.suptitle(title)
    return fig


def plot_flood_map(fm, aoi_id, month=None):
    """
    monthly_sum()'s per-month flood-day-count output for `aoi_id`
    (fm.expected_monthly_outfile(aoi_id)). `month` (e.g. '202408')
    shows that single band; None (default) shows every available month
    as a subplot grid (same row-wrapping as plot_scenes_and_floods,
    target_cols=6), sharing one discrete colorbar (0..max observed
    flood-day count across the shown month(s)) so months are directly
    comparable. Nodata (255, written wherever a pixel had zero valid
    observations all month -- see postprocessing.aggregate_monthly) is
    masked out and rendered gray, not blended into the count colorscale.
    """
    infile = fm.expected_monthly_outfile(aoi_id)
    with rasterio.open(infile) as src:
        band_names = list(src.descriptions)
        if month is not None:
            if month not in band_names:
                raise ValueError(f'Month {month!r} not found in {infile} -- available: {band_names}')
            indices = [band_names.index(month)]
        else:
            indices = list(range(len(band_names)))
        data = [src.read(i + 1) for i in indices]
        labels = [band_names[i] for i in indices]

    masked = [np.ma.masked_equal(d, 255) for d in data]
    vmax = max(int(m.max()) if m.count() > 0 else 0 for m in masked)
    boundaries = np.arange(-0.5, vmax + 1.5, 1)
    cmap = plt.get_cmap('YlGnBu', vmax + 1)
    cmap.set_bad(_FLOOD_MASKED_COLOR)
    norm = BoundaryNorm(boundaries, cmap.N)

    n_panels = len(masked)
    n_cols = min(6, n_panels)
    n_rows = math.ceil(n_panels / n_cols)

    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(n_cols * 2.6, n_rows * 2.6 + 0.6),
        constrained_layout=True, squeeze=False,
    )
    im = None
    for i, (m, label) in enumerate(zip(masked, labels)):
        row, col = divmod(i, n_cols)
        ax = axes[row, col]
        im = ax.imshow(m, cmap=cmap, norm=norm)
        ax.set_title(label)
        ax.set_xticks([])
        ax.set_yticks([])

    for i in range(n_panels, n_rows * n_cols):
        row, col = divmod(i, n_cols)
        axes[row, col].axis('off')

    fig.colorbar(im, ax=axes, fraction=0.03, pad=0.02, label='Flood-day count', ticks=range(vmax + 1))
    fig.suptitle(f'Monthly flood-day count -- AOI {aoi_id}')
    return fig
