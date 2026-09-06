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
from rasterio.enums import Resampling
from skimage.transform import downscale_local_mean

from . import SLOPE_OUTFILE
from .utils import _extract_date_token, linear_to_decibel, sanitize_scene_id_for_filename

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


def _downsample_factor(height, width, max_size):
    """Integer block-size factor so max(height, width) // factor is
    close to `max_size` -- used to shrink a full tile-resolution array
    down to roughly what a small subplot panel actually needs before
    it's ever handed to imshow(). A subplot panel renders at a couple
    hundred pixels at most; matplotlib retains whatever array it's
    given for the lifetime of the Figure (confirmed via a real
    tracemalloc/RSS measurement -- see CHANGELOG.md), so feeding it a
    full ~3500x3500 tile per scene is the actual driver of an OOM risk
    at high scene counts, not the (already correct, one-scene-at-a-time)
    data loading. Returns 1 (no downsampling) if already small enough.
    """
    longer_side = max(height, width)
    if longer_side <= max_size:
        return 1
    return max(1, int(np.ceil(longer_side / max_size)))


def _new_figure_with_legend_row(n_image_rows, n_cols, panel_size=2.2, legend_row_height=0.9,
                                 min_fig_width=None, min_fig_height=None):
    """A grid of `n_image_rows` x `n_cols` image subplots plus one extra
    full-width row reserved for a legend, built via a real GridSpec (not
    a floating fig.legend()) so constrained_layout actually accounts for
    the legend's space and doesn't let it overlap the last row of image
    panels -- the fix for the legend-clipping issue found in real
    rendered output. Returns (fig, axes 2D array of the image subplots,
    legend_ax spanning the full width of the extra row, turned off/no
    ticks -- call legend_ax.legend(...) on it).
    """
    fig_width = max(n_cols * panel_size, min_fig_width or 0)
    fig_height = max(n_image_rows * panel_size, min_fig_height or 0) + legend_row_height

    fig = plt.figure(figsize=(fig_width, fig_height), constrained_layout=True)
    gs = fig.add_gridspec(
        n_image_rows + 1, n_cols,
        height_ratios=[panel_size] * n_image_rows + [legend_row_height],
    )
    axes = np.empty((n_image_rows, n_cols), dtype=object)
    for row in range(n_image_rows):
        for col in range(n_cols):
            axes[row, col] = fig.add_subplot(gs[row, col])
    legend_ax = fig.add_subplot(gs[n_image_rows, :])
    legend_ax.axis('off')
    return fig, axes, legend_ax


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

# plot_flood_map()'s 4-class binning of monthly_sum()'s per-pixel
# flood-day COUNT (an unbounded integer, unlike the fixed 0-3 encoding
# above) -- 0 / 1-3 / 4-7 / 8+, a discrete classification rather than a
# continuous numeric scale, easier to read at a glance than a raw count.
# The last bin's upper boundary (1e6) is a large-but-finite stand-in for
# "unbounded" -- no real flood-day count in a month (max ~31) ever
# approaches it.
_FLOOD_COUNT_BIN_LABELS = ['0', '1-3', '4-7', '8+']
_FLOOD_COUNT_BIN_COLORS = ['#eff3ff', '#bdd7e7', '#6baed6', '#2171b5']  # ColorBrewer Blues, 4-class
_flood_count_cmap = ListedColormap(_FLOOD_COUNT_BIN_COLORS)
_flood_count_cmap.set_bad(_FLOOD_MASKED_COLOR)
_flood_count_norm = BoundaryNorm(boundaries=[-0.5, 0.5, 3.5, 7.5, 1e6], ncolors=4)


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
        f'{aoi_id}_{sanitize_scene_id_for_filename(scene_id)}.tif'
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


def _rgb_composite(scene_path, thumbnail_max_size=400):
    """R=VV, G=VH, B=(VV/VH log-ratio), each in dB and independently
    percentile-clipped for display. B is computed as a dB DIFFERENCE
    (linear_to_decibel(vv) - linear_to_decibel(vh)) rather than
    linear_to_decibel(vv / vh) -- mathematically identical
    (10*log10(vv/vh) == 10*log10(vv) - 10*log10(vh)) but avoids a
    linear-space division that can blow up near-zero VH pixels.

    Downsamples (block-mean, via skimage.transform.downscale_local_mean)
    to roughly `thumbnail_max_size` on the longer side BEFORE computing
    dB/percentile-stretch -- the full tile-resolution array is only ever
    transient here (freed once this function returns), but the
    RETURNED composite is what gets handed to imshow() and retained by
    the Figure for its whole lifetime, so downsampling here (not after)
    is what actually bounds memory at high scene counts -- see
    _downsample_factor()'s docstring."""
    scene = xr.load_dataarray(scene_path)
    vv = scene.sel(band='vv_ds').values
    vh = scene.sel(band='vh_ds').values

    factor = _downsample_factor(vv.shape[0], vv.shape[1], thumbnail_max_size)
    if factor > 1:
        vv = downscale_local_mean(vv, (factor, factor))
        vh = downscale_local_mean(vh, (factor, factor))

    vv_db = linear_to_decibel(vv)
    vh_db = linear_to_decibel(vh)
    ratio_db = vv_db - vh_db

    channels = []
    for channel in (vv_db, vh_db, ratio_db):
        vmin, vmax = _percentile_clip(channel)
        channels.append(_normalize_to_unit(channel, vmin, vmax))
    return np.dstack(channels)


def _read_flood_raster_thumbnail(flood_path, thumbnail_max_size=400):
    """Decimated read of a classified flood raster straight from disk --
    never materializes the full-resolution array at all (unlike a
    full read() followed by downsampling), via rasterio's out_shape=
    with nearest-neighbor resampling (appropriate for this categorical
    0/1/2/3/NaN class data, same convention already used elsewhere in
    this codebase for categorical/threshold rasters)."""
    with rasterio.open(flood_path) as src:
        factor = _downsample_factor(src.height, src.width, thumbnail_max_size)
        if factor > 1:
            out_shape = (max(1, src.height // factor), max(1, src.width // factor))
            return src.read(1, out_shape=out_shape, resampling=Resampling.nearest)
        return src.read(1)


def plot_scenes_and_floods(fm, aoi_id, max_scenes=None, target_cols=6, thumbnail_max_size=400):
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

    `thumbnail_max_size` (default 400, longer side in pixels) bounds
    the resolution each scene/flood panel is downsampled to before
    rendering -- a subplot panel here is a couple hundred pixels at
    most, but matplotlib retains whatever array it's given for the
    Figure's lifetime; without this, real ~3500x3500 tile-resolution
    data across 30-50 scenes measured at multi-GB peak memory (real
    tracemalloc/RSS measurement, see CHANGELOG.md) for no visual
    benefit, since none of it is ever actually visible at this panel
    size. The shared legend is drawn in its own reserved GridSpec row
    (not a floating fig.legend()), so it can't overlap the last row of
    panels regardless of scene count.
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

    fig, axes, legend_ax = _new_figure_with_legend_row(n_rows, n_cols, panel_size=2.2, legend_row_height=0.9)

    for i, (scene_id, wet_path, flood_path) in enumerate(kept):
        row_pair, col = divmod(i, n_cols)
        rgb_ax = axes[row_pair * 2, col]
        flood_ax = axes[row_pair * 2 + 1, col]

        rgb_ax.imshow(_rgb_composite(wet_path, thumbnail_max_size))
        date_token = _extract_date_token(scene_id)
        date_str = f'{date_token[:4]}-{date_token[4:6]}-{date_token[6:8]}'
        rgb_ax.set_title(date_str, fontsize=10)
        rgb_ax.set_xticks([])
        rgb_ax.set_yticks([])

        classified = _read_flood_raster_thumbnail(flood_path, thumbnail_max_size)
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
    legend_ax.legend(handles=legend_handles, loc='center', ncol=5)

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
    target_cols=6).

    The per-pixel count is shown as a discrete 4-class scale --
    0 / 1-3 / 4-7 / 8+ flood-days -- rather than a continuous numeric
    color scale, easier to read at a glance; one shared legend for all
    shown month(s) so they're directly comparable. Nodata (255, written
    wherever a pixel had zero valid observations all month -- see
    postprocessing.aggregate_monthly) is masked out and rendered gray,
    excluded from the count classes entirely.

    A single-panel case (one month, or `month=` given directly) gets a
    fixed, readable minimum figure size rather than being scaled down
    as if it were one cell of a larger grid.
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

    n_panels = len(masked)
    n_cols = min(6, n_panels)
    n_rows = math.ceil(n_panels / n_cols)

    # 5 legend entries wrapped to 3 columns (2 rows) rather than forced
    # onto one row -- at 5 columns, the legend needs more horizontal
    # room than a single narrow panel (the min_fig_width case) has,
    # clipping "8+"/"Masked / no data" off the figure edge.
    fig, axes, legend_ax = _new_figure_with_legend_row(
        n_rows, n_cols, panel_size=3.4, legend_row_height=1.3,
        min_fig_width=4.5, min_fig_height=4.5,
    )

    for i, (m, label) in enumerate(zip(masked, labels)):
        row, col = divmod(i, n_cols)
        ax = axes[row, col]
        ax.imshow(m, cmap=_flood_count_cmap, norm=_flood_count_norm)
        ax.set_title(label)
        ax.set_xticks([])
        ax.set_yticks([])

    for i in range(n_panels, n_rows * n_cols):
        row, col = divmod(i, n_cols)
        axes[row, col].axis('off')

    legend_handles = [
        Patch(facecolor=_FLOOD_COUNT_BIN_COLORS[i], label=_FLOOD_COUNT_BIN_LABELS[i])
        for i in range(4)
    ] + [Patch(facecolor=_FLOOD_MASKED_COLOR, label=_FLOOD_MASKED_LABEL)]
    legend_ax.legend(handles=legend_handles, loc='center', ncol=3, title='Flood-day count')

    fig.suptitle(f'Monthly flood-day count -- AOI {aoi_id}')
    return fig
