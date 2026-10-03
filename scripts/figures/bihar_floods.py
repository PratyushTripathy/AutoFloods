"""
Generates Figure 3 (Bihar flood maps, 2017-2025 small multiples) for the
SoftwareX manuscript. Not part of the autofloods package.

For each year's mosaic (output/bihar_opera_30m/<year>/<year>_mosaic.tif,
bands = calendar months, pixel value = count of flooded observations that
month), sums the wet-season months into one per-year flood-frequency
raster, clips every panel to the Bihar state boundary (so tile-fishnet
edges/gaps never show, only the real state outline), reprojects to WGS84
for display (map figures here are for visualization, not area
measurement -- Figure 4's flooded-area calculation is what needs an
equal-area projection, and that stays in the mosaics' native World
Mollweide), and plots all years as small multiples sharing one colour
scale and one legend/colourbar.

Requires a Bihar boundary GeoPackage that is NOT committed to this repo --
see scripts/figures/_boundaries.py's docstring for where to get it and
where to put it (or set AUTOFLOODS_BOUNDARY_DIR).

Usage:
    python scripts/figures/bihar_floods.py
Writes:
    figures/bihar_floods.png (300 DPI raster -- map figure; PNG only, no PDF)
"""
import os
import sys

import numpy as np
import geopandas as gpd
import rasterio
import rasterio.mask
import rasterio.warp
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _boundaries import BIHAR_PATH as BIHAR_BOUNDARY_PATH, require_boundaries

require_boundaries()

OUT_DIR = '/home/emlab/projects/current-projects/edge-autofloods/autofloods-manuscript/figures'
MOSAIC_DIR = '/home/emlab/projects/current-projects/edge-autofloods/AutoFloods/output/bihar_opera_30m'
YEARS = [2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025]
TARGET_RESOLUTION_M = 300  # display resolution; coarsened via mean, not decimation
PAD_KM = 8  # physical (not degree) padding between the Bihar outline and each panel's frame
VMAX = 5  # fixed colour-scale ceiling (not data-driven), per manuscript style decision
VMIN = 0.25  # colour floor: cells whose areal mean is below this many flooded dates render white
DISPLAY_CRS = 'EPSG:4326'  # WGS84 -- display only; area stats (Figure 4) use Mollweide
NAN_SENTINEL = -9999.0  # float nodata; averaging produces fractional values, not counts

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 9


def load_year_total(year, bihar_geom_native):
    path = os.path.join(MOSAIC_DIR, str(year), f'{year}_mosaic.tif')
    with rasterio.open(path) as src:
        # Clip to the Bihar cutline FIRST, in the mosaic's native CRS (full
        # resolution, so the mask follows the real boundary precisely).
        clipped, clipped_transform = rasterio.mask.mask(
            src, [bihar_geom_native], crop=True, nodata=src.nodata, all_touched=False)
        nodata = src.nodata
        src_crs = src.crs

    # Sum the wet-season bands per NATIVE pixel first (float32, nodata
    # excluded band by band; a pixel is valid where any band is valid), so
    # the single resample below averages the summed counts.
    clip_h, clip_w = clipped.shape[1], clipped.shape[2]
    band_valid = clipped != nodata
    valid = band_valid.any(axis=0)
    summed = (clipped.astype(np.float32) * band_valid).sum(axis=0)
    summed = np.where(valid, summed, NAN_SENTINEL).astype(np.float32)

    # Final WGS84 display grid, targeted directly: the grid spacing (square
    # in degrees) is chosen so the meridional cell is as close to
    # TARGET_RESOLUTION_M as a whole number of rows allows across the
    # clipped extent. A square cell in degrees is not square in metres at
    # Bihar's latitude, so the zonal cell comes out ~10% smaller.
    wgs_bounds = rasterio.warp.transform_bounds(src_crs, DISPLAY_CRS,
                                                *rasterio.transform.array_bounds(clip_h, clip_w, clipped_transform))
    height_deg = wgs_bounds[3] - wgs_bounds[1]
    width_deg = wgs_bounds[2] - wgs_bounds[0]
    dst_h = max(1, round(height_deg / (TARGET_RESOLUTION_M / 111320.0)))
    cell_deg = height_deg / dst_h
    dst_w = max(1, round(width_deg / cell_deg))
    dst_transform = rasterio.transform.from_origin(wgs_bounds[0], wgs_bounds[3], cell_deg, cell_deg)

    # ONE Resampling.average warp from the native Mollweide sum straight
    # onto the display grid: every valid native pixel contributes to the
    # areal mean of its display cell (GDAL excludes nodata from the mean),
    # with no intermediate integer raster and no second reprojection.
    total = np.full((dst_h, dst_w), NAN_SENTINEL, dtype=np.float32)
    rasterio.warp.reproject(
        source=summed, destination=total,
        src_transform=clipped_transform, src_crs=src_crs, src_nodata=NAN_SENTINEL,
        dst_transform=dst_transform, dst_crs=DISPLAY_CRS, dst_nodata=NAN_SENTINEL,
        resampling=rasterio.warp.Resampling.average)
    bounds = rasterio.transform.array_bounds(dst_h, dst_w, dst_transform)
    total[total == NAN_SENTINEL] = np.nan
    return total, bounds


bihar_gdf = gpd.read_file(BIHAR_BOUNDARY_PATH)  # already EPSG:4326
with rasterio.open(os.path.join(MOSAIC_DIR, str(YEARS[0]), f'{YEARS[0]}_mosaic.tif')) as ref:
    mosaic_crs = ref.crs
bihar_native = bihar_gdf.to_crs(mosaic_crs)
bihar_geom_native = bihar_native.geometry.iloc[0].__geo_interface__

rasters = {}
for year in YEARS:
    total, bounds = load_year_total(year, bihar_geom_native)
    rasters[year] = (total, bounds)

ncols = 3
nrows = int(np.ceil(len(YEARS) / ncols))

# Pad relative to the Bihar *vector* boundary's own bounding box, not the
# raster's -- the raster was clipped in Mollweide (an axis-aligned box in
# that CRS) and then reprojected to WGS84, and a Mollweide-aligned box
# does not reproject to a WGS84 box aligned the same way, so the raster's
# WGS84 bounds sit noticeably off-centre from the vector polygon's own
# WGS84 bounds (checked directly: raster xmin was ~0.86 deg west of the
# polygon's true xmin, versus only ~0.75 deg east on the xmax side --
# enough asymmetry to make left/right padding visibly uneven even though
# top/bottom happened to align closely). All years share one boundary, so
# this is computed once.
raw_left, raw_bottom, raw_right, raw_top = bihar_gdf.total_bounds
mid_lat = (raw_bottom + raw_top) / 2
km_per_deg_lon = 111.32 * np.cos(np.radians(mid_lat))
pad_x_deg = PAD_KM / km_per_deg_lon
pad_y_deg = PAD_KM / 111.32
xlim = (raw_left - pad_x_deg, raw_right + pad_x_deg)
ylim = (raw_bottom - pad_y_deg, raw_top + pad_y_deg)


def _nice_ticks(vmin, vmax, n, inset_frac=0.12):
    """n evenly spaced, round values spanning an INSET of [vmin, vmax]
    -- same helper as kosi_example.py, so the graticule labels match
    that figure's style."""
    span = vmax - vmin
    lo, hi = vmin + inset_frac * span, vmax - inset_frac * span
    for step in (0.5, 0.25, 0.1, 0.05):
        start = np.ceil(lo / step) * step
        candidates = np.arange(start, hi + 1e-9, step)
        if len(candidates) >= n:
            idx = np.round(np.linspace(0, len(candidates) - 1, n)).astype(int)
            return sorted(set(candidates[idx]))
    return list(np.linspace(lo, hi, n))


# Graticule labels in the kosi_example.py style: latitude on the left edge
# of the leftmost column only, longitude on the bottom edge of the bottom
# row only, nothing on interior panels, no gridlines.
LAT_TICKS = _nice_ticks(ylim[0], ylim[1], 3)
LON_TICKS = _nice_ticks(xlim[0], xlim[1], 2)

# Panel (frame) aspect is set to exactly match the displayed data aspect
# (padded extent, corrected for the same 1/cos(latitude) stretch applied
# to each axes below) so matplotlib doesn't have to letterbox the image
# into the frame -- that letterboxing is what previously produced heavy
# horizontal padding alongside a near-zero vertical margin, and (since it
# insets the plotted image within an unchanged box) also reintroduced an
# apparent gap between touching columns.
#
# That letterboxing was triggered by fig.colorbar(..., ax=axes[:9]),
# which silently shrinks the *existing* panel axes horizontally to make
# room for itself -- breaking the width:height ratio those axes were
# built with, so matplotlib's aspect enforcement (adjustable='box') then
# padded the mismatch back in as blank margin. The fix is to reserve the
# colorbar's strip of figure width up front, in the same GridSpec used
# for the panels, and hand the colorbar an explicit `cax` rather than
# letting it resize axes after the fact -- so panel geometry is fixed
# exactly once, correctly, and never touched again.
CBAR_WIDTH_FRAC = 0.045  # fraction of total figure width reserved for the colourbar
CBAR_GAP_FRAC = 0.015    # gap between the grid and the colourbar
GRID_WIDTH_FRAC = 1 - CBAR_WIDTH_FRAC - CBAR_GAP_FRAC
HSPACE_FRAC = 0.06
WSPACE_FRAC = 0.03  # small gap between columns -- a previous pass set this to 0 (touching
                     # columns) per an earlier request, since reversed: some gap reads better

panel_aspect = ((ylim[1] - ylim[0]) / (xlim[1] - xlim[0])) / np.cos(np.radians(mid_lat))
FIG_W = 9.5
panel_width_in = FIG_W * GRID_WIDTH_FRAC / (ncols + (ncols - 1) * WSPACE_FRAC)
panel_height_in = panel_width_in * panel_aspect
FIG_H = panel_height_in * (nrows + (nrows - 1) * HSPACE_FRAC)

fig = plt.figure(figsize=(FIG_W, FIG_H))
gs = fig.add_gridspec(nrows, ncols, left=0.0, right=GRID_WIDTH_FRAC, top=1.0, bottom=0.0,
                       wspace=WSPACE_FRAC, hspace=HSPACE_FRAC)
axes = np.array([[fig.add_subplot(gs[r, c]) for c in range(ncols)] for r in range(nrows)]).flatten()

# Blues sequential ramp, colourblind-safe; force the "no data" pixels to a
# light neutral grey rather than white so the true clipped state outline
# reads as a shape, not just fading into the page background.
cmap = plt.get_cmap('Blues').copy()
cmap.set_bad('#F2F2F2')

im = None
for i, year in enumerate(YEARS):
    ax = axes[i]
    total, bounds = rasters[year]
    left, bottom, right, top = bounds
    im = ax.imshow(total, cmap=cmap, vmin=VMIN, vmax=VMAX,
                    extent=(left, right, bottom, top), interpolation='nearest')
    bihar_gdf.boundary.plot(ax=ax, color='#333333', linewidth=0.6)
    # xlim/ylim use the padded frame computed above (same for every year,
    # since every year clips to the same boundary); imshow's own extent
    # above stays at the true, unpadded raster bounds. The padding strip
    # falls outside imshow's extent, so it needs its own facecolor to
    # match the in-raster nodata colour, or it would show through as the
    # axes' default white instead.
    ax.set_facecolor('#F2F2F2')
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    # In-panel label instead of ax.set_title -- decouples the label from
    # inter-row spacing, so hspace can be tight without the label
    # colliding with the row above. Upper-right placement: Bihar's outline
    # narrows toward the north-east of each panel, leaving open space there
    # that the upper-left corner doesn't have.
    ax.text(0.97, 0.96, str(year), transform=ax.transAxes, ha='right', va='top',
            fontsize=10, weight='bold', color='#222222')
    if i % ncols == 0:
        ax.set_yticks(LAT_TICKS)
        ax.set_yticklabels([f'{v:.1f}°N' for v in LAT_TICKS], fontsize=8.5, rotation=90, va='center')
        ax.tick_params(axis='y', length=3)
    else:
        ax.set_yticks([])
    if i // ncols == nrows - 1:
        ax.set_xticks(LON_TICKS)
        ax.set_xticklabels([f'{v:.1f}°E' for v in LON_TICKS], fontsize=8.5)
        ax.tick_params(axis='x', length=3)
    else:
        ax.set_xticks([])
    ax.set_aspect(1 / np.cos(np.radians((bottom + top) / 2)))  # approx equirectangular at Bihar's latitude
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color('#999999')
        spine.set_linewidth(0.6)

for j in range(len(YEARS), len(axes)):
    axes[j].axis('off')

# Single common colourbar on the right, vertical, flat ends (no pointy
# "extend" wedge) -- the >=5 bin is instead spelled out in the top tick
# label. orientation='vertical' rotates the label 90 degrees by default.
# Uses its own pre-reserved `cax` (see GRID_WIDTH_FRAC/CBAR_* above)
# instead of `ax=`, so it never resizes the panel axes.
cax = fig.add_axes([GRID_WIDTH_FRAC + CBAR_GAP_FRAC, 0.06, CBAR_WIDTH_FRAC * 0.5, 0.88])
cbar = fig.colorbar(im, cax=cax, orientation='vertical', extend='neither')
# The bar's base is the colour floor VMIN, not zero, and says so; cells below
# VMIN render white (see the caption).
cbar.set_ticks([VMIN] + list(range(1, VMAX + 1)))
cbar.set_ticklabels([f'{VMIN:g}'] + [str(v) for v in range(1, VMAX)] + [f'≥{VMAX}'])
cbar.set_label('Flooded observations, July–October (count)', fontsize=9)
cbar.ax.tick_params(labelsize=8)

# Single common scale bar, drawn inside the last (2025) panel's lower-right
# corner rather than as a separate element below the grid, to avoid the
# extra vertical space a dedicated row costs. In lon/lat degrees, a fixed
# metre distance isn't a fixed coordinate span, so the degree-length is
# computed from that panel's own local latitude (WGS84 ~111.32 km per
# degree latitude, longitude scaled by cos(latitude)); the axes-fraction
# width follows directly since imshow's extent maps linearly onto the
# panel's 0-1 axes fraction (set_xlim exactly matches that extent).
last_ax = axes[len(YEARS) - 1]
bar_km = 100
bar_deg = bar_km / km_per_deg_lon
bar_frac = bar_deg / (xlim[1] - xlim[0])

bar_x0 = 0.97 - bar_frac
bar_y0 = 0.08
last_ax.plot([bar_x0, bar_x0 + bar_frac], [bar_y0, bar_y0], color='#222222',
             linewidth=2.5, solid_capstyle='butt', transform=last_ax.transAxes,
             clip_on=False)
for x in (bar_x0, bar_x0 + bar_frac):
    last_ax.plot([x, x], [bar_y0 - 0.02, bar_y0 + 0.02], color='#222222',
                 linewidth=1.2, transform=last_ax.transAxes, clip_on=False)
last_ax.text(bar_x0 + bar_frac / 2, bar_y0 + 0.03, f'{bar_km} km', ha='center',
             va='bottom', fontsize=8, color='#222222', transform=last_ax.transAxes,
             clip_on=False)

os.makedirs(OUT_DIR, exist_ok=True)
png_path = os.path.join(OUT_DIR, 'bihar_floods.png')
fig.savefig(png_path, dpi=300, bbox_inches='tight')
print(f'Wrote {png_path}')

print(f'panel xlim={xlim} ylim={ylim} (padded {PAD_KM} km; raw data bounds '
      f'were ({raw_left:.4f}, {raw_right:.4f}) / ({raw_bottom:.4f}, {raw_top:.4f}), '
      f'same for every year)')
