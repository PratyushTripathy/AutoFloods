"""
Generates the manuscript's output-example figure (three panels: RGB
composite, classified output, monthly aggregate) for the SoftwareX
manuscript. Not part of the autofloods package.

Tile 321 (Kosi basin -- Supaul/Saharsa/Madhepura belt), 2020, date
2020-07-29 (984,345 high-confidence flooded pixels -- the peak date
for that tile-year; 2020-07-23 close behind at 969,695, so this is a
plateau across late July 2020, not a singular spike). Selected as the
strongest year for the Kosi tile (1357.1 sq km flooded footprint,
#6 of 171 tile-years overall) rather than the outright tile-year
maximum (2020/tile325, #1) -- the Kosi is a recognizable,
well-documented flood system and makes a more legible example. The
caption describes it as the Kosi basin, not as "the most flooded
tile" -- this figure is illustrative, not a claim about ranking.

Sourced from output/manuscript_fig_regen_20260906/tile321 -- a
same-config regeneration of the real production tile-year
(scripts/configs/bihar_opera/bihar_2020_tile321.yaml, current code),
needed because production (export_raster=False, pre-streaming
architecture) never persisted raw scenes or per-scene classified
rasters -- only the aggregated per-date/per-month outputs. Verified
bit-identical against the real production record before use (see
verify_regen_matches_production() below, run at import time as a hard
gate -- this script refuses to render if the check fails):
  - the regenerated monthly aggregate is pixel-for-pixel identical to
    production's floodextentstackedDRY_2020_2020_WET_202007_202010_321_monthly.tif
    across all 4 months (July-October 2020)
  - the regenerated 2020-07-29 classified band is pixel-for-pixel
    identical to production's own merged-by-date stack band for that
    date (984,345 high-confidence pixels in both)

Usage:
    python scripts/figures/kosi_example.py
Writes:
    figures/kosi_example.png (300 DPI raster)
    figures/kosi_example.pdf (vector)
"""
import os
import sys

import numpy as np
import rasterio
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.colors import BoundaryNorm, ListedColormap
from pyproj import Transformer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from autofloods.visualize import _rgb_composite

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# autofloods-manuscript is a sibling repo to this one (see CLAUDE.md),
# not a subdirectory -- derived relative to BASE rather than a full
# absolute path hardcoded to one machine's directory layout.
OUT_DIR = os.path.join(os.path.dirname(BASE), 'autofloods-manuscript', 'figures')

TILE_ID = 321
DATE = '20200729'
MONTH = '202007'
UTM_CRS = 'EPSG:32645'
PROD_DIR = os.path.join(BASE, 'output', 'bihar_opera_30m', '2020', f'tile{TILE_ID}')
REGEN_DIR = os.path.join(BASE, 'output', 'manuscript_fig_regen_20260906', f'tile{TILE_ID}')

DATE_TAG = 'DRY_2020_2020_WET_202007_202010'
PROD_STACK = os.path.join(
    PROD_DIR, 'flood_raster', f'floodextentstacked_{DATE_TAG}', f'floodextentstacked{DATE_TAG}_{TILE_ID}.tif')
PROD_MONTHLY = os.path.join(
    PROD_DIR, 'flood_raster', f'monthlyadded_{DATE_TAG}', f'floodextentstacked{DATE_TAG}_{TILE_ID}_monthly.tif')
REGEN_STACK = os.path.join(
    REGEN_DIR, 'flood_raster', f'floodextentstacked_{DATE_TAG}', f'floodextentstacked{DATE_TAG}_{TILE_ID}.tif')
REGEN_MONTHLY = os.path.join(
    REGEN_DIR, 'flood_raster', f'monthlyadded_{DATE_TAG}', f'floodextentstacked{DATE_TAG}_{TILE_ID}_monthly.tif')
REGEN_WET_SCENE = os.path.join(REGEN_DIR, 'wet_scenes_cache', f'wetscene_{TILE_ID}_OPERA_PASS_{DATE}.nc')


def _read_band(path, band_label):
    with rasterio.open(path) as ds:
        idx = list(ds.descriptions).index(band_label)
        arr = ds.read(idx + 1)
        nodata = ds.nodata
        bounds = rasterio.transform.array_bounds(ds.height, ds.width, ds.transform)
    return np.where(arr == nodata, np.nan, arr) if nodata is not None else arr, bounds


def verify_regen_matches_production():
    """Hard gate: refuse to render from the regenerated tile-year unless
    it is proven pixel-for-pixel identical to the real production
    record for the exact bands this figure uses. Not a soft warning --
    raises if either check fails."""
    prod_monthly, prod_bounds = _read_band(PROD_MONTHLY, MONTH)
    regen_monthly, regen_bounds = _read_band(REGEN_MONTHLY, MONTH)
    if prod_bounds != regen_bounds:
        raise RuntimeError(f'bounds mismatch: production={prod_bounds} regen={regen_bounds}')
    if not np.array_equal(prod_monthly, regen_monthly, equal_nan=True):
        raise RuntimeError(f'monthly aggregate ({MONTH}) differs between production and regeneration')

    prod_date, _ = _read_band(PROD_STACK, DATE)
    regen_date, _ = _read_band(REGEN_STACK, DATE)
    if not np.array_equal(prod_date, regen_date, equal_nan=True):
        raise RuntimeError(f'classified band ({DATE}) differs between production and regeneration')

    n_flooded = int((prod_date == 3).sum())
    print(f'Verified: regenerated tile{TILE_ID}/2020 matches production exactly '
          f'(monthly aggregate {MONTH}, classified band {DATE}, '
          f'{n_flooded} high-confidence flooded pixels).')


verify_regen_matches_production()

# ---------------------------------------------------------------- data

rgb = _rgb_composite(REGEN_WET_SCENE, thumbnail_max_size=4000)  # full resolution, no downsampling
classified, bounds = _read_band(REGEN_STACK, DATE)
monthly, _ = _read_band(REGEN_MONTHLY, MONTH)
left, bottom, right, top = bounds
extent = (left, right, bottom, top)

# ---------------------------------------------------------- colormaps
# ColorBrewer values, chosen so lightness decreases monotonically
# within each panel (survives greyscale reproduction) and so panels
# (b) and (c) share no hue (separable under colour-vision deficiency,
# and don't read as the same kind of scale despite sitting side by
# side -- they encode unrelated things: per-date detection confidence
# vs. a monthly flood-day-count bin). Masked/no-data, if present in
# either panel, is #BDBDBD -- distinct from the near-white "0"/"not
# flooded" background class.
#
# Panel (b): not-flooded stays neutral; VV-only and VH-only (both
# "low confidence") share one color instead of two similar shades,
# since the distinction barely reads visually and isn't worth two
# legend entries. Blue only.
_CAT_COLORS = ['#EFEFEF', '#6BAED6', '#6BAED6', '#08306B']
_FLOOD_CLASS_LABELS = {0: 'Not flooded', 1: 'Low confidence (VV- or VH-only)', 3: 'High-confidence flood'}
_MASKED_COLOR = '#BDBDBD'
_flood_cmap = ListedColormap(_CAT_COLORS)
_flood_cmap.set_bad(_MASKED_COLOR)
_flood_norm = BoundaryNorm(boundaries=[-0.5, 0.5, 1.5, 2.5, 3.5], ncolors=4)

# Panel (c): yellow -> orange -> red escalation, off blue entirely so
# it shares no hue with panel (b) above.
_COUNT_BIN_LABELS = ['0', '1-3', '4-7', '8+']
_COUNT_BIN_COLORS = ['#EFEFEF', '#FED976', '#FD8D3C', '#BD0026']
_count_cmap = ListedColormap(_COUNT_BIN_COLORS)
_count_cmap.set_bad(_MASKED_COLOR)
_count_norm = BoundaryNorm(boundaries=[-0.5, 0.5, 3.5, 7.5, 1e6], ncolors=4)

# -------------------------------------------------------- lat ticks
# Latitude tick VALUES computed by projecting real EPSG:4326 points
# into this tile's UTM CRS (EPSG:32645), not drawn as gridlines --
# just used to place 3 tick labels on panel (a)'s left edge and panel
# (c)'s right edge. Labels are approximate (read at a representative
# point along that edge, since a UTM edge isn't a true meridian),
# standard practice for a small-extent inset map like this one.
to_ll = Transformer.from_crs(UTM_CRS, 'EPSG:4326', always_xy=True)

corner_x = [left, right, right, left]
corner_y = [bottom, bottom, top, top]
corner_lon, corner_lat = to_ll.transform(corner_x, corner_y)
lon_min, lon_max = min(corner_lon), max(corner_lon)
lat_min, lat_max = min(corner_lat), max(corner_lat)


def _nice_ticks(vmin, vmax, n, inset_frac=0.12):
    """n evenly spaced, round values spanning an INSET of [vmin, vmax]
    -- kept inset_frac away from both ends so a tick label never lands
    right at a panel corner (where it would collide with the
    perpendicular axis's own corner label). Tries progressively finer
    round steps (0.5 deg, then 0.25, then 0.1, then 0.05) until at
    least n round candidates fall in the inset range, then picks n of
    them evenly spaced. Falls back to a plain linspace over the inset
    range if it's too small for any round step to yield n candidates."""
    span = vmax - vmin
    lo, hi = vmin + inset_frac * span, vmax - inset_frac * span
    for step in (0.5, 0.25, 0.1, 0.05):
        start = np.ceil(lo / step) * step
        candidates = np.arange(start, hi + 1e-9, step)
        if len(candidates) >= n:
            idx = np.round(np.linspace(0, len(candidates) - 1, n)).astype(int)
            return sorted(set(candidates[idx]))
    return list(np.linspace(lo, hi, n))


LAT_TICKS = _nice_ticks(lat_min, lat_max, 3)
LON_TICKS = _nice_ticks(lon_min, lon_max, 2)


def _lat_ticks_along_edge(x_edge):
    """(y_position, label) pairs for LAT_TICKS, evaluated at x_edge
    (the panel's left or right edge in UTM meters)."""
    y_samples = np.linspace(bottom, top, 400)
    _, lat_samples = to_ll.transform(np.full_like(y_samples, x_edge), y_samples)
    order = np.argsort(lat_samples)
    ticks = []
    for lat_val in LAT_TICKS:
        if lat_samples[order].min() <= lat_val <= lat_samples[order].max():
            y_pos = np.interp(lat_val, lat_samples[order], y_samples[order])
            ticks.append((y_pos, f'{lat_val:.1f}°N'))
    return ticks


def _lon_ticks_along_edge(y_edge):
    """(x_position, label) pairs for LON_TICKS, evaluated at y_edge
    (the panel's bottom edge in UTM meters)."""
    x_samples = np.linspace(left, right, 400)
    lon_samples, _ = to_ll.transform(x_samples, np.full_like(x_samples, y_edge))
    order = np.argsort(lon_samples)
    ticks = []
    for lon_val in LON_TICKS:
        if lon_samples[order].min() <= lon_val <= lon_samples[order].max():
            x_pos = np.interp(lon_val, lon_samples[order], x_samples[order])
            ticks.append((x_pos, f'{lon_val:.1f}°E'))
    return ticks


# ------------------------------------------------------------- figure

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10.5

fig = plt.figure(figsize=(11.5, 5.1))
gs = fig.add_gridspec(2, 4, width_ratios=[1, 1, 1, 0.09], height_ratios=[1, 0.22],
                       hspace=0.06, wspace=0.045)
ax_a = fig.add_subplot(gs[0, 0])
ax_b = fig.add_subplot(gs[0, 1])
ax_c = fig.add_subplot(gs[0, 2])
ax_arrow = fig.add_subplot(gs[0, 3])
info_a_ax = fig.add_subplot(gs[1, 0])
leg_b_ax = fig.add_subplot(gs[1, 1])
leg_c_ax = fig.add_subplot(gs[1, 2])
for ax in (ax_arrow, info_a_ax, leg_b_ax, leg_c_ax, fig.add_subplot(gs[1, 3])):
    ax.axis('off')

ax_a.imshow(rgb, extent=extent, origin='upper')
ax_a.set_title('(a) Sentinel-1 RGB composite', fontsize=10.5)

ax_b.imshow(np.ma.masked_invalid(classified), extent=extent, origin='upper', cmap=_flood_cmap, norm=_flood_norm)
ax_b.set_title('(b) Classified flood extent, 29 July 2020', fontsize=10.5)

ax_c.imshow(np.ma.masked_invalid(monthly), extent=extent, origin='upper', cmap=_count_cmap, norm=_count_norm)
ax_c.set_title('(c) Monthly sum, July 2020', fontsize=10.5)

# panel a: 3 latitude ticks on the left; panel c: 3 latitude ticks on
# the right only; panel b: no y ticks. All three get 2 longitude
# ticks on the bottom. No gridlines inside any panel.
for ax in (ax_a, ax_b, ax_c):
    ax.set_xlim(left, right)
    ax.set_ylim(bottom, top)
    for spine in ax.spines.values():
        spine.set_visible(False)

bottom_ticks = _lon_ticks_along_edge(bottom)
for ax in (ax_a, ax_b, ax_c):
    ax.set_xticks([x for x, _ in bottom_ticks])
    ax.set_xticklabels([lbl for _, lbl in bottom_ticks], fontsize=8.5)
    ax.tick_params(axis='x', length=3)

a_ticks = _lat_ticks_along_edge(left)
ax_a.set_yticks([y for y, _ in a_ticks])
ax_a.set_yticklabels([lbl for _, lbl in a_ticks], fontsize=8.5, rotation=90, va='center')
ax_a.tick_params(axis='y', length=3)

ax_b.set_yticks([])

c_ticks = _lat_ticks_along_edge(right)
ax_c.yaxis.tick_right()
ax_c.set_yticks([y for y, _ in c_ticks])
ax_c.set_yticklabels([lbl for _, lbl in c_ticks], fontsize=8.5, rotation=90, va='center')
ax_c.tick_params(axis='y', length=3)

# ------------------------------------------------------- north arrow
# Drawn outside all three map panels, in its own column to the right
# of panel (c), so it never overlaps image content.
ax_arrow.set_xlim(0, 1)
ax_arrow.set_ylim(0, 1)
ax_arrow.annotate('', xy=(0.5, 0.96), xytext=(0.5, 0.82),
                   arrowprops=dict(facecolor='black', edgecolor='black', width=3, headwidth=11, headlength=11))
ax_arrow.text(0.5, 0.975, 'N', ha='center', va='bottom', fontsize=12, fontweight='bold', color='black')

# ------------------------------------------------------- scale bar
# Overlaid inside panel (c), in real UTM data coordinates (meters) --
# bottom-left corner, over that panel's mostly-light "0" background.
bar_km = 50
bar_m = bar_km * 1000
bar_x0 = left + 0.06 * (right - left)
bar_y0 = bottom + 0.06 * (top - bottom)
ax_c.plot([bar_x0, bar_x0 + bar_m], [bar_y0, bar_y0], color='#222222', linewidth=2.2,
          solid_capstyle='butt', zorder=5)
for x in (bar_x0, bar_x0 + bar_m):
    ax_c.plot([x, x], [bar_y0 - 0.015 * (top - bottom), bar_y0 + 0.015 * (top - bottom)],
              color='#222222', linewidth=1.3, zorder=5)
ax_c.text(bar_x0 + bar_m / 2, bar_y0 + 0.03 * (top - bottom), f'{bar_km} km',
          ha='center', va='bottom', fontsize=8.5, color='#222222', zorder=5)

# RGB band-assignment explanation, in place of the old scale-bar slot.
info_a_ax.text(0.5, 0.72, 'R = VV, G = VH, B = VV/VH log-ratio (dB),\nper-band percentile-stretched',
               ha='center', va='top', fontsize=8, color='#333333', transform=info_a_ax.transAxes)

# ----------------------------------------------------------- legends
class_handles = [
    Patch(facecolor=_CAT_COLORS[key], label=label, edgecolor='#999999', linewidth=0.5)
    for key, label in _FLOOD_CLASS_LABELS.items()
]
leg_b_ax.legend(handles=class_handles, loc='upper center', ncol=1, frameon=False, fontsize=8,
                 handlelength=1.2, handleheight=1.2, bbox_to_anchor=(0.5, 1.0))

count_handles = [
    Patch(facecolor=_COUNT_BIN_COLORS[i], label=_COUNT_BIN_LABELS[i], edgecolor='#999999', linewidth=0.5)
    for i in range(4)
]
leg_c_ax.legend(handles=count_handles, loc='upper center', ncol=2, frameon=False, fontsize=8,
                 handlelength=1.2, handleheight=1.2, columnspacing=1.0, bbox_to_anchor=(0.5, 1.0),
                 title='Flood-day count', title_fontsize=8)

os.makedirs(OUT_DIR, exist_ok=True)
png_path = os.path.join(OUT_DIR, 'kosi_example.png')
pdf_path = os.path.join(OUT_DIR, 'kosi_example.pdf')
fig.savefig(png_path, dpi=300, bbox_inches='tight')
fig.savefig(pdf_path, bbox_inches='tight')
print(f'wrote {png_path}')
print(f'wrote {pdf_path}')
