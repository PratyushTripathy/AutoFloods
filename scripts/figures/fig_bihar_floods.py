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
    python scripts/figures/fig_bihar_floods.py
Writes:
    figures/fig_bihar_floods.pdf (vector)
    figures/fig_bihar_floods.png (300 DPI raster -- map figure)
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
VMAX = 5  # fixed colour-scale ceiling (not data-driven), per manuscript style decision
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

    # Coarsen to TARGET_RESOLUTION_M via MEAN (not nearest-neighbour
    # decimation, which silently drops 15 of every 16 source pixels --
    # `average` resampling genuinely averages every valid pixel in each
    # output cell, and GDAL excludes nodata pixels from that average
    # rather than corrupting it, since src_nodata is set below).
    clip_h, clip_w = clipped.shape[1], clipped.shape[2]
    clip_bounds = rasterio.transform.array_bounds(clip_h, clip_w, clipped_transform)
    clip_width_m = clip_bounds[2] - clip_bounds[0]
    clip_height_m = clip_bounds[3] - clip_bounds[1]
    out_w = max(1, round(clip_width_m / TARGET_RESOLUTION_M))
    out_h = max(1, round(clip_height_m / TARGET_RESOLUTION_M))

    down = np.full((clipped.shape[0], out_h, out_w), NAN_SENTINEL, dtype=np.float32)
    for b in range(clipped.shape[0]):
        with rasterio.io.MemoryFile() as memfile:
            profile = {
                'driver': 'GTiff', 'dtype': clipped.dtype, 'count': 1, 'nodata': nodata,
                'height': clip_h, 'width': clip_w,
                'transform': clipped_transform, 'crs': src_crs,
            }
            with memfile.open(**profile) as tmp:
                tmp.write(clipped[b], 1)
                masked = tmp.read(1, out_shape=(out_h, out_w), masked=True,
                                   resampling=rasterio.enums.Resampling.average)
                down[b] = masked.astype(np.float32).filled(NAN_SENTINEL)
    down_transform = clipped_transform * clipped_transform.scale(clip_w / out_w, clip_h / out_h)

    # Reproject to WGS84 for display (mean resampling again, for the same
    # reason -- the data is already continuous after averaging above, so
    # this just avoids re-introducing nearest-neighbour dropout).
    dst_transform, dst_w, dst_h = rasterio.warp.calculate_default_transform(
        src_crs, DISPLAY_CRS, out_w, out_h, *rasterio.transform.array_bounds(out_h, out_w, down_transform))
    data = np.full((down.shape[0], dst_h, dst_w), NAN_SENTINEL, dtype=np.float32)
    for b in range(down.shape[0]):
        rasterio.warp.reproject(
            source=down[b], destination=data[b],
            src_transform=down_transform, src_crs=src_crs, src_nodata=NAN_SENTINEL,
            dst_transform=dst_transform, dst_crs=DISPLAY_CRS, dst_nodata=NAN_SENTINEL,
            resampling=rasterio.warp.Resampling.average)
    bounds = rasterio.transform.array_bounds(dst_h, dst_w, dst_transform)

    valid = data[0] != NAN_SENTINEL
    total = np.zeros(data.shape[1:], dtype=np.float32)
    for b in range(data.shape[0]):
        band_valid = data[b] != NAN_SENTINEL
        total[band_valid] += data[b][band_valid]
    total[~valid] = np.nan
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
sample_h, sample_w = rasters[YEARS[0]][0].shape
panel_aspect = sample_h / sample_w
fig, axes = plt.subplots(nrows, ncols, figsize=(9.5, 9.5 / ncols * panel_aspect * nrows))
axes = axes.flatten()
plt.subplots_adjust(wspace=0.02, hspace=0.06)

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
    im = ax.imshow(total, cmap=cmap, vmin=0, vmax=VMAX,
                    extent=(left, right, bottom, top), interpolation='nearest')
    bihar_gdf.boundary.plot(ax=ax, color='#333333', linewidth=0.6)
    ax.set_xlim(left, right)
    ax.set_ylim(bottom, top)
    # In-panel label instead of ax.set_title -- decouples the label from
    # inter-row spacing, so hspace can be tight without the label
    # colliding with the row above.
    ax.text(0.03, 0.96, str(year), transform=ax.transAxes, ha='left', va='top',
            fontsize=10, weight='bold', color='#222222')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect(1 / np.cos(np.radians((bottom + top) / 2)))  # approx equirectangular at Bihar's latitude
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color('#999999')
        spine.set_linewidth(0.6)

for j in range(len(YEARS), len(axes)):
    axes[j].axis('off')

# North arrow + scale bar on the first panel only, per the plan's
# "one north arrow and one scale bar for the whole panel" requirement.
# In lon/lat degrees, a fixed metre distance isn't a fixed coordinate
# span, so the scale bar's degree-length is computed from the local
# latitude (WGS84 ~111.32 km per degree latitude, longitude scaled by
# cos(latitude)).
ax0 = axes[0]
left, bottom, right, top = rasters[YEARS[0]][1]
mid_lat = (bottom + top) / 2
km_per_deg_lon = 111.32 * np.cos(np.radians(mid_lat))
bar_km = 50
bar_deg = bar_km / km_per_deg_lon
dx = right - left
dy = top - bottom
bar_x0 = left + 0.06 * dx
bar_y0 = bottom + 0.06 * dy
ax0.plot([bar_x0, bar_x0 + bar_deg], [bar_y0, bar_y0], color='#222222', linewidth=2.5,
          solid_capstyle='butt')
ax0.text(bar_x0 + bar_deg / 2, bar_y0 + 0.015 * dy, f'{bar_km} km', ha='center',
          va='bottom', fontsize=7.5, color='#222222')
ax0.annotate('N', xy=(right - 0.08 * dx, top - 0.10 * dy),
             xytext=(right - 0.08 * dx, top - 0.24 * dy),
             ha='center', va='bottom', fontsize=9, weight='bold', color='#222222',
             arrowprops=dict(arrowstyle='-|>', color='#222222', lw=1.6))

cbar = fig.colorbar(im, ax=axes[:len(YEARS)], orientation='horizontal',
                     fraction=0.035, pad=0.03, aspect=40, extend='max')
cbar.set_label('Flooded observations, July-October (count)', fontsize=9)
cbar.ax.tick_params(labelsize=8)

os.makedirs(OUT_DIR, exist_ok=True)
pdf_path = os.path.join(OUT_DIR, 'fig_bihar_floods.pdf')
png_path = os.path.join(OUT_DIR, 'fig_bihar_floods.png')
fig.savefig(pdf_path, bbox_inches='tight')
fig.savefig(png_path, dpi=300, bbox_inches='tight')
print(f'Wrote {pdf_path}')
print(f'Wrote {png_path}')
