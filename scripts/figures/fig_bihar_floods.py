"""
Generates Figure 3 (Bihar flood maps, 2017-2025 small multiples) for the
SoftwareX manuscript. Not part of the autofloods package.

For each year's mosaic (output/bihar_opera_30m/<year>/<year>_mosaic.tif,
bands = calendar months, pixel value = count of flooded observations that
month), sums the wet-season months into one per-year flood-frequency
raster, clips every panel to the Bihar state boundary (so tile-fishnet
edges/gaps never show, only the real state outline), and plots all years
as small multiples sharing one colour scale and one legend/colourbar.

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
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _boundaries import BIHAR_PATH as BIHAR_BOUNDARY_PATH, require_boundaries

require_boundaries()

OUT_DIR = '/home/emlab/projects/current-projects/edge-autofloods/autofloods-manuscript/figures'
MOSAIC_DIR = '/home/emlab/projects/current-projects/edge-autofloods/AutoFloods/output/bihar_opera_30m'
YEARS = [2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025]
DOWNSAMPLE_FACTOR = 4
VMAX = 5  # fixed colour-scale ceiling (not data-driven), per manuscript style decision

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 9


def load_year_total(year, bihar_geom):
    path = os.path.join(MOSAIC_DIR, str(year), f'{year}_mosaic.tif')
    with rasterio.open(path) as src:
        # Clip to the Bihar cutline FIRST (in native full resolution, so the
        # mask follows the real boundary precisely), then downsample for
        # small-multiple display.
        clipped, clipped_transform = rasterio.mask.mask(
            src, [bihar_geom], crop=True, nodata=src.nodata, all_touched=False)
        nodata = src.nodata

        out_h = clipped.shape[1] // DOWNSAMPLE_FACTOR
        out_w = clipped.shape[2] // DOWNSAMPLE_FACTOR
        data = np.empty((clipped.shape[0], out_h, out_w), dtype=clipped.dtype)
        for b in range(clipped.shape[0]):
            with rasterio.io.MemoryFile() as memfile:
                profile = src.profile.copy()
                profile.update(height=clipped.shape[1], width=clipped.shape[2],
                                transform=clipped_transform, count=1)
                with memfile.open(**profile) as tmp:
                    tmp.write(clipped[b], 1)
                    data[b] = tmp.read(
                        1, out_shape=(out_h, out_w),
                        resampling=rasterio.enums.Resampling.nearest)
        transform = clipped_transform * clipped_transform.scale(
            clipped.shape[2] / out_w, clipped.shape[1] / out_h)
        bounds = rasterio.transform.array_bounds(out_h, out_w, transform)

    valid = data[0] != nodata
    total = np.zeros(data.shape[1:], dtype=np.float32)
    for b in range(data.shape[0]):
        band_valid = data[b] != nodata
        total[band_valid] += data[b][band_valid]
    total[~valid] = np.nan
    return total, bounds


# Reproject the Bihar boundary once, into each mosaic's CRS (all mosaics
# share the same target CRS, ESRI:54009 -- see mosaic_tiles.py).
bihar_gdf = gpd.read_file(BIHAR_BOUNDARY_PATH)
with rasterio.open(os.path.join(MOSAIC_DIR, str(YEARS[0]), f'{YEARS[0]}_mosaic.tif')) as ref:
    mosaic_crs = ref.crs
bihar_proj = bihar_gdf.to_crs(mosaic_crs)
bihar_geom = bihar_proj.geometry.iloc[0].__geo_interface__

rasters = {}
for year in YEARS:
    total, bounds = load_year_total(year, bihar_geom)
    rasters[year] = (total, bounds)

ncols = 3
nrows = int(np.ceil(len(YEARS) / ncols))
sample_h, sample_w = rasters[YEARS[0]][0].shape
panel_aspect = sample_h / sample_w
fig, axes = plt.subplots(nrows, ncols, figsize=(9.5, 9.5 / ncols * panel_aspect * nrows))
axes = axes.flatten()
plt.subplots_adjust(wspace=0.03, hspace=0.22)

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
    bihar_proj.boundary.plot(ax=ax, color='#333333', linewidth=0.6)
    ax.set_xlim(left, right)
    ax.set_ylim(bottom, top)
    ax.set_title(str(year), fontsize=10, weight='bold', pad=3)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect('equal')
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color('#999999')
        spine.set_linewidth(0.6)

for j in range(len(YEARS), len(axes)):
    axes[j].axis('off')

# North arrow + scale bar on the first panel only, per the plan's
# "one north arrow and one scale bar for the whole panel" requirement.
# Scale bar length is computed in real map units (metres, equal-area
# projection) rather than assumed, since panel extent now varies slightly
# per year's clipped bounding box.
ax0 = axes[0]
left, bottom, right, top = rasters[YEARS[0]][1]
panel_width_m = right - left
bar_m = 50_000
bar_x0 = left + 0.06 * panel_width_m
bar_y0 = bottom + 0.06 * (top - bottom)
ax0.plot([bar_x0, bar_x0 + bar_m], [bar_y0, bar_y0], color='#222222', linewidth=2.5,
          solid_capstyle='butt')
ax0.text(bar_x0 + bar_m / 2, bar_y0 + 0.015 * (top - bottom), '50 km', ha='center',
          va='bottom', fontsize=7.5, color='#222222')
ax0.annotate('N', xy=(right - 0.08 * panel_width_m, top - 0.10 * (top - bottom)),
             xytext=(right - 0.08 * panel_width_m, top - 0.24 * (top - bottom)),
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
