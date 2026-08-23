"""
Generates Figure 4 (flooded area time series) for the SoftwareX manuscript.
Not part of the autofloods package.

For each year's mosaic, clips to the Bihar state boundary first (in the
mosaic's native, equal-area Mollweide CRS -- the same cutline
fig_bihar_floods.py uses for Figure 3), then computes total flooded area
(sq km) as the area of pixels flagged as flooded (count >= 1) in at least
one wet-season month, using the mosaic's own per-pixel area in its
equal-area (Mollweide) projection -- see mosaic_tiles.py's docstring for
why Mollweide is used rather than an arbitrary UTM zone.

The clip matters: the raw mosaic is the full 19-tile fishnet footprint
(~209,000 sq km for 2024), which is more than double Bihar's true area
(~94,600 sq km) since the fishnet tiles extend well past the state
border. Without clipping, "flooded area for Bihar" silently includes
flooding in parts of Nepal, UP, Jharkhand, and West Bengal that happen
to fall inside the tile grid.

Requires a Bihar boundary GeoPackage that is NOT committed to this repo --
see scripts/figures/_boundaries.py's docstring for where to get it and
where to put it (or set AUTOFLOODS_BOUNDARY_DIR).

Usage:
    python scripts/figures/fig_area_stats.py
Writes:
    figures/fig_area_stats.pdf (vector)
    figures/fig_area_stats.png (300 DPI raster)
"""
import os
import sys

import numpy as np
import geopandas as gpd
import rasterio
import rasterio.mask
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _boundaries import BIHAR_PATH as BIHAR_BOUNDARY_PATH, require_boundaries

require_boundaries()

OUT_DIR = '/home/emlab/projects/current-projects/edge-autofloods/autofloods-manuscript/figures'
MOSAIC_DIR = '/home/emlab/projects/current-projects/edge-autofloods/AutoFloods/output/bihar_opera_30m'
YEARS = [2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025]

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10.5

bihar_gdf = gpd.read_file(BIHAR_BOUNDARY_PATH)  # EPSG:4326
with rasterio.open(os.path.join(MOSAIC_DIR, str(YEARS[0]), f'{YEARS[0]}_mosaic.tif')) as ref:
    mosaic_crs = ref.crs
bihar_native = bihar_gdf.to_crs(mosaic_crs)
bihar_geom_native = bihar_native.geometry.iloc[0].__geo_interface__

areas = []
for year in YEARS:
    path = os.path.join(MOSAIC_DIR, str(year), f'{year}_mosaic.tif')
    with rasterio.open(path) as src:
        clipped, clipped_transform = rasterio.mask.mask(
            src, [bihar_geom_native], crop=True, nodata=src.nodata, all_touched=False)
        nodata = src.nodata
        px_area_km2 = abs(clipped_transform.a * clipped_transform.e) / 1e6
    valid = clipped != nodata
    flooded_any_month = (np.where(valid, clipped, 0) >= 1).any(axis=0)
    n_flooded_px = flooded_any_month.sum()
    areas.append(n_flooded_px * px_area_km2)
    print(f'{year}: {n_flooded_px * px_area_km2:.1f} sq km')

fig, ax = plt.subplots(figsize=(7.2, 4.2))
ax.fill_between(YEARS, areas, color='#4C72B0', alpha=0.10, zorder=1)
ax.plot(YEARS, areas, marker='o', color='#2C4870', linewidth=2, markersize=6,
        markerfacecolor='#4C72B0', markeredgecolor='#2C4870', markeredgewidth=1, zorder=3)
for x, y in zip(YEARS, areas):
    ax.annotate(f'{y:,.0f}', (x, y), textcoords='offset points', xytext=(0, 10),
                ha='center', fontsize=8.5, color='#333333')

ax.set_ylabel('Flooded area (sq km)', fontsize=10.5)
ax.set_xticks(YEARS)
ax.set_xticklabels(YEARS, rotation=0)
ax.set_xlim(YEARS[0] - 0.4, YEARS[-1] + 0.4)
ax.set_ylim(0, max(areas) * 1.2)
ax.yaxis.set_major_formatter(lambda v, _: f'{v:,.0f}')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_color('#888888')
ax.spines['bottom'].set_color('#888888')
ax.tick_params(colors='#444444', labelsize=9.5)
ax.grid(axis='y', color='#E2E2E2', linewidth=0.8, zorder=0)
ax.set_axisbelow(True)

plt.tight_layout()
os.makedirs(OUT_DIR, exist_ok=True)
pdf_path = os.path.join(OUT_DIR, 'fig_area_stats.pdf')
png_path = os.path.join(OUT_DIR, 'fig_area_stats.png')
fig.savefig(pdf_path, bbox_inches='tight')
fig.savefig(png_path, dpi=300, bbox_inches='tight')
print(f'Wrote {pdf_path}')
print(f'Wrote {png_path}')
