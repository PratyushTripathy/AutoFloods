"""
Generates Figure 4 (flooded area time series) for the SoftwareX manuscript.
Not part of the autofloods package.

For each year's mosaic, computes total flooded area (sq km) as the area of
pixels flagged as flooded (count >= 1) in at least one wet-season month,
using the mosaic's own per-pixel area in its equal-area (Mollweide)
projection -- see mosaic_tiles.py's docstring for why Mollweide is used
rather than an arbitrary UTM zone.

Usage:
    python scripts/figures/fig_area_stats.py
Writes:
    figures/fig_area_stats.pdf (vector)
    figures/fig_area_stats.png (300 DPI raster)
"""
import os

import numpy as np
import rasterio
import matplotlib.pyplot as plt

OUT_DIR = '/home/emlab/projects/current-projects/edge-autofloods/autofloods-manuscript/figures'
MOSAIC_DIR = '/home/emlab/projects/current-projects/edge-autofloods/AutoFloods/output/bihar_opera_30m'
YEARS = [2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025]

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10.5

areas = []
for year in YEARS:
    path = os.path.join(MOSAIC_DIR, str(year), f'{year}_mosaic.tif')
    with rasterio.open(path) as src:
        data = src.read()
        nodata = src.nodata
        px_area_km2 = abs(src.transform.a * src.transform.e) / 1e6
    valid = data != nodata
    flooded_any_month = (np.where(valid, data, 0) >= 1).any(axis=0)
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
