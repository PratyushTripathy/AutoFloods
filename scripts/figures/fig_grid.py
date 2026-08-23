"""
Generates Figure 1 (Bihar fishnet grid, with an India locator inset) for
the SoftwareX manuscript. Not part of the autofloods package.

Main panel: Bihar's 22-tile fishnet grid, tile IDs labeled, Bihar state
boundary overlaid. Inset panel: India country outline with Bihar
highlighted, so a reader unfamiliar with Indian state geography can place
the study area. Both panels carry the relevant boundary layer(s) --
main shows the state outline (the grid's own extent), inset shows the
country outline plus the state outline as the highlighted region.

Requires India/Bihar boundary GeoPackages that are NOT committed to this
repo -- see scripts/figures/_boundaries.py's docstring for where to get
them and where to put them (or set AUTOFLOODS_BOUNDARY_DIR).

Usage:
    python scripts/figures/fig_grid.py
Writes:
    figures/fig_grid.pdf (vector)
    figures/fig_grid.png (300 DPI raster, map figures use PNG per the plan's
    figure style requirements)
"""
import os
import sys

import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrow

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _boundaries import INDIA_PATH, BIHAR_PATH, require_boundaries

require_boundaries()

OUT_DIR = '/home/emlab/projects/current-projects/edge-autofloods/autofloods-manuscript/figures'
GRID_PATH = '/home/emlab/projects/current-projects/edge-autofloods/AutoFloods/resources/india_utm_fishnet_buffer.gpkg'

BIHAR_TILE_IDS = [274, 275, 276, 277, 313, 314, 315, 316, 317, 318, 319, 320,
                  321, 322, 323, 324, 325, 326, 328, 329, 330, 331]

TILE_FILL = '#4C72B0'
TILE_EDGE = '#FFFFFF'
TEXT_COLOR = '#FFFFFF'
BOUNDARY_COLOR = '#222222'
INDIA_FILL = '#E4E4E4'

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 8.5

grid_gdf = gpd.read_file(GRID_PATH)
bihar_tiles = grid_gdf[grid_gdf['ID'].isin(BIHAR_TILE_IDS)].to_crs('EPSG:32645')  # UTM 45N
bihar_boundary = gpd.read_file(BIHAR_PATH).to_crs('EPSG:32645')
india = gpd.read_file(INDIA_PATH)  # kept in EPSG:4326 for the small inset
bihar_4326 = gpd.read_file(BIHAR_PATH)

fig = plt.figure(figsize=(6.4, 8.2))
ax = fig.add_axes([0.03, 0.03, 0.94, 0.94])

# --- Main panel: Bihar grid + state boundary ---
bihar_tiles.plot(ax=ax, facecolor=TILE_FILL, edgecolor=TILE_EDGE, linewidth=1.1, zorder=2)
bihar_boundary.boundary.plot(ax=ax, color=BOUNDARY_COLOR, linewidth=1.4, zorder=3)

for _, row in bihar_tiles.iterrows():
    c = row.geometry.centroid
    ax.text(c.x, c.y, str(int(row['ID'])), ha='center', va='center',
            fontsize=8, color=TEXT_COLOR, weight='bold', zorder=4)

ax.set_axis_off()
ax.set_aspect('equal')

xmin, xmax = ax.get_xlim()
ymin, ymax = ax.get_ylim()
dx = xmax - xmin
dy = ymax - ymin

# North arrow -- bottom-right, clear of the India inset (top-right)
arrow_x = xmax - 0.08 * dx
arrow_y0 = ymin + 0.06 * dy
arrow_len = 0.07 * dy
ax.add_patch(FancyArrow(arrow_x, arrow_y0, 0, arrow_len, width=dx * 0.004,
                         head_width=dx * 0.018, head_length=arrow_len * 0.35,
                         facecolor=BOUNDARY_COLOR, edgecolor='none', zorder=5))
ax.text(arrow_x, arrow_y0 + arrow_len + dy * 0.012, 'N', ha='center', va='bottom',
        fontsize=10, weight='bold', color=BOUNDARY_COLOR, zorder=5)

# Scale bar (50 km)
bar_len_m = 50_000
bar_x0 = xmin + 0.06 * dx
bar_y0 = ymin + 0.05 * dy
ax.plot([bar_x0, bar_x0 + bar_len_m], [bar_y0, bar_y0], color=BOUNDARY_COLOR, linewidth=2.5,
        solid_capstyle='butt', zorder=5)
ax.text(bar_x0 + bar_len_m / 2, bar_y0 + dy * 0.012, '50 km', ha='center', va='bottom',
        fontsize=8, color=BOUNDARY_COLOR, zorder=5)

# --- Inset: India context, Bihar highlighted ---
ax_inset = fig.add_axes([0.66, 0.66, 0.30, 0.30])
india.plot(ax=ax_inset, facecolor=INDIA_FILL, edgecolor=BOUNDARY_COLOR, linewidth=0.7, zorder=1)
bihar_4326.plot(ax=ax_inset, facecolor='#C0392B', edgecolor='#C0392B', linewidth=0.5, zorder=2)
ax_inset.set_axis_off()
ax_inset.set_aspect('equal')
ax_inset.set_title('India', fontsize=7.5, pad=2, style='italic')

plt.tight_layout()
os.makedirs(OUT_DIR, exist_ok=True)
pdf_path = os.path.join(OUT_DIR, 'fig_grid.pdf')
png_path = os.path.join(OUT_DIR, 'fig_grid.png')
fig.savefig(pdf_path, bbox_inches='tight')
fig.savefig(png_path, dpi=300, bbox_inches='tight')
print(f'Wrote {pdf_path}')
print(f'Wrote {png_path}')
print(f'n tiles: {len(bihar_tiles)}')
