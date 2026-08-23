"""
Generates Figure 1 (Bihar fishnet grid, large, with a small India locator
inset) for the SoftwareX manuscript. Not part of the autofloods package.

Main panel: Bihar's fishnet grid, large, tile IDs labeled, Bihar state
boundary overlaid. Only tiles that actually intersect Bihar are shown --
274-277, 313-322, 324-326, 329-330 (19 of the 22-tile OPERA processing
list; 323, 328, 331 sit entirely outside Bihar's border in neighbouring
states and are dropped from this figure, though they were still part of
the actual pipeline run). Inset (small, corner): India with state
boundaries, Bihar highlighted, for geographic context only -- no
mark_inset zoom lines, since Bihar is the primary panel here, not a
sub-region being zoomed into from India.

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
import matplotlib.patheffects as pe
from matplotlib.patches import FancyArrow
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _boundaries import INDIA_STATES_PATH, BIHAR_PATH, require_boundaries

require_boundaries()

OUT_DIR = '/home/emlab/projects/current-projects/edge-autofloods/autofloods-manuscript/figures'
GRID_PATH = '/home/emlab/projects/current-projects/edge-autofloods/AutoFloods/resources/india_utm_fishnet_buffer.gpkg'

ALL_BIHAR_TILE_IDS = [274, 275, 276, 277, 313, 314, 315, 316, 317, 318, 319, 320,
                      321, 322, 323, 324, 325, 326, 328, 329, 330, 331]

TILE_FILL = '#4C72B0'
TILE_EDGE = '#FFFFFF'
TEXT_COLOR = '#FFFFFF'
BOUNDARY_COLOR = '#222222'
STATE_FILL = '#EDEDED'
BIHAR_HIGHLIGHT = '#C0392B'

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10

grid_gdf = gpd.read_file(GRID_PATH)
all_bihar_tiles = grid_gdf[grid_gdf['ID'].isin(ALL_BIHAR_TILE_IDS)]  # EPSG:4326, matches india_states/bihar
india_states = gpd.read_file(INDIA_STATES_PATH)
bihar_boundary = gpd.read_file(BIHAR_PATH)
bihar_geom = bihar_boundary.geometry.iloc[0]

# Keep only tiles that actually touch Bihar -- drops 323, 328, 331, which
# sit entirely in neighbouring states.
bihar_tiles = all_bihar_tiles[all_bihar_tiles.geometry.intersects(bihar_geom)]

fig, ax = plt.subplots(figsize=(9, 9.5))

# --- Main panel: Bihar grid, large ---
bihar_tiles.plot(ax=ax, facecolor=TILE_FILL, edgecolor=TILE_EDGE, linewidth=1.1, zorder=2)
bihar_boundary.boundary.plot(ax=ax, color=BOUNDARY_COLOR, linewidth=1.5, zorder=3)
for _, row in bihar_tiles.iterrows():
    c = row.geometry.centroid
    ax.text(c.x, c.y, str(int(row['ID'])), ha='center', va='center',
            fontsize=10, color=TEXT_COLOR, weight='bold', zorder=4)

bounds = bihar_boundary.total_bounds
pad_x = (bounds[2] - bounds[0]) * 0.06
pad_y = (bounds[3] - bounds[1]) * 0.06
ax.set_xlim(bounds[0] - pad_x, bounds[2] + pad_x)
ax.set_ylim(bounds[1] - pad_y, bounds[3] + pad_y)
ax.set_xticks([])
ax.set_yticks([])
ax.set_aspect('equal')
for spine in ax.spines.values():
    spine.set_edgecolor('#999999')
    spine.set_linewidth(0.6)

# --- North arrow + scale bar, placed in the empty gap in the grid (the
# missing tiles between 313/314 and 324) -- no tile fill/label underneath.
xmin, xmax = ax.get_xlim()
ymin, ymax = ax.get_ylim()
dx = xmax - xmin
dy = ymax - ymin
backing = dict(facecolor='white', alpha=0.75, edgecolor='none', pad=2)

arrow_x = xmin + 0.47 * dx
arrow_y0 = ymax - 0.13 * dy
arrow_len = 0.07 * dy
ax.add_patch(FancyArrow(arrow_x, arrow_y0, 0, arrow_len, width=dx * 0.005,
                         head_width=dx * 0.02, head_length=arrow_len * 0.35,
                         facecolor=BOUNDARY_COLOR, edgecolor='none', zorder=6))
ax.text(arrow_x, arrow_y0 + arrow_len + dy * 0.015, 'N', ha='center', va='bottom',
        fontsize=10, weight='bold', color=BOUNDARY_COLOR, zorder=6, bbox=backing)

bar_km = 50
bar_len_deg = 0.45  # ~50 km at Bihar's latitude
bar_x0 = xmin + 0.30 * dx
bar_y0 = ymax - 0.13 * dy
ax.plot([bar_x0, bar_x0 + bar_len_deg], [bar_y0, bar_y0], color=BOUNDARY_COLOR,
        linewidth=2.5, solid_capstyle='butt', zorder=6,
        path_effects=[pe.withStroke(linewidth=4.5, foreground='white', alpha=0.75)])
ax.text(bar_x0 + bar_len_deg / 2, bar_y0 + dy * 0.015, f'~{bar_km} km', ha='center', va='bottom',
        fontsize=8, color=BOUNDARY_COLOR, zorder=6, bbox=backing)

# --- Inset: India, small, corner locator only -- placed exactly over
# tile 331's own cell (now empty, since 331 was dropped for not touching
# Bihar), using its real coordinates rather than approximate corner
# padding, so it can't drift onto a neighbouring tile's label.
t331_bounds = grid_gdf.loc[grid_gdf['ID'] == 331, 'geometry'].total_bounds
axins = inset_axes(ax, width='100%', height='100%',
                    bbox_to_anchor=(t331_bounds[0], t331_bounds[1], t331_bounds[2] - t331_bounds[0],
                                     t331_bounds[3] - t331_bounds[1]),
                    bbox_transform=ax.transData, loc='lower left', borderpad=0)
india_states.plot(ax=axins, facecolor=STATE_FILL, edgecolor='#999999', linewidth=0.3, zorder=1)
bihar_boundary.plot(ax=axins, facecolor=BIHAR_HIGHLIGHT, edgecolor=BOUNDARY_COLOR,
                     linewidth=0.4, zorder=2)
axins.set_axis_off()
axins.set_aspect('equal')
for spine in axins.spines.values():
    spine.set_edgecolor('#999999')
    spine.set_linewidth(0.5)

os.makedirs(OUT_DIR, exist_ok=True)
pdf_path = os.path.join(OUT_DIR, 'fig_grid.pdf')
png_path = os.path.join(OUT_DIR, 'fig_grid.png')
fig.savefig(pdf_path, bbox_inches='tight')
fig.savefig(png_path, dpi=300, bbox_inches='tight')
print(f'Wrote {pdf_path}')
print(f'Wrote {png_path}')
print(f'n tiles shown: {len(bihar_tiles)} (dropped: {sorted(set(ALL_BIHAR_TILE_IDS) - set(bihar_tiles["ID"]))})')
