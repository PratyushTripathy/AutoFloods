"""
Generates Figure 1 (Bihar fishnet grid, large, with a small India locator
inset) for the SoftwareX manuscript. Not part of the autofloods package.

Main panel: Bihar's fishnet grid, large, tile IDs labeled, Bihar state
boundary overlaid. Only tiles that actually intersect Bihar are shown --
274-277, 313-322, 324-326, 329-330 (19 tiles). Tiles 323, 328, and 331
sit entirely outside Bihar's border in neighbouring states and are
excluded from the analysis entirely (not just this figure) -- see the
manuscript's Section 2.2 grid description and Section 3 statistics,
which were recomputed to match. Inset (small, corner): India with state
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
# 323, 328, 331 don't intersect Bihar's boundary at all -- excluded from
# the analysis entirely (not just this figure), per explicit decision.

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

# Extent from the actual tile grid, not the Bihar boundary -- the tiles
# extend slightly past Bihar's own border on most sides. Extra headroom
# at the top only, to leave clear room for the India inset without it
# overlapping the tiles.
bounds = bihar_tiles.total_bounds
pad_x = (bounds[2] - bounds[0]) * 0.05
pad_y = (bounds[3] - bounds[1]) * 0.05
pad_top = (bounds[3] - bounds[1]) * 0.19
ax.set_xlim(bounds[0] - pad_x, bounds[2] + pad_x)
ax.set_ylim(bounds[1] - pad_y, bounds[3] + pad_top)
ax.set_xticks([])
ax.set_yticks([])
ax.set_aspect('equal')
ax.set_axis_off()  # no plot border

# --- North arrow + scale bar, placed in the empty gap in the grid (the
# missing tiles between 313/314 and 324) -- no tile fill/label underneath.
# Anchored to the tile grid's own top edge (bounds[3]), not the padded
# axes ymax, so it stays just above the tiles regardless of how much
# headroom the India inset needs.
xmin, xmax = ax.get_xlim()
ymin, ymax = ax.get_ylim()
dx = xmax - xmin
dy = ymax - ymin
grid_top = bounds[3]
backing = dict(facecolor='white', alpha=0.75, edgecolor='none', pad=2)

arrow_x = xmin + 0.48 * dx
arrow_y0 = grid_top - 0.16 * (bounds[3] - bounds[1])
arrow_len = 0.07 * (bounds[3] - bounds[1])
ax.add_patch(FancyArrow(arrow_x, arrow_y0, 0, arrow_len, width=dx * 0.005,
                         head_width=dx * 0.02, head_length=arrow_len * 0.35,
                         facecolor=BOUNDARY_COLOR, edgecolor='none', zorder=6))
ax.text(arrow_x, arrow_y0 + arrow_len + dy * 0.01, 'N', ha='center', va='bottom',
        fontsize=10, weight='bold', color=BOUNDARY_COLOR, zorder=6, bbox=backing)

bar_km = 50
bar_len_deg = 0.45  # ~50 km at Bihar's latitude
bar_x0 = xmin + 0.36 * dx
bar_y0 = arrow_y0
ax.plot([bar_x0, bar_x0 + bar_len_deg], [bar_y0, bar_y0], color=BOUNDARY_COLOR,
        linewidth=2.5, solid_capstyle='butt', zorder=6,
        path_effects=[pe.withStroke(linewidth=4.5, foreground='white', alpha=0.75)])
ax.text(bar_x0 + bar_len_deg / 2, bar_y0 + dy * 0.01, f'~{bar_km} km', ha='center', va='bottom',
        fontsize=8, color=BOUNDARY_COLOR, zorder=6, bbox=backing)

# --- Inset: India, small locator, top-right -- inside the extra headroom
# added above the tile grid (pad_top), with a visible gap between the
# inset's own bottom edge and the tiles/Bihar boundary below it.
inset_h = pad_top * 0.85
inset_w = inset_h * 0.9  # roughly India's aspect ratio
inset_x0 = xmax - pad_x - inset_w
inset_y0 = grid_top + pad_top * 0.12  # gap between tiles and inset
axins = inset_axes(ax, width='100%', height='100%',
                    bbox_to_anchor=(inset_x0, inset_y0, inset_w, inset_h),
                    bbox_transform=ax.transData, loc='lower left', borderpad=0)
india_states.plot(ax=axins, facecolor=STATE_FILL, edgecolor='#999999', linewidth=0.3, zorder=1)
bihar_boundary.plot(ax=axins, facecolor=BIHAR_HIGHLIGHT, edgecolor=BOUNDARY_COLOR,
                     linewidth=0.4, zorder=2)
axins.set_axis_off()
axins.set_aspect('equal')

os.makedirs(OUT_DIR, exist_ok=True)
pdf_path = os.path.join(OUT_DIR, 'fig_grid.pdf')
png_path = os.path.join(OUT_DIR, 'fig_grid.png')
fig.savefig(pdf_path, bbox_inches='tight')
fig.savefig(png_path, dpi=300, bbox_inches='tight')
print(f'Wrote {pdf_path}')
print(f'Wrote {png_path}')
print(f'n tiles shown: {len(bihar_tiles)} (dropped: {sorted(set(ALL_BIHAR_TILE_IDS) - set(bihar_tiles["ID"]))})')
