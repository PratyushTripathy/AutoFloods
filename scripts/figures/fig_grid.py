"""
Generates Figure 1 (Bihar fishnet grid, zoomed in from an India context
map) for the SoftwareX manuscript. Not part of the autofloods package.

Same inset-zoom mechanism as the Amazon deforestation-residual-map figure
in land-based-solutions/neural-network-model/slurm_scripts/
figure_residual_map_model4c.py: mpl_toolkits.axes_grid1.inset_locator's
inset_axes() places the zoomed panel, and mark_inset() both draws the
rectangle on the main map showing what's zoomed and the connector lines
from that rectangle to the inset's corners -- instead of a disconnected
small locator map in a corner.

Main panel: India, all state boundaries drawn. Inset: Bihar's 22-tile
fishnet grid, tile IDs labeled, Bihar state boundary overlaid, connected
back to its outline on the main India map.

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
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _boundaries import INDIA_STATES_PATH, BIHAR_PATH, require_boundaries

require_boundaries()

OUT_DIR = '/home/emlab/projects/current-projects/edge-autofloods/autofloods-manuscript/figures'
GRID_PATH = '/home/emlab/projects/current-projects/edge-autofloods/AutoFloods/resources/india_utm_fishnet_buffer.gpkg'

BIHAR_TILE_IDS = [274, 275, 276, 277, 313, 314, 315, 316, 317, 318, 319, 320,
                  321, 322, 323, 324, 325, 326, 328, 329, 330, 331]

TILE_FILL = '#4C72B0'
TILE_EDGE = '#FFFFFF'
TEXT_COLOR = '#FFFFFF'
BOUNDARY_COLOR = '#222222'
STATE_FILL = '#EDEDED'
BIHAR_HIGHLIGHT = '#C0392B'

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 8.5

grid_gdf = gpd.read_file(GRID_PATH)
bihar_tiles = grid_gdf[grid_gdf['ID'].isin(BIHAR_TILE_IDS)]  # kept in EPSG:4326, matches india_states
india_states = gpd.read_file(INDIA_STATES_PATH)
bihar_boundary = gpd.read_file(BIHAR_PATH)

fig, ax = plt.subplots(figsize=(13.5, 8.5))

# --- Main panel: India, all state boundaries ---
india_states.plot(ax=ax, facecolor=STATE_FILL, edgecolor='#999999', linewidth=0.5, zorder=1)
bihar_boundary.plot(ax=ax, facecolor=BIHAR_HIGHLIGHT, edgecolor=BOUNDARY_COLOR,
                     linewidth=0.8, alpha=0.55, zorder=2)
ax.set_axis_off()
ax.set_aspect('equal')

# --- Inset: Bihar grid, zoomed ---
axins = inset_axes(ax, width='55%', height='75%',
                    bbox_to_anchor=(0.42, -0.08, 0.58, 0.75),
                    bbox_transform=ax.transAxes, loc='lower left', borderpad=0)
bihar_tiles.plot(ax=axins, facecolor=TILE_FILL, edgecolor=TILE_EDGE, linewidth=0.9, zorder=2)
bihar_boundary.boundary.plot(ax=axins, color=BOUNDARY_COLOR, linewidth=1.2, zorder=3)
for _, row in bihar_tiles.iterrows():
    c = row.geometry.centroid
    axins.text(c.x, c.y, str(int(row['ID'])), ha='center', va='center',
               fontsize=6.5, color=TEXT_COLOR, weight='bold', zorder=4)

ins_bounds = bihar_tiles.total_bounds  # tile grid, not the state outline -- tiles extend past Bihar's own border
pad_x = (ins_bounds[2] - ins_bounds[0]) * 0.04
pad_y = (ins_bounds[3] - ins_bounds[1]) * 0.04
axins.set_xlim(ins_bounds[0] - pad_x, ins_bounds[2] + pad_x)
axins.set_ylim(ins_bounds[1] - pad_y, ins_bounds[3] + pad_y)
axins.set_xticks([])
axins.set_yticks([])
axins.set_aspect('equal')
for spine in axins.spines.values():
    spine.set_edgecolor('#666666')
    spine.set_linewidth(0.9)

# mark_inset draws BOTH the rectangle on the main map (around Bihar's own
# extent, set as ax's zoom target via loc1/loc2 anchor corners) and the
# connector lines to the inset -- same call the deforestation-residual
# figure uses.
mark_inset(ax, axins, loc1=2, loc2=3, fc='none', ec='0.4', linewidth=0.8)

# North arrow + scale bar on the inset (the panel readers will actually use for measurement)
xmin, xmax = axins.get_xlim()
ymin, ymax = axins.get_ylim()
dx = xmax - xmin
dy = ymax - ymin

# Semi-transparent white backing boxes so these stay legible regardless of
# which tile colour/label sits underneath, positioned in the bottom-left
# corner clear of any tile ID text.
backing = dict(facecolor='white', alpha=0.75, edgecolor='none', pad=2)

# Placed in the genuinely empty gap in the grid (the two missing tiles
# between 313/314 and 323/328 in the top row) -- the only spot with no
# tile fill or label underneath at all, rather than chasing collisions
# with individual tile ID labels around the packed grid's edges.
arrow_x = xmin + 0.47 * dx
arrow_y0 = ymax - 0.16 * dy
arrow_len = 0.09 * dy
axins.add_patch(FancyArrow(arrow_x, arrow_y0, 0, arrow_len, width=dx * 0.006,
                            head_width=dx * 0.026, head_length=arrow_len * 0.35,
                            facecolor=BOUNDARY_COLOR, edgecolor='none', zorder=6))
axins.text(arrow_x, arrow_y0 + arrow_len + dy * 0.02, 'N', ha='center', va='bottom',
           fontsize=8, weight='bold', color=BOUNDARY_COLOR, zorder=6, bbox=backing)

bar_len_deg = 0.45  # ~50 km at Bihar's latitude
bar_x0 = xmin + 0.32 * dx
bar_y0 = ymax - 0.16 * dy
axins.plot([bar_x0, bar_x0 + bar_len_deg], [bar_y0, bar_y0], color=BOUNDARY_COLOR,
           linewidth=2.2, solid_capstyle='butt', zorder=6,
           path_effects=[pe.withStroke(linewidth=4, foreground='white', alpha=0.75)])
axins.text(bar_x0 + bar_len_deg / 2, bar_y0 + dy * 0.02, '~50 km', ha='center', va='bottom',
           fontsize=6.5, color=BOUNDARY_COLOR, zorder=6, bbox=backing)

os.makedirs(OUT_DIR, exist_ok=True)
pdf_path = os.path.join(OUT_DIR, 'fig_grid.pdf')
png_path = os.path.join(OUT_DIR, 'fig_grid.png')
fig.savefig(pdf_path, bbox_inches='tight')
fig.savefig(png_path, dpi=300, bbox_inches='tight')
print(f'Wrote {pdf_path}')
print(f'Wrote {png_path}')
print(f'n tiles: {len(bihar_tiles)}')
