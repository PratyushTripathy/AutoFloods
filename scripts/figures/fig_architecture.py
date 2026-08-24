"""
Generates Figure 2 (architecture diagram) for the SoftwareX manuscript.
Not part of the autofloods package -- a one-off figure-generation script,
committed so the figure is reproducible (per the plan's figure style
requirements).

Shows: the abstract STACSource/FloodDetector interfaces, their concrete
implementations, the shared backend-agnostic pipeline path, and which
artifacts are cached to disk vs recomputed per run.

Usage:
    python scripts/figures/fig_architecture.py
Writes:
    figures/fig_architecture.pdf (vector, for the manuscript)
    figures/fig_architecture.png (300 DPI raster preview)
"""
import os

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.lines import Line2D

# Hardcoded absolute path rather than derived, to avoid path-math errors.
# NOTE: /home/ptripathy/Projects/edge-autofloods is a symlink to
# /home/emlab/projects/current-projects/edge-autofloods -- they are the
# SAME directory, not two independent copies. Never rm -rf one assuming
# the other is untouched.
OUT_DIR = '/home/emlab/projects/current-projects/edge-autofloods/autofloods-manuscript/figures'

# Colourblind-safe, muted 2-tone palette (Wong 2011 palette subset) plus grey.
FLOW = '#4C72B0'       # muted blue -- ordinary pipeline stages
CACHE = '#DD8452'      # muted orange -- cached-to-disk artifacts
INTERFACE = '#55555A'  # dark grey -- abstract interfaces (dashed)
TEXT = '#222222'
BG = '#FFFFFF'

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 9.5


def box(ax, x, y, w, h, label, sub=None, color=FLOW, dashed=False, textcolor='white'):
    style = 'round,pad=0.02,rounding_size=0.06'
    b = FancyBboxPatch(
        (x, y), w, h, boxstyle=style,
        linewidth=1.3,
        edgecolor=color if dashed else 'none',
        facecolor=BG if dashed else color,
        linestyle=(0, (4, 2)) if dashed else 'solid',
    )
    ax.add_patch(b)
    cx, cy = x + w / 2, y + h / 2
    tcolor = color if dashed else textcolor
    if sub:
        ax.text(cx, cy + h * 0.14, label, ha='center', va='center',
                fontsize=9.5, color=tcolor, weight='bold' if not dashed else 'normal',
                style='italic' if dashed else 'normal')
        ax.text(cx, cy - h * 0.20, sub, ha='center', va='center',
                fontsize=7.6, color=tcolor)
    else:
        ax.text(cx, cy, label, ha='center', va='center',
                fontsize=9.5, color=tcolor, weight='bold' if not dashed else 'normal',
                style='italic' if dashed else 'normal')
    return (cx, cy, x, y, w, h)


def arrow(ax, p_from, p_to, side_from='right', side_to='left', color='#666666', style='-|>'):
    def edge_point(p, side):
        cx, cy, x, y, w, h = p
        return {
            'right': (x + w, cy), 'left': (x, cy),
            'top': (cx, y + h), 'bottom': (cx, y),
        }[side]
    a = FancyArrowPatch(
        edge_point(p_from, side_from), edge_point(p_to, side_to),
        arrowstyle=style, mutation_scale=10, linewidth=1.1,
        color=color, shrinkA=2, shrinkB=2, connectionstyle='arc3,rad=0.0',
    )
    ax.add_patch(a)


fig, ax = plt.subplots(figsize=(9.5, 6.2))
ax.set_xlim(0, 9.5)
ax.set_ylim(0, 6.2)
ax.axis('off')

# --- Source interface hierarchy (top) ---
src_abs = box(ax, 0.3, 5.35, 2.2, 0.55, 'STACSource', sub='(abstract interface)',
              color=INTERFACE, dashed=True)
opera = box(ax, 0.15, 4.35, 1.15, 0.7, 'OPERASource', sub='OPERA RTC-S1\nAWS / ASF', color=FLOW)
mpc = box(ax, 1.5, 4.35, 1.15, 0.7, 'MPCSource', sub='sentinel-1-rtc\nAzure / MPC', color=FLOW)
arrow(ax, src_abs, opera, side_from='bottom', side_to='top')
arrow(ax, src_abs, mpc, side_from='bottom', side_to='top')

# --- Detector interface hierarchy (bottom-left, feeds into "detect") ---
# Two concrete implementations, same visual treatment as OPERASource/
# MPCSource under STACSource above -- ZScoreDetector (default; used for
# every result in this paper) and OtsuDetector (single-scene, no
# dry-season baseline: requires_baseline_fitting=False, the first
# detector to exercise that flag's skip path through the orchestrator).
det_abs = box(ax, 3.55, 0.25, 3.0, 0.55, 'FloodDetector', sub='(abstract interface)',
              color=INTERFACE, dashed=True)
zscore = box(ax, 3.65, 1.15, 1.5, 0.62, 'ZScoreDetector', sub='Z-score\nthresholding', color=FLOW)
otsu = box(ax, 5.25, 1.15, 1.15, 0.62, 'OtsuDetector', sub='Otsu\nthresholding', color=FLOW)
future_det = box(ax, 6.5, 1.15, 1.3, 0.62, 'future detector', sub='(pluggable)',
                  color=INTERFACE, dashed=True)
arrow(ax, det_abs, zscore, side_from='top', side_to='bottom')
arrow(ax, det_abs, otsu, side_from='top', side_to='bottom')
arrow(ax, det_abs, future_det, side_from='top', side_to='bottom', color='#AAAAAA')

# --- Shared pipeline, row A (top) ---
row_a_y = 3.35
row_a_h = 0.62
xs_a = [0.15, 1.55, 2.95, 4.55, 6.05]
ws_a = [1.2, 1.2, 1.4, 1.3, 1.3]
labels_a = ['search', 'read VV/VH', 'reproject\n(tile UTM)', 'stack', 'baseline\n(mean / SD)']
colors_a = [FLOW, FLOW, FLOW, FLOW, CACHE]
boxes_a = []
for x, w, lab, c in zip(xs_a, ws_a, labels_a, colors_a):
    b = box(ax, x, row_a_y, w, row_a_h, lab, color=c)
    boxes_a.append(b)
for i in range(len(boxes_a) - 1):
    arrow(ax, boxes_a[i], boxes_a[i + 1])

# link sources down into "search"
arrow(ax, opera, boxes_a[0], side_from='bottom', side_to='top', color='#999999')
arrow(ax, mpc, boxes_a[0], side_from='bottom', side_to='top', color='#999999')

# elbow: baseline (end of row A) down to detect (start of row B)
row_b_y = 2.1
row_b_h = 0.62
xs_b = [6.05, 7.55, 0.15, 1.65, 3.35, 5.05]
# reorder row B left-to-right beneath row A, snaking back
xs_b = [8.15, 6.55, 4.95, 3.35, 1.75, 0.15]
labels_b = ['Mollweide\nmosaic', 'monthly\naggregation', 'merge\nby date', 'export\nCOG', 'slope mask',
            'detect']
colors_b = [FLOW, FLOW, FLOW, FLOW, CACHE, FLOW]
ws_b = [1.2, 1.45, 1.4, 1.2, 1.4, 1.2]
boxes_b = []
for x, w, lab, c in zip(xs_b, ws_b, labels_b, colors_b):
    b = box(ax, x, row_b_y, w, row_b_h, lab, color=c)
    boxes_b.append(b)
# boxes_b is right-to-left in reading flow (detect -> ... -> mosaic), reverse for arrows
flow_b = list(reversed(boxes_b))
for i in range(len(flow_b) - 1):
    arrow(ax, flow_b[i], flow_b[i + 1])

# baseline (row A, last) elbows down to detect (row B, last-in-xs_b list = leftmost = 'detect')
detect_box = boxes_b[-1]
arrow(ax, boxes_a[-1], detect_box, side_from='bottom', side_to='top', color='#666666')

# detector interfaces feed up into "detect"
arrow(ax, zscore, detect_box, side_from='top', side_to='bottom')
arrow(ax, otsu, detect_box, side_from='top', side_to='bottom')

# --- Legend ---
legend_elems = [
    Line2D([0], [0], marker='s', color='w', markerfacecolor=FLOW, markersize=12, label='pipeline stage'),
    Line2D([0], [0], marker='s', color='w', markerfacecolor=CACHE, markersize=12, label='cached to disk, reused across runs'),
    Line2D([0], [0], marker='s', color='w', markerfacecolor='none', markeredgecolor=INTERFACE,
           linestyle='--', markersize=12, label='abstract interface'),
]
ax.legend(handles=legend_elems, loc='lower right', bbox_to_anchor=(0.995, -0.01),
          frameon=False, fontsize=8.2, handletextpad=0.6, borderaxespad=0.0)

plt.tight_layout()
os.makedirs(OUT_DIR, exist_ok=True)
pdf_path = os.path.join(OUT_DIR, 'fig_architecture.pdf')
png_path = os.path.join(OUT_DIR, 'fig_architecture.png')
fig.savefig(pdf_path, bbox_inches='tight')
fig.savefig(png_path, dpi=300, bbox_inches='tight')
print(f'Wrote {pdf_path}')
print(f'Wrote {png_path}')
