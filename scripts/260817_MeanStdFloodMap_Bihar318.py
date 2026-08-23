import sys
sys.path.append('../')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from autofloods import flood_mapper

AOI_ID = 318

fm = flood_mapper(
    grid_shapefile=r'../resources/india_utm_fishnet_buffer.gpkg',
    grid_id_list=[AOI_ID],
    dry_date_col='dry_month',
    id_col='ID',
    dry_years=[2024, 2024],
    slope_dir=r'../resources/slope/',
    wet_duration=['2024/08', '2024/08'],
)

fm.get_dry_dates()
fm.generate_dry_date_ranges()
fm.get_s1_items(dry_wet='dry')
fm.read_scenes(dry_wet='dry', overview_level=2)
fm.generate_mean_std_by_aoi()

fm.prepare_slope(dem_overview=0, buffer=500)

fm.prepare_wet_scenes(overview_level=2)
fm.map_floods(vv_thd=-2.5, vh_thd=-2.5, rel_slope_thd=20,
              export_raster=False, export_vector=False, export_maps=False)
fm.merge_floods_by_date(export_raster=False)

vv_mean = fm.mean_std_by_aoi[AOI_ID].sel(band='vv_mean')
vv_std = fm.mean_std_by_aoi[AOI_ID].sel(band='vv_std')

# a single "final" flood map: max classification value across all wet-period dates
# (0=no flood, 1=VH only, 2=VV only, 3=high-confidence flood in both bands)
final_flood = fm.flood_by_date[AOI_ID].max(dim='date')

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

vv_mean.plot(ax=axes[0], cmap='Greys_r', add_colorbar=True)
axes[0].set_title(f'VV dry-season mean (AOI {AOI_ID})')

vv_std.plot(ax=axes[1], cmap='viridis', add_colorbar=True)
axes[1].set_title(f'VV dry-season std (AOI {AOI_ID})')

final_flood.plot(ax=axes[2], cmap='Blues', vmin=0, vmax=3, add_colorbar=True)
axes[2].set_title(f'Final flood map, Aug 2024 (AOI {AOI_ID})')

for ax in axes:
    ax.set_aspect('equal')

plt.tight_layout()
outfile = '../output/final_output/260817_MeanStdFlood_Bihar318.png'
plt.savefig(outfile, dpi=150, bbox_inches='tight')
print(f'Saved figure to {outfile}')
