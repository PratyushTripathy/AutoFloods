"""
Generate one run_autofloods.py YAML config per (tile, year) for the full
Bihar OPERA production batch. Matches the original production settings
(dry_years=[year,year], wet_duration=[f'{year}/07', f'{year}/10'],
same 19-tile fishnet list, same thresholds) used for the 80m/20m MPC
batches and the tile-321 OPERA soak tests -- just generalized across
years and all tiles via config instead of hand-edited scripts.
"""
import os

BASE = '/home/emlab/projects/current-projects/edge-autofloods/AutoFloods'
CONFIG_DIR = f'{BASE}/scripts/configs/bihar_opera'

# 19 tiles that actually intersect Bihar's boundary. Tiles 323, 328, and 331
# sit entirely outside Bihar in neighbouring states and are excluded from
# the analysis entirely (see scripts/figures/fig_grid.py).
TILES = [274, 275, 276, 277, 313, 314, 315, 316, 317, 318, 319, 320,
         321, 322, 324, 325, 326, 329, 330]

YEARS = [2024, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2025, 2026]

TEMPLATE = """\
aoi:
  grid_shapefile: {base}/resources/india_utm_fishnet_buffer.gpkg
  grid_id_list: [{tile}]
  dry_date_col: dry_month
  id_col: ID

dates:
  dry_years: [{year}, {year}]
  wet_duration: ['{year}/07', '{year}/10']

source:
  type: opera

detector:
  type: zscore

detection:
  vv_thd: -2.5
  vh_thd: -2.5
  rel_slope_thd: 20

read:
  overview_level: null
  max_workers: 6

slope:
  dem_overview: 0
  buffer: 500

output_dir: {base}/output/bihar_opera_30m/{year}
slope_dir: {base}/resources/slope
"""

if __name__ == '__main__':
    os.makedirs(CONFIG_DIR, exist_ok=True)
    n = 0
    for year in YEARS:
        for tile in TILES:
            path = os.path.join(CONFIG_DIR, f'bihar_{year}_tile{tile}.yaml')
            with open(path, 'w') as f:
                f.write(TEMPLATE.format(base=BASE, tile=tile, year=year))
            n += 1
    print(f'Wrote {n} config files to {CONFIG_DIR}')
