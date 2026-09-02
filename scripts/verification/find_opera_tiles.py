"""
Sample: find OPERA RTC-S1 passes (groups of same-track/subswath bursts)
covering a bbox over a date range, using the existing OPERASource.

Swap OPERASource() for MPCSource() to run the identical query against
Microsoft Planetary Computer instead -- both implement the same
STACSource interface (search_sentinel1(bbox, start_date, end_date)).
"""
import sys
import datetime

sys.path.append('/home/emlab/projects/current-projects/edge-autofloods/AutoFloods')

from autofloods.sources import OPERASource

# Patna tile (ID 318), WGS84 bbox
BBOX = {
    'type': 'Polygon',
    'coordinates': [[
        [84.998, 24.998], [86.002, 24.998], [86.002, 26.002], [84.998, 26.002], [84.998, 24.998]
    ]],
}
START_DATE = datetime.date(2024, 7, 1)
END_DATE = datetime.date(2024, 7, 31)

if __name__ == '__main__':
    source = OPERASource()
    source.authenticate()  # optional -- search_sentinel1() will lazily do this too

    passes = source.search_sentinel1(BBOX, START_DATE, END_DATE)

    print(f'Found {len(passes)} passes covering the bbox, {START_DATE} to {END_DATE}\n')
    for p in passes:
        print(f'{p.id}  ({len(p.bursts)} bursts)')
        for b in p.bursts:
            print(f'    {b.id}')

    # Read pixel data for the first pass: mosaics its bursts via a GDAL
    # VRT, then opens that VRT lazily (rioxarray/rasterio) -- actual HTTP
    # transfer happens once you touch .values / .load() / .compute().
    first_pass = passes[0]
    print(f'\nReading {first_pass.id} ...')
    vv_da, vh_da = source.read_vv_vh(first_pass)
    print(f'VV: shape={vv_da.shape}, dtype={vv_da.dtype}, crs={vv_da.rio.crs}')
    print(f'VH: shape={vh_da.shape}, dtype={vh_da.dtype}, crs={vh_da.rio.crs}')
