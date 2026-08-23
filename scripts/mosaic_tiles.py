"""
Mosaic each tile's final monthly flood-day-count raster (produced by
flood_mapper.monthly_sum() / autofloods.postprocessing.aggregate_monthly(),
one *_monthly.tif per tile under <tile_dir>/flood_raster/monthlyadded*/)
into a single raster covering all tiles under a given parent folder: build
a GDAL VRT mosaic, then render it to a DEFLATE-compressed GeoTIFF via
gdal_translate. Plain script outside the autofloods package -- not part
of the pipeline, just a convenience for using the tiled output as one
raster.

Tiles are each reprojected to their own local UTM zone by the pipeline
(see flood_mapper's cell_size/grid handling), so a state-wide run spans
multiple UTM zones. `gdalbuildvrt` requires one common CRS and silently
DROPS any input whose CRS doesn't match the first one it sees -- so tiles
are first reprojected (via `gdalwarp -of VRT`, nearest-neighbor since
these are integer flood-day counts, not continuous data) to one common
target CRS before mosaicking. Target CRS defaults to ESRI:54009 (World
Mollweide, equal-area) rather than picking one tile's own UTM zone as the
"true" one for every other tile -- avoids that arbitrary distortion bias
and matches this being a multi-UTM-zone mosaic, not a single-zone extension.
Override with --t_srs (e.g. a specific UTM zone, or another equal-area CRS).

Uses the GDAL command-line tools `gdalwarp`/`gdalbuildvrt`/`gdal_translate`
(not Python bindings), so GDAL's CLI must be on PATH. nodata (read from
the first source tile, e.g. 255) is passed EXPLICITLY to every step
(-srcnodata/-dstnodata/-vrtnodata/-a_nodata) rather than relying on each
tool to implicitly carry it over from the previous one -- letting it
propagate implicitly is exactly what silently produced huge float64
sentinel values (+/-1.7e308) in an earlier version of this script. Band
descriptions (e.g. "202407") are a similar exception: `gdalbuildvrt` drops
them when mosaicking, and this GDAL build's `gdal_edit` CLI has no
per-band -description option to restore them, so they're re-applied with
a short rasterio patch at the end (rasterio is already a project
dependency).

Usage:
    python scripts/mosaic_tiles.py /path/to/output/bihar_opera_30m/2024
    python scripts/mosaic_tiles.py /path/to/output/bihar_opera_30m/2024 --out mosaic_2024.tif
    python scripts/mosaic_tiles.py /path/to/output/bihar_opera_30m/2024 --t_srs EPSG:32645
"""
import argparse
import glob
import os
import subprocess
import sys

import rasterio


def find_tile_rasters(parent_dir):
    pattern = os.path.join(parent_dir, 'tile*', 'flood_raster', 'monthlyadded*', '*_monthly.tif')
    matches_by_tile = {}
    for path in sorted(glob.glob(pattern)):
        tile_dir = path.split(os.sep + 'flood_raster' + os.sep)[0]
        matches_by_tile.setdefault(tile_dir, []).append(path)

    rasters = []
    for tile_dir, paths in sorted(matches_by_tile.items()):
        if len(paths) > 1:
            print(f"WARNING: {tile_dir} has {len(paths)} monthly rasters, using the most recent: {paths[-1]}",
                  file=sys.stderr)
        rasters.append(sorted(paths, key=os.path.getmtime)[-1])
    return rasters


def get_epsg(raster_path):
    result = subprocess.run(['gdalsrsinfo', '-o', 'epsg', raster_path],
                             capture_output=True, text=True, check=True)
    return result.stdout.strip().splitlines()[0]  # e.g. "EPSG:32645"


def main():
    parser = argparse.ArgumentParser(
        description='Mosaic tile-level monthly flood rasters (spanning multiple UTM zones) into one VRT.')
    parser.add_argument('parent_dir', help='Folder containing tile<ID> subfolders (e.g. output/bihar_opera_30m/2024)')
    parser.add_argument('--out', default=None,
                         help='Output GeoTIFF filename (default: <parent_dir_basename>_mosaic.tif, written inside parent_dir)')
    parser.add_argument('--t_srs', default='ESRI:54009',
                         help='Common target CRS for all tiles (default: ESRI:54009, World Mollweide equal-area)')
    args = parser.parse_args()

    parent_dir = args.parent_dir.rstrip('/')
    rasters = find_tile_rasters(parent_dir)

    if not rasters:
        print(f"No tile*/flood_raster/monthlyadded*/*_monthly.tif found under {parent_dir}", file=sys.stderr)
        sys.exit(1)

    t_srs = args.t_srs
    with rasterio.open(rasters[0]) as src:
        nodata = src.nodata
    print(f"Found {len(rasters)} tile rasters under {parent_dir}. Common target CRS: {t_srs}, nodata: {nodata}")

    warp_dir = os.path.join(parent_dir, '.mosaic_warped_vrts')
    os.makedirs(warp_dir, exist_ok=True)

    warped_rasters = []
    for raster in rasters:
        tile_name = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(raster))))  # tile<ID>
        raster_epsg = get_epsg(raster)
        warped_vrt = os.path.join(warp_dir, f'{tile_name}.vrt')

        if raster_epsg == t_srs:
            # Already in the target CRS -- a plain VRT passthrough avoids an
            # unnecessary warp/resample of data that doesn't need reprojecting.
            cmd = ['gdalbuildvrt', '-srcnodata', str(nodata), '-vrtnodata', str(nodata), warped_vrt, raster]
        else:
            cmd = ['gdalwarp', '-of', 'VRT', '-t_srs', t_srs, '-r', 'near',
                   '-srcnodata', str(nodata), '-dstnodata', str(nodata), raster, warped_vrt]

        subprocess.run(cmd, check=True, capture_output=True, text=True)
        warped_rasters.append(warped_vrt)

    out_name = args.out or f"{os.path.basename(parent_dir)}_mosaic.tif"
    out_path = os.path.join(parent_dir, out_name)
    mosaic_vrt_path = os.path.join(parent_dir, f"{os.path.basename(parent_dir)}_mosaic.vrt")

    file_list_path = os.path.join(parent_dir, '.mosaic_input_files.txt')
    with open(file_list_path, 'w') as f:
        f.write('\n'.join(warped_rasters) + '\n')

    cmd = ['gdalbuildvrt', '-srcnodata', str(nodata), '-vrtnodata', str(nodata),
           '-input_file_list', file_list_path, mosaic_vrt_path]
    print('Running:', ' '.join(cmd))
    subprocess.run(cmd, check=True)
    os.remove(file_list_path)

    cmd = ['gdal_translate', '-of', 'GTiff', '-co', 'COMPRESS=DEFLATE', '-co', 'TILED=YES',
           '-a_nodata', str(nodata), mosaic_vrt_path, out_path]
    print('Running:', ' '.join(cmd))
    subprocess.run(cmd, check=True)

    # gdalbuildvrt drops band descriptions (e.g. "202407"); restore them
    # from the first source raster -- all tiles share the same wet-season
    # months, so any tile's band descriptions apply to every band here.
    with rasterio.open(rasters[0]) as src:
        band_descriptions = src.descriptions
    with rasterio.open(out_path, 'r+') as dst:
        dst.descriptions = band_descriptions

    print(f"Wrote {out_path}")


if __name__ == '__main__':
    main()
