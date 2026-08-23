"""
Shared boundary-file resolution for scripts/figures/fig_grid.py and
fig_bihar_floods.py. Not part of the autofloods package.

resources/boundaries/{india,bihar}_outline.gpkg are NOT committed to this
repo (they're derived from a private internal project's data, and we don't
vendor third-party boundary data into an open-source package). Get your
own copy from a public source, e.g. GADM (https://gadm.org/download_country.html,
country = India, level 1 = states) or Natural Earth's admin-1 states dataset,
then extract India's outline (dissolved, whole country) and Bihar's outline
(dissolved, NAME_1 == 'Bihar' or equivalent field) as EPSG:4326 GeoPackages,
and place them at the paths below (or point AUTOFLOODS_BOUNDARY_DIR at
wherever you put them). Example, starting from a GADM level-1 file:

    import geopandas as gpd
    states = gpd.read_file('gadm41_IND_1.json')
    states.dissolve()[['geometry']].to_file('india_outline.gpkg', driver='GPKG')
    bihar = states[states['NAME_1'] == 'Bihar'].dissolve()[['geometry']]
    bihar.to_file('bihar_outline.gpkg', driver='GPKG')
"""
import os

BOUNDARY_DIR = os.environ.get(
    'AUTOFLOODS_BOUNDARY_DIR',
    '/home/emlab/projects/current-projects/edge-autofloods/AutoFloods/resources/boundaries',
)
INDIA_PATH = os.path.join(BOUNDARY_DIR, 'india_outline.gpkg')
BIHAR_PATH = os.path.join(BOUNDARY_DIR, 'bihar_outline.gpkg')


def require_boundaries():
    """Raise a clear, actionable error (not a cryptic fiona/pyogrio
    traceback) if the boundary files aren't where expected."""
    missing = [p for p in (INDIA_PATH, BIHAR_PATH) if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(
            "Boundary file(s) not found:\n  " + "\n  ".join(missing) +
            "\n\nThese aren't committed to the repo (see this module's "
            "docstring for why and how to get your own copy -- short "
            "version: get India admin-1 boundaries from GADM or Natural "
            "Earth, dissolve to a single India outline and a single Bihar "
            "outline, save as EPSG:4326 GeoPackages at the paths above). "
            f"Current search directory: {BOUNDARY_DIR} "
            "(override with the AUTOFLOODS_BOUNDARY_DIR env var)."
        )
