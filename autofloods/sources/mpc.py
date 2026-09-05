# autofloods/sources/mpc.py

from __future__ import annotations

import logging
import os
from urllib.parse import urlsplit, urlunsplit

import geopandas as gpd
import pystac_client
from shapely.geometry import box

from .base import STACSource

logger = logging.getLogger(__name__)

# Buffer (meters, in the item's own native CRS) added around a
# reprojected AOI bbox before windowing a read -- see
# _windowed_bbox_for_item()'s docstring for why this is needed at all.
# 5km is generous relative to the sub-pixel rounding effect it guards
# against (a few tens of meters at most, per clip_xarray_using_id()'s
# own docstring), so this is a safety margin, not a tuned value.
_WINDOW_READ_BUFFER_M = 5000


class MPCSource(STACSource):
    """
    Microsoft Planetary Computer (MPC) implementation of STACSource.

    MPC's STAC search and asset signing both work anonymously. A
    subscription key is optional -- it only raises the request rate
    limit, it is never required for search or download. If no key is
    passed explicitly, the MPC_SUBSCRIPTION_KEY environment variable is
    used if set; otherwise access proceeds anonymously. A missing key
    never raises an error here, matching planetary_computer.sign_inplace's
    own default behavior.
    """

    def __init__(
        self,
        subscription_key: str | None = None,
        collection: str = "sentinel-1-rtc",
        vv_asset_key: str = "vv",
        vh_asset_key: str = "vh",
        dem_collection: str = "nasadem",
        dem_asset_key: str = "elevation",
    ):
        """
        Parameters
        ----------
        subscription_key : MPC API subscription key, or None for
                            anonymous access (falls back to the
                            MPC_SUBSCRIPTION_KEY env var, then anonymous).
        collection        : STAC collection ID for Sentinel-1 RTC search.
        vv_asset_key      : Asset dict key for the VV band on each item.
        vh_asset_key      : Asset dict key for the VH band on each item.
        dem_collection    : STAC collection ID for DEM search.
        dem_asset_key     : Asset dict key for the elevation band.

        All defaults match MPC's current collection/asset naming --
        override any of them to point this same class at a differently-
        named collection (e.g. a future MPC collection version) without
        needing a new subclass. A source with genuinely different search
        or auth mechanics still belongs in its own STACSource subclass.
        """
        self._subscription_key = subscription_key or os.environ.get("MPC_SUBSCRIPTION_KEY")
        self._collection = collection
        self._vv_asset_key = vv_asset_key
        self._vh_asset_key = vh_asset_key
        self._dem_collection = dem_collection
        self._dem_asset_key = dem_asset_key
        self._catalog = None

    def authenticate(self) -> None:
        import planetary_computer

        if self._subscription_key:
            planetary_computer.settings.set_subscription_key(self._subscription_key)
        else:
            logger.info(
                "No MPC subscription key set (MPC_SUBSCRIPTION_KEY not found); "
                "proceeding with anonymous access. This works fine at low "
                "volume, but a key raises your rate limit for large-scale runs."
            )

        # No timeout= here: pinned pystac-client==0.6.1's Client.open()
        # doesn't accept one (added in a later release) -- passing it
        # raised TypeError on every call. Same bug as OPERASource.authenticate()
        # (see its comment); found here by the same test-coverage push,
        # not caught by any release before 0.1.0a9. See CLAUDE.md's Future
        # To-Dos.
        self._catalog = pystac_client.Client.open(
            "https://planetarycomputer.microsoft.com/api/stac/v1",
            modifier=planetary_computer.sign_inplace,
        )

    def search_sentinel1(self, bbox, start_date, end_date):
        if self._catalog is None:
            self.authenticate()

        all_results = []
        date_range = f'{start_date.strftime("%Y-%m-%dT00:00:00Z")}/{end_date.strftime("%Y-%m-%dT23:59:59Z")}'

        results = self._catalog.search(
            collections=[self._collection],
            intersects=bbox,
            datetime=date_range,
        )

        for item in results.get_items():
            if (self._vh_asset_key in item.assets) and (self._vv_asset_key in item.assets):
                all_results.append(item)

        return all_results

    def vv_vh_hrefs(self, item):
        return item.assets[self._vv_asset_key].href, item.assets[self._vh_asset_key].href

    def read_vv_vh(self, item, overview_level=None, bbox=None):
        """
        Open and return (vv_dataarray, vh_dataarray) for item.

        Re-signs each href immediately before every read attempt (not once
        up front), since Azure SAS tokens are short-lived (~45 min) and a
        queued read can outlive a token that was fresh when this method was
        called.

        bbox, if given (minx, miny, maxx, maxy in EPSG:4326 -- see
        STACSource.read_vv_vh's docstring), is reprojected into item's
        own native CRS (from its STAC proj:code/proj:epsg properties --
        a Sentinel-1 scene's UTM zone is not necessarily the AOI/tile's
        own) and used to windowed-read only that portion of the asset,
        instead of the full scene. MPC's Sentinel-1 RTC assets are real
        COGs (STAC-declared "profile=cloud-optimized", confirmed via
        rasterio: internally tiled, 512x512 blocks, 6 overview levels),
        so this is a real network-transfer reduction, not just an
        in-memory one -- measured ~540x fewer bytes for a 5km AOI window
        vs. a full ~1.86GB scene (see tests/test_sources.py and this
        session's investigation notes). Falls back to a full,
        unwindowed read if item doesn't carry proj:code/proj:epsg --
        windowing is an optimization, not a correctness requirement.
        """
        from ..utils import open_rasterio_with_retry

        native_bbox = self._windowed_bbox_for_item(item, bbox) if bbox is not None else None

        # Re-sign before opening, rather than trusting the SAS token
        # vv_vh_hrefs() returns from search time -- that token was minted
        # once, when the item was searched/signed via sign_inplace, and
        # Azure SAS tokens are short-lived (~45min observed).
        #
        # Signing once here (before the retry loop) is NOT enough on its
        # own: under bounded worker concurrency (read_scenes()'s
        # ThreadPoolExecutor), a scene can sit queued and then take many
        # minutes to actually read (especially at native, non-overview
        # resolution), so a token that was fresh when this function was
        # entered can still expire mid-read, and open_rasterio_with_retry's
        # own backoff (up to 480s across 5 attempts) was retrying that same
        # now-expired token every time -- this crashed the first Bihar
        # batch run (jobs 1078191/1078192, Quebracho tiles 315-317) and, in
        # a slower variant (every attempt failing, never recovering, job
        # hitting its SLURM time limit before the resulting exception could
        # even propagate through the executor), a later dual-backend
        # verification run. Passing a callable instead of a plain string
        # makes open_rasterio_with_retry call self.sign() again before
        # EVERY attempt, so a token that expires mid-retry-loop is simply
        # replaced by a fresh one on the next attempt rather than retried.
        vv_href, vh_href = self.vv_vh_hrefs(item)
        vv_ds = open_rasterio_with_retry(
            lambda: self.sign(vv_href), overview_level=overview_level, masked=True, bbox=native_bbox,
        )
        vh_ds = open_rasterio_with_retry(
            lambda: self.sign(vh_href), overview_level=overview_level, masked=True, bbox=native_bbox,
        )
        return vv_ds, vh_ds

    @staticmethod
    def _windowed_bbox_for_item(item, bbox_4326):
        """
        Reproject bbox_4326 (minx, miny, maxx, maxy in EPSG:4326) into
        item's own native CRS, buffered by _WINDOW_READ_BUFFER_M meters.

        The buffer matters: clip_xarray_using_id()'s later reproject+
        interp step already extrapolates a little past the raw
        reprojected extent to cover a sub-pixel rounding gap at the
        tile edge (see its own docstring) -- a full-scene read always
        had a whole scene's worth of margin for that to draw on. A
        tightly-windowed read has none by default, so this buffer
        preserves the same safety margin instead of risking a new
        edge-of-window NaN gap that a full read never produced.

        Returns None (falls back to a full, unwindowed read) if item
        doesn't carry proj:code/proj:epsg.
        """
        item_crs = item.properties.get('proj:code')
        if item_crs is None:
            epsg = item.properties.get('proj:epsg')
            item_crs = f'EPSG:{epsg}' if epsg is not None else None
        if item_crs is None:
            return None

        minx, miny, maxx, maxy = bbox_4326
        native_bounds = gpd.GeoSeries([box(minx, miny, maxx, maxy)], crs='EPSG:4326').to_crs(item_crs).iloc[0].bounds
        minx, miny, maxx, maxy = native_bounds
        return (
            minx - _WINDOW_READ_BUFFER_M, miny - _WINDOW_READ_BUFFER_M,
            maxx + _WINDOW_READ_BUFFER_M, maxy + _WINDOW_READ_BUFFER_M,
        )

    def search_dem(self, bbox):
        if self._catalog is None:
            self.authenticate()

        results = self._catalog.search(
            collections=[self._dem_collection],
            intersects=bbox,
        )

        return [item for item in results.get_items() if self._dem_asset_key in item.assets]

    def dem_href(self, item):
        return item.assets[self._dem_asset_key].href

    def sign(self, href):
        """
        Return a freshly-signed, fetchable URL for href.

        Strips any existing signature query string from href first, so a
        previously-signed (and possibly expired) URL is always re-signed
        rather than returned unchanged.
        """
        import planetary_computer

        # planetary_computer.sign_url() short-circuits and returns href
        # UNCHANGED if it already has st/se/sp query params -- it only
        # checks whether the href *looks* signed, never whether that
        # existing signature has actually expired. Our hrefs are already
        # signed once at search time (sign_inplace), so calling sign() on
        # them as-is was a silent no-op that never refreshed an expired
        # token -- this is why the first re-signing fix (read_vv_vh
        # calling self.sign() on the as-returned href) did not actually
        # work: it kept returning the same stale token past its ~45min
        # expiry. Stripping the existing query string first forces
        # sign_url() to treat it as unsigned and fetch a genuinely fresh
        # token from planetary_computer's own (correctly TTL-aware)
        # per-container token cache.
        parts = urlsplit(href)
        unsigned_href = urlunsplit((parts.scheme, parts.netloc, parts.path, '', ''))
        return planetary_computer.sign(unsigned_href)
