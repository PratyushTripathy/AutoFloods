# autofloods/sources/mpc.py

import logging
import os
from urllib.parse import urlsplit, urlunsplit

import pystac_client

from .base import STACSource

logger = logging.getLogger(__name__)


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

        self._catalog = pystac_client.Client.open(
            "https://planetarycomputer.microsoft.com/api/stac/v1",
            modifier=planetary_computer.sign_inplace,
            # bound the STAC API's own requests-level HTTP calls (connect
            # timeout, read timeout) so a stalled search request fails
            # loudly instead of hanging indefinitely; this is separate
            # from the GDAL-level timeout used for actual raster reads
            # (see autofloods.utils.open_rasterio_with_retry), since a
            # VSI curl hang wouldn't be caught by this alone.
            timeout=(15, 30),
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

    def read_vv_vh(self, item, overview_level=None):
        from ..utils import open_rasterio_with_retry

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
        vv_ds = open_rasterio_with_retry(lambda: self.sign(vv_href), overview_level=overview_level, masked=True)
        vh_ds = open_rasterio_with_retry(lambda: self.sign(vh_href), overview_level=overview_level, masked=True)
        return vv_ds, vh_ds

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
