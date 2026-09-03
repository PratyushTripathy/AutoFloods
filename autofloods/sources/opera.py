# autofloods/sources/opera.py

import logging
import os
import shutil
import subprocess
import tempfile
import time

import pystac_client
import requests
from shapely.geometry import box, mapping, shape
from shapely.ops import unary_union

from .base import STACSource
from .mpc import MPCSource

logger = logging.getLogger(__name__)

CMR_ASF_STAC_URL = "https://cmr.earthdata.nasa.gov/stac/ASF"
OPERA_RTC_S1_COLLECTION = "OPERA_L2_RTC-S1_V1_1"


class OperaPass:
    """
    A group of same-day OPERA RTC-S1 burst items covering an AOI that no
    single burst covers alone. Stands in for a pystac.Item wherever
    autofloods only needs `.id` as a dict key; OPERASource.read_vv_vh()
    mosaics `.bursts` into one dataset per band.

    Grouped by calendar date only, not by track/subswath, so a date's
    full AOI coverage lands in one composite image even when it spans
    multiple tracks -- a deliberate simplification that can introduce
    backscatter seams from differing incidence angles across tracks.
    """

    def __init__(self, pass_id, bursts):
        self.id = pass_id
        self.bursts = bursts
        # A GeoJSON-like footprint (bounding box of all bursts' own
        # geometries), needed because utils.s1item_footprint() /
        # seggregate_sentinel_search() -- used by flood_mapper.get_s1_items()
        # to determine which AOI(s) a scene intersects -- expect every
        # search result to have a `.geometry` attribute, same shape as a
        # real pystac.Item's. OperaPass is a synthetic grouping with no
        # geometry of its own otherwise. A bbox (not the exact multi-burst
        # outline) is a deliberate simplification, consistent with how AOI
        # intersection is already approximated elsewhere in this codebase
        # (see utils.gpd_to_json); any burst outside the exact AOI polygon
        # still just contributes a few extra no-op pixels beyond the tile
        # edge in read_vv_vh()'s mosaic, not lost data.
        self.geometry = mapping(box(*unary_union([shape(b.geometry) for b in bursts]).bounds))


class OPERASource(STACSource):
    """
    NASA OPERA RTC-S1 implementation of STACSource (search via NASA CMR's
    STAC API for the ASF DAAC, assets served from asf.alaska.edu / AWS
    us-west-2). Same RTC (radiometrically terrain corrected) product type
    as MPCSource -- no reprocessing gap -- but hosted in the US rather
    than MPC's Azure West Europe storage.

    Auth: NASA Earthdata Login. Asset hrefs returned by CMR
    (datapool.asf.alaska.edu) require resolving a redirect chain
    (datapool -> cumulus.asf.alaska.edu -> urs.earthdata.nasa.gov OAuth
    login -> a time-limited CloudFront/S3-signed URL) before they can be
    opened by GDAL. This class does that resolution once per href inside
    vv_vh_hrefs()/sign(), so callers (autofloods.preprocessing) get back
    a plain, ready-to-fetch URL exactly like MPCSource does.

    Requires a working ~/.netrc entry for urs.earthdata.nasa.gov (the
    standard way NASA Earthdata scripted access is set up) -- `requests`
    applies it automatically on the redirect to urs.earthdata.nasa.gov as
    long as no explicit `auth` is passed and the session's `trust_env` is
    left at its default (True).

    DEM search/read is delegated to an internal MPCSource -- OPERA/ASF
    does not host NASADEM, and DEM download is a small, once-per-tile
    cost unrelated to the per-scene S1 transfer bottleneck this source
    exists to fix.
    """

    def __init__(self, collection=OPERA_RTC_S1_COLLECTION, vv_asset_key="0_VV", vh_asset_key="0_VH"):
        """
        Parameters
        ----------
        collection    : STAC collection ID to search on the CMR ASF
                        endpoint. Defaults to the current OPERA RTC-S1
                        collection; override if NASA publishes a new
                        collection version with a different ID.
        vv_asset_key  : Asset dict key for the VV band on each burst item.
        vh_asset_key  : Asset dict key for the VH band on each burst item.
                        Both default to OPERA's current naming ("0_VV" /
                        "0_VH"); override if a future product version (or
                        a differently-structured S3/CMR source reusing
                        this class) names its assets differently, without
                        needing a new subclass.
        """
        self._collection = collection
        self._vv_asset_key = vv_asset_key
        self._vh_asset_key = vh_asset_key
        self._catalog = None
        self._session = None
        self._dem_source = MPCSource()

    def authenticate(self) -> None:
        # No timeout= here: pinned pystac-client==0.6.1's Client.open()
        # doesn't accept one (added in a later release) -- passing it
        # raised TypeError on every call, undetected for 3+ releases
        # since sources/ had zero test coverage. See CLAUDE.md's Future
        # To-Dos.
        self._catalog = pystac_client.Client.open(CMR_ASF_STAC_URL)

        self._session = requests.Session()
        netrc_path = os.path.expanduser("~/.netrc")
        if not os.path.exists(netrc_path):
            logger.warning(
                "No ~/.netrc found; OPERASource requires a NASA Earthdata "
                "Login entry for urs.earthdata.nasa.gov to resolve asset "
                "downloads. Requests to datapool.asf.alaska.edu will 401/403 "
                "without one."
            )

    def search_sentinel1(self, bbox, start_date, end_date):
        """
        Return one OperaPass per calendar date with burst coverage of bbox
        within [start_date, end_date] -- not one pystac.Item per scene as
        the base contract's literal type suggests; OperaPass duck-types as
        an Item via `.id` and `.geometry`. Reprocessed granules are deduped,
        keeping only the most recently processed version of each
        (tile, acquisition_time) pair.
        """
        if self._catalog is None:
            self.authenticate()

        date_range = f'{start_date.strftime("%Y-%m-%dT00:00:00Z")}/{end_date.strftime("%Y-%m-%dT23:59:59Z")}'
        results = self._catalog.search(
            collections=[self._collection],
            intersects=bbox,
            datetime=date_range,
        )

        candidates = [
            item for item in results.items()
            if (self._vv_asset_key in item.assets) and (self._vh_asset_key in item.assets)
        ]

        # OPERA reprocesses granules; the same tile/acquisition can appear
        # more than once with a different processing timestamp (the 6th
        # underscore-delimited token in the item id). Keep only the most
        # recently processed version of each (tile, acquisition_time) pair,
        # otherwise the same scene is double-counted in baseline/detection.
        latest_by_acquisition = {}
        for item in candidates:
            parts = item.id.split("_")
            acquisition_key = (parts[3], parts[4]) if len(parts) > 4 else (item.id,)
            processing_time = parts[5] if len(parts) > 5 else ""
            existing = latest_by_acquisition.get(acquisition_key)
            if existing is None or processing_time > existing[0]:
                latest_by_acquisition[acquisition_key] = (processing_time, item)

        deduped = [item for _, item in latest_by_acquisition.values()]

        # A single OPERA burst (~85x33km) is much smaller than one AOI
        # tile (~111x111km) -- MPC's one-scene-covers-the-tile assumption
        # doesn't hold here. Group ALL bursts covering the AOI on the same
        # calendar date (regardless of track/subswath) into one OperaPass;
        # read_vv_vh() mosaics each pass's bursts into a single per-date
        # dataset covering the AOI -- see OperaPass's docstring for the
        # tradeoff this implies when a date is covered by multiple tracks.
        passes = {}
        for item in deduped:
            parts = item.id.split("_")
            acquisition_date = parts[4][:8]
            passes.setdefault(acquisition_date, []).append(item)

        return [
            OperaPass(
                pass_id=f"OPERA_PASS_{date}",
                bursts=sorted(bursts, key=lambda it: it.id),
            )
            for date, bursts in passes.items()
        ]

    def vv_vh_hrefs(self, item):
        """
        Not supported. OPERASource distributes AOI coverage across multiple
        burst files per pass, so no single href pair represents a full
        scene -- use read_vv_vh() instead. Always raises NotImplementedError.
        """
        raise NotImplementedError(
            "OPERASource distributes AOI coverage across multiple burst "
            "files per pass; a single href pair can't represent a full "
            "scene here. Use read_vv_vh(item, overview_level) instead, "
            "which mosaics the pass's bursts via a GDAL VRT before "
            "returning DataArrays."
        )

    def read_vv_vh(self, item, overview_level=None):
        """
        Download item's bursts to local disk, mosaic each band into a VRT,
        and return (vv_dataarray, vh_dataarray) covering the pass's full
        footprint. overview_level is accepted for interface compatibility
        but ignored -- OPERA RTC-S1 GeoTIFFs ship without an internal
        overview pyramid.
        """
        from ..utils import open_rasterio_with_retry
        import shutil

        if overview_level is not None:
            logger.info(
                "OPERASource.read_vv_vh: ignoring overview_level=%s -- OPERA "
                "RTC-S1 GeoTIFFs ship without an internal overview pyramid "
                "(unlike MPC's COGs), and native 30m is already fast enough "
                "not to need one.",
                overview_level,
            )

        # Download each burst to local disk first, then mosaic/open from
        # local files -- NOT GDAL /vsicurl/ streaming. This mirrors the
        # pattern validated by the sister edge-india-crop-mapping project
        # (see its satellite-download-aws/09_Download_UID_S1.py) and by
        # this project's own smoke test (46/46 real burst downloads, 0
        # errors), both against this same ASF/CloudFront endpoint.
        # /vsicurl/ streaming was never proven at scale here; downloading
        # to a local temp file before opening avoids GDAL's HTTP-driver
        # behavior (open-time directory probes, range-request patterns)
        # entirely -- only a plain sequential requests.get() touches the
        # network.
        tmp_dir = tempfile.mkdtemp(prefix="opera_dl_")
        try:
            vv_paths = [self._download_burst(b.assets[self._vv_asset_key].href, tmp_dir) for b in item.bursts]
            vh_paths = [self._download_burst(b.assets[self._vh_asset_key].href, tmp_dir) for b in item.bursts]
            vv_vrt = self._build_vrt(vv_paths)
            vh_vrt = self._build_vrt(vh_paths)
            # open_rasterio_with_retry() already calls .load() internally,
            # so the returned DataArrays are fully in memory before we
            # clean up the temp dir the VRT points at.
            vv_ds = open_rasterio_with_retry(vv_vrt, overview_level=None, masked=True)
            vh_ds = open_rasterio_with_retry(vh_vrt, overview_level=None, masked=True)
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)
        return vv_ds, vh_ds

    def _download_burst(self, href, tmp_dir, max_attempts=3, backoff_seconds=10):
        """
        Download one burst asset to a local file under tmp_dir via a plain
        sequential GET (not GDAL /vsicurl/ streaming), retrying up to
        max_attempts times on request failures. self.sign() resolves auth
        before the download, so no further auth is needed here.
        """
        signed_url = self.sign(href)
        dest = os.path.join(tmp_dir, href.rsplit("/", 1)[-1])

        last_exc = None
        for attempt in range(1, max_attempts + 1):
            try:
                with requests.get(signed_url, stream=True, timeout=120) as r:
                    r.raise_for_status()
                    with open(dest, "wb") as f:
                        for chunk in r.iter_content(chunk_size=4 * 1024 * 1024):
                            f.write(chunk)
                return dest
            except requests.RequestException as exc:
                last_exc = exc
                if attempt < max_attempts:
                    logger.warning(
                        "Attempt %d/%d to download %s failed (%r); retrying in %ds.",
                        attempt, max_attempts, href, exc, backoff_seconds,
                    )
                    time.sleep(backoff_seconds)
        raise last_exc

    def _build_vrt(self, paths):
        """Build a VRT mosaic from local file paths (no /vsicurl/)."""
        if shutil.which("gdalbuildvrt") is None:
            raise RuntimeError(
                "gdalbuildvrt not found -- install GDAL command-line tools "
                "(e.g. `apt-get install gdal-bin` on Debian/Ubuntu/Colab, "
                "`conda install -c conda-forge gdal`, or see "
                "https://gdal.org/download.html for other platforms). "
                "This is a system dependency separate from the rasterio "
                "Python bindings pip installs."
            )

        vrt_path = tempfile.NamedTemporaryFile(suffix=".vrt", delete=False).name

        subprocess.run(
            ["gdalbuildvrt", vrt_path] + paths,
            check=True,
            capture_output=True,
            text=True,
        )
        return vrt_path

    def search_dem(self, bbox):
        return self._dem_source.search_dem(bbox)

    def dem_href(self, item):
        return self._dem_source.dem_href(item)

    def sign(self, href, max_attempts=3, backoff_seconds=5):
        """
        Resolve datapool.asf.alaska.edu's auth-redirect chain to a
        fetchable, time-limited signed URL. A 1-byte range request is used
        so resolving the redirect doesn't pull the asset body twice.

        The multi-hop OAuth redirect (datapool -> cumulus ->
        urs.earthdata.nasa.gov -> signed CloudFront URL) occasionally
        403s on the first touch of a given asset in a session even with
        valid credentials; a short retry clears this without masking a
        real, persistent auth failure (which still raises after
        max_attempts).
        """
        if self._session is None:
            self.authenticate()

        last_exc = None
        for attempt in range(1, max_attempts + 1):
            response = self._session.get(
                href,
                headers={"Range": "bytes=0-0"},
                stream=True,
                allow_redirects=True,
                timeout=(15, 30),
            )
            response.close()
            try:
                response.raise_for_status()
                return response.url
            except requests.exceptions.HTTPError as exc:
                last_exc = exc
                if attempt < max_attempts:
                    logger.warning(
                        "Attempt %d/%d to sign %s failed (%r); retrying in %ds.",
                        attempt, max_attempts, href, exc, backoff_seconds,
                    )
                    time.sleep(backoff_seconds)
        raise last_exc
