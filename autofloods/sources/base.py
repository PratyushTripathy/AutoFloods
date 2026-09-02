# autofloods/sources/base.py

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import date

import pystac


class STACSource(ABC):
    """
    Contract a STAC-based Sentinel-1/DEM provider must satisfy so that
    autofloods.flood_mapper can search and retrieve imagery regardless of
    which catalog it's talking to.
    """

    @abstractmethod
    def authenticate(self) -> None:
        """
        Perform any auth/signing setup needed before search_sentinel1(),
        search_dem(), or sign() will work. Called once, lazily, on first
        use by the pipeline -- not at import time.
        """

    @abstractmethod
    def search_sentinel1(
        self, bbox: dict, start_date: date, end_date: date
    ) -> list[pystac.Item]:
        """
        Return STAC Items with both a VV and VH asset, intersecting bbox,
        within [start_date, end_date].
        """

    @abstractmethod
    def vv_vh_hrefs(self, item: pystac.Item) -> tuple[str, str]:
        """
        Return (vv_href, vh_href) for a search result item, resolving
        this catalog's actual asset key naming. Only meaningful for a
        source whose search results are already single-file-per-band
        (see read_vv_vh for the general case).
        """

    @abstractmethod
    def read_vv_vh(self, item, overview_level: int | None = None):
        """
        Return (vv_dataarray, vh_dataarray) covering item's full extent,
        in this source's native CRS/resolution, unreprojected.

        For a source whose search results already cover the requested
        AOI in one file per band (e.g. MPC), this is just opening
        vv_vh_hrefs(item). For a source that distributes AOI coverage
        across multiple co-registered files (e.g. a per-burst product),
        this is where mosaicking those files back into one dataset per
        band belongs -- callers (autofloods.preprocessing) treat the
        return value as one scene either way, regardless of how many
        underlying files it took to build it.
        """

    @abstractmethod
    def search_dem(self, bbox: dict) -> list[pystac.Item]:
        """Return STAC Items covering bbox for this source's DEM collection."""

    @abstractmethod
    def dem_href(self, item: pystac.Item) -> str:
        """Return the elevation asset href for a DEM search result item."""

    @abstractmethod
    def sign(self, href: str) -> str:
        """
        Return a fetchable URL for href, applying any catalog-specific
        signing. A fully public catalog can return href unchanged.
        """
