# autofloods/sources/__init__.py

"""
STAC data source implementations for autofloods.

Implemented:

- MPCSource: Microsoft Planetary Computer (Sentinel-1 RTC via Azure Blob
  Storage, West Europe). The original, longest-tested backend.
- OPERASource: NASA OPERA RTC-S1 via ASF/CMR (AWS us-west-2). Same RTC
  product type as MPCSource; measured substantially faster per-band reads
  from this project's compute clusters. Requires a NASA Earthdata Login
  ~/.netrc entry.

Extending with a new source
----------------------------

- Same access pattern, different names/IDs (e.g. a new MPC collection
  version, or another Azure/AWS STAC catalog with a similar one-item-
  per-band shape and auth model): instantiate MPCSource/OPERASource with
  different constructor args rather than writing a new class.
- Genuinely different mechanics (auth flow, search API, asset-to-scene
  mapping): subclass STACSource directly; see OPERASource for an example
  whose read_vv_vh() does real mosaicking work.

Not implemented: Earth Search (Element84) and Copernicus Data Space
Ecosystem (CDSE) were evaluated and ruled out -- both serve raw GRD
rather than RTC and would add a terrain-correction step; see the
project's design notes/history for the full comparison.
"""

from .base import STACSource
from .mpc import MPCSource
from .opera import OPERASource

__all__ = ["STACSource", "MPCSource", "OPERASource"]
