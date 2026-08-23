# autofloods/sources/__init__.py

"""
STAC data source implementations for autofloods.

Implemented:

- MPCSource (Microsoft Planetary Computer): the original, longest-tested
  backend. Assets served from Azure Blob Storage, West Europe region.
- OPERASource (NASA OPERA RTC-S1, via ASF/CMR): same RTC product type as
  MPCSource, hosted in the US (AWS us-west-2). Measured ~10x faster
  end-to-end per-band read than MPCSource from this project's compute
  clusters (both GRIT and Quebracho see slow, roughly 3-6 MB/s single-
  connection throughput to MPC's West Europe storage; OPERA's US-hosted
  storage measured 13-56 MB/s in the same tests). Requires a NASA
  Earthdata Login ~/.netrc entry. Reads via download-to-local-file-then-
  open rather than GDAL's /vsicurl/ streaming (see read_vv_vh's
  docstring) -- soak-tested at real production volume (102 real
  scenes/tile, 0 failures) after /vsicurl/ streaming was found to fail
  under sustained load on the MPC side.

Extending with a new source
----------------------------

Two ways to add a source, depending on how different it actually is from
an existing one:

1. Same access pattern, different names/IDs -- e.g. a future MPC
   collection version, or another Azure/AWS-hosted STAC catalog with the
   same one-item-per-band-per-scene (or per-burst) shape and a similar
   auth model. Just instantiate the existing MPCSource/OPERASource with
   different constructor args (collection, vv_asset_key, vh_asset_key,
   etc. -- see each class's __init__ docstring) rather than writing a new
   class. This is the common case and costs nothing to maintain.

2. Genuinely different mechanics -- different auth flow, different
   search API, different asset-to-scene mapping (e.g. OPERA's per-burst
   mosaicking vs MPC's one-file-per-band). Subclass STACSource directly
   and implement its abstract methods; see OPERASource for a worked
   example of a source whose read_vv_vh() has to do real work beyond
   "open this href."

Future work, not yet implemented:

- Earth Search (Element84): ruled out, not just unimplemented. It has no
  public Sentinel-1 RTC-equivalent collection (only raw GRD, which would
  need its own terrain-correction step), the underlying bucket
  (sentinel-s1-l1c) is in AWS eu-central-1 -- the same regional-distance
  problem OPERASource was built to avoid -- and it's requester-pays,
  adding a real per-run cloud cost with no offsetting benefit.
- Copernicus Data Space Ecosystem (CDSE): requires OAuth2 token-based
  authentication, distinct from MPC's optional-static-key model. Also
  serves raw GRD, not RTC, and is EU-hosted like MPC. A CDSESource would
  need a token-refresh concern inside authenticate()/sign() and a
  terrain-correction step; lower priority than OPERASource given both
  downsides are already solved by OPERA.
"""

from .base import STACSource
from .mpc import MPCSource
from .opera import OPERASource

__all__ = ["STACSource", "MPCSource", "OPERASource"]
