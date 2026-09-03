# Changelog

## 0.1.0a9

**Critical fixes -- upgrade if you're on 0.1.0a8 or earlier:**

- **`MPCSource.authenticate()` was broken** (the *default* source used
  by `flood_mapper` when no `source=` is given): it passed a `timeout=`
  keyword argument to `pystac_client.Client.open()` that the pinned
  `pystac-client==0.6.1` doesn't support, raising `TypeError` on every
  use since `0.1.0a1`. This is the same class of bug fixed for
  `OPERASource` in `0.1.0a8`, found immediately once a real
  `authenticate()` test was written for `MPCSource` too. If you never
  passed `source=OPERASource()` explicitly, you were hitting this one.
- Re-verified the `0.1.0a8` `OPERASource.authenticate()` fix is still
  correct and now has direct regression-test coverage (see below), so
  it can't silently regress again.

**Also in this release:**

- Added 85 new unit tests across `sources/`, `utils/`, `preprocessing/`,
  `postprocessing/`, `mapfloods/`, and `authenticate/` (all previously
  untested; only `detectors/` had coverage before). All network calls
  mocked, no live credentials required. This is what caught the
  `MPCSource` bug above.
- Fixed Southern Hemisphere support: UTM zone resolution
  (`utils.zone_to_epsg`) previously hardcoded the Northern Hemisphere
  EPSG prefix regardless of an AOI's actual location, silently
  producing the wrong CRS for any Southern-Hemisphere tile.
  `generate_grid()` no longer refuses Southern-Hemisphere AOIs.
- Added Python 3.11 support (`requires-python` now `>=3.9,<3.12`) --
  root cause of the earlier 3.11 install failures was an internal
  `dask` bug, not the numpy/pandas/matplotlib/scikit-learn pins;
  fixed by bumping `dask` to `2024.4.1`. CI now tests 3.9, 3.10, and
  3.11 on every push/PR.
- `scripts/verification/`'s internal scripts and configs no longer
  hardcode a personal machine path; they resolve the repo root
  automatically instead.

## 0.1.0a8

- **Fixed a critical bug**: `OPERASource.authenticate()` passed a
  `timeout=` keyword argument to `pystac_client.Client.open()` that the
  pinned `pystac-client==0.6.1` doesn't support, raising `TypeError` on
  every `OPERASource` use -- including the documented Getting Started
  quickstart. Present since 0.1.0a1; fixed by dropping the unsupported
  kwarg.

## 0.1.0a7

- Corrected the author contact email in package metadata.

## 0.1.0a6

- Added automatic grid generation: `autofloods.grid.generate_grid()`
  builds a tiling grid for an AOI when you don't already have one, in
  two modes -- `mode='mgrs'` (MGRS 100km-aligned tiles, the default for
  `OPERASource`) and `mode='utm_fishnet'` (configurable fixed-size
  fishnet). `flood_mapper` accepts `aoi=` as an additive alternative to
  a pre-made `grid_shapefile=`, generating the grid on the fly.
- Added CI: a GitHub Actions test matrix (Python 3.9/3.10) on every
  push/PR to `main`, and a release workflow that publishes to PyPI via
  Trusted Publishing (OIDC) on every published GitHub Release.

## 0.1.0

Initial public release.

- Pluggable SAR-based flood mapping pipeline: Sentinel-1 VV/VH Z-score
  anomaly detection against a dry-season baseline (`ZScoreDetector`), plus
  a baseline-free Otsu alternative (`OtsuDetector`).
- Two interchangeable data source backends implementing a common
  `STACSource` interface: `MPCSource` (Microsoft Planetary Computer) and
  `OPERASource` (NASA OPERA RTC-S1 via ASF/CMR).
- Tiled, resumable processing designed to scale to large areas.
- Applied and validated on a 2017-2025 Bihar (India) flood reprocess
  across both backends.
