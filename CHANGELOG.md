# Changelog

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
