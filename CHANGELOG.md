# Changelog

## 0.1.0a14

Usability improvement: one-line progress feedback per pipeline step,
configurable via standard Python logging.

- **`flood_mapper`'s pipeline methods now each log a one-line INFO
  status summary** on completion (`get_dry_dates`, `generate_dry_date_ranges`,
  `get_s1_items`, `read_scenes`, `generate_mean_std_by_aoi`,
  `prepare_slope`, `prepare_wet_scenes`, `map_floods`,
  `merge_floods_by_date`, `generate_number_of_scenes`, `monthly_sum`) --
  e.g. `Found 42 dry-season scene(s)`, `Flood maps generated for 1
  AOI(s), 12 scene(s): 8431 high-confidence flooded pixels`. Uses a
  real, named `logging.getLogger('autofloods')` logger with a default
  handler so this prints out of the box in a notebook/script with zero
  setup, but is still fully standard logging -- silence it with
  `logging.getLogger('autofloods').setLevel(logging.WARNING)`, redirect
  it by replacing `.handlers`, or mute it entirely with `.disabled =
  True`. No `logging.basicConfig()` call, so it never touches global
  logging state.
- Added 12 new tests confirming each message fires (`tests/
  test_pipeline_logging.py`).

## 0.1.0a13

**Fixes a regression introduced in 0.1.0a12** (the Welford's-algorithm
dry-season baseline rewrite) -- **upgrade immediately if you're on
0.1.0a12**, which crashes on the very first dry scene of any tile,
every time, for both `OPERASource` and `MPCSource`.

- **`generate_mean_std_by_aoi()` raised `ValueError: Dimension band
  already exists`**, from `compute_dry_baseline_stats()`. Root cause:
  every real VV/VH scene (opened via `open_rasterio_with_retry()`, used
  identically by both sources) carries a real, incidental leading
  `'band'` dimension of size 1 -- an artifact of reading a single-band
  GeoTIFF/VRT via rioxarray, not a "which scene" axis -- and this
  survives reprojection/regridding unchanged. 0.1.0a12's `grid_ref`
  construction called `.expand_dims(band=[0])`, which assumes no
  `'band'` dim exists yet; on real data, it already does. Every
  synthetic test fixture exercising this path used a `(y, x)`-only
  array that never had this dimension, so 0.1.0a12 shipped with all 139
  tests green despite the bug -- caught only once real Sentinel-1 data
  hit the new code path on Colab.
- Fixed by squeezing that incidental `'band'` dim off each scene
  immediately after alignment, before folding it into the Welford
  accumulators -- restoring the exact `(y, x)` shape the pre-0.1.0a12
  stack-then-reduce path always produced.
- **Closed the actual testing gap, not just the one bug**: the
  synthetic fixtures feeding this code path now carry a real `band=1`
  dimension, matching what `rioxarray.open_rasterio()` actually
  returns, instead of a simplified `(y, x)`-only shape. Confirmed these
  corrected fixtures do reproduce the exact original crash if the fix
  is reverted, before re-verifying the fix against them. Full test
  suite (139 tests) passing with real shapes exercised throughout.

## 0.1.0a12

**Fixes an out-of-memory crash a user hit on Google Colab** while fitting
the dry-season Z-score baseline on a tile with many dry scenes.

- **`generate_mean_std_by_aoi()`'s dry-season baseline fit no longer
  loads every dry-season scene into memory at once.** The old path
  (`stack_images()`, `xr.concat`, then `.mean()`/`.std()`) held every
  aligned scene for a tile simultaneously, then a second full copy in
  the concatenated stack -- peak memory scaled linearly with the number
  of dry scenes, and grew without bound as more scenes were added to a
  tile-year. Replaced with
  `autofloods.preprocessing.compute_dry_baseline_stats()`, which folds
  each scene into a running per-pixel mean/variance one at a time via
  Welford's online algorithm (chosen over a plain running-sum/
  sum-of-squares for numerical stability -- SAR backscatter's std is a
  small fraction of its mean, which is exactly where the naive formula's
  cancellation error is worst), through a bounded sliding-window thread
  pool so at most `max_workers` scenes are ever resident at once instead
  of all of them. Verified numerically identical (within floating-point
  tolerance) to the old stack-then-reduce result. Measured on a synthetic
  2000x2000px tile: peak memory stayed roughly flat (~370-450 MB) from
  20 to 120 dry scenes, versus the old path's ~1.0 GB to ~1.8 GB linear
  growth over the same range.
  `autofloods.detectors.FloodDetector.fit_baseline()`'s interface
  changed accordingly (now takes `{'mean', 'std'}` stat dicts rather
  than a pre-stacked array) -- affects custom `FloodDetector`
  subclasses, not typical usage via `flood_mapper`.

## 0.1.0a11

Two usability fixes, both surfaced by real first-time-user testing
(including on Google Colab):

- **Added `autofloods.authenticate.setup_earthdata_login()`**, the
  recommended way to set up NASA Earthdata Login credentials for
  `OPERASource`. Prompts for a username (`input()`) and password
  (`getpass.getpass()`, never echoed or logged) -- or accepts them
  directly as arguments for scripted/non-interactive use -- and
  writes/updates the `urs.earthdata.nasa.gov` entry in `~/.netrc`
  without disturbing any other entries already there. Sets the file to
  owner-only permissions (`chmod 600`, skipped on Windows). The manual
  `.netrc` setup instructions in the docs remain as an alternative for
  anyone who prefers editing the file directly.
- **`OPERASource` now fails with a clear, actionable error** if the
  `gdalbuildvrt` CLI tool isn't installed, instead of a raw
  `FileNotFoundError` from deep inside a `subprocess` call. `rasterio`'s
  pip wheel bundles its own GDAL library but doesn't expose the CLI
  binaries, so this is a real gap for a plain `pip install autofloods`
  user -- most visibly on Google Colab. The new error message points
  directly at the fix (`apt-get install gdal-bin` / `conda install -c
  conda-forge gdal`), and this system dependency is now documented
  prominently in the Installation section of the docs.

## 0.1.0a10

**Heads up if you're pinning dependencies**: this release moves core
dependencies to NumPy 2.x and pandas 2.x -- a bigger jump than usual.
If your own project pins `numpy<2` or `pandas<2` alongside
`autofloods`, check compatibility before upgrading.

- **Dropped Python 3.9 support.** `requires-python` is now
  `>=3.10,<3.14`. Forced by NumPy 2.x and scikit-image 0.25+ (both
  needed for Python 3.12/3.13 support below) themselves requiring
  Python `>=3.10` -- no single dependency set could span 3.9 through
  3.13. Python 3.9 also reaches its own upstream end-of-life in
  October 2025.
- **Added Python 3.12 and 3.13 support.** Required bumping 12
  dependencies together: `numpy` (1.23.2 -> 2.1.0), `pandas` (1.5.0 ->
  2.2.3), `matplotlib` (3.6.0 -> 3.9.2), `rasterio` (1.3.3 -> 1.3.11),
  `shapely` (2.0.1 -> 2.0.6), `fiona` (1.9.6 -> 1.10.1), `geopandas`
  (0.13.2 -> 0.14.4), `xarray` (2023.6.0 -> 2024.3.0), `xarray-spatial`
  (0.3.7 -> 0.4.0), `scikit-image` (0.19.3 -> 0.25.0), `scikit-learn`
  (1.1.3 -> 1.5.2), and `dask`'s extras (`[array]` -> `[array,dataframe]`).
  Full test suite (125 tests) verified passing on real, clean installs
  across Python 3.10, 3.11, 3.12, and 3.13.
- Python 3.14 was investigated and is explicitly **not** supported yet:
  `fiona` has not published `cp314` wheels for any release. Will
  revisit once it does.

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
