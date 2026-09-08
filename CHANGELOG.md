# Changelog

## 0.1.0a28

**Lowered the default concurrency for `max_workers`/`reproject_max_workers`
to 2 everywhere they're exposed**, replacing the old CPU-count-derived
default. Peak memory in the bounded read/reproject pipeline is bounded
by this concurrency window: a real sweep on a 49-scene tile-year
measured **14.35 GB** peak RSS at the old defaults (6, 7) versus
**8.42 GB** at (2, 2) -- roughly 2x wall-clock for that ~39% memory
reduction. A conservative default matters more than throughput for a
first run on an unfamiliar machine; raise both once you know how much
memory is actually available.

Affected: `flood_mapper.read_scenes()`, `prepare_wet_scenes()`,
`prepare_slope()`, `generate_mean_std_by_aoi()`;
`preprocessing.reproject_clip_stac()`, `stack_images()`,
`compute_dry_baseline_stats()`, `compute_dry_baseline_stats_from_paths()`;
`utils.download_nasadem()`. `utils.default_max_workers()` (the old
CPU-scaled resolver) is unchanged and still available for anyone who
wants that behavior explicitly.

A one-time INFO log fires the first time a `flood_mapper` instance
actually falls back to this default (never when the caller passes
either value explicitly): "Running with max_workers=2,
reproject_max_workers=2 for low memory use; on a machine with more RAM
available, raising these will speed up processing considerably." Fires
once per instance, not per scene or per tile.

**Also fixed**: `scripts/run_autofloods.py` (the config-driven runner)
and `scripts/verification/manuscript_stage_profile.py` both had their
own hardcoded `.get('max_workers', 6)`-style fallback, silently
overriding the new default before it could ever apply to a config that
simply omits the key -- the exact path production config-driven runs
use. Both now fall through to `None`, letting the new default resolve
and log as intended.

Full test suite: 233 passing (6 new: `TestResolveConcurrency` in
`test_pipeline_logging.py`, covering default resolution, explicit-value
passthrough with no log, one-log-per-instance across multiple calls,
partial-override still resolving+logging, and two end-to-end checks
through `read_scenes()`).

## 0.1.0a27

**Documentation correction: Microsoft Planetary Computer subscription
keys are no longer obtainable.** The developer portal that issued them
has been retired, so every mention of getting a free key to raise
MPC's request rate limit was stale and pointed users to a dead
process. `getting-started.qmd`, `index.qmd`, and `PYPI_README.md` now
say plainly that `MPCSource` needs no credentials at all, and note that
anonymous access carries lower rate limits and shorter-lived access
tokens than a key would have, which can cause failures on large
concurrent runs -- with no workaround via a key, since none can be
issued anymore.

**Removed the `MPCSource` subscription-key code path** to match:
`authenticate()` no longer reads the `MPC_SUBSCRIPTION_KEY` environment
variable or calls `planetary_computer.settings.set_subscription_key()`
-- both dead code now that no key can exist. The `subscription_key`
constructor parameter is retained as a deprecated no-op (passing a
non-`None` value emits a `DeprecationWarning` and is otherwise ignored)
so existing code that passes this argument doesn't break.

Also trimmed `getting-started.qmd`'s `OPERASource` setup section (cut
the paragraph explaining `setup_earthdata_login()`'s internals; reduced
the manual `~/.netrc` alternative to the entry format plus `chmod`) and
`index.qmd`'s `OPERASource` data-source bullet (cut the download-then-
open-vs-streaming implementation rationale, which is not something a
user needs to know to use the package). Reworded `index.qmd`'s Overview
paragraph on per-AOI tiled processing to describe it as the workflow's
design, not a workaround for a reliability limitation.

Committed the manuscript-work measurement scripts and configs
(`scripts/verification/manuscript_*.py`,
`scripts/configs/manuscript_*/`, `scripts/run_manuscript_profile.sbatch`)
as permanent, reproducible tooling -- they produced the real
before/after memory and wall-clock numbers cited for 0.1.0a26's
`read_scenes()`/`merge_floods_by_date()` fixes, so they should stay
runnable rather than be discarded as scratch work. Updated the same
`BASE = pathlib.Path(__file__).resolve().parents[2]` repo-relative path
convention already used by every other `scripts/verification/*.py`
script, replacing the hardcoded absolute path they were written with.

Full test suite: 227 passing (net -1 from 0.1.0a26: the 4 old
subscription-key tests in `TestMPCSourceAuthenticate` were replaced
with 3 new ones covering the deprecated-no-op behavior).

## 0.1.0a26

**Correctness fix: per-scene flood raster filenames could collide for
OPERA-sourced runs, causing `merge_floods_by_date()` to aggregate
repeated copies of one scene instead of distinct dates.**
`map_floods()` derived each scene's output filename from
`scene_id.split('_')[4:]` -- a positional token slice that assumed an
MPC-style compound scene ID. For an OPERA-style scene_id
(`OPERA_PASS_{YYYYMMDD}`, only 3 underscore-separated tokens), index 4
is past the end, so every scene in a run produced the exact same empty
suffix and wrote to (and overwrote) the exact same output file. Since
`merge_floods_by_date()`/`generate_number_of_scenes()` read these
per-scene rasters back from disk (unconditional export since 0.1.0a19's
streaming rewrite), an affected run's per-date merged stack and monthly
aggregate would have held N copies of whichever scene was written last,
not N distinct dates.

This is live on the default path for **any OPERA-sourced run on
0.1.0a21 through 0.1.0a25** (export became unconditional in that
range). It was NOT live before 0.1.0a19's streaming rewrite (raster
export was opt-in and merge worked from an in-memory dict, not
re-parsed files) and was NOT live for MPC-sourced runs (MPC's much
longer compound scene IDs happen to leave a distinct trailing token
sequence, by luck of ID shape, not because the slicing logic was
correct). Checked the published Bihar production record (2017-2025,
198 tile-years, all OPERA-sourced) directly against this: it predates
0.1.0a19 and used `export_raster=False` throughout, so per-scene
rasters were never written and `merge_floods_by_date()` worked from the
in-memory dict -- confirmed unaffected by exhaustively scanning the
entire production output tree for per-scene `floodextent_*.tif` files
(zero found anywhere, consistent with `export_raster=False`, not with
the collision ever firing).

Fixed by deriving the filename from the full scene_id, sanitized for
filesystem safety (`utils.sanitize_scene_id_for_filename`), instead of
a positional slice -- unique by construction for both OPERA's and MPC's
real ID shapes, with no per-source special-casing. A fourth,
independent occurrence of the same positional-slice bug was found and
fixed in `autofloods.visualize._flood_scene_raster_path()` (used by
`plot_scenes_and_floods()` to reload a scene's raster), which
recomputed the same broken formula rather than reading it from
`flood_dict`.

**Fixed the real memory bottleneck in the bulk scene-read step**
(`read_scenes()`, shared by both dry- and wet-season phases): it used
to hold every scene's raw VV+VH arrays in memory at once before
anything downstream consumed them, even though the reproject/accumulate
loops downstream of it were already bounded by earlier fixes. Real
profiling across 10 tile-years showed this peaking at 4.6-7.4GB
(`read_scenes(dry)`) and 12.7-20.5GB (`prepare_wet_scenes()`). Rewrote
it into a two-stage bounded pipeline (a read pool feeding a
reproject-and-cache pool, each independently sized via `max_workers`/
`reproject_max_workers`), writing each scene straight to a persistent
per-(AOI, scene) disk cache and releasing it before the next read is
submitted -- residency is now bounded by the worker-pool sizes, not by
scene count, and a crashed run resumes from its cache for free. An
earlier version of this rewrite had a real backpressure bug (reads
could race ahead of the slower reproject stage and pile up an unbounded
backlog, silently reproducing the same problem); caught via real
before/after peak-memory measurement, not assumed, and fixed with
explicit backlog capacity gating -- see the new
`TestReadScenesBackpressure` regression test.

**Fixed the transient 2x memory blowup in `merge_floods_by_date()`**:
`flood_data_3dstack_from_paths()` used to build a dict holding every
date's combined raster, then `np.stack()` a second, same-size 3D array
on top of it before the dict was freed. Rewrote it to write each date's
combined raster directly into a preallocated output array as it's
computed, so only one date's per-scene rasters plus the single
accumulating output are ever resident. Measured: this stage's peak
dropped from ~5.0GB to ~2.7GB on a real 49-scene tile-year, matching
the predicted ~2.40GB single-copy size closely.

**Renamed** `map_floods()`'s `rel_slope_thd` parameter to `slope_thd`
(deprecated alias kept, forwards with a `DeprecationWarning`) -- the
value is compared as a flat absolute per-pixel slope cutoff in degrees,
nothing normalized against a neighborhood/percentile/tile-wide
reference, so the `rel_` prefix was a leftover misnomer.

Removed the now-unused `scikit-learn` dependency (it was pinned solely
for the since-removed `smoothen_slope()`'s patch-extraction call).

Full test suite: 228 passing (16 new: the filename-collision
regression and its `merge_floods_by_date()` consumer check, 4
`sanitize_scene_id_for_filename()` unit tests, 4 new dry-season
streaming tests including the backpressure regression, 2 new
`flood_data_3dstack_from_paths()` correctness/memory-bound tests, plus
fixture updates across existing tests for the new read/reproject
concurrency contract).

## 0.1.0a25

Four fixes to `autofloods.visualize`, found via actual rendered
output/measurement, not assumed.

**Fixed a real memory issue in `plot_scenes_and_floods()`.**
Investigated first (per this codebase's established practice): reading
was already correct and one-scene-at-a-time (no upfront-list bug); a
real tracemalloc/RSS measurement showed the growth was matplotlib
itself retaining every panel's full-resolution array for the returned
`Figure`'s lifetime -- confirmed by measuring memory drop back to
baseline immediately after `plt.close(fig)`. At 1200x1200px synthetic
scenes (a scaled stand-in for real ~3500x3500px tiles), 30 scenes
peaked at **1358MB**; extrapolated to real tile resolution, ~11-19GB
for 30-50 scenes. Fixed by downsampling each panel to a new
`thumbnail_max_size=400` (default, longer side in pixels) BEFORE
rendering -- block-mean for the RGB composite (raw scene is transient
either way; only the size of what's retained by the Figure matters),
a decimated `rasterio` read straight from disk for the flood raster
(never materializes full resolution at all). Re-measured after the
fix: same 30-scene case now peaks at **187MB** -- a ~7.3x reduction,
matching the predicted 150-250MB range.

**Fixed the shared legend overlapping the last row of image panels**
in `plot_scenes_and_floods()` (confirmed via a real rendered
screenshot, not just "no error"). Root cause: the legend was a
floating `fig.legend()` with no space reserved for it by
`constrained_layout`. Fixed by drawing it in its own dedicated
`GridSpec` row instead (a real subplot slot, not a floating artist),
so its space is now genuinely reserved regardless of scene count --
verified both visually and with a new geometric test asserting the
legend's bounding box never extends above the lowest image panel's box.

**Fixed `plot_flood_map()`'s single-panel sizing and squeezed
colorbar.** A single month (or `month=` given directly) was scaled
down as if it were one cell of a larger grid (2.6in), with a
`fraction=0.03` colorbar sized for a wide multi-panel row -- both
illegibly small. Fixed with a fixed readable minimum figure size
(floor of 4.5in) regardless of panel count, and (see next item)
replaced the squeezed colorbar with a properly-spaced legend.

**Replaced `plot_flood_map()`'s continuous color scale with a discrete
4-class one**: **0**, **1-3**, **4-7**, **8+** flood-days, via a
`ListedColormap`+`BoundaryNorm` (same categorical pattern already used
for `plot_scenes_and_floods()`'s flood-classification panels, for
consistency) with a clearly labeled legend instead of a raw numeric
colorbar. Nodata (255) handling unchanged -- masked gray, excluded
from the class scale entirely.

Full test suite: 213 passing (8 new tests: legend-overlap geometry,
downsampling behavior for both the RGB composite and the flood-raster
decimated read, the 4-class bin-boundary logic, and the single-panel
sizing/discrete-colormap/legend-label checks for `plot_flood_map()`).

## 0.1.0a24

**Breaking change**: `grid.generate_grid()` no longer accepts a
`dry_months` parameter. It's now purely geometric -- geometry and
season-assignment are separate concerns, and baking a single uniform
value into grid generation also precluded the more realistic case of
per-tile variation. Every tile's `dry_date_col` (default `dry_month`)
now always comes back as the placeholder `"REQUIRED"`, matching its
existing behavior when `dry_months` was previously omitted -- just
unconditionally now, regardless of how many tiles are generated.

If your script calls `generate_grid(..., dry_months='04,05')`, drop
that argument -- passing it now raises `TypeError`. Set the column
yourself afterward instead:

```python
gdf = generate_grid(aoi, mode='mgrs')
gdf['dry_month'] = '04,05'  # same value for every tile
# or, for per-tile variation:
gdf.loc[gdf['zone'].str.startswith('43'), 'dry_month'] = '04,05'
gdf.loc[gdf['zone'].str.startswith('44'), 'dry_month'] = '05,06'
```

**`flood_mapper`'s own `grid_dry_months=` convenience parameter (the
`aoi=` on-the-fly grid generation path) is unaffected** -- it still
requires one value and stamps it across every tile exactly as before,
just implemented at that convenience layer (setting the grid's
`dry_month` column directly after `generate_grid()` returns) instead
of inside `generate_grid()` itself. Verified end-to-end, not just at
the unit level.

Also fixed a stale, unrelated doc claim in `examples.qmd`'s Grid
Generation section ("Only the Northern Hemisphere is currently
supported") -- wrong since Southern Hemisphere support was added
2026-09-03, just never updated in this doc until now.

Full test suite: 205 passing.

## 0.1.0a23

Two independent changes in this release.

**Added `postprocessing.smoothen_flood_raster(flood_array, kernel_size=3)`**,
a majority/mode filter (`skimage.filters.rank.majority`) for the
classified flood raster's discrete 0/1/2/3/NaN class data -- the
correct filter choice for categorical data, unlike a mean/Gaussian
filter, which would average class labels into meaningless values.
Removes isolated single-pixel speckle misclassifications while
preserving real class boundaries.

- NaN (data-gap) pixels are excluded from every neighbor's vote via
  `rank.majority()`'s `mask=` parameter -- verified directly (not just
  from its docs) that a masked-out pixel's own value never influences
  a neighbor's result. A gap pixel itself is resolved (filled in) via
  majority vote of its valid neighbors wherever it has at least one
  within `kernel_size`; only where its entire neighborhood is also gap
  does it stay NaN -- `skimage`'s own fallback-to-`0` for a
  fully-masked neighborhood is explicitly detected (via a second
  `rank.sum()` pass counting valid neighbors) and overridden back to
  NaN, since a silent `0` there would misread as a real "not flooded"
  classification.
- `kernel_size` must be a positive odd integer (e.g. `3`, `5`); a
  clear `ValueError` otherwise.
- Wired into `map_floods()` as new `smooth=False`/`smooth_kernel_size=3`
  parameters (opt-in, no behavior change unless requested), applied
  right after slope-masking and before export. The logged
  "high-confidence flooded pixels" count reflects the post-smoothing
  raster when `smooth=True`, matching what's actually written to disk.
- Also fixed a stale code comment in `examples.qmd` left over from the
  0.1.0a21 fix (`generate_number_of_scenes()`'s comment still said
  "count of scenes with data gaps" instead of "valid observations").

**Trimmed the PyPI README** (`PYPI_README.md`, PyPI's rendered
long_description) to just the intro paragraph and a minimal Quickstart
(authentication note, pre-release install line, and a link to the full
docs) -- dropped the logo image and the entire "Basic usage" code
block (which had also drifted out of date, e.g. still showing the
`export_raster` kwarg removed in 0.1.0a19). Full usage now lives only
in the docs site (`examples.qmd`), not duplicated (and liable to rot)
in two places.

Full test suite: 204 passing.

## 0.1.0a22

**Added `autofloods.visualize`**, a new module of quick-look
matplotlib plots for pipeline outputs -- useful for a fast sanity
check in a notebook without opening each raster in GIS software. All
four functions read directly from `output_dir` on disk (reconstructing
file paths deterministically from the `flood_mapper` instance's own
constructor attributes), not from in-memory pipeline state -- they
work against a completed run's output in a **fresh session**, as long
as `fm` is constructed with the same
`grid_shapefile`/`dry_years`/`wet_duration`/`output_dir`/`slope_dir`
the original run used (the same assumption
`flood_mapper.is_fully_processed` already makes). Each function
returns a `matplotlib.figure.Figure` rather than displaying it, so it
renders normally as a notebook cell's last expression or via
`fig.savefig(...)`.

- **`plot_baseline(fm, aoi_id)`** -- 2x2 grid of the dry-season Z-score
  baseline (VV mean, VV std, VH mean, VH std). Mean panels are
  converted to dB for display (the on-disk baseline is linear power --
  dB gives SAR backscatter's standard, visually sane dynamic range)
  and share one percentile-clipped color scale; std panels stay in
  native linear units with their own separate scale.
- **`plot_terrain(fm, aoi_id)`** -- terrain slope (degrees). Slope-only,
  not DEM+slope side by side: the DEM mosaic `prepare_slope()` uses is
  transient (read once to derive slope, then discarded) -- nothing in
  the pipeline caches a DEM file to disk, so there is currently no DEM
  for this function to read in a fresh session.
- **`plot_scenes_and_floods(fm, aoi_id, max_scenes=None, target_cols=6)`**
  -- the most complex layout: one column per wet-season scene, an RGB
  composite (R=VV, G=VH, B=the VV/VH log-ratio, all dB, percentile-
  stretched) directly above that scene's flood classification, columns
  wrapping into additional row-pairs beyond `target_cols` (e.g. 11
  scenes -> two row-pairs, 6 then 5, with unused trailing axes turned
  off rather than left blank). One fixed color scale and shared legend
  for the flood-classification panels across the whole figure, so
  panels are directly comparable. `max_scenes` caps how many
  (chronologically earliest) scenes are rendered; the figure title
  reports how many were shown and, if any were skipped (cap, or a
  scene with no matching flood raster on disk yet), how many and why.
- **`plot_flood_map(fm, aoi_id, month=None)`** -- `monthly_sum()`'s
  per-month flood-day-count output; a single month or every available
  month as a subplot grid, sharing one discrete colorbar so months are
  directly comparable. Nodata (255) is masked and rendered gray, kept
  out of the count colorscale entirely.

Also documented the classified-raster pixel-value encoding
(0=not flooded, 1=VH-only, 2=VV-only, 3=high-confidence flood,
NaN=data gap) prominently in `examples.qmd`'s Output section -- this
was previously only in the `FloodDetector`/`ZScoreDetector` docstrings,
not anywhere a user browsing the docs would see it directly.

Added to the API reference (`quartodocs/_quarto.yml`) and a new
"Visualizing results" section in `examples.qmd` with a real example
per function, including a detailed walkthrough of
`plot_scenes_and_floods`'s layout. 14 new tests in
`tests/test_visualize.py`, including parametrized column-wrap cases
for scene counts that do and don't evenly divide `target_cols`.

Full test suite: 194 passing.

## 0.1.0a21

Two independent changes in this release: a correctness fix and a UX
addition. Documented separately since one is a bug fix, not cosmetic.

**Bug fix: `generate_number_of_scenes()` was computing the opposite of
what it's documented and needed for.** It's meant to return, per pixel,
how many wet-season scenes had a VALID (non-NaN) observation there --
the denominator needed to normalize `monthly_sum()`'s per-month
flood-day count into a fraction (e.g. "flooded on 3 of N valid dates"
requires knowing N per pixel). It was actually returning a per-pixel
GAP count (scenes with a missing/NaN observation) -- exactly inverted.

- **This is a longstanding bug, not a regression from the 0.1.0a19
  wet-season streaming rewrite.** Checked the pre-streaming
  stack-then-reduce implementation (present since at least `d906374`,
  and likely from this method's original implementation): it computed
  the identical `np.any(np.isnan(...), axis=0)` gap count from a fully
  materialized stack. The streaming rewrite faithfully reproduced the
  same (already wrong) computation, just accumulated instead of
  materialized -- it did not introduce this bug.
- Fixed at the source: `prepare_wet_scenes()`'s streaming accumulator
  (renamed `self._wet_scene_gap_count_by_aoi` ->
  `self._wet_scene_valid_count_by_aoi`) now folds in `~np.any(np.isnan(...))`
  (valid) instead of `np.any(np.isnan(...))` (gap), one scene at a
  time, same as before. `generate_number_of_scenes()`'s public name
  and signature are unchanged -- only what it computes internally.
- Output filename changed to make the semantics explicit and prevent
  this confusion recurring: `floodscenescount_<tag>_<id>.tif` ->
  `floodvalidscenecount_<tag>_<id>.tif`.
- Verified: valid count + gap count == total scenes processed, at
  every pixel, for a synthetic case with known per-pixel gaps. Also
  added a test exercising the actual intended use case -- combining
  `generate_number_of_scenes()`'s output with `monthly_sum()`'s output
  to compute "fraction of valid observations flooded" per pixel, on a
  synthetic multi-scene wet season with known flood/valid/gap pixels.

**Added `tqdm` progress bars** across every real per-item pipeline
loop -- `read_scenes()`, `generate_mean_std_by_aoi()`, `prepare_slope()`,
`prepare_wet_scenes()`, `map_floods()`, `merge_floods_by_date()`,
`generate_number_of_scenes()`, `monthly_sum()`, and
`utils.download_nasadem()` -- using `tqdm.auto` (renders correctly in
both notebooks and plain terminals) with `disable=None` (tqdm's own
non-TTY auto-detection, so CI logs and any other non-interactive
output stay clean -- verified the full test suite produces zero tqdm
output under pytest's captured output). The existing one-line
`logger.info(...)` summary at the end of each method is unchanged --
tqdm shows live per-item progress, the log line still gives the final
summary. `get_dry_dates()`, `generate_dry_date_ranges()`, and
`get_s1_items()` were checked but have no per-item loop worth
instrumenting (single combined STAC search / small nested
comprehensions over date ranges, not a per-scene/per-AOI loop).
Added `tqdm==4.70.0` as an explicit dependency (was not previously
installed even transitively).

Full test suite: 180 passing.

## 0.1.0a20

**Fix two crashes in the streaming `map_floods()`'s `export_maps=True`/
`export_vector=True` paths, introduced by 0.1.0a19's disk-cache
restructure and untested until now** (every `map_floods()` call in the
test suite passed `export_vector=False, export_maps=False`).

- **Band-dimension crash (`export_maps=True`)**: `map_floods()` now
  reloads each scene's just-written classified raster from disk via
  `xr.load_dataarray(..., engine='rasterio')`, which returns a
  `(band, y, x)` array (band size 1) instead of the `(y, x)` array
  `flood_images()` got directly from memory in the old architecture --
  `height, width = flood_xarray.shape` then raised
  `ValueError: too many values to unpack (expected 2)`. Fixed by
  squeezing the band dimension after reload.
- **Missing-CRS crash (`export_vector=True`)**: the same reload also
  came back with `rio.crs = None`, crashing
  `postprocessing.polygonize_flood_raster()` at
  `gdf.crs = data.rio.crs.to_string()`. Root cause:
  `utils.export_xarray()` never writes a CRS into the GeoTIFF it
  produces, even though the classified array has one in memory (it's
  inherited from the dry-season baseline) -- this is a pre-existing
  gap in `export_xarray()` itself, not something the streaming
  restructure introduced, just never previously hit because nothing
  reloaded and re-used a flood raster's CRS before. **Fixed for flood
  exports specifically**: the `export_vector=True` reload now borrows
  CRS from `self.mean_std_by_aoi[id]` (same tile/UTM zone), matching
  the pattern already used for the slope reload. The underlying
  `export_xarray()` gap is not fixed at the root and may affect other
  export call sites (slope, the merged-by-date flood stack, scene
  counts) that haven't hit this exact crash only because nothing
  currently reloads and polygonizes them -- see the repo's internal
  notes for the full list of affected call sites, deferred to keep
  this fix scoped to the reported crash.
- Added `TestMapFloodsExportMapsAndVector` (`tests/test_wet_season_streaming.py`),
  running the real streaming `map_floods()` with both flags on, closing
  the test-coverage gap that let both bugs ship in 0.1.0a19.

Full test suite: 176 passing.

## 0.1.0a19

**Major architectural fix: the wet-season pipeline (`prepare_wet_scenes()`
-> `map_floods()` -> `merge_floods_by_date()`/`generate_number_of_scenes()`)
now processes one scene at a time via a disk cache, instead of holding
every wet-season scene in memory at once.** This was the same class of
OOM risk as the dry-season baseline (0.1.0a12) and slope (0.1.0a16)
fixes, just not yet addressed on the wet-season side.

- **`prepare_wet_scenes()`** now reprojects each scene through a
  bounded sliding window (same pattern as the earlier DEM/baseline
  fixes) and writes it straight to a **persistent, resumable** disk
  cache (`wet_scenes_cache/wetscene_<aoi_id>_<scene_id>.nc`) instead of
  building a full in-memory `self.wet_scenes_by_aoi` dict -- re-running
  it skips any scene whose cache file already exists. A per-pixel
  gap/NaN count is accumulated in the same pass for
  `generate_number_of_scenes()` to use later, instead of being
  re-derived from a full in-memory scene dict that no longer exists.
- **`map_floods()`** now reads one scene at a time from that cache,
  classifies it, slope-masks it, and writes it to disk -- `self.flood_dict`
  is now `{scene_id: filepath}`, not arrays.
- **`merge_floods_by_date()`** and **`generate_number_of_scenes()`**
  now read from disk (one date's files, or a small pre-computed
  accumulator) instead of stacking every scene in memory.
- **Real measured impact** (tile 318, 35 real wet scenes, ~204MB each):
  peak memory per stage is now **839MB / 1059MB / 1939MB / 153MB**
  across the four stages -- flat regardless of scene count, versus an
  estimated **~7.1GB** the old architecture would have needed just to
  hold all 35 raw scenes at once (before even adding the classified
  output it also held simultaneously).
- **`map_floods()` can still be safely re-run with different
  `vv_thd`/`vh_thd`/`rel_slope_thd` values** without re-triggering
  `prepare_wet_scenes()`'s read/reproject work -- it just re-reads the
  same cached scenes and overwrites its own output. Verified explicitly
  with a call-count spy (no re-reprojection across two `map_floods()`
  calls with very different thresholds).
- Verified bit-identical (not just "runs without crashing") against
  the old in-memory computation for per-scene classification,
  merge-by-date max-combine, and the gap count, on a synthetic
  multi-scene, multi-date case.

**Breaking change**: `map_floods()`'s `export_raster` parameter has
been **removed** -- every scene's classified raster is now always
written to disk (there's no longer an in-memory result to keep around
if it weren't, since `merge_floods_by_date()`/`generate_number_of_scenes()`
now read from disk too). **If your own script calls `map_floods(...,
export_raster=False)` or `export_raster=True`, remove that argument --
passing it now raises `TypeError`.** `merge_floods_by_date()`'s and
`generate_number_of_scenes()`'s own `export_raster` parameters are
unaffected.

Docs (`quartodocs/examples.qmd`) updated to document the new
`wet_scenes_cache/` directory and its resumability, the map_floods()
re-run behavior, and this breaking change -- not just the CHANGELOG.

## 0.1.0a17

New opt-in flag; **default behavior is unchanged**.

- **Added `keep_intermediate_in_memory` (default `False`) to
  `flood_mapper.__init__()`.** Following the two memory leaks fixed in
  0.1.0a15/0.1.0a16 (`self.s1_dry_dict`, `self.slope`), an audit found
  both were genuinely never needed again in-process once written to
  disk -- but keeping them around instead of re-reading from disk is a
  legitimate choice on a RAM-rich machine (HPC node, local dev), where
  the same amount of memory the default frees up doesn't matter and
  the repeated disk I/O does.
  - Default `False`: current behavior, unchanged -- `self.s1_dry_dict`
    and `self.slope` are `del`eted (+ `gc.collect()`) as soon as their
    pipeline step no longer needs them, and downstream steps that want
    that data again (`map_floods()`'s slope mask) re-read it from disk.
    Still the right choice for a constrained environment like Google
    Colab (~12-13GB).
  - `True`: both attributes stay resident, and `map_floods()` checks
    `self.slope[id]` first, only falling back to its disk read if the
    in-memory copy isn't there. Recommended only where the extra
    memory is genuinely not a concern.
- Audited the rest of the pipeline for the same "compute/read once,
  then redundantly re-read from disk" pattern while implementing this.
  Found one more real instance -- `monthly_sum()` ->
  `postprocessing.aggregate_monthly()` re-reading
  `self.flood_by_date[id]`'s data from disk -- deliberately **not**
  fixed here (it needs restructuring `aggregate_monthly()`'s file-based
  internals, not a simple check-self-first branch); tracked in
  CLAUDE.md's Future To-Dos as its own follow-up.
- Added 7 new tests covering both modes, including that `map_floods()`
  never trusts a stale `self.slope` in the default (`False`) mode even
  if one happens to exist.

## 0.1.0a16

**Fixes the `prepare_slope()` out-of-memory crash reported on Google
Colab** -- the two fixes shipped in 0.1.0a15 (leftover dry-season
scene cleanup, windowed DEM reads) were real and are still in effect,
but investigation found the actual dominant cause was a third,
separate, more severe bug: `preprocessing.smoothen_slope()`'s
neighborhood-averaging kernel.

- **Removed `smoothen_slope()` entirely** (not just stopped calling
  it) and replaced it with `preprocessing.compute_slope()`, which
  returns raw, unsmoothed slope. The removed kernel used
  `sklearn.feature_extraction.image.extract_patches_2d`, which
  materializes a full array copy for *every* overlapping window
  position rather than using an O(1)-memory sliding-window filter --
  for a real ~100km tile at the default `buffer=500`/`cell_size=30`
  (a 33x33 kernel), this required **~101 GB**, computed precisely from
  real tile geometry and confirmed against a live `sklearn` call.
  Independent of and much larger than the 0.1.0a15 fixes.
- **Verified fixed, not just "should be fixed"**: re-ran the exact
  crash scenario (tile 318, real data, `prepare_slope(dem_overview=1,
  buffer=500)`) end-to-end -- previously a reliable `SIGKILL` even on
  a 503GB-RAM machine, now completes with a real, bounded, measured
  peak (945 MB).
- **Known open question, not a regression**: `map_floods()`'s
  `rel_slope_thd=20` default was implicitly tuned against *smoothed*
  slope (the only place this was ever documented was the already-
  deprecated `mapfloods.map_floods()` free function's docstring, now
  corrected for accuracy). Whether 20 degrees still correctly separates
  real terrain from flat ground against raw (noisier) slope is
  genuinely unvalidated -- flagged in CLAUDE.md's Future To-Dos for
  empirical validation against known flood/non-flood ground truth, not
  guessed at here. If your results look different after upgrading,
  this threshold is the first thing to check.
- `scikit-learn` is now an unused dependency (kept in `pyproject.toml`
  for this release; a future release may drop it).
- Updated/renamed test coverage (`tests/test_preprocessing.py::
  TestComputeSlope`, replacing `TestSmoothenSlope`), including a
  regression guard confirming the removed kernel's imports can't
  silently creep back in.

## 0.1.0a15

**Windowed reads for `MPCSource`**: reading a scene no longer always
downloads the full asset before clipping to a tile.

- `MPCSource.read_vv_vh()` now reprojects the AOI(s) being processed
  into the scene's own native CRS (read from its STAC `proj:code`/
  `proj:epsg` properties) and clips the still-lazy `rioxarray` handle
  to that window -- via `.rio.clip_box()` -- **before** materializing
  pixel data, instead of always calling `.load()` on the full scene
  first. MPC's Sentinel-1 RTC assets are real COGs (internally tiled,
  512x512 blocks, 6 overview levels, STAC-declared
  `profile=cloud-optimized`), so this is a genuine reduction in bytes
  fetched over the network, not just an in-memory optimization.
- **Measured, live, through the real `read_vv_vh()` code path**: for a
  ~5km AOI, the old (always-full-scene) path transfers 3,708,221,241
  bytes (~3.71 GB, both VV+VH bands, from the assets' own STAC
  `file:size`); the new windowed path transfers 33,320,222 bytes
  (~33.3 MB) for the same AOI -- **~111x fewer bytes**. Verified
  numerically identical to a full-read-then-clip result first, both on
  a synthetic array and against the real remote asset (clip-before-load
  and load-then-clip produce bit-identical pixels and coordinates).
- A 5km buffer (in the scene's native CRS) is added around the
  windowed extent, so the later reproject+resample step
  (`clip_xarray_using_id()`) still has the same edge margin a
  full-scene read always implicitly provided.
- `OPERASource` is intentionally unchanged: it downloads whole burst
  files to local disk by deliberate design (a prior reliability
  choice, not a limitation being revisited here), so `bbox` is
  accepted for interface compatibility but ignored.
- Falls back to a full, unwindowed read if a scene lacks
  `proj:code`/`proj:epsg` metadata -- windowing is an optimization,
  not a correctness requirement.
- Added 9 new tests covering the reprojection/buffer math and the
  windowed-read call path (mocked network layer).

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
