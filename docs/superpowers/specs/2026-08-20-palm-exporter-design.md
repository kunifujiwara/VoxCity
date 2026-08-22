# PALM Static Driver Exporter — Design

**Date:** 2026-08-20
**Status:** Approved for planning
**Scope decisions (user-confirmed):**
- Static driver only — no `_p3d` namelist, no dynamic driver.
- Buildings: `buildings_2d` + `building_id` always; `buildings_3d` when needed (auto) or on request.
- Vegetation: 3D LAD field + surface classification (not surface types alone).
- Implementation approach: direct netCDF4 writer, cityles-style module (pure builders + writer).
- Validation: format tests + built-in PIDS/runtime consistency validation in the
  automated suite; a real PALM run against an exported file will be performed
  later as a separate end-to-end validation step (outside this implementation).

## Background

PALM (palm.muk.uni-hannover.de) is an LES model for atmospheric boundary-layer and
urban-climate simulation. Its terrain/building/vegetation input is the **static
driver**: a NetCDF file named `<domain_name>_static` conforming to the PALM Input
Data Standard (PIDS). VoxCity already holds every required input in memory
(building heights/IDs, land cover, DEM, tree canopy top/bottom, voxel grid,
meshsize, geolocation), so the exporter is a translation layer, structurally
parallel to the existing CityLES exporter ([src/voxcity/exporter/cityles.py](../../../src/voxcity/exporter/cityles.py)).

Key conveniences confirmed against the codebase:

- **Orientation is a direct match.** VoxCity's internal contract
  (`voxcity.utils.orientation`: axis 0 = northward with row 0 at the south edge,
  axis 1 = eastward) equals PALM's `(y, x)` convention. Grids are written as
  `var[y, x] = grid[i, j]` with **no flip**. `buildings_3d(z, y, x)` is NOT a
  transpose of `city.voxels.classes(i, j, k)` -- the implementation never reads
  the voxel grid for this field. It is built directly from
  `city.buildings.heights`/`min_heights` (per-cell `[min, max]` segments,
  extruded to a byte 0/1 mask) in `_build_buildings_3d`, at whatever `nz` the
  tallest height/segment requires. (Update 2026-08-22: this is no longer the
  whole story for building *presence* -- see "Voxel-grid reconciliation" below.
  `min_heights` itself is now reconciled against `city.voxels.classes` before
  either LOD1 or LOD2 building fields are built, so `buildings_3d` for a
  reconciled column IS effectively derived from the voxel grid, via
  synthesized `min_heights` segments.)
- **Rotation conventions agree.** VoxCity's `rotation_angle` (degrees clockwise,
  derived from `rectangle_vertices` via `compute_rotation_angle`) matches the PIDS
  `rotation_angle` global attribute definition.
- **Dependencies already present.** `netCDF4` and `pyproj` are hard dependencies.

## Module & public API

New file `src/voxcity/exporter/palm.py`, registered in
`src/voxcity/exporter/__init__.py` (`__all__`: `PalmExporter`, `export_palm`).

```python
export_palm(
    city: VoxCity,
    output_directory: str = "output/palm",
    domain_name: str = "voxcity",            # output file: <domain_name>_static
    origin_time: str = "2000-01-01 00:00:00 +00",
    lad: float = 1.0,                        # constant leaf area density, m2/m3
    trunk_height_ratio: float = <sentinel>,  # same semantics as export_cityles
    canopy_bottom_height_grid: np.ndarray | None = None,
    building_type: int = 3,                  # PALM building class for all buildings
    buildings_3d: bool | str = "auto",       # True | False | "auto"
    under_tree_vegetation_type: int = 3,     # PALM class for ground beneath canopy
    soil_type: int = 3,                      # medium soil wherever veg/pavement
    land_cover_source: str | None = None,    # auto from city.extras
    author: str | None = None,
    comment: str | None = None,
) -> str                                     # path to the written _static file

class PalmExporter:
    """Adapter matching the exporter Protocol (same shape as CityLesExporter)."""
    def export(self, obj, output_directory, base_filename, **kwargs) -> str
```

Internal structure, top to bottom of the module:

1. **Mapping tables** — per-land-cover-source class-name → `(category, palm_code)`.
2. **Pure builder functions** — each returns numpy arrays (+ metadata), no I/O:
   `_build_zt`, `_build_buildings`, `_build_surface_types`, `_build_lad`,
   `_build_georeference`.
3. **Validator** — `_validate_static_fields(...)` enforcing PIDS/runtime rules.
4. **Writer** — `_write_static_driver(path, fields, attrs)` using `netCDF4`
   directly (`format="NETCDF4"`).
5. **`export_palm`** orchestrator + `PalmExporter` adapter.

## Grid geometry & georeferencing

- Dimensions `x`, `y`; coordinate variables are cell centers
  `(index + 0.5) * meshsize`, `dx = dy = dz = meshsize` (uniform, from
  `city.voxels.meta.meshsize`).
- `z` (for `buildings_3d`) and `zlad` (for `lad`) both follow PALM's zu-grid
  convention: a surface level at 0.0, then cell centres `(k - 0.5) * dz` —
  one shared constructor (`_zu_levels`) produces both. This is a hard PALM
  requirement, not a style choice: PALM compares each coordinate against its
  own `zu` grid within `0.001 * dz` on read (`topography_mod`, error
  PAC0337, and `plant_canopy_model_mod`) and aborts on mismatch. Verified
  against a real PALM v25.04 run: an earlier draft wrote `z` as pure cell
  centres `[1, 3, 5, ...]` with no surface level and PALM rejected the file
  at topography setup. A `buildings_3d` level is set when its zu height lies
  inside a building segment's `[min, max]` — the same "level inside
  geometry" rule the LAD field uses for crowns, matching the reference
  driver's convention (a 10 m building at dz = 2 is `[1,1,1,1,1,1]`:
  surface + centres 1..9 m). `zlad` covers ground to the highest canopy
  layer; `z` covers ground to the highest building layer.
- Global attributes actually written: `Conventions = "CF-1.7"`, `title`,
  `source = "VoxCity"`, `origin_time`, `creation_time`, `origin_lon`,
  `origin_lat`, `origin_x`, `origin_y`, `origin_z`, `rotation_angle`,
  `comment` (always; the user-supplied `comment` argument, when given, is
  appended to it); `author` only when provided. `data_content` and
  `version` are not written -- an earlier draft of this spec promised
  them; they were dropped during implementation and never added back.
  - `origin_lon`, `origin_lat`: SW corner from
    `normalize_rectangle_vertices(city.extras["rectangle_vertices"])`.
  - `origin_x`, `origin_y`: that corner projected to the auto-detected UTM zone
    (pyproj, from origin lon/lat); `crs`-related info recorded in a comment
    attribute naming the EPSG code used.
  - `rotation_angle`: from `compute_rotation_angle(rect)` (conventions agree).
  - `origin_z`: the vertical shift applied to the terrain (see below).
- `zt(y, x)` float32: DEM with NaN replaced by the grid minimum, then shifted so
  the minimum is 0. The shift amount is recorded as `origin_z`. Negative
  post-shift values cannot occur by construction.

## Surface classification

PIDS rule: every ground-surface cell must have **exactly one** of
`vegetation_type`, `pavement_type`, `water_type` set (byte, `_FillValue = -127`);
building-footprint cells have none (building surfaces take over).

Mapping follows the cityles pattern: raw per-source land-cover indices are
resolved to class names via `get_land_cover_classes(land_cover_source)`, then a
per-source table maps names → `(category, palm_code)`:

| VoxCity class (standard names) | Category | PALM code |
|---|---|---|
| Bareland, Moss and lichen | vegetation | 1 (bare soil) |
| Rangeland | vegetation | 3 (short grass) |
| Shrub | vegetation | 16 (deciduous shrubs) |
| Agriculture land | vegetation | 2 (crops, mixed farming) |
| Tree, Mangroves | vegetation | 7 (deciduous broadleaf trees) |
| Wet land | vegetation | 14 (bogs and marshes) |
| Snow and ice | vegetation | 13 (ice caps and glaciers) |
| Water | water | 1 (lake) |
| Road | pavement | 2 (asphalt) |
| Developed space | pavement | 3 (concrete) |
| Building (no recorded height) | pavement | 3 (concrete) -- remapped, not skipped: a footprint with no height is a sealed surface but no obstacle (building mask, which DOES win on a cell with a recorded height, applies separately) |
| No Data | vegetation | 3 (short grass) |

Analogous tables cover the same sources as cityles (OpenStreetMap/Standard,
Urbanwatch, OpenEarthMapJapan, ESA WorldCover, ESRI 10m, Dynamic World V1);
unknown sources fall back to the OSM table with a logged warning.

Precedence per cell (highest first):

1. Building footprint (`buildings.heights > 0`) → all three types stay fill.
2. Tree canopy present (no building, and the land-cover mapping below does
   not resolve to water) → `vegetation_type = under_tree_vegetation_type`
   (default 3, short grass): the resolved LAD field represents the trees
   themselves; the surface below is ground vegetation. Overhanging or
   riparian canopy over water keeps the water surface (tier 3) instead —
   water differs from short grass in albedo, heat capacity, and
   evaporation by margins that dominate a microclimate result, and the
   LAD field already resolves the trees independently of surface type.
3. Otherwise → mapped `(category, code)` from land cover (this is also
   where a water classification under canopy is resolved, per tier 2's
   exception).

Derived fields:

- `soil_type(y, x)` byte: `soil_type` parameter (default 3, medium) wherever
  `vegetation_type` or `pavement_type` is set; fill elsewhere.
- `surface_fraction(nsurface_fraction=3, y, x)` float32: one-hot
  (vegetation, pavement, water) matching the assigned category; fill on
  building cells.

A mapping summary (per source class → PALM assignment, cell counts) is logged,
like the cityles exporter does.

## Buildings

**Presence predicate (single source of truth).** A cell is a building iff it has
a positive height OR LOD2 geometry rising above ground. Formally, per cell
`segment_top_m = max(max(0.0, seg_top) for seg in segments)` (so a segment
entirely below the terrain datum contributes nothing), and
`mask = (height > 0) | (segment_top_m > 0)`. This one mask feeds both the LOD1
and LOD2 builders, so the two cannot disagree by construction. Without it two
PIDS-invalid outputs are reachable: a segment with no recorded height yields a
filled 3D column with no `building_id`/`building_type`, and a building shorter
than half a voxel yields a `buildings_2d` entry whose 3D column is empty.

- `buildings_2d(y, x)` float32, attribute `lod = 1`, `_FillValue = -9999.0`:
  `max(height, segment_top_m)` on the mask, fill elsewhere. NaN/inf heights are
  treated as no building. Taking the max repairs the orphan case — a cell whose
  geometry is taller than its recorded height gets the height its geometry
  implies — and logs at info level when that repair fires, so malformed input is
  visible rather than silent.
- `building_id(y, x)` int32, `_FillValue = -9999`: from `city.buildings.ids`
  where a building exists; cells with a height but no ID get a generated unique
  ID (max existing ID + running counter). IDs are assumed to be small positive
  integers, as the VoxCity pipeline produces them.
- `building_type(y, x)` byte, `_FillValue = -127`: constant `building_type`
  parameter on building cells (default 3, "Residential, > 2000" per PALM's
  `building_type` table; 1950-2000 is code 2, not the default).
- `buildings_3d(z, y, x)` byte 0/1, attribute `lod = 2`, `_FillValue = -127`:
  built from the per-cell `[min, max]` segments in `city.buildings.min_heights`,
  falling back to ground extrusion for cells without segments. Written when
  `buildings_3d=True`, or under `"auto"` when any segment starts above ground —
  the case `buildings_2d` cannot represent. Level conversion uses the
  voxelizer's rounding (`int(h / meshsize + 0.5)`) so the mask aligns with
  VoxCity's own voxel grid. `nz` covers the taller of the height-derived and
  segment-derived tops, so tall geometry is never truncated; segment bounds are
  clamped to `[0, nz]` so below-ground geometry starts at ground rather than
  wrapping via negative-index slicing.
- **Sub-voxel guarantee:** every masked cell gets at least one filled level. A
  building shorter than half a voxel would otherwise vanish from the LOD2 field,
  and when `buildings_3d` is present PALM reads it as authoritative — so
  dropping the column would silently delete a building the user asked to export.
  Rounding up by less than one `dz` is a bounded error at the vertical
  resolution limit; silent data loss is not. This deliberately diverges from
  VoxCity's own voxelizer, which drops sub-voxel buildings.

## Vegetation (LAD)

`lad(zlad, y, x)` float32, `_FillValue = -9999.0`: for each canopy cell, the
constant `lad` parameter value in voxel layers whose centers lie between canopy
bottom and canopy top; fill (not zero) outside vegetation columns.

Canopy bottom resolution replicates `export_cityles` exactly, including the
sentinel semantics: an explicitly passed `trunk_height_ratio` always recomputes
`bottom = top * ratio`; otherwise an explicit `canopy_bottom_height_grid`
argument wins; otherwise `city.tree_canopy.bottom` when present; otherwise
default ratio 0.3. Canopy on building cells is cleared (building precedence),
matching cityles.

## Validation & error handling

`export_palm` raises `ValueError` (before writing anything, via
`_check_export_inputs`) for seven kinds of bad user input:

- `city.extras["rectangle_vertices"]` is missing — georeferencing is required;
  a (0,0)-defaulted PALM file would be useless.
- A 2D grid (land cover, DEM, canopy top, `buildings.ids`,
  `buildings.min_heights`, `canopy_bottom_height_grid`) or the voxel grid's
  horizontal shape disagrees with the building heights grid's shape.
- `meshsize` is non-positive or non-finite (a zero mesh size divides by zero
  inside a builder; a negative one exports "successfully" with negative
  coordinates).
- `building_type`/`soil_type`/`under_tree_vegetation_type` fall outside
  their PIDS class range (`BYTE_RANGES`).
- `lad` is negative or non-finite.
- `trunk_height_ratio`, when explicitly passed, is not a finite number in
  `[0, 1]`.
- `buildings_3d` is not exactly `True`, `False`, or `"auto"`.

Built-in consistency validation (the rules PALM checks at runtime, encoded so
files pass PALM's own checks — user-selected in lieu of a live PALM run):

- Exactly-one-surface-type invariant on every non-building cell; none on
  building cells.
- `soil_type` present wherever vegetation or pavement is set.
- `surface_fraction` sums to 1.0 on classified cells.
- **Presence**-consistency between LOD1 and LOD2: a cell is a building in every
  building output or in none of them. Concretely, for every cell,
  `buildings_3d.any(axis=0)`, `buildings_2d != fill`, `building_id != fill` and
  `building_type != fill` must all agree. This is **not** a magnitude check:
  `buildings_2d` may legitimately exceed the 3D column's top (e.g. a recorded
  height of 20 m alongside a single `[0, 4]` segment is valid LOD2 data), so the
  validator must not compare heights against 3D extents or it will reject valid
  cities. The builders establish this invariant by construction via the shared
  presence mask (see Buildings), so a violation here indicates an exporter bug.
- All written arrays free of NaN/inf; byte fields within valid PIDS class
  ranges; `zt` minimum exactly 0.
- `lad` values (where not fill) are finite and non-negative.

**Not enforced, deliberately:** "LAD values only within `[bottom, top]`
layers" -- an earlier draft of this spec listed this as validated, but
`_validate_static_fields` only ever receives the finished `lad` array, not
the canopy top/bottom inputs `_build_lad` placed it from, so it has no
independent reference to re-derive the expected layer range against; and
`_build_lad`'s own empty-crown fallback (a crown too thin to catch a zlad
level) deliberately fills exactly one level outside `[bottom, top]` rather
than dropping the canopy entirely. Both functions' docstrings already say
this; this spec previously did not agree with them.

These run as an internal `_validate_static_fields()` step before the writer; a
violation is an exporter bug, so it raises `AssertionError`-style `RuntimeError`
with a description (not silently corrected).

Unknown `land_cover_source` → OSM fallback table + logged warning (cityles
behavior). NaN in DEM → replaced with grid minimum before shifting.

## Testing

`tests/test_exporter_palm.py`, synthetic small `VoxCity` fixture (pattern from
the cityles/envimet tests):

- **File contract:** re-open the written file with `netCDF4` and pin dimensions,
  coordinate values, dtypes, `_FillValue`s, per-variable attributes (`lod`,
  `units`, `long_name`), and global attributes (origin lon/lat, UTM origin,
  `rotation_angle`, `origin_z`, `origin_time`).
- **Invariants:** surface-type mutual exclusivity, soil-type coverage,
  surface_fraction one-hot, LAD layer placement, canopy-under-building
  clearing, building precedence over land cover.
- **buildings_3d auto:** absent for a purely extruded city; present when the
  fixture contains a floating building voxel; forced on/off by parameter.
- **Georeferencing:** SW-corner selection, UTM projection round-trip sanity,
  rotation angle for a rotated fixture.
- **Error paths:** missing rectangle_vertices, mismatched shapes, unknown land
  cover source (warning + fallback).
- **Pure builders:** direct unit tests without disk I/O.

## Out of scope

- `_p3d` namelist, dynamic driver, `palm_csd`-style child domains / nesting.
- Street types (`street_type`), surface albedo/emissivity pars
  (`building_pars`, `vegetation_pars`, ...), water temperature pars.
- Running PALM inside the automated test suite; automated validation is
  format-level plus encoded consistency rules.

## End-to-end PALM validation (performed 2026-08-21)

The planned real-PALM validation was carried out against **PALM v25.04**
(WSL2 Ubuntu, gfortran + OpenMPI + netCDF-Fortran, `palmrun -X4`) using a
synthetic 20x20-cell city exported by `export_palm` (four building blocks,
an elevated walkway, roads, a water pond, a plaza, street trees, sloped
terrain) and a minimal `_p3d` namelist (no dynamic driver, clear-sky
radiation, LSM + USM + plant canopy from file).

**It caught one real bug**: the `z` coordinate was written as pure cell
centres `[1, 3, 5, ...]`, and PALM aborted at topography setup with error
PAC0337 — its `topography_mod` requires `z` to equal PALM's own zu grid
(surface level 0.0, then centres) within `0.001 * dz`. `zlad` already
used the correct convention. Fixed by giving both coordinates one shared
constructor (`_zu_levels`) and rebuilding the LOD2 mask on zu levels
(a level is building when its height lies inside a segment), matching the
reference driver's convention. See the grid-geometry section above.

After the fix, PALM ran the case to completion (27 timesteps, t = 20 s,
exit 0) and the output was checked against the input city:

- `u` is topography-masked inside the 18 m building and ~2 m/s above roof.
- `pcm_lad` has exactly 36 nonzero cells = 12 trees x 3 crown levels, at
  the exported LAD value 1.0; the header shows the canopy profile
  `[0.00, 1.00, 1.00, 1.00]` read from file.
- Surface temperatures order physically (water 283.9 K coolest, road
  288.9 K, plaza 289.1 K, grass 289.9 K warmest at summer noon +20 s), so
  the corrected `pavement_type` codes and the water classification reach
  PALM's energy balance as intended.

**Scale validation (same day):** a real 1 km x 1 km Tokyo model built by
VoxCityGML from PLATEAU LOD2 (500x500 cells at 2 m, 136,133 building cells,
tallest 237.6 m, 21,847 cells with elevated LOD2 segments, OpenEarthMapJapan
land cover) exported in 1.9 s to a 2.2 MiB compressed driver and ran in PALM
(500x500x141 grid, dynamics + plant canopy, 21 timesteps to t = 4 s, exit 0).
The output's solid fraction matches the input exactly at ground sections
(0.998 = 0.998 at zu = 3 m including terrain) and within terrain-rounding at
height (0.420 vs 0.373 at 23 m; 0.102 vs 0.096 at 119 m).

**Namelist interaction discovered by the scale run:** building footprints
carry no ground surface class (deliberately -- PALM's own reference driver
does the same, all 100 of its building cells unclassified). With the LSM
enabled but the USM off, PALM rejects such a driver with DRV0021 at every
footprint cell. A domain with buildings therefore needs the urban surface
model enabled alongside the LSM and a radiation scheme; dynamics-only
namelists accept the driver as plain topography. Recorded in the README's
PALM section and the module docstring.

**Cross-version confirmation (2026-08-22):** the same small-city driver was
run unchanged under **PALM v25.10.1** (fresh gfortran build) and completed
identically (27 timesteps, exit 0, all outputs). A source diff confirmed the
input contracts the exporter relies on -- the zu-grid check (PAC0337), the
pavement class table, the building-footprint/USM rule (DRV0021), and the zlad
check -- are byte-identical between v25.04 and v25.10.1. Also established:
PALM's OpenACC GPU mode rejects building-resolved topography outright
(PAC0359), so urban drivers are CPU-only as of v25.10.1.

## Voxel-grid reconciliation (fixed 2026-08-22): 149 missing building columns

The first real PALM run through VoxCityApp (a 50x50 Tokyo LOD2 tile) exposed
a root-caused data bug: the exported static driver omitted **149 of 1155**
building columns present in the model's voxel grid. Every one of the 149
columns had `city.buildings.heights == 0.0` and an **empty**
`min_heights` list, yet a real `building_id` (> 0) and real `-3`
(building) voxels in `city.voxels.classes` -- some up to ~160 m of tower,
some 1-2 voxel-tall shell fragments.

**Root cause:** VoxCity's LOD2 pipeline voxelizes the detailed building
SHELLS (partial-coverage edge cells, complex geometry, small structures)
directly into `city.voxels.classes`, while the 2-D `heights`/`min_heights`
grids are a separate footprint rasterization that can miss those same
cells entirely. `export_palm` derived building presence exclusively from
`_build_building_mask(heights, min_heights)` -- the voxel grid was never
consulted for presence, so PALM silently simulated open street through
149 real building columns (wrong local radiation/wind, pedestrian comfort
computed "inside" buildings). Every other simulator in this ecosystem
(wind LBM, solar, view) runs on the voxel grid, so this was PALM running
on a materially different city than the rest of the toolchain.

**Fix (additive, not a rewrite of the presence rule):** a new function,
`_reconcile_buildings_with_voxels`, runs once, immediately before the
single existing `_build_building_mask` call in `export_palm`. For every
column where `city.voxels.classes` contains a `-3` voxel that the
existing heights/min_heights mask does not already cover, it derives LOD2
segments directly from that column's contiguous `-3` runs (same datum
`generator/voxelizer.py`/`importer/integrate.py` already use: segment top
in meters = `(top_k_building - terrain_top_k) * meshsize`, terrain being
the topmost ground/land-cover solid voxel, never `-2` trees) and appends
them to that column's (previously empty) `min_heights`. From that point
on the reconciled column is an ordinary orphan-segment cell -- the
pre-existing `_build_buildings` repair path raises `buildings_2d` to the
segment top (and logs the existing WARNING), and `_build_buildings_3d`
turns the same segments into exact zu-level occupancy, so the LOD2 mask
for a reconciled column reflects the real voxel shape, not a blind
ground-up extrusion. `building_id`/`building_type` need no special
handling: `city.buildings.ids` already carries the real id at these
columns, so the existing `_build_buildings` id/type logic picks it up
once the column is inside `building_mask`. The one presence predicate
(`_build_building_mask`) still feeds both LOD1 and LOD2 -- reconciliation
happens on its *input* (`min_heights`), not as a second, parallel rule.

Guard: when `city.voxels` is `None`, reconciliation is a no-op (today's
behavior is preserved exactly) -- currently unreachable via `export_palm`
itself since `_check_export_inputs` already requires `city.voxels.classes`
unconditionally for the pre-existing shape check.

Logged: `PALM driver: N building column(s) reconciled from the voxel grid
...` at INFO whenever N > 0.

**Real-model verification** (the scenario that surfaced the bug, re-run
after the fix): voxel-grid building columns = 1155; driver building
columns before the fix = 1006 (gap 149); after the fix = 1155 (gap 0).

**Tests:** `tests/test_exporter_palm.py::TestVoxelBuildingReconciliation`
-- a voxel-only column is reconciled into `buildings_2d`/`building_id`/
`buildings_3d`; a tree-only (`-2`) voxel column is never reconciled into a
building; an all-air voxel grid (no `city.voxels` building data) leaves
the pre-existing 2-D-derived buildings unchanged; the reconciled-column
count is logged. Full suite: 300 -> 304 tests, all passing.
