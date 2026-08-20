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
  `var[y, x] = grid[i, j]` with **no flip**. The voxel grid `(i, j, k)` maps to
  `buildings_3d(z, y, x)` by a `(2, 0, 1)` transpose.
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
- `z` (for `buildings_3d`) and `zlad` (for `lad`) are cell-center height
  coordinates. Per PIDS, `zlad` covers ground to the highest canopy layer; `z`
  covers ground to the highest building layer.
- Global attributes:
  - `Conventions = "CF-1.7"`, `origin_time`, `data_content`, `version`,
    `creation_time`; `author`/`comment` when provided.
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
| Road | pavement | 1 (asphalt) |
| Developed space | pavement | 2 (concrete) |
| Building | — (skipped; building mask wins) | — |
| No Data | vegetation | 3 (short grass) |

Analogous tables cover the same sources as cityles (OpenStreetMap/Standard,
Urbanwatch, OpenEarthMapJapan, ESA WorldCover, ESRI 10m, Dynamic World V1);
unknown sources fall back to the OSM table with a logged warning.

Precedence per cell (highest first):

1. Building footprint (`buildings.heights > 0`) → all three types stay fill.
2. Tree canopy present (and no building) → `vegetation_type =
   under_tree_vegetation_type` (default 3, short grass): the resolved LAD field
   represents the trees themselves; the surface below is ground vegetation.
3. Otherwise → mapped `(category, code)` from land cover.

Derived fields:

- `soil_type(y, x)` byte: `soil_type` parameter (default 3, medium) wherever
  `vegetation_type` or `pavement_type` is set; fill elsewhere.
- `surface_fraction(nsurface_fraction=3, y, x)` float32: one-hot
  (vegetation, pavement, water) matching the assigned category; fill on
  building cells.

A mapping summary (per source class → PALM assignment, cell counts) is logged,
like the cityles exporter does.

## Buildings

- `buildings_2d(y, x)` float32, attribute `lod = 1`, `_FillValue = -9999.0`:
  `heights` where `> 0`, fill elsewhere. NaN heights are treated as no building.
- `building_id(y, x)` int32, `_FillValue = -9999`: from `city.buildings.ids`
  where a building exists; cells with a height but no ID get a generated unique
  ID (max existing ID + running counter).
- `building_type(y, x)` byte, `_FillValue = -127`: constant `building_type`
  parameter on building cells (default 3, residential 1950–2000).
- `buildings_3d(z, y, x)` byte 0/1, attribute `lod = 2`, `_FillValue = -127`:
  derived from the voxel grid's building class via transpose `(2, 0, 1)`.
  Written when `buildings_3d=True`, or under `"auto"` when the voxel grid
  contains non-ground-based building geometry (some building voxel column has
  air below building — the case `buildings_2d` cannot represent). PIDS requires
  `building_id` and `building_type` on every column where `buildings_3d` has any
  set voxel; the builder enforces this.

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

`export_palm` raises `ValueError` (before writing anything) when:

- `city.extras["rectangle_vertices"]` is missing — georeferencing is required;
  a (0,0)-defaulted PALM file would be useless.
- 2D grid shapes disagree with each other or with the voxel grid's horizontal
  shape.

Built-in consistency validation (the rules PALM checks at runtime, encoded so
files pass PALM's own checks — user-selected in lieu of a live PALM run):

- Exactly-one-surface-type invariant on every non-building cell; none on
  building cells.
- `soil_type` present wherever vegetation or pavement is set.
- `surface_fraction` sums to 1.0 on classified cells.
- `buildings_3d` columns all have `building_id`/`building_type`; `buildings_2d`
  is consistent with `buildings_3d` where both exist.
- All written arrays free of NaN/inf; byte fields within valid PIDS class
  ranges; `zt` minimum exactly 0.
- LAD values only within `[bottom, top]` layers and only where fill isn't
  expected.

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
- Running PALM itself within this implementation; automated validation is
  format-level plus encoded consistency rules. A real PALM run on an exported
  static driver is planned as a follow-up end-to-end validation once the
  exporter lands — findings from that run feed back as fixes/tests here.
