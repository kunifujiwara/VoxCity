# Rhino/OBJ Building Import — Design Spec

**Date:** 2026-06-23
**Status:** Approved design, ready for implementation planning
**Author:** VoxCity (brainstormed with Claude)

## 1. Purpose

Let users add buildings authored in Rhino (3D models) into an existing
("base") VoxCity voxel model. The Rhino geometry supplies *buildings only*;
the base model continues to provide terrain (DEM), land cover, and trees from
VoxCity's normal data sources. The imported buildings are voxelized directly
from their 3D mesh form (preserving cantilevers, sloped roofs, atria) and
integrated so they participate in voxel- and grid-based simulations and
exports.

This closes the loop with VoxCity's existing Rhino interoperability (it already
*exports* OBJ for use in Rhino).

## 2. Scope

**In scope**
- Import building geometry from a Wavefront **OBJ** file.
- Add it to an existing in-memory `VoxCity` object (post-processing, not part of
  initial generation).
- Georeference via an **anchor** (a model point ↔ world lon/lat/elevation),
  plus **rotation** (true-north) and **horizontal + vertical move** adjustments,
  plus **units**.
- **Direct 3D mesh voxelization** into BUILDING voxels.
- Update derived metadata grids: `building_id_grid`, `building_height_grid`,
  and the per-cell `buildings.min_heights` segment lists.
- Placement in **VoxCity core** as a new `voxcity/importer/` subpackage.

**Out of scope (explicitly)**
- Footprint+height extraction mode (we voxelize the true 3D mesh instead).
- Synthesizing `building_gdf` footprint polygons for imported buildings.
- Importing terrain, land cover, or vegetation from the Rhino file.
- Native `.3dm` parsing (OBJ only for this iteration).
- Integrating import as a `building_source` inside `get_voxcity` (it is a
  separate post-processing step).
- Interactive map-based placement UI.

## 3. Decisions (from brainstorming)

| Topic | Decision |
|---|---|
| Role of imported model | Buildings only; added to a base voxel city model |
| Integration style | Post-processing function on a `VoxCity` object (Approach A) |
| Georeferencing | Anchor (model point ↔ lon/lat/elevation) + rotation + horizontal & vertical move + units |
| Voxelization | Direct 3D mesh voxelization (occupancy) |
| File format | OBJ |
| Metadata depth | Voxels **+** derived metadata (`building_id_grid`, `building_height_grid`, `min_heights`); **no** `building_gdf` synthesis |
| Vertical placement | User supplies anchor elevation; mapped relative to base model's ground datum |
| Voxelization backend | **trimesh** default (MIT-clean, already a dependency); optional lazy `backend="meshlib"` for robust SDF voxelization (MeshLib is non-commercial-licensed, so never required/default) |
| Code placement | VoxCity core, `voxcity/importer/` subpackage |

## 4. Architecture

New subpackage parallel to `voxcity/exporter/`:

```
src/voxcity/importer/
    __init__.py        # exports add_buildings_from_obj
    rhino_obj.py       # public function + orchestration
    transform.py       # anchor/rotation/move/units -> 4x4 affine (model->voxel index)
    voxelize.py        # mesh -> occupied (i,j,k) cells; trimesh + optional meshlib backends
    integrate.py       # stamp voxels + update metadata grids on the VoxCity object
```

Each internal piece is independently testable (mesh-in → indices-out, etc.).

### 4.1 Public API

```python
from voxcity.importer import add_buildings_from_obj

voxcity = get_voxcity(rectangle_vertices, meshsize, ...)   # base model

voxcity = add_buildings_from_obj(
    voxcity,                              # base VoxCity object
    obj_path="design.obj",
    anchor_lonlat=(139.7536, 35.6841),   # world (lon, lat) the anchor maps to
    anchor_elevation=12.0,               # world elevation (m) of the anchor
    anchor_model_point=(0.0, 0.0, 0.0),  # the point IN the model = the anchor (default: model origin)
    rotation=0.0,                        # degrees clockwise; model +Y -> true north at 0
    move=(0.0, 0.0, 0.0),                # extra nudge (east_m, north_m, up_m)
    units="m",                           # "m" | "cm" | "mm" | "ft" | "in"
    backend="trimesh",                   # "trimesh" (default) | "meshlib" (optional, lazy)
    z_up=True,                           # model Z is vertical; set False / use swap_yz for Y-up exports
    swap_yz=False,                       # escape hatch for Y-up OBJ exports
    overwrite=True,                      # imported buildings win in cells they occupy
    solid_fill=True,                     # fill interiors (solid masses) vs hollow shells
    gridvis=False,
) -> VoxCity
```

Returns an updated `VoxCity` object (same dataclass), so all existing exporters
and simulators work unchanged.

## 5. Coordinate transform (`transform.py`)

Build one 4×4 affine `M: model coordinates → voxel-index space`, composed of:

1. **Units → meters & anchor to origin.**
   `p_m = (V − anchor_model_point) × unit_scale`
   where `unit_scale`: m=1, cm=0.01, mm=0.001, ft=0.3048, in=0.0254.
   If `swap_yz`/`z_up=False`, swap axes so model Z is vertical first.

2. **Horizontal rotation** about the vertical axis. `rotation` degrees clockwise
   so that model **+Y = true north**, **+X = east** at `rotation=0`. Fold in the
   **domain's own rotation** (a rotated `rectangle_vertices` means north ≠ +u):
   net rotation maps model (x→east, y→north) into the domain (u along `side_1`,
   v along `side_2`) frame.

3. **Translate to the anchor's domain position.**
   - Horizontal: `GridProjector.lon_lat_to_uv_m(lon_a, lat_a)` → `(u_a, v_a)`
     meters from grid origin; add horizontal `move`.
   - Vertical: scene-z is measured from the domain ground reference
     (`min(DEM)`), matching the voxelizer. `z_a = anchor_elevation − dem_min_abs
     + move_z`. A model point at local z=0 lands at scene height `z_a`.

4. **Meters → voxel index.** Divide by `meshsize`
   (invariant: `voxel[i,j,k] ↔ scene (i·meshsize, j·meshsize, k·meshsize)`).

Composite:
`M = Scale(1/meshsize) · Translate(u_a+move_u, v_a+move_v, z_a) · Rot_z(θ_net) · Scale(unit_scale) · Translate(−anchor_model_point)`

**Vertical datum fallback:** if the absolute DEM min is unrecoverable (e.g. an
`.h5`-loaded model), seat the anchor on the DEM value at the anchor's `(i,j)`
cell and treat `anchor_elevation` as height-above-that-ground. Detect and warn.

**Grid geometry source:** derive `origin`, `u_vec`, `v_vec`, `adj_mesh`,
`grid_size` from `compute_grid_geometry(rectangle_vertices, meshsize)`.
`rectangle_vertices` is read from `voxcity.extras['rectangle_vertices']`, which
the pipeline stores and which **persists through `.h5` save/load** (serialized in
`extras_json`). This works for rotated domains too (axis-aligned `bounds` alone
would not). If `rectangle_vertices` is absent (older/hand-built objects), raise a
clear error instructing the user to regenerate or supply it.

## 6. Mesh voxelization (`voxelize.py`)

Input: a placed mesh (already in voxel-index space via `M`). Output: set of
occupied integer `(i, j, k)` cells.

**Default backend — trimesh:**
1. Apply `M` so 1 unit = 1 voxel, grid origin = index origin.
2. Voxelize at `pitch=1.0` aligned to the integer lattice
   (`trimesh.voxel.creation.voxelize`), then `solid_fill` the interior to get
   solid masses (not hollow shells).
3. Collect occupied indices; drop cells outside `(nx, ny)` with a warning
   reporting how many were clipped.

**Watertightness handling (Rhino meshes are often not closed):**
- Watertight → solid fill works directly.
- Not watertight → attempt `trimesh` repair (fill holes, fix normals).
- Still open → fall back to a **z-ray column-fill**: for each `(i,j)` column,
  intersect a vertical ray with the mesh and fill between entry/exit pairs.
  Emit one summarizing warning. This matches VoxCity's per-column nature.

**Optional backend — meshlib (lazy import):**
- Selected only via `backend="meshlib"`. SDF-based `meshToVolume` voxelization;
  more robust/faster on imperfect CAD meshes.
- If not installed → actionable `ImportError` ("install `meshlib` or use
  backend='trimesh'"). Never imported otherwise (keeps default install
  MIT-clean; MeshLib is non-commercial-licensed).

**Per-building identity:** each OBJ object/group is voxelized separately so its
cells carry that building's identity. Unnamed geometry → `imported_building_1…N`.

**Resolution note:** voxelizing at `meshsize` loses sub-voxel architectural
detail — inherent to voxel models. No separate downsampling pass.

## 7. Integration into the VoxCity object (`integrate.py`)

1. **Grow Z if needed.** If `max_k` of imported cells exceeds
   `voxels.classes.shape[2]`, pad the array with air (0) along Z. X/Y are fixed
   by the domain (out-of-XY cells already clipped).

2. **Stamp building voxels.** Set `voxels.classes[i,j,k] = building_value` (−3,
   `BUILDING_CODE`).
   - `overwrite=True` (default): imported buildings overwrite whatever is there;
     log a count of collisions with existing non-air voxels.
   - `overwrite=False`: fill only air cells (imported building yields to existing
     geometry).
   - Terrain/ground below is left untouched (no carving/backfill).

3. **Update metadata grids:**
   - `building_id_grid` (2D): for each touched `(i,j)` column, assign a **new**
     ID = `current_max_id + n`, unique per OBJ object/group; never reuse existing
     IDs. Written where `overwrite` allows.
   - `building_height_grid` (2D): set to the imported building's **top height in
     meters** at that column = `max_k_in_column × meshsize` (relative to the
     column's ground).
   - `buildings.min_heights` (per-cell segment lists): append the imported
     building's vertical span(s) per column. Multiple `[min, max]` segments per
     column are supported (e.g. an arch with a gap).
   - `building_gdf`: **not** synthesized. Strictly footprint-GDF-only features
     won't see imported buildings; voxel- and grid-based ones will.

4. **Provenance.** Record `voxcity.extras['imported_buildings']` — a manifest of
   source file, anchor, rotation, move, units, backend, and per-building
   id↔name map — for reproducibility and inspection.

## 8. Rhino export guide (user-facing docs + docstring)

1. **Model buildings as closed solids/meshes.** Watertight geometry voxelizes
   most reliably.
2. **Organize by building** — one layer/named object per building. Each OBJ
   object/group → one building identity (ID + name).
3. **Establish the anchor:** pick one identifiable point; record its model
   coordinates (`anchor_model_point`), its real-world lon/lat + elevation
   (`anchor_lonlat`, `anchor_elevation`), and the **rotation** so model +Y =
   true north (0 if already north-aligned). Docs include a +Y-north diagram.
4. **Check units** (Rhino `Units`) → pass `units=...`.
5. **Export OBJ** (`File ▸ Export Selected ▸ .obj`): mesh (tris or quads), keep
   **Rhino Z-up** (VoxCity assumes Z = vertical; `swap_yz`/`z_up` available for
   Y-up exports), export **with object names/groups**.
6. **(Optional) MeshLib:** pass `backend="meshlib"` if installed — note the
   non-commercial license.

Docs include a minimal end-to-end snippet (`get_voxcity` →
`add_buildings_from_obj` → `export_obj`/visualize) so users verify placement
visually and iterate on `rotation`/`move`.

## 9. Error handling

- Missing/unreadable OBJ → clear `FileNotFoundError`/`ValueError` naming path.
- Empty mesh / no faces → `ValueError`.
- `backend="meshlib"` not installed → actionable `ImportError`.
- Invalid `units`/`anchor`/`rotation` → validated up front with descriptive
  messages.
- Anchor outside domain → warn (not error).
- All geometry clipped out of domain (0 in-bounds cells) → loud warning with the
  computed anchor `(i,j)` and bbox (usually a bad anchor/rotation/units).
- Collisions under `overwrite=True` → info-level count.
- Non-watertight mesh → repair, then column-fill fallback, with one warning.
- Vertical datum unrecoverable → warn and use anchor-on-DEM-cell seating.

## 10. Testing (TDD, pytest, matching existing `tests/` style)

- **Transform unit tests:** synthetic anchor/rotation/move/units → known model
  points land at expected `(i,j,k)`; axis-aligned and rotated domains; non-1 m
  meshsizes; ft/mm units.
- **Voxelization tests:** unit cube and L-shape → assert occupied-cell
  counts/extents; open (non-watertight) box → fallback fills solid.
- **Integration tests:** tiny base VoxCity (flat DEM) + small OBJ → BUILDING
  voxels at right cells; `building_id_grid` new non-overlapping IDs;
  `building_height_grid` matches; Z grows when building taller than base;
  `overwrite=False` yields to existing geometry.
- **Datum test:** nonzero/sloped DEM → building base seats at correct k.
- **Error-path tests:** missing file, empty OBJ, meshlib-not-installed,
  all-clipped.
- **Backend parity (optional):** `@pytest.mark.skipif` on meshlib absence; same
  cube voxelizes equivalently under both backends.
- **Fixtures:** tiny `.obj` files generated in-test via trimesh (cube, L-shape,
  open box) — no large binary assets committed.

## 11. Dependencies

- **trimesh** — already a dependency; no change.
- **meshlib** — optional, never installed/imported unless `backend="meshlib"`.
  Document its non-commercial license clearly. Not added to required deps.

## 12. Open questions / future work (not in this iteration)

- Native `.3dm` import (preserves layers/blocks; adds `rhino3dm` dep).
- Footprint+height extraction mode as an alternative to mesh voxelization.
- `building_gdf` synthesis for full footprint-feature parity.
- Material/color mapping from OBJ to VoxCity classes.
- A plugin/importer-family extraction if SketchUp/IFC importers are later added.
