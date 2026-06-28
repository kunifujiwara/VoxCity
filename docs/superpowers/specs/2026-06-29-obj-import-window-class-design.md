# Window class on imported OBJ building voxels

**Date:** 2026-06-29
**Status:** Approved (design)
**Related:** [2026-06-23-rhino-obj-building-import-design.md](2026-06-23-rhino-obj-building-import-design.md), [2026-06-24-app-obj-import-tab-design.md](2026-06-24-app-obj-import-tab-design.md)

## Summary

When importing an OBJ, window-role groups are currently **discarded**
(`select_building_groups` keeps only role `"building"`). This feature instead
voxelizes window geometry as a **glass skin** and stamps it into the voxel grid
as code **`-16`** (the existing "glass" class used by the palette,
`utils/material.py`'s `glass_id`, and the GPU renderer's dielectric material).

Buildings are voxelized solid (`-3`) first; window geometry then **recolors the
facade voxels it touches** to `-16`. At voxel resolution a facade cell is
*either* glass or wall, and glass wins where a window covers it — the wall
behind stays solid. This is purely a reclassification of already-solid cells.

Window groups are auto-detected by **group/layer name OR assigned material
name** (case-insensitive keyword match), with an explicit `roles` override.

## Goals

- Imported windows become a first-class semantic voxel class (`-16`) that is
  visible everywhere the palette is used (3D viewer, exports).
- Auto-detection minimizes manual per-group configuration: a group is treated
  as a window if its name *or* its material name matches a window keyword.
- Existing building-only imports are byte-for-byte unchanged (all new behavior
  is opt-in via defaults that preserve current results when no windows exist).

## Non-goals (explicit follow-ups)

- **Simulator wiring.** Solar/visibility simulators are *not* updated to treat
  `-16` as glass-but-occupied. `-16` is already a recognized glass code in the
  palette / `utils/material.py` / `renderer_gpu.py`, so *visualization* works,
  but simulator semantics (transmission, reflectance, view factors through
  glass) are a separate task.
- **Per-face material splitting.** Groups that *mix* glass and opaque faces in
  a single object are classified at the group level only (one role per group).
  Splitting a mixed group by per-face material is a documented future option,
  not implemented here.

## Background: current architecture

- `importer/loader.py`
  - `load_obj_groups(obj_path, swap_yz)` → `list[(name, trimesh.Trimesh)]`
    (loads with `split_objects=True, group_material=False`).
  - `classify_roles(names, roles=None)` → `{name: role}`, defaulting to
    `"building"`; explicit `roles` entries override (exact string match).
  - `select_building_groups(groups, roles=None)` → keeps role `"building"`,
    logs+drops everything else.
- `importer/rhino_obj.py::add_buildings_from_obj(...)` loads groups, selects
  building groups, builds a placement transform `M`, voxelizes each group
  (`importer/voxelize.py::voxelize_mesh`, vertical-ray column fill), and stamps
  via `importer/integrate.py::stamp_buildings` (writes `-3`, updates
  `buildings.ids` / `heights` / `min_heights`, appends an `imported_buildings`
  manifest entry).
- App: `POST /api/model/import_obj/upload` calls `classify_roles` and returns
  per-group `role`; the frontend `ImportTab` shows a per-group dropdown
  (today: **building / skip**). `POST /api/model/import_obj/commit` passes
  `roles` to `add_buildings_from_obj` and reports `n_building_voxels_added`.

Verified during design: with the importer's exact load options, trimesh exposes
the per-group material name as `mesh.visual.material.name` (e.g. a group
`WindowPane` with `usemtl GlassMat` → `'GlassMat'`). `ColorVisuals`/untextured
meshes have no `.material.name`, so access is guarded.

## Design

### 1. Role routing (`importer/loader.py`)

Add a guarded material-name reader:

```python
def group_material_name(mesh) -> str | None:
    """Return the group's assigned OBJ material name, or None.

    Reads ``mesh.visual.material.name`` when present (TextureVisuals);
    returns None for ColorVisuals / untextured meshes or any missing attr.
    """
```

Window keyword set (module constant, English-only default):

```python
DEFAULT_WINDOW_KEYWORDS = ("window", "glass", "glazing")
```

Extend `classify_roles` with auto window detection and an optional parallel
material-name mapping:

```python
def classify_roles(
    names,
    roles=None,
    *,
    auto_window=True,
    window_keywords=DEFAULT_WINDOW_KEYWORDS,
    material_names=None,   # optional {name: material_name}
) -> dict[str, str]:
```

Resolution order per group `name`:

1. If `name` in `roles` → that explicit role (overrides everything).
2. Else if `auto_window` and (`name` OR `material_names.get(name)` contains any
   keyword, case-insensitive substring) → `"window"`.
3. Else → `"building"`.

Backwards compatible: with `material_names=None` and no window-named groups,
output is identical to today (everything `"building"`).

Replace the building-only selector with a role-grouping selector:

```python
def select_groups_by_role(groups, roles=None, *, auto_window=True,
                          window_keywords=DEFAULT_WINDOW_KEYWORDS) -> dict[str, list[(name, mesh)]]:
    """Return {"building": [...], "window": [...]} (other/unknown roles dropped+logged).

    Reads each mesh's material name internally (via group_material_name) so
    callers don't have to.
    """
```

Keep a thin `select_building_groups(...)` wrapper (returns the `"building"`
bucket) for back-compat with any existing callers/tests.

**Role vocabulary:** `building` / `window` / `skip`. Any other/unknown role is
treated as excluded (current behavior preserved), logged at INFO.

### 2. Window stamping (new `importer/windows.py`)

`stamp_windows(voxcity, window_groups, M, *, window_value=-16, skin_radius=1)`
runs **after** `stamp_buildings` has produced the solid `-3` model. Per window
group:

1. Copy the mesh and apply placement transform `M` → voxel-index space.
2. **Surface-voxelize** it: `trimesh.voxel.creation.voxelize_subdivide(mesh,
   pitch=1.0)` → the set of index cells the (possibly thin/vertical) pane passes
   through. (Robust to thin panes precisely because it rasterizes the *surface*,
   not a closed volume — unlike the column-fill `voxelize_mesh`.)
3. Recolor to `window_value` every cell `(i,j,k)` of `voxcity.voxels.classes`
   that **is a building (`-3`) cell** AND is within Chebyshev distance
   `skin_radius` (default 1) of a window-touched cell. The radius absorbs
   sub-voxel offsets between the pane plane and the wall surface.
4. Window-touched cells with **no** building cell within `skin_radius` are
   **counted and logged, then skipped** (no floating glass — keeps "replace
   outer skin" literal).

Return the number of cells recolored (per call / per group, for reporting).

**Metadata is untouched.** Glass cells are still solid occupancy, merely
reclassified, so `buildings.ids` / `heights` / `min_heights` (and their span
lists from `stamp_buildings`) remain valid and unchanged. A per-import window
voxel count is recorded in the latest `imported_buildings` manifest entry, e.g.
`manifest["n_window_voxels"]`.

Window groups do **not** receive their own building id; they recolor cells that
already belong to whatever building group claimed that column.

### 3. Public API (`importer/rhino_obj.py`)

`add_buildings_from_obj` gains keyword args (defaults preserve current behavior):

- `auto_window: bool = True`
- `window_keywords: tuple[str, ...] | None = None` (None → `DEFAULT_WINDOW_KEYWORDS`)
- `window_value: int = -16`

New flow:

1. `load_obj_groups(...)`.
2. `buckets = select_groups_by_role(groups, roles, auto_window=..., window_keywords=...)`.
3. If no building groups → unchanged early-return (deep copy), as today.
4. Build transform `M`; voxelize + `stamp_buildings` the `"building"` bucket (`-3`).
5. `stamp_windows(out, buckets["window"], M, window_value=window_value)`.
6. Append/extend the manifest with `n_window_voxels` and continue (gridvis etc.).

If there are window groups but **no** building groups, log a warning and skip
window stamping (windows can only recolor existing building cells).

### 4. Web app

- **`models.py`:** add `n_window_voxels_added: int = 0` to
  `ImportObjCommitResponse`. (`ImportObjGroup.role` already exists.)
- **Upload endpoint (`main.py`):** build a `{name: material_name}` map from the
  loaded meshes via `group_material_name`, pass it to `classify_roles(...,
  material_names=...)`. Auto-detected windows (by name or material) surface with
  `role="window"`.
- **Commit endpoint (`main.py`):** count `-16` before/after
  (`np.sum(classes == -16)`) → `n_window_voxels_added`; include it in the
  response and in the success-message text. `roles` already passes through to
  `add_buildings_from_obj` unchanged.
- **Frontend `ImportTab.tsx`:** add a **`window`** option to the per-group role
  `<select>` (building / window / skip). Auto-detected windows preselect
  `window` (from the upload response `role`). Update the success message to
  mention window voxels, e.g. *"Imported N building(s); X voxel(s) added, Y
  window voxel(s)."*
- The committed 3D figure already renders `-16` via the palette → windows show
  as glass with **no renderer change**.

## Testing

- **`tests/importer/test_loader.py`**
  - Auto window detection by **group name** (`"Windows_01"` → `"window"`).
  - Auto window detection by **material name** (generic group name, `Glass`
    material → `"window"`), using `group_material_name`.
  - Explicit `roles` override beats auto-detection (both directions:
    force-building a glass group, force-window a plain group).
  - `skip` role still excluded; unknown role excluded+logged.
  - `material_names=None` / no window groups → identical to legacy output.
- **`tests/importer/test_windows.py`** (new)
  - Thin vertical pane coincident with a `-3` wall column recolors those cells
    to `-16`; building `ids`/`heights`/`min_heights` unchanged before/after.
  - Window cells with no nearby building cell are skipped and counted (returned
    count reflects only recolored cells).
  - `skin_radius` absorbs a 1-cell offset between pane and wall.
- **`tests/importer/test_add_buildings_from_obj.py`**
  - End-to-end: box building + separate window-named group → result contains
    both `-3` and `-16`; manifest has `n_window_voxels > 0`.
  - Building-only OBJ → no `-16` cells; result identical to pre-feature.
- **App (`app/backend/test_import_obj.py`)**
  - Upload of an OBJ with a glass-material group reports that group's
    `role == "window"`.
  - Commit returns `n_window_voxels_added > 0` for an OBJ with windows.

## Risks / edge cases

- **Untextured / `ColorVisuals` meshes:** `group_material_name` returns `None`;
  detection falls back to name-only. No crash.
- **Generic trimesh material names** (e.g. `material_0`): won't match keywords,
  so no false positives; name-based detection still applies.
- **Mixed-material groups:** classified by a single group-level material name
  (whichever trimesh retains). Acceptable per non-goals; per-face split is a
  future follow-up.
- **Windows but no buildings:** skipped with a warning (nothing to recolor).
- **`skin_radius` too large** could bleed glass onto non-facade cells; default
  `1` is conservative and only ever recolors existing `-3` cells.
