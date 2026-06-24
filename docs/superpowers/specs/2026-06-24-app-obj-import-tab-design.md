# Design: OBJ Import Tab for the VoxCity App

**Date:** 2026-06-24
**Status:** Approved (brainstorming)
**Scope:** Add a dedicated "Import" tab to the VoxCity web app (`app/`) that imports
Rhino/OBJ buildings into the current in-memory VoxCity model, wrapping the existing
`voxcity.importer.add_buildings_from_obj` engine.

## Goal

Let a user, after generating a base VoxCity model, upload an OBJ file (authored in
Rhino or similar), position it over the site, and stamp its buildings into the model
— with both **precise numeric** placement and an **interactive 3D gizmo**. The
imported buildings become first-class buildings (they update the 2D building grids,
so they survive re-voxelization and remain editable/deletable in the Edit tab).

## Why a new tab (not part of the Edit tab)

The Edit tab's model is: buffer small vector edits → batch-commit → mutate the 2D
component grids in place → `regenerate_voxels`. OBJ import is architecturally
different: `add_buildings_from_obj` performs true 3D mesh voxelization and returns a
**whole new VoxCity** that replaces `app_state.voxcity`. It also carries a larger UI
surface (file upload, per-group role table, dual numeric + interactive placement).
A dedicated tab fits the operation and gives the UI room, while still reusing the
existing map projection and 3D viewer infrastructure. Imported buildings remain
editable in the Edit tab afterward because the importer also updates the 2D grids.

## Decisions (from brainstorming)

- **New "Import" tab**, registered in `App.tsx` alongside Target Area / Generation /
  Edit / Solar / View / Landmark / Zoning / Export.
- **Multi-group OBJ + role routing**: each named OBJ group becomes a separate
  building; a per-group role table lets the user mark groups as `building` or `skip`
  (non-building, e.g. windows). Uses the importer's `load_obj_groups` /
  `classify_roles` / `select_building_groups`.
- **Placement: numeric form AND interactive gizmo**, sharing one placement-state
  object (single source of truth).
- **Interactive gizmo = 3D axis-arrow translate gizmo + rotation ring** on the live
  building preview, via three.js `TransformControls` in the R3F `SceneViewer`:
  - X arrow → `move.east`, Y arrow → `move.north`, Z arrow → `move.up`
  - rotation ring → `rotation`
  - `anchor_lonlat` set by an **initial click on the 2D map** (rough placement),
    `anchor_elevation` **auto-sampled from the DEM** at the anchor (manual override
    in Advanced). The gizmo then fine-tunes via `move` + `rotation`.
- **Auto-detect groups + role table; sensible defaults with an Advanced section**
  (units, `z_up`, `swap_yz`, anchor-elevation override, `overwrite`).
- **Live preview = visual/approximate (client-side); commit = authoritative
  (server-side).** No re-voxelization during drag.

## Architecture

### Coordinate convention

Per `src/voxcity/exporter/obj.py`, the app's 3D scene world frame is
**X = east (v), Y = north (u), Z = up (height)**, in metres, with the SOUTH_UP cell
layout used by `SceneViewer`. The gizmo therefore maps:
`X→move.east`, `Y→move.north`, `Z→move.up`. The exact axis sign/orientation is
verified against `SceneViewer`'s existing scene transform during implementation
(it must match how the static city geometry is already laid out).

### Backend (FastAPI — `app/backend/`)

Two new endpoints in `main.py`, models in `models.py`.

1. **`POST /api/model/import_obj/upload`** (multipart) — requires a base model.
   - Saves the uploaded `.obj` (+ optional `.mtl`) to a per-session temp dir under
     `BASE_OUTPUT_DIR/import_obj/<import_id>/`.
   - Parses via `load_obj_groups` (default `swap_yz=False`) and `classify_roles`.
   - Returns `ImportObjUploadResponse`:
     - `import_id: str` — token referencing the stored file (used by commit).
     - `groups: [{name, role, n_faces, bbox_model:[[xmin,ymin,zmin],[xmax,ymax,zmax]]}]`
     - `model_bounds:[[..],[..]]` — overall XYZ bounds in model units.
     - `preview`: lightweight geometry for the frontend:
       - `footprints`: per-group XY footprint rings in **model coordinates**
         (convex/outline polygon from projecting the group's vertices to XY), plus
         z-extents.
       - `mesh`: a **decimated** combined mesh (`vertices`, `indices`) in model
         coordinates, capped (e.g. ≤ ~20k triangles) for the 3D preview. Decimation
         only affects the preview; commit re-reads the original file.
   - Errors: invalid/empty OBJ → 400 (importer raises `ValueError`/`FileNotFoundError`);
     no base model → 400 (mirror `_require_model`).

2. **`POST /api/model/import_obj/commit`** — body `ImportObjCommitRequest`:
   ```
   {
     import_id: str,
     placement: {
       anchor_lonlat: [lon, lat],
       anchor_elevation: float | null,   # null → auto-sample DEM at anchor cell
       anchor_model_point: [x, y, z] = [0,0,0],
       rotation: float = 0.0,            # degrees
       move: [east, north, up] = [0,0,0],
       units: "m"|"cm"|"mm"|"ft"|"in" = "m",
       z_up: bool = true,
       swap_yz: bool = false
     },
     roles: { <group_name>: "building"|"skip"|<other> },
     overwrite: bool = true
   }
   ```
   - Resolves the stored file for `import_id` (404 if missing — server may have
     restarted; prompt re-upload).
   - If `anchor_elevation is null`: project `anchor_lonlat` → cell `(i,j)` using the
     same grid geometry as `/api/model/geo`, sample `app_state.voxcity.dem.elevation[i,j]`.
   - Calls `add_buildings_from_obj(app_state.voxcity, path, anchor_lonlat=...,
     anchor_elevation=..., anchor_model_point=..., rotation=..., move=..., units=...,
     roles=..., z_up=..., swap_yz=..., overwrite=..., backend="trimesh")`.
   - Replaces `app_state.voxcity = result`; calls `app_state.refresh_raw_cache()`.
   - Returns `ImportObjCommitResponse`: `figure_json` (via `_render_edit_preview`),
     `imported_building_ids: [int]`, `n_building_voxels_added: int`,
     `warning: str | null` (e.g. "voxelized to 0 cells inside the domain").
   - The role map is passed straight through to the importer's `roles` argument; any
     group mapped to a non-`building` role is skipped by `select_building_groups`.

No `regenerate_voxels` call here — the importer does its own voxelization and updates
the 2D grids, so the result is internally consistent and the new buildings persist
through later Edit-tab re-voxelization.

### Frontend (React/Vite — `app/frontend/src/`)

- **`tabs/ImportTab.tsx`** (new), registered in `App.tsx`. Guards on `hasModel`
  ("Generate a model first", mirroring EditTab). Three-column layout reusing the
  app's panel structure:
  - **Controls column:** drag-drop OBJ uploader → group/role table (role `<select>`
    per group: building / skip) → placement form (anchor lon/lat read-only-from-map
    or editable, rotation, move E/N/Up, units) → collapsible **Advanced**
    (`z_up`, `swap_yz`, anchor-elevation auto/override, `overwrite`) → **Import**
    button (+ status/warning area).
  - **2D map:** reuse `PlanMapEditor`. New mode: an `ObjPlacementLayer` overlay that
    draws the current footprint at the live placement and lets the user (a) click to
    set the initial `anchor_lonlat`, (b) optionally drag the footprint horizontally
    (mirrors the gizmo's X/Y). Footprint reprojects live via the existing `grid_geom`
    projection (`polygonToCells`/projector already in `lib/grid.ts`).
  - **3D viewer:** reuse the R3F `SceneViewer` to render the current city + a live
    **imported-mesh overlay** transformed by the placement, with a three.js
    `TransformControls` gizmo (translate X/Y/Z arrows + rotate ring) attached to it.
    Gizmo deltas update the shared placement state. On commit, the authoritative
    figure replaces the preview.
- **Shared placement state:** one `Placement` object in `ImportTab`; the numeric
  form, the 2D footprint, and the 3D gizmo all read/write it, so they stay in sync by
  construction.
- **TS preview transform** (`lib/objPlacement.ts`): a TypeScript port of the
  *visual* portion of `build_placement_transform` (anchor→uv_m, units scale,
  rotation + domain rotation, move) used only to position the preview footprint/mesh.
  It is explicitly an approximation; the committed voxelization uses the exact Python
  transform. Vertical (ground offset) is treated as a constant for preview.
- **`api.ts`:** `uploadImportObj(file): ImportObjUploadResponse`,
  `commitImportObj(req): ImportObjCommitResponse`. On successful commit, call the
  parent `onModelEdited` so Solar/View/Landmark/Export figures invalidate (same hook
  the Edit tab uses).

## Data flow

1. User uploads OBJ → backend parses (single source of truth for group names/roles) →
   returns `import_id`, groups+roles, preview geometry.
2. Frontend shows the role table; renders the footprint on the 2D map and the
   decimated mesh in the 3D viewer.
3. User clicks the 2D map to set the initial anchor; fine-tunes with the 3D
   axis-arrow gizmo (move E/N/Up) and rotation ring, and/or the numeric form. The TS
   preview transform repositions footprint + mesh live (no backend calls).
4. User clicks **Import** → `commit` → backend runs `add_buildings_from_obj` → new
   VoxCity replaces state → authoritative figure renders in 3D; `onModelEdited`
   invalidates downstream tabs.

## Error handling

- Invalid/empty OBJ on upload → 400 with the importer's message.
- No base model loaded → tab shows guard; endpoints return 400.
- Stale `import_id` (server restarted) → 404 → frontend prompts re-upload.
- Building voxelizes to 0 cells inside the domain → commit succeeds but returns a
  `warning`; frontend surfaces it as a non-fatal notice (placement likely off-site).
- Invalid `units` / malformed placement → 400 (importer + endpoint validation).

## Testing

- **Backend** (`app/backend/test_import_obj.py`, mirroring existing `test_*.py`):
  - upload returns groups with default roles + an `import_id`; invalid file → 400.
  - commit applies, replaces the model, assigns new building id(s), auto-samples
    elevation when `anchor_elevation` is null, and honors the role map (a `skip`
    group contributes no voxels).
  - commit with a placement that lands off-domain returns a `warning`.
  - The voxelization engine itself is already covered by the importer's 36 tests
    (`tests/importer/`), so these tests focus on the app wiring.
- **Frontend** (`src/lib/objPlacement.test.ts`, mirroring existing `*.test.ts`):
  - placement-state sync (gizmo delta ↔ numeric fields ↔ footprint) is consistent.
  - the TS preview transform matches expected positions for representative
    anchor/rotation/move/units inputs (sanity vs. hand-computed cases).

## New work, by size

1. **`ObjPlacementLayer`** (2D footprint overlay: initial-click anchor + optional
   horizontal drag, live reprojection) — moderate.
2. **3D gizmo preview** (R3F `SceneViewer` + imported-mesh overlay + `TransformControls`
   wiring to placement state) — largest single item.
3. **Two backend endpoints** + pydantic models + DEM elevation sampling + decimated
   preview geometry extraction — moderate.
4. **TS preview transform** port (`lib/objPlacement.ts`) — small/moderate.
5. **`ImportTab` shell** (uploader, role table, placement form, Advanced, API wiring,
   App.tsx registration) — moderate.

## Reuse

- Engine: `voxcity.importer.add_buildings_from_obj` (unchanged).
- Backend: `_require_model`, `_render_edit_preview`, the `/api/model/geo` grid
  geometry builder (for anchor→cell projection), the EPW multipart upload pattern,
  `app_state.refresh_raw_cache`.
- Frontend: `PlanMapEditor`, R3F `SceneViewer`, `lib/grid.ts` projection, guided
  components (`GuidedSection`, `ChoiceGroup`), `api.ts` patterns, `onModelEdited` hook.

## Out of scope (v1)

- Geometry-driven windows/glazing (importer "Path B"); roles only *skip* non-building
  groups for now (consistent with the importer's current behavior).
- The optional `meshlib` voxelization backend (commit always uses `backend="trimesh"`).
- Persisting uploaded OBJ files across server restarts / into saved sessions.
- Editing/moving a building *after* it has been committed (delete + re-import, or use
  the Edit tab's existing delete/height tools).
