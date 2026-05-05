# Building Surface Zones - Design Spec

**Date:** 2026-05-05
**Status:** Draft (pre-implementation)
**Scope:** `app/` FastAPI backend and React frontend, with small helper use from `src/voxcity/geoprocessor/mesh.py`

---

## 1. Goal

Extend the existing Zoning tab so users can define zones from building vertical
and horizontal surfaces, not only 2D horizontal map footprints.

The primary workflow is:

1. Generate a VoxCity model.
2. Open the Zoning tab.
3. Create a `Building surfaces` zone row.
4. Click buildings in the 3D viewer to add or remove whole buildings from the
   active zone.
5. Enter refine mode to replace or narrow a selected building to roofs, walls,
   directional walls, or specific clicked faces.
6. Run a building-surface Solar, View, or Landmark simulation and see stats for
   those surface zones.

Building-surface zones are independent of simulation runs. Users can define them
immediately after model generation because the Zoning tab uses a dedicated
selection mesh, not the latest simulation overlay.

## 2. Non-goals

- Mixing 2D horizontal footprint geometry and building-surface selectors inside
  one logical zone row.
- Using building-surface zones for ground-target simulation statistics.
- Importing or exporting zones to external GIS formats.
- Persisting zones across page reloads or backend restarts.
- Editing raw mesh vertices or changing building geometry from the Zoning tab.
- Preventing overlaps between zones. Overlap is allowed and each zone is
  evaluated independently.
- Replacing the current three-column Zoning tab with a separate 3D-first
  inspector workspace.

## 3. User Decisions Captured

| Topic | Decision |
| --- | --- |
| Default picking granularity | Whole building by click. |
| Active zone behavior | Clicks add or remove from the active zone row. |
| Refine mode | Bulk controls plus direct clicking of specific surfaces/faces. |
| Ground-target behavior | Building-surface zones show no data for ground simulations. |
| Overlap | Same building or face can belong to multiple zones. |
| Zone type mixing | 2D horizontal and 3D building-surface zones stay separate. |
| Surface mesh availability | Dedicated selectable building-surface mesh, independent of simulations. |

## 4. Existing Context

The current app already has most of the pieces needed for this design:

- `ZoningTab.tsx` owns the 2D zone editor UI and grouped zone list.
- `types/zones.ts` defines frontend zone records and grouping helpers.
- `SceneViewer` renders static voxel geometry, optional simulation overlays,
  zone outlines, and supports click picking through `Picker`.
- `LandmarkTab.tsx` already demonstrates building click selection with direct
  `buildingId` metadata and a `/api/buildings/at` fallback.
- `scene_geometry.py` builds Three.js `MeshChunk` payloads and already carries
  `face_to_building` metadata for building simulation overlays.
- `zoning.py` computes ground and building-surface stats from cached simulation
  results, including area-weighted means for building faces.
- `src/voxcity/geoprocessor/mesh.py:create_voxel_mesh` can build a building mesh
  with per-face building IDs and provided face normals. This is the right source
  for the dedicated surface selection mesh.

## 5. Design Summary

Use Approach A: extend the current three-column Zoning tab.

The left panel gains a zone type control and surface refinement controls. The
center panel remains the 2D map editor for horizontal zones and becomes a
context-only building footprint map for building-surface zones. The right panel
becomes an interactive building-surface picker when the active zone type is
`building_surface`.

Zones become a discriminated union:

- `horizontal` zones keep the existing `ring_lonlat` polygon data.
- `building_surface` zones store compact surface selectors.

Backend stats also branch by zone type:

- Horizontal zones use the current polygon/cell or polygon/face-centroid paths.
- Building-surface zones filter cached building simulation meshes by selector
  metadata: building ID, surface kind, wall orientation, and face key.

## 6. Frontend Data Model

### 6.1 Zone Types

`app/frontend/src/types/zones.ts` should evolve from one `Zone` interface into a
discriminated union while preserving shared UI metadata.

```ts
export type ZoneType = 'horizontal' | 'building_surface';
export type ZoneShape = 'rect' | 'polygon';

export interface BaseZone {
  id: string;
  groupId?: string;
  name: string;
  color: string;
  type: ZoneType;
}

export interface HorizontalZone extends BaseZone {
  type: 'horizontal';
  shape: ZoneShape;
  ring_lonlat: [number, number][];
}

export interface BuildingSurfaceZone extends BaseZone {
  type: 'building_surface';
  selectors: SurfaceSelector[];
}

export type Zone = HorizontalZone | BuildingSurfaceZone;
```

Existing helper functions such as `makeZoneId`, `makeZoneGroupId`,
`nextZoneName`, `nextZoneColor`, `zoneGroupKeys`, and `hashZones` remain useful.
`hashZones` must include both polygon coordinates and surface selectors so stats
refresh when a surface selection changes.

### 6.2 Surface Selectors

Surface selectors intentionally store user intent compactly instead of storing a
large list of every triangle for common actions.

```ts
export type WallOrientation = 'N' | 'E' | 'S' | 'W';

export type SurfaceSelector =
  | { buildingId: number; mode: 'whole' }
  | { buildingId: number; mode: 'roof' }
  | { buildingId: number; mode: 'all_walls' }
  | { buildingId: number; mode: 'wall_orientation'; orientation: WallOrientation }
  | { buildingId: number; mode: 'faces'; faceKeys: string[] }
  | { buildingId: number; mode: 'exclude_faces'; faceKeys: string[] };
```

Selector normalization should run whenever the user edits a surface zone:

- If a building has `whole`, other positive selectors for that building are
  redundant and should be removed. `exclude_faces` is the one exception because
  it can subtract specific faces from a whole-building selection.
- If `all_walls` is present, directional wall selectors for the same building
  are redundant.
- Empty `faces` selectors should be removed.
- Empty `exclude_faces` selectors should be removed.
- `exclude_faces` is retained when it subtracts faces from `whole`, `roof`,
  `all_walls`, or directional wall selectors.
- Ordering should be stable by `buildingId`, `mode`, orientation, then face key
  so hashing and display are deterministic.

### 6.3 API Serialization

Frontend state uses camelCase because the existing React code does. API payloads
use snake_case to match existing FastAPI models. The mapping must be explicit in
`app/frontend/src/api.ts`; `useZoneStats` should call this helper instead of
serializing `Zone` objects directly.

```ts
export interface SurfaceSelectorDto {
  building_id: number;
  mode: SurfaceSelector['mode'];
  orientation?: WallOrientation | null;
  face_keys?: string[] | null;
}

export interface ZoneSpecDto {
  id: string;
  group_id?: string | null;
  name: string;
  type: ZoneType;
  ring_lonlat?: [number, number][] | null;
  selectors?: SurfaceSelectorDto[] | null;
}

export function toZoneSpecDto(zone: Zone): ZoneSpecDto;
```

The inverse mapping is not needed for v1 because zones are not persisted by the
backend.

## 7. Backend Data Model

### 7.1 API Models

`app/backend/models.py` should mirror the frontend discriminated union. Pydantic
models can use a `type` field to branch validation.

```python
from typing import List, Literal, Optional

ZoneType = Literal["horizontal", "building_surface"]
SurfaceSelectorMode = Literal[
    "whole",
    "roof",
    "all_walls",
    "wall_orientation",
    "faces",
    "exclude_faces",
]
WallOrientation = Literal["N", "E", "S", "W"]

class SurfaceSelector(BaseModel):
    building_id: int
    mode: SurfaceSelectorMode
    orientation: Optional[WallOrientation] = None
    face_keys: Optional[List[str]] = None

class ZoneSpec(BaseModel):
    id: str
    group_id: Optional[str] = None
    name: str
    type: ZoneType = "horizontal"
    ring_lonlat: Optional[List[List[float]]] = None
    selectors: Optional[List[SurfaceSelector]] = None
```

Validation rules:

- `type == "horizontal"` requires `ring_lonlat` with at least three points.
- `type == "building_surface"` requires `selectors`, which may be empty for a
  pending row but produce no stats.
- `orientation` is required only for `wall_orientation`.
- `face_keys` is required only for `faces` and `exclude_faces`.
- `orientation` and `face_keys` must be absent or null for modes that do not use
  them.

`ZoneStatsRequest` uses `group_id` to validate logical groups before computing
stats. A missing `group_id` means the zone is its own group, matching current
frontend behavior.

### 7.2 Selection Metadata

Each selectable building-surface face needs metadata with the same indexing
order as the mesh chunk triangles returned to the frontend.

```python
class SurfaceFaceMeta(BaseModel):
    face_key: str
    building_id: int
    surface_kind: str       # "roof" | "wall" | "bottom" | "other"
    orientation: Optional[str] = None  # "N" | "E" | "S" | "W" for walls
```

The frontend only needs `roof` and `wall` for selection controls. `bottom` and
`other` are retained so the backend can explicitly avoid accidental matching.

Face keys should be stable within one generated model and recomputable when a
simulation mesh has the same underlying building mesh topology. A practical key
is based on source mesh face index plus quantized centroid and normal, for
example:

```text
b{building_id}:c{x}_{y}_{z}:n{nx}_{ny}_{nz}:i{face_index}
```

The exact format is internal; it must be deterministic for a model and identical
between the selection mesh and the building simulation mesh built from the same
`create_voxel_mesh(..., mesh_type='open_air')` call pattern.

The canonical owner for face-key construction should be one backend helper, for
example `make_surface_face_key(meta_input)`, used by `/api/buildings/surfaces`,
simulation metadata attachment, and tests. The frontend treats face keys as
opaque strings.

### 7.3 Metadata Invariants For Simulation Meshes

Selector-based stats require the cached building simulation mesh to expose the
same logical surface metadata as the selection mesh. This is an explicit backend
contract, not a best-effort frontend assumption.

For every building-target Solar, View, or Landmark result cached in `AppState`,
the backend must ensure one of these is true before `store_sim_result` completes:

1. The returned simulation mesh already has face-aligned `SurfaceFaceMeta`
  metadata.
2. The backend reattaches `SurfaceFaceMeta` by classifying the returned mesh's
  faces and matching its face count/order to a freshly built open-air building
  mesh for the current model.
3. The backend reconstructs equivalent metadata directly from the returned
  mesh's face centers, normals, and `building_id`/`building_face_ids` metadata.

The concrete storage contract is `mesh.metadata["surface_face_meta"]`: a
face-aligned list of plain dictionaries matching `SurfaceFaceMeta`, with length
equal to `len(mesh.faces)`. `mesh.metadata["surface_face_meta_version"]` should
be set to `1` so future changes can be detected. `SimulationResultCache` does
not need a new field in v1; the stats path reads metadata from `cached.mesh`.
If a simulation function returns a new mesh object, the caller is responsible for
copying or reconstructing this metadata onto that returned mesh before caching.

The stats path must not silently reinterpret a building-surface zone as a 2D
footprint if metadata is unavailable. If no reliable face metadata can be
attached to a cached building simulation mesh, building-surface zones return
zero-count/no-data stats and the backend logs enough context to diagnose which
simulation type lacked metadata.

Exact face-key matching depends on the same key algorithm being used for both
selection and simulation meshes. If a simulation function returns a mesh with a
different topology from the selection mesh, bulk selectors (`whole`, `roof`,
`all_walls`, `wall_orientation`) can still work from classified metadata, while
exact `faces` selectors only match when their keys can be reconstructed.

## 8. Backend Endpoints

### 8.1 `GET /api/buildings/surfaces`

Returns selectable building-surface geometry independent of simulations.

Response shape:

```ts
interface BuildingSurfaceGeometryResponse {
  chunk: MeshChunkDto;
  face_to_surface: SurfaceFaceMeta[];
  buildings: BuildingInfo[];
}
```

`BuildingInfo` reuses the existing frontend/backend shape returned by
`/api/buildings/list`: `id`, `cx`, `cy`, `cz`, and `top_z`.

Implementation outline:

1. Require a generated model.
2. Build the open-air building mesh with `create_voxel_mesh`:
   - `voxel_data = app_state.voxcity.voxels.classes`
   - `class_id = -3`
  - `meshsize = app_state.meshsize`
  - `building_id_grid = app_state.voxcity.buildings.ids`
   - `mesh_type = 'open_air'`
3. Convert the mesh into one `MeshChunk` or triangle-soup chunk for the
   frontend.
4. Derive `surface_kind` from normals:
   - `nz > 0.5`: `roof`
   - mostly horizontal downward normals: `bottom`
   - vertical normals: `wall`
   - otherwise: `other`
5. Derive wall orientation from the XY normal. Scene convention is X=east,
   Y=north, Z=up. The dominant cardinal direction maps to `E`, `W`, `N`, or
   `S`.
6. Return `face_to_surface` aligned to the selectable mesh face order.

If the model has no building mesh, return an empty chunk and empty metadata
instead of raising a server error.

The empty chunk must still satisfy the `MeshChunk` DTO shape:

```json
{
  "name": "building_surfaces",
  "positions": [],
  "indices": [],
  "color": null,
  "colors": null,
  "opacity": 0.0,
  "flat_shading": false,
  "metadata": {}
}
```

The frontend should refetch this endpoint whenever the same `geometryToken` used
by `SceneViewer` changes after model generation or edits.

### 8.2 `POST /api/zones/stats`

The existing endpoint remains the stats entry point but accepts the expanded
`ZoneSpec` model.

Behavior by cached simulation target:

| Cached target | Horizontal zones | Building-surface zones |
| --- | --- | --- |
| `ground` | Existing ground stats path | Empty/no-data stats |
| `building` | Existing polygon-to-building-face path | Selector-filtered building stats |

This preserves current horizontal-zone behavior while adding the requested
surface-zone behavior.

## 9. Backend Selector Matching

Add pure helpers to `app/backend/zoning.py` or a new focused module imported by
`zoning.py`.

Suggested units:

- `classify_surface_faces(mesh) -> list[SurfaceFaceMeta]`
- `surface_selector_mask(face_meta, selector) -> np.ndarray`
- `surface_zone_mask(face_meta, selectors) -> np.ndarray`
- `stats_for_surface_zone(zone_id, values, areas, mask) -> ZoneStat`

Matching rules:

- `whole`: all faces with matching `building_id` and selectable surface kind.
- `roof`: matching `building_id` and `surface_kind == "roof"`.
- `all_walls`: matching `building_id` and `surface_kind == "wall"`.
- `wall_orientation`: matching `building_id`, `surface_kind == "wall"`, and
  matching orientation.
- `faces`: matching `building_id` and `face_key in face_keys`.
- `exclude_faces`: matching `building_id` and `face_key in face_keys`, applied
  after positive selectors as a subtraction.

Positive selectors in one zone combine by OR. `exclude_faces` selectors combine
by OR and are subtracted from the positive mask. If the same face matches more
than one positive selector, it is counted once for that zone. Different zones
can match the same face independently.

Selectable surface kinds are exactly `roof` and `wall`. `bottom` and `other`
metadata values can be returned for transparency and diagnostics, but `whole`,
bulk selectors, exact face includes, and exact face excludes only operate on
`roof` and `wall` faces. This keeps the feature aligned with the requested
horizontal and vertical building surfaces.

Example building-surface zone DTO:

```json
{
  "id": "z_surface_1",
  "group_id": "g_surface_1",
  "name": "Zone 1",
  "type": "building_surface",
  "selectors": [
    { "building_id": 12, "mode": "whole" },
    { "building_id": 18, "mode": "roof" },
    { "building_id": 18, "mode": "exclude_faces", "face_keys": ["b18:c10_20_30:n0_0_1:i42"] }
  ]
}
```

When this zone is evaluated against a ground-target simulation, the corresponding
stat row is no-data by design:

```json
{
  "zone_id": "z_surface_1",
  "cell_count": 0,
  "valid_count": 0,
  "mean": null,
  "min": null,
  "max": null,
  "std": null
}
```

Building-surface means stay area-weighted. Min, max, and standard deviation use
the selected finite values, matching current behavior.

## 10. Frontend UI Behavior

### 10.1 Zoning Tab Layout

Keep the current three-column layout:

1. Left panel: zone controls and zone list.
2. Center panel: 2D editor or context map.
3. Right panel: 3D viewer and surface picker.

### 10.2 Left Panel

Add a `Zone type` segmented control:

- `2D area`
- `Building surfaces`

The selected type controls what `+ Add a zone` creates and what interactions are
enabled.

Zone rows show type and counts:

- Horizontal: `2 polygons` or `pending`.
- Building surfaces: `3 buildings`, `roof + east wall`, `12 faces`, or
  `pending`.

Clicking a zone row makes it active. Building and surface clicks always mutate
the active building-surface zone row.

### 10.3 Center Panel

For horizontal zones:

- Keep the current `PlanMapEditor` draw behavior.
- Existing map-drawn zones continue to render as `paint_zone` overlays.

For building-surface zones:

- Disable drawing interactions.
- Show building footprints and selected-building highlights for context.
- Footprint highlighting is a UI aid only; correctness depends on the 3D surface
  highlight and backend selector metadata.

### 10.4 Right Panel

For horizontal zones:

- Keep the current 3D preview with zone outlines.

For building-surface zones:

- Render the dedicated `/api/buildings/surfaces` selection mesh on top of the
  static city geometry.
- Enable `SceneViewer` picking.
- Normal click toggles whole-building membership in the active zone.
- Refine mode scopes editing to one selected building and exposes controls:
  - `Whole`
  - `Roof`
  - `All walls`
  - `N`, `E`, `S`, `W`
  - `Clear building`
- In refine mode, direct surface/face clicks toggle exact `faces` selectors for
  the refined building.

Refine mode is a small UI state machine owned by `ZoningTab`:

| State | Meaning | Entry | Exit |
| --- | --- | --- | --- |
| `idle` | Whole-building click toggles the active zone. | Default. | Enter refine from a selected-building chip or a picked building. |
| `refining(buildingId)` | Bulk controls and face clicks edit only that building in the active zone. | User clicks `Refine` on a building chip, or clicks `Refine` then picks a building. | `Done`, switching active zone, deleting the building from the zone, or changing zone type. |

Refine editing semantics:

- In `idle`, clicking a building with no selector in the active zone adds
  `{ mode: 'whole' }` for that building.
- In `idle`, clicking a building that already has any selector in the active
  zone removes all selectors for that building. This makes whole-building click
  behavior a true add/remove toggle even for refined buildings.
- To replace a refined building with a whole-building selector without removing
  it first, enter refine mode and use `Whole`.

- `Whole` replaces all selectors for the refined building with `{ mode: 'whole' }`.
- `Clear building` removes all selectors for the refined building.
- `Roof`, `All walls`, and directional wall buttons toggle their selectors. If
  the building currently has `whole`, toggling a narrower selector first removes
  `whole` so the result is genuinely refined.
- Exact face clicks toggle membership in a `faces` selector for the refined
  building when the clicked face is not already selected by a bulk selector.
- If the clicked face is already selected by `whole`, `roof`, `all_walls`, or a
  directional wall selector, the click toggles an `exclude_faces` override for
  that face. This makes direct face clicks able to remove individual faces from
  a bulk selection.
- If an excluded face is clicked again, the exclusion is removed.
- Selector normalization removes redundant narrower selectors when `whole` or
  `all_walls` makes them unnecessary.

Selected surfaces render as translucent colored overlays using the active zone
color. Hover preview is optional polish and is not required for the first
implementation plan.

## 11. Scene Viewer Changes

`SceneViewer` and `Picker` need a small generalization.

Current `PickResult` returns `target`, optional `cell`, optional `buildingId`, and
world point. Extend it with optional surface metadata:

```ts
interface PickResult {
  target: 'ground' | 'building';
  cell: [number, number] | null;
  buildingId: number | null;
  point: [number, number, number];
  surface?: SurfaceFaceMeta | null;
}
```

`MeshLayer` already accepts `userData`. The selectable surface mesh can attach
`faceToSurface` to `userData`, and `Picker` can read it by `faceIndex` just as it
does for `faceToCell` and `faceToBuilding`.

`SceneViewer` should accept optional surface-selection props rather than making
Zoning-specific business logic part of the core renderer. The prop boundary is:

```ts
interface SurfaceSelectionLayerProps {
  surfaceChunk: MeshChunkDto | null;
  faceToSurface: SurfaceFaceMeta[];
  activeZoneColor: string | null;
  selectedSelectors: SurfaceSelector[];
  hoverSurface?: SurfaceFaceMeta | null;
  enabled: boolean;
}
```

Ownership is split deliberately:

- `ZoningTab` fetches `/api/buildings/surfaces`, owns active-zone state,
  computes `selectedSelectors`, and handles `onPick` mutations.
- `SceneViewer` only renders optional surface-selection geometry and forwards
  picked metadata through `onPick`.
- A focused renderer component, for example `SurfaceSelectionLayer`, converts
  `selectedSelectors` plus `faceToSurface` into highlight material or overlay
  geometry. It does not mutate zones or call APIs.

This keeps the renderer reusable and keeps zoning rules in the Zoning tab.

## 12. Zone Statistics Display

`useZoneStats` should send the full zone DTO, not only `id`, `name`, and
`ring_lonlat`.

Aggregation by `groupId` remains only for horizontal multi-polygon zones. A
building-surface logical zone is represented as one `BuildingSurfaceZone` record
whose `selectors` array contains all selected buildings and surfaces. This
avoids incorrect frontend aggregation of already area-weighted surface means.

Validation rules:

- Frontend normalization must prevent a `groupId` from containing both
  `horizontal` and `building_surface` members.
- If a user action would mix types, create a new group for the new zone type
  instead of attaching it to the active group.
- Backend validation defensively rejects a request where one logical group
  contains mixed zone types. The backend groups by `group_id` when present and
  by `id` otherwise. This catches malformed clients and prevents stats
  aggregation from combining unrelated scopes.
- Backend validation also rejects multiple `building_surface` records with the
  same `group_id`; those selectors must be merged into one submitted zone.

`ZoneStatsTable` can keep the same columns in the first iteration. For
building-surface stats, the current `cells` column label is acceptable because
the backend already uses `cell_count` for cells or faces. A later UI polish can
rename it dynamically to `faces` when `stats.target === 'building'`.

No-data rows are expected when:

- A building-surface zone is shown after a ground simulation.
- A pending surface zone has no selectors.
- Stale selectors no longer match the cached simulation mesh.

`/api/zones/stats` returns one raw stat per submitted zone id. Frontend
aggregation by `groupId` remains in `useZoneStats` for horizontal zones only, as
it does today. Backend `group_id` validation exists only to reject malformed
mixed-type or multi-record building-surface logical groups, not to aggregate
response rows.

## 13. Error Handling And State Reset

Selections are model-specific. When the target rectangle changes or a new model
is generated, all zones should be cleared alongside current cached simulation
state. When the existing model is edited, horizontal zones can be preserved, but
building-surface zones should be cleared because face keys and building IDs may
no longer match the rebuilt geometry.

If `/api/buildings/surfaces` returns no faces:

- Disable `Building surfaces` interactions.
- Show a normal info or warning alert in the Zoning tab.
- Keep existing horizontal-zone functionality available.

If a picked 3D object has no surface metadata:

- For whole-building mode, fallback to `/api/buildings/at` with the world point,
  matching the Landmark tab behavior.
- For refine mode, ignore the exact-face toggle and leave the current selection
  unchanged.

If a selector no longer matches any face while computing stats:

- Ignore that selector and continue evaluating the other selectors in the same
  zone.
- Return a zero-count/no-data `ZoneStat` only when no positive selected face
  remains after all matching and exclusions for that zone.
- Do not fail the entire request.

## 14. Testing Strategy

### 14.1 Backend Tests

Add focused tests for pure selector and stats helpers using small synthetic
meshes and metadata arrays:

- Whole-building selector matches all selectable faces for a building.
- Roof selector matches only upward roof faces.
- All-walls selector excludes roofs.
- Directional wall selector matches only the requested wall orientation.
- Exact face selector matches only requested face keys.
- Exclude-face selector subtracts faces from a whole-building or roof/wall
  selection.
- Multiple selectors combine with OR and do not double-count faces.
- Overlapping zones can produce independent stats over the same face.
- Empty or stale selectors return zero-count/no-data stats.
- Ground-target stats return no data for building-surface zones.

Add endpoint-level tests where existing backend test patterns make that cheap:

- `/api/buildings/surfaces` returns aligned `chunk` and `face_to_surface`
  lengths for a model with buildings.
- `/api/zones/stats` accepts a request containing horizontal zones and
  building-surface zones in separate logical groups, and returns one stat per
  zone. A request that mixes zone types inside the same `group_id` is rejected.

### 14.2 Frontend Tests

Add tests where the existing test setup supports React components:

- Creating a `Building surfaces` zone row stores `type: 'building_surface'`.
- Whole-building click toggles the active zone's selector.
- Refine controls normalize selectors correctly.
- Exact face click adds or removes a `faces` selector.
- Clicking a bulk-selected face creates or removes an `exclude_faces` selector.
- `useZoneStats` serializes both horizontal and surface zones.
- Zone list count labels handle whole buildings, roof/wall selectors, and faces.

### 14.3 Manual Verification

Manual checks should include:

- Generate a model with buildings.
- Create a horizontal zone and confirm current behavior still works.
- Create a building-surface zone by clicking whole buildings.
- Refine one building to roof and one directional wall.
- Run Solar/View/Landmark building-surface simulations and confirm zone stats
  update.
- Run a ground-target simulation and confirm building-surface zones show no
  data rather than footprint-derived values.
- Run `npx tsc --noEmit` in `app/frontend`.
- Verify a non-default mesh size still aligns the selectable building-surface
  mesh with the static scene.

## 15. Implementation Boundaries

The implementation should be split into units with clear responsibilities:

- Zone type definitions and normalization in `types/zones.ts`.
- Surface selection API models in `models.py`.
- Surface metadata classification and selector masks in backend pure helpers.
- Selectable surface mesh endpoint in `main.py` or a small endpoint helper.
- Surface picking and highlight rendering in frontend Three.js components.
- Zoning tab UI integration in `ZoningTab.tsx`.
- Stats serialization and display updates in `useZoneStats` and
  `ZoneStatsTable`.
- API DTO conversion helpers in `api.ts`, including camelCase-to-snake_case
  mapping for zones and selectors.

Avoid coupling surface-selection behavior to Solar, View, or Landmark tabs.
Those tabs should continue to consume zones through the existing shared zone
stats mechanism.
