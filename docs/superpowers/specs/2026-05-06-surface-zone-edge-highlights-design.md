# Surface Zone Edge Highlights Design

Date: 2026-05-06

## Goal

Replace the current simulation-tab building-surface zone boundary display with a cleaner, architectural edge highlight. The new highlight should be shown only in Solar, View, and Landmark simulation viewers. It should represent the selected surface zone itself, use the voxel building model as the source, merge adjacent voxel faces into larger rectangles, remove diagonal triangle edges, and render only edges parallel to the scene X, Y, or Z axes.

Zoning remains unchanged: it keeps the translucent fill display used for surface-zone editing.

## Current Context

The current surface-zone rendering path uses `/api/buildings/surfaces`, which returns a non-indexed triangle-soup `MeshChunk` plus one `SurfaceFaceMeta` per triangle. `SurfaceSelectionLayer` derives boundary lines by counting triangle edges. This correctly removes internal edges between adjacent selected triangles, but the visual result still exposes too much of the triangle and voxel structure in simulation views.

The backend already has voxel-derived building geometry paths:

- `create_voxel_mesh()` builds exposed building faces from the voxel grid and building ID grid.
- `/api/buildings/surfaces` uses that mesh for selectable surface metadata.
- `/api/buildings/highlight` builds voxel-face highlight chunks for whole building selection.

The new edge highlight should use this voxel/building-ID source instead of deriving user-facing outlines from the frontend triangle soup.

## User-Approved Decisions

- The overlay represents the selected building-surface zone only, not the whole building.
- The change applies only to Solar, View, and Landmark simulation viewers.
- `ZoningTab` keeps the existing translucent fill editing display.
- The visual style is solid zone-colored edges with a dark halo.
- The backend should generate a selected, voxel-merged edge payload.

## Backend API

Add a backend endpoint for selected surface-zone edge geometry. Suggested route:

```http
POST /api/buildings/surface-zone-edges
```

Request body:

```json
{
  "zones": [
    {
      "id": "zone-id",
      "name": "Zone name",
      "group_id": "optional-group-id",
      "type": "building_surface",
      "selectors": []
    }
  ]
}
```

The endpoint should ignore non-`building_surface` zones. It should validate selector shapes using the same rules already used for zone requests.

Response body:

```json
{
  "zones": [
    {
      "id": "zone-id",
      "segments": [
        [0, 0, 0, 10, 0, 0]
      ]
    }
  ]
}
```

Each segment is `[x1, y1, z1, x2, y2, z2]` in the same scene coordinate system as existing `MeshChunk` geometry. The backend returns geometry only. The frontend hook attaches each zone color from the existing frontend zone state before rendering.

Horizontal zones are valid in the request but ignored by this endpoint. They should not cause an error and should not appear in the response.

If the model is missing, the building ID grid is unavailable, or a zone has no matching selected surfaces, the endpoint should return an empty zone list or a zone with an empty `segments` list. Malformed requests should still return normal API validation errors.

## Backend Geometry Builder

Create a pure backend helper, preferably in a focused module such as `app/backend/surface_zone_edges.py` unless the implementation stays very small. The helper should not depend on FastAPI or `app_state`.

Inputs:

- `voxcity_grid`: 3D voxel class grid.
- `building_id_grid`: 2D building ID grid in uv layout.
- `meshsize`: voxel size in metres.
- `zones`: `ZoneSpec` values.

Output:

- A list of per-zone edge payloads containing zone ID, color, and axis-aligned line segments.

The helper should use exposed building voxel faces only, matching `create_voxel_mesh(..., class_id=-3, mesh_type="open_air")` and `/api/buildings/surfaces`. A building voxel has an exposed face when the adjacent location is out of bounds or the adjacent in-bounds voxel class is `0` (air) or `-2` (tree). Adjacent land-cover, terrain, or building voxels do not produce selectable faces.

### Surface Records

Build an intermediate list of voxel-surface records before merging. Each record should represent one exposed voxel face rectangle and include:

- building ID
- plane (`+x`, `-x`, `+y`, `-y`, `+z`, `-z`)
- grid coordinate `(u, v, k)` of the source voxel
- surface kind (`roof`, `wall`, or ignored bottom/other)
- wall orientation when applicable
- four rectangle corners in scene coordinates
- the two triangle face keys that correspond to the existing `/api/buildings/surfaces` metadata

Suggested Python shape:

```python
@dataclass(frozen=True)
class VoxelSurfaceRecord:
  building_id: int
  plane: str              # "+x", "-x", "+y", "-y", "+z", "-z"
  voxel_uvw: tuple[int, int, int]
  surface_kind: str       # "roof" | "wall" | "bottom" | "other"
  orientation: str | None # "N" | "E" | "S" | "W" for walls
  corners: tuple[tuple[float, float, float], ...]  # four scene-space corners
  face_keys: tuple[str, str]                       # triangle keys for this rectangle
```

The face keys must remain compatible with current `faces` and `exclude_faces` selectors. Do not query the existing HTTP endpoint from the new endpoint. Instead, share or factor the same voxel surface extraction and face-key generation used by `/api/buildings/surfaces` so both payloads derive keys from one implementation.

If implementation starts as a private helper, it must reproduce the current order and key formula:

- iterate building voxels with `np.argwhere(voxel_array == -3)` order
- process directions in the same order as `create_voxel_mesh`: `+x`, `-x`, `+y`, `-y`, `+z`, `-z`
- create two triangle indices per voxel-face rectangle: `2 * quad_index` and `2 * quad_index + 1`
- generate triangle keys with `make_surface_face_key(building_id, centroid, normal, face_index)`

Add a regression test that verifies the new record extraction produces the same face keys as `/api/buildings/surfaces` for a small fixture.

Because current face keys are per triangle while the new display is per rectangle, selector handling should promote triangle-level matches to the full voxel-face rectangle:

- `faces`: select the voxel-face rectangle if either of its two triangle keys is selected.
- `exclude_faces`: exclude the voxel-face rectangle if either of its two triangle keys is excluded.

This avoids rendering diagonal half-face highlights while keeping face-specific selections meaningful.

### Selector Semantics

Preserve existing selector meanings:

- `whole`: all selectable roof and wall faces for the building.
- `roof`: exposed top faces only.
- `all_walls`: all exposed vertical faces.
- `wall_orientation`: exposed vertical faces matching the requested orientation.
- `faces`: selected voxel-face rectangles derived from matching triangle face keys.
- `exclude_faces`: removes matching voxel-face rectangles after positive selection.

Selection is evaluated per zone. Do not merge or deduplicate across zones, because overlapping zones must preserve separate colors.

### Greedy Merge

For each zone, merge selected faces only when they share all of these properties:

- same building ID
- same plane
- same constant coordinate
- same zone

Merge adjacent coplanar selected faces into deterministic axis-aligned rectangles in that plane. Use this row-major greedy algorithm unless a shared helper already exists:

1. Build a 2D boolean mask for a single `(zone, building_id, plane, constant_coordinate)` group.
2. Visit cells in sorted row-major order.
3. When an unvisited selected cell is found, extend the rectangle width along the first in-plane axis until the next cell is unselected or visited.
4. Extend rectangle height along the second in-plane axis while every cell across the chosen width remains selected and unvisited.
5. Mark all cells in the rectangle visited and continue.

This does not need to be globally optimal for every concave mask, but it must be deterministic, must merge a fully selected rectangle into one rectangle, and must not emit per-voxel internal cell edges inside a merged rectangle.

Convert each merged rectangle to four perimeter segments. Do not emit internal voxel-cell edges inside the merged rectangle.

### Edge Filtering

Use `AXIS_EPSILON_M = 1e-6` metres for axis-alignment and zero-length checks. A segment is axis-aligned only when exactly one coordinate delta is greater than `AXIS_EPSILON_M` and the other two coordinate deltas are less than or equal to `AXIS_EPSILON_M`. Drop diagonal triangle edges and any malformed zero-length segment.

Normalize segments for deduplication by rounding coordinates to 6 decimal places and ordering the two endpoints lexicographically. Deduplicate identical normalized segments within a zone. Do not deduplicate between zones.

## Frontend Data Flow

Add TypeScript DTOs and an API function in `app/frontend/src/api.ts`, for example:

```ts
export interface SurfaceZoneEdgePayload {
  id: string;
  segments: [number, number, number, number, number, number][];
}

export interface SurfaceZoneEdgesResponse {
  zones: SurfaceZoneEdgePayload[];
}

export async function getSurfaceZoneEdges(zones: Zone[]) { ... }
```

The API response payload does not need to carry color. The hook should map each response `id` back to the original frontend zone and build render specs with `{ id, color, segments }` for the Three.js layer.

Create a separate hook, for example `useSurfaceZoneEdges`, instead of extending `useSurfaceZoneSelection`. The existing selection hook carries surface chunks and face metadata for picking/fill rendering. Edge highlights are a different payload and do not need the pickable surface mesh.

The hook should:

- filter to `building_surface` zones with selectors
- skip fetching when `hasModel` is false
- skip fetching when `enabled` is false
- skip fetching when no building-surface zone has selectors
- refetch when `geometryToken` or relevant zones change
- suppress endpoint errors in simulation tabs
- return `null` or an empty payload when no edge overlay should render

`SolarTab`, `ViewTab`, and `LandmarkTab` should call this hook with `enabled: showZones3D` and pass the result to `SceneViewer`.

`ZoningTab` should not use this hook.

## Frontend Rendering

Add a small edge layer component for the returned segment payload. It can live under `app/frontend/src/three`, for example `SurfaceZoneEdgeLayer.tsx`.

Rendering rules:

- For each non-empty zone, render a black halo line first.
- Render a solid zone-colored line above the halo.
- Use `depthTest={false}` and `depthWrite={false}` so edges stay visible over simulation colors.
- Use solid lines only; no dashed styling for this path.
- Skip zones with empty segment arrays.

`SceneViewer` should accept the edge payload separately from `surfaceSelection`. This keeps the existing fill/pickable-surface path intact and avoids overloading `displayMode: 'boundary'` with two unrelated geometry sources.

## Error Handling

Surface-zone edge highlighting is optional decoration in simulation views. If the endpoint fails, the frontend should keep rendering the simulation normally and omit the edge overlay. The failure should not block simulation results, color settings, class visibility, landmark picking, or existing horizontal zones.

Backend validation should still reject malformed selector payloads, but normal empty states should be quiet:

- no model
- no building ID grid
- no building-surface zones
- no matching selected surfaces

## Testing

Backend tests should cover the pure edge builder:

- a 2x2 selected wall merges into one rectangle with four perimeter edges
- internal voxel-cell edges are not emitted after merge
- emitted edges are axis-aligned only
- diagonal triangle edges never appear
- `whole` selects all selectable roof and wall records for the requested building
- `roof` selects only `+z` roof records for the requested building
- `all_walls` selects `+x`, `-x`, `+y`, and `-y` wall records for the requested building
- `wall_orientation` selects only walls whose orientation matches `N`, `E`, `S`, or `W`
- `faces` selects the full voxel-face rectangle when either triangle face key matches
- `exclude_faces` removes the full voxel-face rectangle when either triangle face key matches, after positive selectors are applied
- face-key selectors promote matching triangle keys to the full voxel-face rectangle
- new record extraction produces the same triangle face keys as `/api/buildings/surfaces` for a small fixture
- merging does not cross building IDs, planes, or zones
- empty inputs return empty payloads

Frontend tests should cover:

- the hook skips fetches when disabled, no model exists, or no building-surface zones with selectors exist
- the hook suppresses errors for optional simulation overlays
- `SceneViewer` renders the edge layer independently from `surfaceSelection`
- the edge layer renders halo plus solid color line for non-empty zones and skips empty zones

Verification should include the existing frontend TypeScript check and tests, plus backend unit tests for the new pure geometry helper.

## Out Of Scope

- Changing the ZoningTab fill overlay.
- Changing horizontal zone display.
- Whole-building landmark selection highlight behavior.
- Replacing simulation mesh rendering.
- Adding UI controls for line style or width.
