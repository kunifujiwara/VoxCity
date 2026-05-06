# Building Surface Zone Highlights On Simulation Viewers

## Goal

Show building surface zones in the Solar, View, and Landmark 3D simulation viewers without reducing the readability of simulation results. Horizontal zones should continue to render as existing zone outlines. Building surface zones should become visible in simulation viewers as non-obscuring annotations.

The approved visual direction is boundary-only marking for building-surface simulation results: selected surfaces are identified by their outer selected-surface boundaries, not by filled faces and not by dense mesh edge overlays.

## Current Context

`SceneViewer` is shared by the Zoning tab and the simulation tabs. It already renders simulation overlays, horizontal `ZoneOutlines`, optional building highlights, and the `SurfaceSelectionLayer` used by Zoning.

`ZoningTab` currently fetches `/buildings/surfaces`, converts `face_to_surface` into `SurfaceFaceMeta[]`, and passes building surface zones to `SceneViewer.surfaceSelection`. The simulation tabs already pass `zones` and `showZones` into `SceneViewer`, but they do not fetch surface geometry or pass `surfaceSelection`, so building surface zones are missing from the simulation 3D viewers.

## Architecture

Keep zone ownership in the existing tab/App state and keep rendering responsibility in `SceneViewer` and its Three.js layers.

Extend the existing surface-zone rendering path rather than creating a separate simulation-only overlay system. `SurfaceSelectionLayer` should support two display modes:

- `fill`: existing Zoning behavior with translucent selected surface faces.
- `boundary`: simulation-viewer behavior with selected-surface boundaries only.

Simulation tabs should continue using their existing `Show zones in 3D` checkbox. When enabled, it should show both horizontal zone outlines and building surface zone boundary marks. No new checkbox is needed in the first version.

## Components And Boundaries

### Shared Surface Selection Hook

Introduce a small shared hook or helper, likely `useSurfaceZoneSelection`, used by Zoning and simulation tabs. It should:

- fetch `/buildings/surfaces` only when a model exists and surface-zone marks are needed;
- convert the response's `face_to_surface` map into `SurfaceFaceMeta[]`;
- filter `Zone[]` to `BuildingSurfaceZone[]`;
- return the `surfaceSelection` payload expected by `SceneViewer`;
- accept a display mode so callers choose `fill` or `boundary`.

This avoids duplicating surface-geometry fetch and DTO conversion logic in Solar, View, Landmark, and Zoning.

### SceneViewer

`SceneViewer.surfaceSelection` should remain the public bridge into surface-zone rendering. Its payload should include:

```ts
type SurfaceSelectionDisplayMode = 'fill' | 'boundary';

interface SurfaceSelectionLayerSpec {
	id: string;
	color: string;
	selectors: SurfaceSelector[];
	active: boolean;
}

interface SceneSurfaceSelectionSpec {
	surfaceChunk: MeshChunkDto | null;
	faceToSurface: SurfaceFaceMeta[];
	zones: SurfaceSelectionLayerSpec[];
	enabled: boolean;
	displayMode: SurfaceSelectionDisplayMode;
}
```

`SceneViewer` should not need to know whether the caller is Solar, View, Landmark, or Zoning. It should forward the display mode to `SurfaceSelectionLayer`.

### SurfaceSelectionLayer

`SurfaceSelectionLayer` should keep selector semantics unchanged:

- `whole`: all faces of a building;
- `roof`: roof faces only;
- `all_walls`: wall faces only;
- `wall_orientation`: walls in one orientation;
- `faces`: specific face keys;
- `exclude_faces`: excluded from the positive selection.

In `fill` mode, preserve the current translucent face rendering.

In `boundary` mode, render only selected-surface boundary marks. Do not fill selected faces and do not draw all mesh edges.

## Boundary Rendering Behavior

Boundary mode should first determine the selected triangle set for each surface-zone layer. From that selected set, build boundary edges by counting triangle edges: an edge is a boundary edge when it appears only once in the selected set after vertex-key normalization. Edges shared by two selected triangles are internal and must not be drawn.

Boundary marks should use each zone's existing color. To preserve legibility over arbitrary colormaps, draw a thin contrast halo behind the colored line, such as a slightly wider dark or light line, then the dashed zone-color line above it.

For building-surface simulation results, the default should be boundary-only marks with no face fill. Do not add anchor points in the first implementation; they are reserved for a later UX iteration if manual verification shows boundary lines are still hard to find.

Boundary mode requirements:

- no face fill in boundary mode;
- thin dashed colored boundary line;
- contrast halo drawn as a black line behind the colored line, wider than the colored line and partially transparent;
- `depthTest={false}` and `depthWrite={false}` so zone marks remain visible;
- render above the simulation mesh;
- avoid dense internal triangle edges.

## Simulation Tab Behavior

Solar, View, and Landmark tabs should pass surface-zone marks to `SceneViewer` when all of these are true:

- a model exists;
- `Show zones in 3D` is enabled;
- there is at least one `building_surface` zone;
- `/buildings/surfaces` returned usable geometry.

Usable surface geometry means: `surfaceChunk` exists, `surfaceChunk.positions` contains at least one triangle, `faceToSurface` contains at least one metadata entry, and at least one building-surface zone has selectors. A zone that selects no matching faces should simply render no boundary marks; it should not be treated as a fatal error.

The behavior should apply to both ground-level and building-surface simulation targets. The first implementation should use boundary mode for both, because it is conservative and avoids surprising color mixing. If a later version needs stronger context for ground-level results, it can add a separate visual tuning option after the base behavior is proven.

Horizontal zones continue to render through `ZoneOutlines` as they do today.

## Error Handling

Surface-zone rendering must not block simulation viewing. If `/buildings/surfaces` fails or returns no usable geometry in a simulation tab, the tab should continue rendering the simulation and horizontal zones. Building surface zone marks should silently be absent.

This quiet failure behavior is specific to simulation viewers, where the primary user task is reading simulation output. Zoning can keep its current behavior and controls.

## Testing And Verification

Automated tests should focus on shared logic:

- unit-test selected-face boundary extraction so internal triangle edges are omitted and only outer selected-set edges remain;
- unit-test selector behavior for `whole`, `roof`, `all_walls`, `wall_orientation`, `faces`, and `exclude_faces` in the shared surface selection logic;
- type-check the expanded `surfaceSelection` payload and all tab call sites.

Manual/browser verification should cover:

- building surface zones visible in Solar, View, and Landmark simulation viewers when `Show zones in 3D` is enabled;
- building-surface simulation results remain readable because zone marks do not fill faces and do not show dense internal mesh edges;
- horizontal zone outlines continue to work;
- disabling `Show zones in 3D` hides both horizontal and building-surface zone marks;
- simulation viewers still render if surface-zone geometry fetch fails.

## Non-Goals

- No separate per-zone visibility UI in the first version.
- No dense wireframe overlay for all selected mesh triangles.
- No simulation-specific backend endpoint for zone visualization unless existing `/buildings/surfaces` proves insufficient.
- No changes to zone statistics behavior.
- No changes to the simulation computation itself.