# Three.js / R3F Migration Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking. Each chunk ends with a verification checkpoint — STOP and ask the user to validate before starting the next chunk.

**Goal:** Replace the hand-rolled Three.js viewer (`app/frontend/src/components/ThreeViewer.tsx`) with a modern `@react-three/fiber` + `@react-three/drei` viewer fed by raw BufferGeometry-friendly JSON from new backend endpoints. Eliminate the Plotly `Mesh3d` JSON transport format.

**Why:**
- The current viewer decodes Plotly's binary `Mesh3d` format on every load — pure indirection, no benefit. Native R3F + raw arrays is simpler and faster.
- Zone outlines must be visible through buildings/trees (`material.depthTest = false`). Trivial in R3F via `<Line depthTest={false} renderOrder={999}>` from drei.
- R3F gives us a sane component model for layering (buildings, trees, terrain, zones, sim overlay, picker) instead of a 1100-line imperative class.

**Tech Stack:** FastAPI (backend), React 18 + TypeScript + Vite + `three@^0.170` + `@react-three/fiber` + `@react-three/drei` (frontend).

**Reference:** `reference/optree_app/frontend/src/components/ThreeViewer.tsx` uses raw three.js with `depthTest = false` lines for zone outlines — same technique, but we're going declarative.

---

## File structure

**New:**

| File | Responsibility |
| --- | --- |
| `app/backend/scene_geometry.py` | Pure builders: `build_voxel_buffers(grid, meshsize) -> SceneGeometry`, `build_sim_overlay_buffers(...) -> OverlayGeometry`. Reuses voxel→mesh internals from `src/voxcity/visualizer/renderer.py` and `src/voxcity/geoprocessor/mesh.py` but emits raw `{positions, indices, colors}` arrays. |
| `app/frontend/src/three/SceneViewer.tsx` | R3F `<Canvas>` viewer. Props: `geometry`, `overlay`, `zones`, `selection`, callbacks. |
| `app/frontend/src/three/MeshLayer.tsx` | One `<mesh>` per geometry chunk. Builds `BufferGeometry` from `{positions, indices, colors?}`. |
| `app/frontend/src/three/ZoneOutlines.tsx` | drei `<Line>` per zone. `depthTest={false}`, `renderOrder={999}`. |
| `app/frontend/src/three/Picker.tsx` | Invisible R3F mesh that intercepts clicks for face/building selection. |
| `app/frontend/src/three/ColorBar.tsx` | HTML overlay (absolute-positioned `<div>`) showing colormap legend for sim overlays. |
| `app/frontend/src/three/CameraControls.tsx` | drei `<OrbitControls>` with Z-up + auto-fit on bbox change + reset/screenshot buttons. |
| `app/frontend/src/three/types.ts` | `SceneGeometry`, `OverlayGeometry`, `MeshChunk` types matching backend payloads. |

**Modified:**

| File | Change |
| --- | --- |
| `app/backend/models.py` | Add `SceneGeometryResponse`, `OverlayGeometryResponse`, `MeshChunk`, `OverlayKind`. |
| `app/backend/main.py` | Add `GET /api/scene/geometry` and `POST /api/sim/{kind}/geometry` routes. Keep existing `figure_json` routes alive during transition. |
| `app/frontend/src/api.ts` | Add `getSceneGeometry()`, `getSolarGeometry()`, `getViewGeometry()`, `getLandmarkGeometry()`. Keep existing `runSolar/runView/runLandmark` returning sim metadata only. |
| `app/frontend/src/App.tsx` | Replace `figureJson` state with `sceneGeometry` + per-tab `overlayGeometry` state. |
| `app/frontend/src/tabs/ZoningTab.tsx` | Replace `<ThreeViewer figureJson=...>` with `<SceneViewer geometry={scene} zones={zones}/>`. Drop `useZoneOverlay`. |
| `app/frontend/src/tabs/SolarTab.tsx` | Same shape; pass `overlay={solarOverlay}`. |
| `app/frontend/src/tabs/ViewTab.tsx` | Same. |
| `app/frontend/src/tabs/LandmarkTab.tsx` | Same; wire `onFacePick` for click-selection. |
| `app/frontend/src/tabs/EditTab.tsx` | Same. |
| `app/frontend/src/tabs/GenerationTab.tsx` | Same. |
| `app/frontend/src/hooks/useZoneOverlay.ts` | **Delete** — replaced by `<ZoneOutlines>` inside `<SceneViewer>`. |
| `app/frontend/src/lib/zoneTraces.ts` | **Delete** — was only ever a Plotly-trace builder. |
| `app/frontend/src/components/ThreeViewer.tsx` | **Delete** after all tabs migrated. |
| `app/frontend/package.json` | Add `@react-three/fiber`, `@react-three/drei`. |

---

## Chunk 1 — Backend: raw geometry endpoints

Goal: produce JSON payloads the new viewer can hand directly to `BufferGeometry`. Keep the existing `figure_json` endpoints working in parallel so we can ship chunks incrementally.

### Task 1.1 Pydantic models

**Files:**
- Modify: `app/backend/models.py`

- [ ] **Step 1.1.1: Append models**

```python
# ---------------------------------------------------------------------------
# Three.js raw geometry
# ---------------------------------------------------------------------------

class MeshChunk(BaseModel):
    """One BufferGeometry-friendly chunk."""
    name: str                                  # e.g. "buildings", "trees+x", "terrain"
    positions: List[float]                     # flat XYZ, length = 3*N
    indices: List[int]                         # flat triangle indices, length = 3*M
    color: Optional[List[float]] = None        # uniform RGB [0..1] if no per-face color
    colors: Optional[List[float]] = None       # per-vertex RGB, length = 3*N (when set, overrides color)
    opacity: float = 1.0
    flat_shading: bool = False
    metadata: Dict[str, Any] = Field(default_factory=dict)


class SceneGeometryResponse(BaseModel):
    chunks: List[MeshChunk]
    bbox_min: List[float]                      # [x, y, z]
    bbox_max: List[float]
    meshsize_m: float


class OverlayGeometryResponse(BaseModel):
    chunk: MeshChunk                           # single overlay mesh, per-vertex colors
    face_to_cell: Optional[List[List[int]]] = None  # [[i, j], ...] per face, for ground sims
    face_to_building: Optional[List[int]] = None    # [building_id, ...] per face, for building sims
    value_min: float
    value_max: float
    colormap: str
    unit_label: str
```

### Task 1.2 Geometry builders

**Files:**
- Create: `app/backend/scene_geometry.py`

- [ ] **Step 1.2.1: Voxel buffers**
  - Refactor `add_faces` from `src/voxcity/visualizer/renderer.py` into a pure function `voxel_face_arrays(mask, plane, meshsize, x, y, z) -> (positions, indices)` that returns numpy arrays.
  - In `scene_geometry.py`, add `build_voxel_buffers(voxcity_grid, meshsize, *, downsample=1) -> SceneGeometryResponse` that iterates classes (buildings, trees, land cover groups), calls `voxel_face_arrays` per (class, plane), and emits one `MeshChunk` per (class, plane). Color from `palette.get_voxel_color_map()`.

- [ ] **Step 1.2.2: Sim overlay buffers**
  - Add `build_ground_overlay_buffers(sim_grid, dem_grid, meshsize, cmap, vmin, vmax, z_offset, unit_label) -> OverlayGeometryResponse`. One quad (2 tris) per non-NaN cell. Per-vertex colors via `matplotlib.cm.ScalarMappable`. Build `face_to_cell` mapping.
  - Add `build_building_overlay_buffers(mesh, sim_type, cmap, vmin, vmax, unit_label) -> OverlayGeometryResponse`. Convert per-face values to per-vertex colors (replicate face color to all 3 vertices). Build `face_to_building` from `mesh.metadata['building_face_ids']` if present.

### Task 1.3 Routes

**Files:**
- Modify: `app/backend/main.py`

- [ ] **Step 1.3.1: `GET /api/scene/geometry`** → `SceneGeometryResponse`. Reads `app_state.last_voxcity_grid`. 404 if no model loaded. Add `?downsample=1` query param.
- [ ] **Step 1.3.2: `POST /api/sim/{kind}/geometry`** where `kind ∈ {solar, view, landmark}`. Body: `{ colormap, vmin?, vmax? }`. Reads `app_state.last_sim_grid` (ground) or `app_state.last_sim_mesh` (building) and returns `OverlayGeometryResponse`. 404 if no sim run yet.

### Task 1.4 Tests

**Files:**
- Create: `tests/app/test_scene_geometry.py`

- [ ] **Step 1.4.1:** `build_voxel_buffers` returns at least one chunk for a 2×2×2 grid with one building voxel. Positions length divisible by 3. Indices in valid range.
- [ ] **Step 1.4.2:** `build_ground_overlay_buffers` returns one face per non-NaN cell. `face_to_cell` length = `chunk.indices` length / 3 / 2.
- [ ] **Step 1.4.3:** API integration test: generate small model, GET `/api/scene/geometry`, assert response shape.

### ✅ Verification checkpoint 1
- `pytest tests/app/test_scene_geometry.py -v` passes
- Manual: `curl http://localhost:8000/api/scene/geometry` after generating a model returns valid JSON with non-empty chunks
- **STOP. Ask user to confirm before proceeding to Chunk 2.**

---

## Chunk 2 — Frontend: scaffolding + dependencies

### Task 2.1 Install deps

- [ ] **Step 2.1.1:** `cd app/frontend; npm i @react-three/fiber@^8 @react-three/drei@^9`. Note: drei v9 is for R3F v8 + three v0.170 (matches our existing three version).

### Task 2.2 Types

**Files:**
- Create: `app/frontend/src/three/types.ts`

- [ ] **Step 2.2.1:** Mirror backend models — `MeshChunk`, `SceneGeometry`, `OverlayGeometry`. Use `Float32Array | number[]` for positions/colors and `Uint32Array | number[]` for indices.

### Task 2.3 API client

**Files:**
- Modify: `app/frontend/src/api.ts`

- [ ] **Step 2.3.1:** Add `getSceneGeometry()`, `getSimGeometry(kind, params)` returning the new shapes.

### Task 2.4 Core viewer components

**Files:**
- Create: `app/frontend/src/three/MeshLayer.tsx`
- Create: `app/frontend/src/three/ZoneOutlines.tsx`
- Create: `app/frontend/src/three/CameraControls.tsx`
- Create: `app/frontend/src/three/ColorBar.tsx`
- Create: `app/frontend/src/three/Picker.tsx`
- Create: `app/frontend/src/three/SceneViewer.tsx`

- [ ] **Step 2.4.1: `MeshLayer`** — `useMemo` constructs a `BufferGeometry` from a `MeshChunk`. Returns `<mesh geometry={geom}><meshStandardMaterial vertexColors={!!chunk.colors} color={chunk.color} flatShading={chunk.flat_shading} transparent={chunk.opacity < 1} opacity={chunk.opacity}/></mesh>`.
- [ ] **Step 2.4.2: `ZoneOutlines`** — for each zone, project lon/lat ring to local-grid XY using a `gridToLocal(lon, lat)` helper. Emit a drei `<Line>` per zone with `depthTest={false}`, `renderOrder={999}`, `lineWidth={selected ? 4 : 2}`, `color={zone.color}`. Two rings per zone: at z = 0.5 m and z = `bbox_max.z + 2 m` so the outline is visible from any angle.
- [ ] **Step 2.4.3: `CameraControls`** — drei `<OrbitControls makeDefault />`. On `geometry` prop change, fit camera to bbox. Z-up: `<Canvas camera={{ up: [0,0,1] }}>`. Render two HTML buttons (reset, screenshot) absolutely positioned over canvas.
- [ ] **Step 2.4.4: `ColorBar`** — pure HTML/CSS gradient bar with min/max labels, fed by `OverlayGeometry.value_min/max/colormap/unit_label`.
- [ ] **Step 2.4.5: `Picker`** — invisible mesh that wraps the overlay. `onClick={(e) => onFacePick({ faceIndex: e.faceIndex, ...mapping })}`.
- [ ] **Step 2.4.6: `SceneViewer`** — composes everything. Props: `geometry: SceneGeometry`, `overlay?: OverlayGeometry`, `zones?: Zone[]`, `selectedZoneId?: string`, `selectionMode?: 'none'|'face'|'building'`, `onFacePick?(...)`. Layout: `<Canvas>` + ambient + directional light + `MeshLayer` per chunk + `MeshLayer` for overlay (if any) + `ZoneOutlines` + `CameraControls` + HTML overlay (`ColorBar`, buttons).

### ✅ Verification checkpoint 2
- `npx tsc --noEmit` clean
- Storybook-style smoke: render `<SceneViewer>` with a hand-built tiny `geometry` (one cube) in `App.tsx` temporarily, verify it appears
- **STOP. Ask user to validate visually before proceeding.**

---

## Chunk 3 — Vertical slice: ZoningTab on the new viewer

### Task 3.1 App-level state

**Files:**
- Modify: `app/frontend/src/App.tsx`

- [ ] **Step 3.1.1:** Add `sceneGeometry: SceneGeometry | null` state. Fetch via `getSceneGeometry()` after generation/edit completes (replace whatever currently sets `figureJson` for the model preview). Pass to `<ZoningTab>`.

### Task 3.2 ZoningTab swap

**Files:**
- Modify: `app/frontend/src/tabs/ZoningTab.tsx`

- [ ] **Step 3.2.1:** Replace `<ThreeViewer figureJson=...>` with `<SceneViewer geometry={sceneGeometry} zones={zones} selectedZoneId={selectedId}/>`. Delete the `useZoneOverlay` / `buildZoneTraces` paths from this tab.

### ✅ Verification checkpoint 3
- App restarts, ZoningTab loads, model is visible, drawing a zone immediately shows a polygon outline that **penetrates buildings/trees** (this is the original requirement)
- `npx tsc --noEmit` clean
- **STOP. Ask user to validate before propagating to other tabs.**

---

## Chunk 4 — Sim tab migration

### Task 4.1 SolarTab

**Files:**
- Modify: `app/frontend/src/tabs/SolarTab.tsx`

- [ ] **Step 4.1.1:** After `runSolar()` succeeds, call `getSimGeometry('solar', { colormap, vmin, vmax })` to get the overlay. Pass to `<SceneViewer overlay={overlay}/>`. Drop `useZoneOverlay`.
- [ ] **Step 4.1.2:** Wire `<ColorBar>` (already inside `SceneViewer`) using `overlay.value_min/max/colormap/unit_label`.

### Task 4.2 ViewTab + LandmarkTab

- [ ] **Step 4.2.1:** Same as Solar.
- [ ] **Step 4.2.2:** LandmarkTab also wires `onFacePick={handleBuildingSelect}` — `Picker` returns `face_to_building[faceIndex]`.

### ✅ Verification checkpoint 4
- All 3 sim tabs render correctly with overlay + colorbar + zones
- LandmarkTab building selection still works
- **STOP. Ask user to validate.**

---

## Chunk 5 — Remaining tabs + cleanup

### Task 5.1 EditTab + GenerationTab

- [ ] **Step 5.1.1:** Migrate to `<SceneViewer>`. After edits/generation finish, refetch `getSceneGeometry()`.

### Task 5.2 Delete dead code

- [ ] **Step 5.2.1:** Delete `app/frontend/src/components/ThreeViewer.tsx`.
- [ ] **Step 5.2.2:** Delete `app/frontend/src/hooks/useZoneOverlay.ts`.
- [ ] **Step 5.2.3:** Delete `app/frontend/src/lib/zoneTraces.ts` (if it exists on disk; it was gitignored).
- [ ] **Step 5.2.4:** Backend: leave the existing `figure_json` endpoints in place but mark deprecated in their docstrings. Removing them is a separate cleanup pass.

### Task 5.3 Final validation

- [ ] **Step 5.3.1:** `npx tsc --noEmit` clean
- [ ] **Step 5.3.2:** `pytest tests/app/ -v` clean
- [ ] **Step 5.3.3:** Manual smoke: every tab, every interaction.

### ✅ Verification checkpoint 5
- Whole app working end-to-end on the new viewer.
- **STOP. Ask user for sign-off.**

---

## Out of scope

- Removing `figure_json` endpoints from the backend (deprecate now, delete in a follow-up).
- Saving camera state to URL/localStorage.
- Per-tab voxel-class visibility filter — currently lives in `useDebouncedRerender`; will continue calling the legacy `/api/rerender` route. Migration of that flow is a separate plan.
- Box-selection in LandmarkTab (currently in `ThreeViewer`). Re-implement only if the user explicitly asks; click-selection covers the common case.
