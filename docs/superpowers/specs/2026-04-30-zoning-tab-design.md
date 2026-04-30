# Zoning Tab — Design Spec

**Date:** 2026-04-30
**Status:** Draft (pre-implementation)
**Scope:** `app/` (FastAPI backend + React frontend)

---

## 1. Goal

Add a **Zoning** tab to the VoxCity web app that lets the user define one or
more 2D footprint zones over the current model. Each subsequent simulation tab
(Solar, View, Landmark) then displays per-zone summary statistics computed
from its cached results, plus a 3D outline of each zone in the model viewer.

## 2. Non-goals (v1)

- 3D-volume zones (z-min / z-max per zone)
- Categories / tags for grouping zones
- Building-group zones (zone defined by selecting building footprints)
- Per-zone histograms or percentile statistics
- Backend persistence of zones (saved across page reloads or restarts)
- Importing zones from GeoJSON / Shapefile
- Real "always-on-top" rendering through occluders (Plotly limitation)

These are intentionally deferred and the data model leaves room to add them
later without breaking the v1 contract.

## 3. User stories

1. *As a researcher,* after generating a model I open the **Zoning** tab,
   draw a rotated rectangle around a courtyard, and a polygon along a street.
   I see both zones outlined on the 2D map and as colored vertical curtains
   in the 3D viewer.
2. *I run a Solar simulation.* On the Solar tab I see, beneath the run
   controls, a table with `count, mean, min, max, std` of irradiance for each
   zone, plus the same zone outlines drawn over the colored 3D result.
3. *I tweak a zone* (rename, recolor, delete, draw a new one) on the Zoning
   tab and return to Solar — the table refreshes against the existing sim
   result without re-running it.
4. *I change the target area* on the Target Area tab — my zones are cleared
   (they belonged to a different region).

## 4. Architectural decisions

| Decision | Choice | Rationale |
| --- | --- | --- |
| Zone shape | 2D polygon footprint (no height range) | Simplest model that covers the dominant use case; aggregates cleanly against existing `last_sim_grid` / `last_sim_mesh` caches. |
| Drawing primitives | Rotated rectangle (default) + polygon | Matches existing `EditTab` building-draw primitives in `PlanMapEditor`. No new primitives needed. |
| Multiple zones | Flat list, auto-named "Zone N", overlap allowed | Matches the rest of the app; categories add UI weight without evidence of need. |
| Persistence | Frontend state in `App.tsx`, lon/lat polygons | Survives Edit/Generation re-runs; cleared when target rectangle changes. Mirrors existing patterns (`rectangle`, `figureJson`, edits buffer). |
| Stats compute | New stateless `POST /api/zones/stats` | Single endpoint; sim handlers untouched; supports "edit zones without re-running sim". |
| Tab order | `Area · Generation · Edit · Zoning · Solar · View · Landmark · Export` | Natural flow; same `hasModel` gating as other post-model tabs. |
| Layout | `.three-col` shell (config | 2D editor | 3D viewer) | Mirrors `EditTab`. |

## 5. Data model

### 5.1 Frontend (`app/frontend/src/types/zones.ts`, new)

```ts
export type ZoneShape = 'rect' | 'polygon';

export interface Zone {
  id: string;                          // uuid (client-generated)
  name: string;                        // "Zone 1" by default; user-editable
  color: string;                       // hex from default palette; user-editable
  shape: ZoneShape;                    // informational
  ring_lonlat: [number, number][];     // [[lon, lat], ...], not closed
}
```

State lifted into `App.tsx` next to `rectangle` / `figureJson`:

```ts
const [zones, setZones] = useState<Zone[]>([]);
```

Wired with an explicit effect that also fixes a pre-existing UX bug (cached
sim figures currently survive a target-rectangle change until the next
`/generate`):

```ts
// Clear zones AND any cached sim figures when the target rectangle changes.
// The old results no longer correspond to the new area.
useEffect(() => {
  setZones([]);
  setFigureJson('');
  setEditFigureJson('');
  setSolarFigureJson('');
  setViewFigureJson('');
  setLandmarkFigureJson('');
}, [rectangle]);
```

`zones` is then passed read-only into `<ZoningTab>`, `<SolarTab>`,
`<ViewTab>`, and `<LandmarkTab>`.

### 5.2 Backend (`app/backend/models.py`)

```python
class ZoneSpec(BaseModel):
    id: str
    name: str
    ring_lonlat: list[list[float]]      # [[lon, lat], ...] (>=3 pts)

class ZoneStatsRequest(BaseModel):
    zones: list[ZoneSpec]

class ZoneStat(BaseModel):
    zone_id: str
    cell_count: int
    valid_count: int                    # cells/faces with finite values
    mean: float | None
    min:  float | None
    max:  float | None
    std:  float | None

class ZoneStatsResponse(BaseModel):
    target:     str                     # "ground" | "building" | "none"
    sim_type:   str | None              # "solar" | "view" | "landmark" | None
    unit_label: str | None              # mirrors AppState.last_colorbar_title
    stats:      list[ZoneStat]
```

No server-side zone state; reuses existing `AppState.last_sim_*` caches.

## 6. Backend endpoint

### 6.1 Route

`POST /api/zones/stats` → `ZoneStatsResponse`

### 6.2 Handler outline

```python
@app.post("/api/zones/stats", response_model=ZoneStatsResponse)
def zone_stats(req: ZoneStatsRequest):
    if app_state.voxcity is None:
        raise HTTPException(400, "No model loaded")
    if app_state.last_sim_type is None:
        raise HTTPException(400, "Run a simulation first")
    if app_state.last_sim_target == "ground":
        return _zone_stats_ground(req.zones)
    if app_state.last_sim_target == "building":
        return _zone_stats_building(req.zones)
    raise HTTPException(400, f"Unsupported target: {app_state.last_sim_target}")
```

### 6.3 Aggregation

**Ground** — each lon/lat ring → grid cells via a Python port of the
frontend `polygonToCells` (in `app/frontend/src/lib/grid.ts`). Index
`app_state.last_sim_grid[i, j]`, drop non-finite, compute `mean/min/max/std`.

**Building surfaces** — for each face on `app_state.last_sim_mesh` compute
its centroid in grid-local meters, project to lon/lat, test point-in-polygon
against each zone. Aggregate face values **area-weighted** for `mean`;
unweighted for `min`, `max`, `std`. (Confirmed area-weighted is the desired
behaviour; face-centroid containment is the desired inclusion rule.)

### 6.4 New module

`app/backend/zoning.py` housing:

- `polygon_lonlat_to_cells(ring, grid_geom) -> list[(int, int)]`
- `grid_xy_to_lonlat(xy, grid_geom) -> ndarray`
- `points_in_polygon(points_lonlat, ring) -> ndarray[bool]`
- `_stats_from(zone_id, count, values, mask) -> ZoneStat`

Keeps `main.py` clean.

### 6.5 Edge cases

| Input | Response |
| --- | --- |
| Empty `zones` list | `200`, `stats: []` |
| Zone fully outside grid | row with `cell_count: 0`, all metrics `null` |
| Zone whose values are all NaN/inf | `valid_count: 0`, all metrics `null` |
| No model loaded | `400 No model loaded` |
| No sim cached | `400 Run a simulation first` |

## 7. Frontend — Zoning tab

**File:** `app/frontend/src/tabs/ZoningTab.tsx` (new)

### 7.1 Layout (`.three-col`, mirrors `EditTab`)

```
.three-col
├── .panel  (left)             ← config + zone list
│   ├── <h2>Zoning</h2>
│   ├── Toolbar (mode + shape buttons)
│   ├── Default-color palette indicator
│   ├── ── Zones ──
│   │     ● Zone 1  [✎][🗑]
│   │     ● Zone 2  [✎][🗑]
│   │   [+ Add new zone]   [Clear all]
│   └── (info / error banners)
│
├── .map-pane  (center)        ← <PlanMapEditor>
│   - drawColor   = active draw color (next zone's color)
│   - interaction = 'draw_rect_3pt' | 'draw_polygon'
│   - pendingEdits = zones rendered as `paint_zone` overlays
│
└── .three-pane (right)        ← <ThreeViewer>
    - figureJson from parent
    - + zone curtain traces (client-side)
```

### 7.2 Toolbar

- **Mode**: `[Add new zone] | [Replace selected]` (default Add).
- **Shape**: `[Rectangle] (default) | [Polygon]`.
- **Default palette**: 8-color cycle assigned to new zones automatically.
- **Clear all** (with confirm).

### 7.3 List interactions

- `●` swatch click → small palette popover.
- `✎` inline rename.
- `🗑` delete with confirm.
- Row click selects the zone (highlighted in 2D + 3D).

### 7.4 3D curtain rendering ("luminescent, see-through-best-effort")

For each zone:

1. **Geometry**: extrude `ring_lonlat → grid xy` from `z = 0` to
   `ceiling = max(0.1 * max_building_height_m, 3.0)`.
2. **Surface trace**: `Mesh3d` of the curtain panel.
   - `color = zone.color`, `opacity = 0.35` (selected: `0.5`).
   - `flatshading: true`, `lighting: { ambient: 1, diffuse: 0, specular: 0 }`
     (emissive look — bright regardless of normals/light).
3. **Edge trace**: `Scatter3d` `mode: 'lines'`, ring + corner verticals.
   - `line.width: 6` (selected: `8`), `line.color = zone.color` at full opacity.
4. **Draw order**: append zone traces *last* in `figure.data` so Plotly's GL
   renderer favors them over translucent voxel meshes.
5. **Constraint**: Plotly does not expose a true `depthTest=false` mode.
   When tall buildings stand between the camera and a zone, the lower part
   of the curtain may be occluded. The low ceiling, saturated color, and
   thick edge line maximize the chance of an upper-rim silhouette remaining
   visible. A full always-on-top fix requires swapping the 3D backend
   (out of scope).

### 7.5 Curtain ceiling source

`max_building_height_m` from `ModelGeoResult` if present; otherwise computed
once on tab mount from `building_height_grid`.

### 7.6 Empty / gating states

- `!hasModel` → existing "Please generate a VoxCity model first" warning.
- Model exists, no zones → editor active, list shows hint "Draw a zone on
  the map →".

## 8. Frontend — simulation tab integration

Applies identically to `SolarTab`, `ViewTab`, `LandmarkTab`.

### 8.1 Wiring

`App.tsx` passes `zones` (read-only) into each sim tab in addition to current
props.

### 8.2 New shared pieces

- `app/frontend/src/hooks/useZoneStats.ts` — debounced fetch of
  `/api/zones/stats` keyed by `(simRunNonce, hash(zones))`.
- `app/frontend/src/components/ZoneStatsTable.tsx` — renders the table,
  `Export CSV` button (client-side blob download).
- `app/frontend/src/lib/zoneTraces.ts` — pure builder used by both Zoning
  tab and sim tabs to produce the curtain `Mesh3d` + edge `Scatter3d` traces
  from `(zones, geo, max_h)`.

### 8.3 Refresh matrix

| Trigger | Refetch zone stats? |
| --- | --- |
| Run sim | Yes (`simRunNonce++`) |
| Color setting rerender | No (values unchanged) |
| User edits zones, returns to sim tab | Yes (zones hash changed) |
| Target area changes | Both zones *and* sim cleared by existing logic |

### 8.4 Stats table UI (under run controls)

```
┌── Zone statistics ─────────────── (W/m², ground) ──┐
│ Zone        cells   mean    min    max    std       │
│ ●  Zone 1   1,284   412.1   12.3   980.0  217.5     │
│ ●  Zone 2     631   356.8    4.7   902.1  198.0     │
│ ●  Zone 3     402     —       —      —      —       │
└─────────────────────────────────────────────────────┘
[ Export CSV ]
```

Header unit pulled from `ZoneStatsResponse.unit_label`. Rows with
`valid_count == 0` show `—` and a muted "no data" hint.

### 8.5 3D outlines on sim tabs

Reuses `lib/zoneTraces.ts`. The sim tab appends zone traces to `figure.data`
before passing the figure to `ThreeViewer`. A `[x] Show zones in 3D` checkbox
above the table toggles their visibility (client-side, no refetch).

### 8.6 Empty / disabled states

- `zones.length === 0` → no extra request, no table, no checkbox; sim tab
  identical to today.
- Sim not yet run → table absent (matches today's "no result" state).
- `/api/zones/stats` returns `400 Run a simulation first` → swallowed
  silently; table only appears after a successful sim.

## 9. Lifecycle / state coupling

| Event | Effect on `zones` |
| --- | --- |
| Page load | `[]` (default) |
| `setRectangle(...)` (Target Area tab edit) | Cleared to `[]`. Also clears all cached sim figures (`figureJson`, `editFigureJson`, `solarFigureJson`, `viewFigureJson`, `landmarkFigureJson`) via a `useEffect([rectangle])` in `App.tsx` — see §5.1. |
| Generation completes | Untouched (zones lon/lat valid for the same area) |
| Edit-tab commit → `onModelEdited` | Untouched (rectangle unchanged) |
| `resetSession()` | Cleared to `[]` (page-load reset path already runs this) |

## 10. Testing

### 10.1 Backend (`tests/app/test_zones.py`, new)

- `polygon_lonlat_to_cells`: rectangle, polygon, fully-outside, degenerate.
- `_zone_stats_ground`: synthetic ramp grid; assert `mean/min/max/std`.
- `_zone_stats_building`: synthetic mesh with hand-set face values + areas;
  verify area-weighted mean.
- `/api/zones/stats` integration: 400 (no model), 400 (no sim), 200 (empty),
  200 (mixed valid + outside-grid zones).

### 10.2 Cross-stack consistency

- Snapshot a hand-computed cell set for one polygon and assert the backend
  port of `polygonToCells` returns the same set.

### 10.3 Frontend (manual smoke flow — no Vitest harness today)

1. Generate model → Zoning → draw rect → confirm 2D overlay + 3D curtain.
2. Add polygon → rename → recolor → delete one → list/2D/3D stay in sync.
3. Run Solar (ground) → Zoning, edit a zone → back to Solar → table refreshes
   without re-running sim.
4. Run Solar (building) → confirm area-weighted mean on a sun-facing wall.
5. Change target rectangle → confirm zones cleared.
6. Export CSV round-trip.

## 11. Risks & mitigations

| Risk | Mitigation |
| --- | --- |
| Plotly cannot render through occluders | Documented; mitigated by low ceiling, saturated emissive color, last-in-data ordering, thick edge lines. |
| Sim mesh interfaces may differ across solar/view/landmark | Implementation step 1 inspects each simulator's mesh and writes per-sim adapters in `zoning.py` that normalize to `(centroids_xy, values, areas)`. |
| Large zones could slow `/api/zones/stats` | Vectorized numpy. Add a soft cap (e.g., 50 zones) only if profiling shows an issue. |
| Lon/lat → cell math drift between client/server | Snapshot test asserts identical cell sets. |
| `paint_zone` in `PlanMapEditor` was speculative | Zoning tab is the first real consumer; fix any rough edges as they surface. |

## 12. Open items to confirm during implementation

1. `ModelGeoResult` does **not** currently expose `max_building_height_m`
   (verified). Implementation will compute it on Zoning-tab mount from
   `voxcity.buildings.heights.max()` (or the equivalent already in the
   client-side `geo` payload's `building_height_grid`).
2. Each simulator's mesh return shape — verified to vary by sim type and
   target. `app/backend/zoning.py` will normalize to a uniform
   `(centroids_xy, values, areas)` tuple via per-sim adapters; first
   implementation step is a dry-run inspection of solar/view/landmark
   building-target mesh outputs.

## 13. File touch list

**New:**

- `app/backend/zoning.py`
- `app/frontend/src/tabs/ZoningTab.tsx`
- `app/frontend/src/types/zones.ts`
- `app/frontend/src/hooks/useZoneStats.ts`
- `app/frontend/src/components/ZoneStatsTable.tsx`
- `app/frontend/src/lib/zoneTraces.ts`
- `tests/app/test_zones.py`

**Modified:**

- `app/backend/main.py` (add `/api/zones/stats` route)
- `app/backend/models.py` (add `ZoneSpec`, `ZoneStatsRequest`, `ZoneStat`,
  `ZoneStatsResponse`)
- `app/frontend/src/App.tsx` (add `Zoning` tab + `zones` state + clear on
  rectangle change)
- `app/frontend/src/api.ts` (add `getZoneStats` helper)
- `app/frontend/src/tabs/SolarTab.tsx`,
  `app/frontend/src/tabs/ViewTab.tsx`,
  `app/frontend/src/tabs/LandmarkTab.tsx` (consume `zones`, render table +
  curtain outlines)
- `app/frontend/src/index.css` (any minor styles for `ZoneStatsTable`)
