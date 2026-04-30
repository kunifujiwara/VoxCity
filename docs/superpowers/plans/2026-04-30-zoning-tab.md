# Zoning Tab Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a Zoning tab to the VoxCity web app that lets users draw 2D zone footprints over the current model and view per-zone summary statistics on every simulation tab (Solar, View, Landmark).

**Architecture:** Frontend owns zone state (lon/lat polygons in `App.tsx`). A single new stateless backend endpoint `POST /api/zones/stats` aggregates the zones against the cached `last_sim_grid` (ground) or `last_sim_mesh` (building-surface) result. 3D zone outlines are rendered client-side as Plotly `Mesh3d` curtains + `Scatter3d` edge lines layered on top of the existing model figure. Existing simulation endpoints are unchanged.

**Tech Stack:** FastAPI + Pydantic v2 (backend), React + TypeScript + Vite + Leaflet + Plotly.js (frontend), pytest (tests).

**Spec:** [docs/superpowers/specs/2026-04-30-zoning-tab-design.md](docs/superpowers/specs/2026-04-30-zoning-tab-design.md)

---

## File structure

**New:**

| File | Responsibility |
| --- | --- |
| `app/backend/zoning.py` | Pure helpers: `polygon_lonlat_to_cells`, `points_in_polygon_lonlat`, `stats_from_values`, `stats_from_mesh_faces`. |
| `app/frontend/src/types/zones.ts` | `Zone` interface + helpers (`makeZoneId`, `nextZoneName`, `nextZoneColor`). |
| `app/frontend/src/lib/zoneTraces.ts` | Pure builder: `(zones, geo, max_h) → Plotly traces[]` (curtain `Mesh3d` + edge `Scatter3d`). |
| `app/frontend/src/hooks/useZoneStats.ts` | Debounced fetch keyed by `(simRunNonce, hash(zones))`. |
| `app/frontend/src/components/ZoneStatsTable.tsx` | Renders the stats table + CSV export button. Used by all 3 sim tabs. |
| `app/frontend/src/tabs/ZoningTab.tsx` | The new tab itself. |
| `tests/app/test_zones.py` | Backend unit + integration tests. |

**Modified:**

| File | Change |
| --- | --- |
| `app/backend/models.py` | Add `ZoneSpec`, `ZoneStatsRequest`, `ZoneStat`, `ZoneStatsResponse`. |
| `app/backend/main.py` | Add `POST /api/zones/stats` route. |
| `app/frontend/src/api.ts` | Add `getZoneStats()` + types `ZoneStat`, `ZoneStatsResponse`. |
| `app/frontend/src/App.tsx` | Add `zones` state, `useEffect([rectangle])` clearing, register `Zoning` tab, pass `zones` to sim tabs + Zoning tab. |
| `app/frontend/src/tabs/SolarTab.tsx` | Accept `zones` prop, render `<ZoneStatsTable>`, add curtain traces to figure. |
| `app/frontend/src/tabs/ViewTab.tsx` | Same as Solar. |
| `app/frontend/src/tabs/LandmarkTab.tsx` | Same as Solar. |
| `app/frontend/src/index.css` | Minor styles for `.zone-list`, `.zone-row`, `.zone-stats-table`. |

---

## Chunk 1: Backend — endpoint + tests

### Task 1.1: Add Pydantic models

**Files:**
- Modify: `app/backend/models.py` (append after `StatusResponse`)

- [ ] **Step 1.1.1: Append models**

```python
# ---------------------------------------------------------------------------
# Zoning
# ---------------------------------------------------------------------------

class ZoneSpec(BaseModel):
    """A 2D zone footprint as a lon/lat ring (does not need to be closed)."""
    id: str
    name: str
    ring_lonlat: List[List[float]] = Field(..., min_length=3)


class ZoneStatsRequest(BaseModel):
    zones: List[ZoneSpec]


class ZoneStat(BaseModel):
    zone_id: str
    cell_count: int            # cells/faces inside the zone
    valid_count: int           # of those, with finite values
    mean: Optional[float] = None
    min:  Optional[float] = None
    max:  Optional[float] = None
    std:  Optional[float] = None


class ZoneStatsResponse(BaseModel):
    target:     str            # "ground" | "building"
    sim_type:   Optional[str] = None  # "solar" | "view" | "landmark"
    unit_label: Optional[str] = None
    stats:      List[ZoneStat]
```

- [ ] **Step 1.1.2: Commit**

```bash
git add app/backend/models.py
git commit -m "feat(api): add zoning request/response pydantic models"
```

---

### Task 1.2: Pure aggregation helpers (`zoning.py`)

**Files:**
- Create: `app/backend/zoning.py`
- Test: `tests/app/test_zones.py`

- [ ] **Step 1.2.1: Write the failing tests**

Create `tests/app/__init__.py` (empty) if it does not exist, then `tests/app/test_zones.py`:

```python
"""Unit tests for app.backend.zoning helpers."""
import numpy as np
import pytest

from app.backend.zoning import (
    polygon_lonlat_to_cells,
    points_in_polygon_lonlat,
    stats_from_values,
)


# ---- Fixtures --------------------------------------------------------------

@pytest.fixture
def grid_geom_axis_aligned():
    """A trivial 10×10 axis-aligned grid: cell centres at (i+0.5, j+0.5)."""
    return {
        "origin":    [0.0, 0.0],
        "u_vec":     [1.0, 0.0],
        "v_vec":     [0.0, 1.0],
        "adj_mesh":  [1.0, 1.0],
        "grid_size": [10, 10],
    }


# ---- polygon_lonlat_to_cells ----------------------------------------------

def test_rect_polygon_returns_inside_cells(grid_geom_axis_aligned):
    # Square covering centres (0.5..3.5, 0.5..3.5) ⇒ 4×4 = 16 cells.
    ring = [[0.0, 0.0], [4.0, 0.0], [4.0, 4.0], [0.0, 4.0]]
    cells = polygon_lonlat_to_cells(ring, grid_geom_axis_aligned)
    assert len(cells) == 16
    assert (0, 0) in cells and (3, 3) in cells

def test_polygon_outside_grid_returns_empty(grid_geom_axis_aligned):
    ring = [[100.0, 100.0], [101.0, 100.0], [101.0, 101.0]]
    assert polygon_lonlat_to_cells(ring, grid_geom_axis_aligned) == []

def test_degenerate_polygon_returns_empty(grid_geom_axis_aligned):
    assert polygon_lonlat_to_cells([[0.0, 0.0], [1.0, 0.0]], grid_geom_axis_aligned) == []


# ---- stats_from_values -----------------------------------------------------

def test_stats_finite_values():
    vals = np.array([1.0, 2.0, 3.0, 4.0])
    s = stats_from_values("z1", cell_count=4, values=vals)
    assert s.cell_count == 4
    assert s.valid_count == 4
    assert s.mean == pytest.approx(2.5)
    assert s.min == 1.0 and s.max == 4.0
    assert s.std == pytest.approx(np.std(vals))

def test_stats_with_nan_values():
    vals = np.array([1.0, np.nan, 3.0, np.inf])
    s = stats_from_values("z1", cell_count=4, values=vals)
    assert s.cell_count == 4
    assert s.valid_count == 2
    assert s.mean == pytest.approx(2.0)

def test_stats_empty_zone():
    s = stats_from_values("z1", cell_count=0, values=np.array([], dtype=float))
    assert s.cell_count == 0 and s.valid_count == 0
    assert s.mean is None and s.std is None


# ---- points_in_polygon_lonlat ---------------------------------------------

def test_points_in_polygon():
    ring = [[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]]
    pts = np.array([[5.0, 5.0], [-1.0, 5.0], [11.0, 5.0]])
    mask = points_in_polygon_lonlat(pts, ring)
    assert mask.tolist() == [True, False, False]
```

- [ ] **Step 1.2.2: Run tests to verify they fail**

Run: `pytest tests/app/test_zones.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.backend.zoning'`.

- [ ] **Step 1.2.3: Implement `app/backend/zoning.py`**

```python
"""Pure helpers for zone aggregation. No FastAPI / state imports here."""
from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np
from matplotlib.path import Path as MplPath

from .models import ZoneStat


def _cell_centres_lonlat(grid_geom: dict) -> np.ndarray:
    """Return an (nx*ny, 2) array of cell-centre lon/lat coords."""
    nx, ny = grid_geom["grid_size"]
    dx, dy = grid_geom["adj_mesh"]
    o = np.asarray(grid_geom["origin"], dtype=float)
    u = np.asarray(grid_geom["u_vec"], dtype=float)
    v = np.asarray(grid_geom["v_vec"], dtype=float)
    ii, jj = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
    centres = (
        o[None, None, :]
        + (ii[..., None] + 0.5) * dx * u[None, None, :]
        + (jj[..., None] + 0.5) * dy * v[None, None, :]
    )
    return centres.reshape(-1, 2), ii.ravel(), jj.ravel()


def polygon_lonlat_to_cells(
    ring: Sequence[Sequence[float]],
    grid_geom: dict,
) -> List[Tuple[int, int]]:
    """Rasterize a closed lon/lat ring to NORTH_UP (i, j) cells.

    Mirrors the JS `polygonToCells` in `app/frontend/src/lib/grid.ts`.
    """
    if len(ring) < 3:
        return []
    centres, ii, jj = _cell_centres_lonlat(grid_geom)
    path = MplPath(np.asarray(ring, dtype=float))
    inside = path.contains_points(centres)
    if not inside.any():
        return []
    return list(zip(ii[inside].tolist(), jj[inside].tolist()))


def points_in_polygon_lonlat(
    points_lonlat: np.ndarray,
    ring: Sequence[Sequence[float]],
) -> np.ndarray:
    """Vectorized point-in-polygon test. `points_lonlat` is (N, 2)."""
    if len(ring) < 3 or points_lonlat.size == 0:
        return np.zeros(len(points_lonlat), dtype=bool)
    return MplPath(np.asarray(ring, dtype=float)).contains_points(points_lonlat)


def stats_from_values(
    zone_id: str,
    cell_count: int,
    values: np.ndarray,
    weights: np.ndarray | None = None,
) -> ZoneStat:
    """Compute count/valid/mean/min/max/std. `weights` enables area-weighted mean."""
    if values.size == 0:
        return ZoneStat(zone_id=zone_id, cell_count=int(cell_count), valid_count=0)
    finite = np.isfinite(values)
    if not finite.any():
        return ZoneStat(zone_id=zone_id, cell_count=int(cell_count), valid_count=0)
    v = values[finite]
    if weights is not None:
        w = weights[finite]
        if w.sum() > 0:
            mean = float((v * w).sum() / w.sum())
        else:
            mean = float(v.mean())
    else:
        mean = float(v.mean())
    return ZoneStat(
        zone_id=zone_id,
        cell_count=int(cell_count),
        valid_count=int(finite.sum()),
        mean=mean,
        min=float(v.min()),
        max=float(v.max()),
        std=float(v.std()),
    )
```

- [ ] **Step 1.2.4: Run tests to verify they pass**

Run: `pytest tests/app/test_zones.py -v`
Expected: 6 passed.

- [ ] **Step 1.2.5: Commit**

```bash
git add app/backend/zoning.py tests/app/__init__.py tests/app/test_zones.py
git commit -m "feat(zoning): add pure aggregation helpers"
```

---

### Task 1.3: Investigate building-surface mesh shape

This is a **read-only investigation** before writing the building-surface adapter — the spec marked this as an open item.

**Files:** none (investigation only)

- [ ] **Step 1.3.1: Identify mesh return shapes**

In each of `app/backend/main.py` ≈ lines 1006-1010 (solar building), 1155 (view building), and the equivalent landmark building handler:
- Read what type of object is assigned to `app_state.last_sim_mesh`.
- Identify whether per-face centroids, values, and areas can be obtained directly or need to be computed from triangle vertices.

Common voxcity surface-result patterns to look for: `trimesh.Trimesh`-like objects with `.triangles` / `.area_faces`, or a tuple `(vertices, faces, values)`. Check `voxcity.simulator_gpu.solar` and `voxcity.simulator_gpu.visibility` modules.

- [ ] **Step 1.3.2: Decide adapter strategy**

Document in your scratch notes which one of these applies:
1. All three return the same mesh type → write **one** adapter function `mesh_face_data(mesh) → (centroids_xy_local, values, areas)`.
2. They differ → write per-sim adapters dispatched by `app_state.last_sim_type`.

This decision drives Task 1.4.

---

### Task 1.4: Endpoint handler

**Files:**
- Modify: `app/backend/main.py`
- Modify: `tests/app/test_zones.py`

- [ ] **Step 1.4.1: Write failing endpoint tests**

Append to `tests/app/test_zones.py`:

```python
from fastapi.testclient import TestClient

from app.backend.main import app
from app.backend.state import app_state


@pytest.fixture
def client():
    return TestClient(app)


def test_zone_stats_no_model(client, monkeypatch):
    monkeypatch.setattr(app_state, "voxcity", None)
    r = client.post("/api/zones/stats", json={"zones": []})
    assert r.status_code == 400
    assert "No model" in r.json()["detail"]


def test_zone_stats_no_sim(client, monkeypatch):
    monkeypatch.setattr(app_state, "voxcity", object())
    monkeypatch.setattr(app_state, "last_sim_type", None)
    r = client.post("/api/zones/stats", json={"zones": []})
    assert r.status_code == 400
    assert "simulation" in r.json()["detail"].lower()


def test_zone_stats_ground_basic(client, monkeypatch):
    """A 4×4 ramp grid + 2 zones (one inside, one outside)."""
    grid = np.arange(16, dtype=float).reshape(4, 4)
    monkeypatch.setattr(app_state, "voxcity", object())
    monkeypatch.setattr(app_state, "last_sim_type", "solar")
    monkeypatch.setattr(app_state, "last_sim_target", "ground")
    monkeypatch.setattr(app_state, "last_sim_grid", grid)
    monkeypatch.setattr(app_state, "last_sim_mesh", None)
    monkeypatch.setattr(app_state, "last_colorbar_title", "W/m²")
    monkeypatch.setattr(app_state, "rectangle_vertices", [[0, 0], [4, 0], [4, 4], [0, 4]])
    monkeypatch.setattr(app_state, "meshsize", 1.0)
    # The handler must call compute_grid_geometry — for the test we patch it.
    import app.backend.main as main_mod
    monkeypatch.setattr(
        main_mod, "_grid_geom_for_zoning",
        lambda: {"origin":[0.0,0.0],"u_vec":[1.0,0.0],"v_vec":[0.0,1.0],
                 "adj_mesh":[1.0,1.0],"grid_size":[4,4]},
        raising=False,
    )
    r = client.post("/api/zones/stats", json={"zones": [
        {"id": "z1", "name": "all",     "ring_lonlat": [[0,0],[4,0],[4,4],[0,4]]},
        {"id": "z2", "name": "outside", "ring_lonlat": [[100,100],[101,100],[101,101]]},
    ]})
    assert r.status_code == 200
    body = r.json()
    assert body["target"] == "ground"
    assert body["unit_label"] == "W/m²"
    by_id = {s["zone_id"]: s for s in body["stats"]}
    assert by_id["z1"]["cell_count"] == 16
    assert by_id["z1"]["mean"] == pytest.approx(grid.mean())
    assert by_id["z2"]["cell_count"] == 0
    assert by_id["z2"]["mean"] is None
```

- [ ] **Step 1.4.2: Run to verify failure**

Run: `pytest tests/app/test_zones.py::test_zone_stats_no_model tests/app/test_zones.py::test_zone_stats_no_sim tests/app/test_zones.py::test_zone_stats_ground_basic -v`
Expected: 3 failures (`/api/zones/stats` does not exist).

- [ ] **Step 1.4.3: Implement the endpoint**

In `app/backend/main.py`, near the other sim handlers, add (and update the imports at the top of the file accordingly):

```python
from .models import (
    # ... existing imports ...
    ZoneSpec,
    ZoneStatsRequest,
    ZoneStat,
    ZoneStatsResponse,
)
from .zoning import (
    polygon_lonlat_to_cells,
    points_in_polygon_lonlat,
    stats_from_values,
)


def _grid_geom_for_zoning() -> dict:
    """Build the same grid_geom dict that /api/model/geo returns."""
    from voxcity.geoprocessor.draw._common import compute_grid_geometry
    rect = app_state.rectangle_vertices
    if rect is None:
        raise HTTPException(status_code=400, detail="No rectangle_vertices on model")
    gg = compute_grid_geometry(rect, float(app_state.meshsize))
    if gg is None:
        raise HTTPException(status_code=500, detail="compute_grid_geometry returned None")
    return {
        "origin":    [float(gg["origin"][0]),  float(gg["origin"][1])],
        "u_vec":     [float(gg["u_vec"][0]),   float(gg["u_vec"][1])],
        "v_vec":     [float(gg["v_vec"][0]),   float(gg["v_vec"][1])],
        "adj_mesh":  [float(gg["adj_mesh"][0]),float(gg["adj_mesh"][1])],
        "grid_size": [int(gg["grid_size"][0]), int(gg["grid_size"][1])],
    }


def _zone_stats_ground(zones: list[ZoneSpec]) -> list[ZoneStat]:
    sim = app_state.last_sim_grid
    grid_geom = _grid_geom_for_zoning()
    out: list[ZoneStat] = []
    for z in zones:
        cells = polygon_lonlat_to_cells(z.ring_lonlat, grid_geom)
        if not cells:
            out.append(stats_from_values(z.id, 0, np.array([], dtype=float)))
            continue
        ii = np.fromiter((c[0] for c in cells), dtype=int, count=len(cells))
        jj = np.fromiter((c[1] for c in cells), dtype=int, count=len(cells))
        # Clip in case mesh shrank (defensive).
        nx, ny = sim.shape
        valid_idx = (ii >= 0) & (ii < nx) & (jj >= 0) & (jj < ny)
        ii = ii[valid_idx]; jj = jj[valid_idx]
        vals = sim[ii, jj].astype(float, copy=False)
        out.append(stats_from_values(z.id, len(cells), vals))
    return out


def _zone_stats_building(zones: list[ZoneSpec]) -> list[ZoneStat]:
    """Building-surface aggregation. Adapter shape decided in Task 1.3."""
    # Implement per the strategy chosen in Task 1.3 step 1.3.2.
    # Expected to produce, for the cached mesh:
    #   centroids_lonlat: (F, 2)  -- face centroids projected to lon/lat
    #   values:          (F,)     -- per-face metric value
    #   areas:           (F,)     -- per-face area in m² (for area-weighted mean)
    raise HTTPException(status_code=501, detail="building-surface zoning not yet implemented")


@app.post("/api/zones/stats", response_model=ZoneStatsResponse)
def zone_stats(req: ZoneStatsRequest) -> ZoneStatsResponse:
    if app_state.voxcity is None:
        raise HTTPException(status_code=400, detail="No model loaded")
    if app_state.last_sim_type is None:
        raise HTTPException(status_code=400, detail="Run a simulation first")

    target = app_state.last_sim_target
    if target == "ground":
        stats = _zone_stats_ground(req.zones)
    elif target == "building":
        stats = _zone_stats_building(req.zones)
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported target: {target}")

    return ZoneStatsResponse(
        target=target,
        sim_type=app_state.last_sim_type,
        unit_label=app_state.last_colorbar_title,
        stats=stats,
    )
```

- [ ] **Step 1.4.4: Run tests to verify they pass**

Run: `pytest tests/app/test_zones.py -v`
Expected: all 9 tests pass (6 helpers + 3 endpoint).

- [ ] **Step 1.4.5: Commit**

```bash
git add app/backend/main.py tests/app/test_zones.py
git commit -m "feat(api): add POST /api/zones/stats for ground-target sims"
```

---

### Task 1.5: Implement `_zone_stats_building` per investigation

**Files:**
- Modify: `app/backend/main.py` (`_zone_stats_building`)
- Optionally: `app/backend/zoning.py` (helpers like `mesh_face_data`)
- Modify: `tests/app/test_zones.py`

- [ ] **Step 1.5.1: Add a small helper(s) in `zoning.py`**

Based on Task 1.3 outcome, expose a single normalizer (e.g.
`def mesh_face_data(mesh, sim_type: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]`)
that returns `(centroids_xy_local, values, areas)` where `centroids_xy_local`
are in grid-local meters (matching the convention used by
`compute_grid_geometry`'s `u_vec`/`v_vec` axes), then a separate
`grid_xy_to_lonlat(xy_local, grid_geom) -> ndarray` helper.

- [ ] **Step 1.5.2: Write a focused unit test for the building path**

Construct a synthetic mesh-like object (e.g., a `SimpleNamespace` or a
small dataclass) with hand-set face centroids, values, and areas; assert
`stats_from_values(...)` with `weights=areas` produces the expected
area-weighted mean.

- [ ] **Step 1.5.3: Implement the endpoint branch**

Replace the `501` body with:

```python
def _zone_stats_building(zones: list[ZoneSpec]) -> list[ZoneStat]:
    mesh = app_state.last_sim_mesh
    centroids_xy, values, areas = mesh_face_data(mesh, app_state.last_sim_type)
    grid_geom = _grid_geom_for_zoning()
    centroids_lonlat = grid_xy_to_lonlat(centroids_xy, grid_geom)
    out: list[ZoneStat] = []
    for z in zones:
        mask = points_in_polygon_lonlat(centroids_lonlat, z.ring_lonlat)
        v = values[mask]; a = areas[mask]
        out.append(stats_from_values(z.id, int(mask.sum()), v, weights=a))
    return out
```

- [ ] **Step 1.5.4: Run all backend tests**

Run: `pytest tests/app/test_zones.py -v`
Expected: all pass.

- [ ] **Step 1.5.5: Commit**

```bash
git add app/backend/main.py app/backend/zoning.py tests/app/test_zones.py
git commit -m "feat(api): implement building-surface zoning aggregation"
```

---

## Chunk 2: Frontend — Zoning tab

### Task 2.1: Zone types + API client

**Files:**
- Create: `app/frontend/src/types/zones.ts`
- Modify: `app/frontend/src/api.ts`

- [ ] **Step 2.1.1: Create `types/zones.ts`**

```ts
export type ZoneShape = 'rect' | 'polygon';

export interface Zone {
  id: string;
  name: string;
  color: string;
  shape: ZoneShape;
  ring_lonlat: [number, number][];
}

export const ZONE_PALETTE: string[] = [
  '#e6194B', '#3cb44b', '#ffe119', '#4363d8',
  '#f58231', '#911eb4', '#42d4f4', '#f032e6',
];

export function makeZoneId(): string {
  return `z_${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 8)}`;
}

export function nextZoneName(existing: Zone[]): string {
  const used = new Set(existing.map((z) => z.name));
  let n = existing.length + 1;
  while (used.has(`Zone ${n}`)) n += 1;
  return `Zone ${n}`;
}

export function nextZoneColor(existing: Zone[]): string {
  return ZONE_PALETTE[existing.length % ZONE_PALETTE.length];
}

export function hashZones(zones: Zone[]): string {
  // Stable, cheap hash for hook dep keys.
  return zones
    .map((z) => `${z.id}:${z.ring_lonlat.map((p) => p.join(',')).join('|')}`)
    .join(';');
}
```

- [ ] **Step 2.1.2: Add API client + types**

Append to `app/frontend/src/api.ts`:

```ts
export interface ZoneStat {
  zone_id: string;
  cell_count: number;
  valid_count: number;
  mean: number | null;
  min:  number | null;
  max:  number | null;
  std:  number | null;
}

export interface ZoneStatsResponse {
  target: 'ground' | 'building';
  sim_type: 'solar' | 'view' | 'landmark' | null;
  unit_label: string | null;
  stats: ZoneStat[];
}

export interface ZoneSpecDto {
  id: string;
  name: string;
  ring_lonlat: [number, number][];
}

export async function getZoneStats(zones: ZoneSpecDto[]) {
  return request<ZoneStatsResponse>('/zones/stats', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ zones }),
  });
}
```

- [ ] **Step 2.1.3: Type-check**

Run: `cd app/frontend && npx tsc --noEmit`
Expected: no new errors.

- [ ] **Step 2.1.4: Commit**

```bash
git add app/frontend/src/types/zones.ts app/frontend/src/api.ts
git commit -m "feat(zoning): add Zone type and zone stats API client"
```

---

### Task 2.2: 3D curtain trace builder

**Files:**
- Create: `app/frontend/src/lib/zoneTraces.ts`

- [ ] **Step 2.2.1: Implement**

```ts
import type { Zone } from '../types/zones';
import type { GridGeom } from './grid';

/** Convert a single lon/lat point to grid-local meters using grid_geom.
 *  Inverse of the centre formula in `cellCentre` (lib/grid.ts). */
function lonlatToGridXy(lon: number, lat: number, g: GridGeom): [number, number] {
  // The grid is parameterized by origin + i*dx*u + j*dy*v.
  // We approximate (i, j) by solving the 2×2 linear system and convert
  // back to grid-local meters by multiplying by meshsize. We don't need
  // exact metric coords for visualization — only the *relative* layout
  // expected by Plotly's voxel mesh. The voxel figure already places voxels
  // in (i, j, k)*meshsize, so we just need (i, j) in cell units * meshsize.
  // Solve [u_vec[0] dx, v_vec[0] dy] [i] = [lon - o_lon]
  //       [u_vec[1] dx, v_vec[1] dy] [j]   [lat - o_lat]
  const [dx, dy] = g.adj_mesh;
  const a = g.u_vec[0] * dx, b = g.v_vec[0] * dy;
  const c = g.u_vec[1] * dx, d = g.v_vec[1] * dy;
  const det = a * d - b * c;
  if (Math.abs(det) < 1e-30) return [0, 0];
  const dl = lon - g.origin[0];
  const dla = lat - g.origin[1];
  const i = (dl * d - dla * b) / det;
  const j = (a * dla - c * dl) / det;
  return [i, j];
}

export interface ZoneTraceOpts {
  meshsize: number;          // meters per cell
  ceilingM: number;          // curtain top (m above ground)
  selectedZoneId?: string | null;
}

/** Build Plotly traces (one Mesh3d + one Scatter3d edge per zone). */
export function buildZoneTraces(
  zones: Zone[],
  geo: { grid_geom: GridGeom },
  opts: ZoneTraceOpts,
): any[] {
  const traces: any[] = [];
  for (const z of zones) {
    const ringIJ = z.ring_lonlat.map(([lon, lat]) => lonlatToGridXy(lon, lat, geo.grid_geom));
    const ringXY = ringIJ.map(([i, j]) => [i * opts.meshsize, j * opts.meshsize] as [number, number]);
    if (ringXY.length < 3) continue;
    const top = opts.ceilingM;
    const isSel = z.id === opts.selectedZoneId;

    // Mesh3d: a vertical curtain panel for each ring edge, two triangles per edge.
    const x: number[] = [], y: number[] = [], zz: number[] = [];
    const i0: number[] = [], i1: number[] = [], i2: number[] = [];
    for (let k = 0; k < ringXY.length; k++) {
      const [ax, ay] = ringXY[k];
      const [bx, by] = ringXY[(k + 1) % ringXY.length];
      const base = x.length;
      // 4 corners of the panel: (a, 0) (b, 0) (b, top) (a, top)
      x.push(ax, bx, bx, ax);
      y.push(ay, by, by, ay);
      zz.push(0, 0, top, top);
      i0.push(base, base);
      i1.push(base + 1, base + 2);
      i2.push(base + 2, base + 3);
    }
    traces.push({
      type: 'mesh3d',
      x, y, z: zz, i: i0, j: i1, k: i2,
      color: z.color,
      opacity: isSel ? 0.5 : 0.35,
      flatshading: true,
      lighting: { ambient: 1, diffuse: 0, specular: 0 },
      hoverinfo: 'skip',
      name: z.name,
      showlegend: false,
    });

    // Edge trace: top ring + corner verticals.
    const ex: number[] = [], ey: number[] = [], ez: number[] = [];
    // top loop
    for (let k = 0; k <= ringXY.length; k++) {
      const [ax, ay] = ringXY[k % ringXY.length];
      ex.push(ax); ey.push(ay); ez.push(top);
    }
    // verticals (use null gaps to break the line between segments)
    for (const [ax, ay] of ringXY) {
      ex.push(NaN as any, ax, ax);
      ey.push(NaN as any, ay, ay);
      ez.push(NaN as any, 0, top);
    }
    traces.push({
      type: 'scatter3d',
      mode: 'lines',
      x: ex, y: ey, z: ez,
      line: { color: z.color, width: isSel ? 8 : 6 },
      hoverinfo: 'skip',
      name: z.name,
      showlegend: false,
    });
  }
  return traces;
}
```

- [ ] **Step 2.2.2: Type-check**

Run: `cd app/frontend && npx tsc --noEmit`
Expected: no new errors.

- [ ] **Step 2.2.3: Commit**

```bash
git add app/frontend/src/lib/zoneTraces.ts
git commit -m "feat(zoning): add 3D curtain + edge trace builder"
```

---

### Task 2.3: Wire `App.tsx` and add Zoning to tab list

**Files:**
- Modify: `app/frontend/src/App.tsx`

- [ ] **Step 2.3.1: Add `zones` state, lifecycle effect, and tab registration**

Make these surgical edits in `App.tsx`:

1. Import the type:

```ts
import type { Zone } from './types/zones';
import ZoningTab from './tabs/ZoningTab';
```

2. Add to `TABS` array, between `'edit'` and `'solar'`:

```ts
{ id: 'zoning', label: 'Zoning' },
```

3. Add state next to other state:

```ts
const [zones, setZones] = useState<Zone[]>([]);
```

4. Add a clearing effect (this also fixes the stale-figure UX bug surfaced in spec review):

```ts
// When the user changes the target rectangle, the previous zones and any
// cached simulation figures no longer correspond to the area on screen.
useEffect(() => {
  setZones([]);
  setFigureJson('');
  setEditFigureJson('');
  setSolarFigureJson('');
  setViewFigureJson('');
  setLandmarkFigureJson('');
}, [rectangle]);
```

5. Render the new tab and pass `zones` to consumers:

```tsx
{activeTab === 'zoning' && (
  <ZoningTab hasModel={hasModel} figureJson={figureJson} zones={zones} onZonesChange={setZones} />
)}
{activeTab === 'solar'    && <SolarTab    hasModel={hasModel} figureJson={solarFigureJson}    onFigureChange={setSolarFigureJson}    zones={zones} />}
{activeTab === 'view'     && <ViewTab     hasModel={hasModel} figureJson={viewFigureJson}     onFigureChange={setViewFigureJson}     zones={zones} />}
{activeTab === 'landmark' && <LandmarkTab hasModel={hasModel} figureJson={landmarkFigureJson} onFigureChange={setLandmarkFigureJson} zones={zones} />}
```

- [ ] **Step 2.3.2: Type-check (will surface mismatched signatures we fix next)**

Run: `cd app/frontend && npx tsc --noEmit`
Expected: errors about `zones` prop on Solar/View/Landmark/Zoning tabs (those tabs don't accept it yet) — leave them; the next tasks fix them.

- [ ] **Step 2.3.3: Commit**

```bash
git add app/frontend/src/App.tsx
git commit -m "feat(app): add zones state, Zoning tab, and rectangle-change clearing"
```

---

### Task 2.4: Build the Zoning tab

**Files:**
- Create: `app/frontend/src/tabs/ZoningTab.tsx`
- Modify: `app/frontend/src/index.css`

- [ ] **Step 2.4.1: Implement `ZoningTab.tsx`**

Use `EditTab.tsx` as the structural template. The component receives:

```ts
interface ZoningTabProps {
  hasModel: boolean;
  figureJson: string;            // current model figure (cached at parent)
  zones: Zone[];
  onZonesChange: (z: Zone[]) => void;
}
```

Behavior contract for this step:

1. Same gating banner as `EditTab` when `!hasModel`.
2. `useEffect(() => { getModelGeo().then(setGeo) }, [hasModel])`.
3. Compute `maxBuildingHeightM` once on mount from `geo` (fall back: walk
   `geo.building_geojson` features for `properties.height`; default 30 m).
4. Layout: `<div className="three-col">`:
   - **Left `.panel`** — title, toolbar (Mode = Add/Replace; Shape = Rect/Polygon; Clear all), `ZoneList` (rows: swatch • name [✎] [🗑]).
   - **Center** — `<PlanMapEditor>` with `interaction = currentInteraction`, `drawColor = activeColor`, `pendingEdits = zones.map(z => ({ kind: 'paint_zone', cells: polygonToCells(z.ring_lonlat, geo.grid_geom), color: z.color, target: 'evaluation' }))`, `onPolygonComplete = handleDrawComplete`.
   - **Right** — `<ThreeViewer>` rendering a parsed `figureJson` with `buildZoneTraces(zones, geo, { meshsize: geo.meshsize_m, ceilingM: Math.max(0.1 * maxBuildingHeightM, 3.0), selectedZoneId })` appended to `figure.data` before render.
5. `handleDrawComplete(ring)`:
   - In `Add` mode: append a new `Zone` (id, name, color via helpers) and set it selected.
   - In `Replace` mode (and `selectedZoneId` set): replace `ring_lonlat` of that zone.
6. `ZoneList` row interactions:
   - Click row → `setSelectedZoneId(z.id)`.
   - `✎` toggles inline `<input>` for renaming.
   - `🗑` confirms then `onZonesChange(zones.filter(...))`.
   - Color swatch click → small popover listing `ZONE_PALETTE` colors.

Implementation note: keep it self-contained. Do not introduce new state-management libraries. Local `useState` for transient UI state (selected id, mode, shape, rename buffer) is sufficient.

- [ ] **Step 2.4.2: Add minimal CSS**

Append to `app/frontend/src/index.css`:

```css
.zone-list { display: flex; flex-direction: column; gap: 4px; margin-top: 12px; }
.zone-row  { display: flex; align-items: center; gap: 8px; padding: 4px 6px; border-radius: 4px; cursor: pointer; }
.zone-row.selected { background: rgba(0, 0, 0, 0.06); }
.zone-row .swatch  { width: 14px; height: 14px; border-radius: 3px; flex: 0 0 14px; border: 1px solid rgba(0,0,0,0.2); cursor: pointer; }
.zone-row .name    { flex: 1; }
.zone-row button   { background: none; border: none; cursor: pointer; padding: 2px 4px; }
```

- [ ] **Step 2.4.3: Type-check + smoke run**

Run: `cd app/frontend && npx tsc --noEmit`
Expected: only the pre-existing errors from Task 2.3 about `zones` on sim tabs (fixed in Chunk 3).

Then run the dev server and manually exercise:
1. Generate a model → switch to Zoning.
2. Draw a rotated rectangle → row appears in list, overlay on 2D, curtain in 3D.
3. Add a polygon zone → second row.
4. Rename, recolor, delete a zone → list/2D/3D stay in sync.
5. Change target rectangle on Target Area tab → zones cleared.

- [ ] **Step 2.4.4: Commit**

```bash
git add app/frontend/src/tabs/ZoningTab.tsx app/frontend/src/index.css
git commit -m "feat(zoning): add Zoning tab UI"
```

---

## Chunk 3: Simulation tab integration

### Task 3.1: Shared stats hook + table component

**Files:**
- Create: `app/frontend/src/hooks/useZoneStats.ts`
- Create: `app/frontend/src/components/ZoneStatsTable.tsx`
- Modify: `app/frontend/src/index.css`

- [ ] **Step 3.1.1: Implement `useZoneStats.ts`**

```ts
import { useEffect, useState } from 'react';
import { getZoneStats, ZoneStatsResponse } from '../api';
import { Zone, hashZones } from '../types/zones';

export function useZoneStats(
  zones: Zone[],
  simRunNonce: number,
): { stats: ZoneStatsResponse | null; loading: boolean; error: string | null } {
  const [stats, setStats] = useState<ZoneStatsResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const key = `${simRunNonce}|${hashZones(zones)}`;

  useEffect(() => {
    if (zones.length === 0 || simRunNonce === 0) {
      setStats(null);
      return;
    }
    let cancelled = false;
    setLoading(true);
    setError(null);
    const handle = window.setTimeout(() => {
      getZoneStats(zones.map((z) => ({ id: z.id, name: z.name, ring_lonlat: z.ring_lonlat })))
        .then((r) => { if (!cancelled) setStats(r); })
        .catch((e: Error) => {
          if (!cancelled) {
            // Swallow the "no sim" 400; treat all other errors as visible.
            if (/simulation/i.test(e.message)) { setStats(null); setError(null); }
            else setError(e.message);
          }
        })
        .finally(() => { if (!cancelled) setLoading(false); });
    }, 150);
    return () => { cancelled = true; window.clearTimeout(handle); };
  }, [key]);

  return { stats, loading, error };
}
```

- [ ] **Step 3.1.2: Implement `ZoneStatsTable.tsx`**

```tsx
import React from 'react';
import { Zone } from '../types/zones';
import { ZoneStatsResponse } from '../api';

interface Props {
  zones: Zone[];
  stats: ZoneStatsResponse | null;
  loading?: boolean;
}

function fmt(n: number | null, d = 2): string {
  return n === null || !isFinite(n) ? '—' : n.toFixed(d);
}

export const ZoneStatsTable: React.FC<Props> = ({ zones, stats, loading }) => {
  if (zones.length === 0) return null;
  const unit = stats?.unit_label ? ` (${stats.unit_label})` : '';
  const byId = new Map((stats?.stats ?? []).map((s) => [s.zone_id, s]));

  return (
    <div className="zone-stats-table">
      <div className="header">Zone statistics{unit}{loading ? ' …' : ''}</div>
      <table>
        <thead>
          <tr><th>Zone</th><th>cells</th><th>mean</th><th>min</th><th>max</th><th>std</th></tr>
        </thead>
        <tbody>
          {zones.map((z) => {
            const s = byId.get(z.id);
            const noData = !s || s.valid_count === 0;
            return (
              <tr key={z.id} className={noData ? 'muted' : ''}>
                <td>
                  <span className="swatch" style={{ background: z.color }} /> {z.name}
                </td>
                <td>{s?.cell_count ?? 0}</td>
                <td>{fmt(s?.mean ?? null)}</td>
                <td>{fmt(s?.min ?? null)}</td>
                <td>{fmt(s?.max ?? null)}</td>
                <td>{fmt(s?.std ?? null)}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
      <button
        className="btn-secondary"
        disabled={!stats}
        onClick={() => downloadCsv(zones, stats!)}
      >
        Export CSV
      </button>
    </div>
  );
};

function downloadCsv(zones: Zone[], stats: ZoneStatsResponse): void {
  const rows = [['zone_id', 'name', 'cell_count', 'valid_count', 'mean', 'min', 'max', 'std']];
  const byId = new Map(stats.stats.map((s) => [s.zone_id, s]));
  for (const z of zones) {
    const s = byId.get(z.id);
    rows.push([
      z.id, JSON.stringify(z.name), String(s?.cell_count ?? 0), String(s?.valid_count ?? 0),
      String(s?.mean ?? ''), String(s?.min ?? ''), String(s?.max ?? ''), String(s?.std ?? ''),
    ]);
  }
  const csv = rows.map((r) => r.join(',')).join('\n');
  const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url; a.download = 'zone_stats.csv'; a.click();
  URL.revokeObjectURL(url);
}

export default ZoneStatsTable;
```

- [ ] **Step 3.1.3: Add minimal CSS**

Append:

```css
.zone-stats-table { margin-top: 12px; font-size: 13px; }
.zone-stats-table .header { font-weight: 600; margin-bottom: 4px; }
.zone-stats-table table { width: 100%; border-collapse: collapse; }
.zone-stats-table th, .zone-stats-table td { padding: 4px 6px; text-align: right; border-bottom: 1px solid rgba(0,0,0,0.08); }
.zone-stats-table th:first-child, .zone-stats-table td:first-child { text-align: left; }
.zone-stats-table tr.muted td { color: #888; }
.zone-stats-table .swatch { display: inline-block; width: 10px; height: 10px; border-radius: 2px; margin-right: 6px; vertical-align: middle; }
```

- [ ] **Step 3.1.4: Type-check**

Run: `cd app/frontend && npx tsc --noEmit`
Expected: no new errors.

- [ ] **Step 3.1.5: Commit**

```bash
git add app/frontend/src/hooks/useZoneStats.ts app/frontend/src/components/ZoneStatsTable.tsx app/frontend/src/index.css
git commit -m "feat(zoning): add useZoneStats hook + ZoneStatsTable component"
```

---

### Task 3.2: Solar tab integration

**Files:**
- Modify: `app/frontend/src/tabs/SolarTab.tsx`

- [ ] **Step 3.2.1: Add prop + hook + table + 3D outlines**

Edits:

1. Extend `SolarTabProps`:
```ts
interface SolarTabProps {
  hasModel: boolean;
  figureJson: string;
  onFigureChange: (json: string) => void;
  zones: Zone[];                  // NEW
}
```
Import `Zone`, `useZoneStats`, `ZoneStatsTable`, `buildZoneTraces`, `getModelGeo`, `ModelGeoResult`.

2. Add a sim-run nonce + geo fetch:
```ts
const [simRunNonce, setSimRunNonce] = useState(0);
const [geo, setGeo] = useState<ModelGeoResult | null>(null);
const [maxH, setMaxH] = useState(30);
const [showZones3D, setShowZones3D] = useState(true);
useEffect(() => { if (hasModel) getModelGeo().then((g) => {
  setGeo(g);
  // Fallback: scan building_geojson for max height.
  let m = 0;
  for (const f of g.building_geojson?.features ?? []) {
    const h = Number(f.properties?.height ?? 0);
    if (h > m) m = h;
  }
  setMaxH(m > 0 ? m : 30);
}); }, [hasModel]);
```

3. In `handleRun` after successful `runSolar` call, bump the nonce:
```ts
setSimRunNonce((n) => n + 1);
```
Same in the `useManualRerender` callback only if rerender returns new values — for the spec-described "no refetch on color settings" behavior, **do not** bump nonce there (color changes don't change values).

4. Use the hook:
```ts
const { stats, loading: statsLoading } = useZoneStats(zones, simRunNonce);
```

5. Augment the rendered figure with curtain traces (only when `showZones3D` and `geo` available). Wherever the existing code parses `figureJson` and passes to `<ThreeViewer>`, splice in:
```ts
const figure = useMemo(() => {
  if (!figureJson) return null;
  const fig = JSON.parse(figureJson);
  if (showZones3D && geo && zones.length) {
    fig.data = [...(fig.data ?? []), ...buildZoneTraces(zones, geo, {
      meshsize: geo.meshsize_m,
      ceilingM: Math.max(0.1 * maxH, 3),
    })];
  }
  return fig;
}, [figureJson, showZones3D, geo, zones, maxH]);
```
Pass `figure` (or its stringified form) to `ThreeViewer` as it already accepts.

6. Render the controls under the form:
```tsx
{zones.length > 0 && (
  <>
    <label className="checkbox">
      <input type="checkbox" checked={showZones3D}
             onChange={(e) => setShowZones3D(e.target.checked)} />
      Show zones in 3D
    </label>
    <ZoneStatsTable zones={zones} stats={stats} loading={statsLoading} />
  </>
)}
```

- [ ] **Step 3.2.2: Type-check + manual smoke**

Run: `cd app/frontend && npx tsc --noEmit`
Expected: pass.

Manual smoke:
1. Generate model, draw 2 zones in Zoning tab.
2. Run Solar (ground). Stats table appears under controls; zones visible as curtains in 3D.
3. Change colormap → no refetch (table unchanged values).
4. Switch to Zoning, edit a zone (rename/move) → return to Solar; the table refreshes without re-running the sim.

- [ ] **Step 3.2.3: Commit**

```bash
git add app/frontend/src/tabs/SolarTab.tsx
git commit -m "feat(zoning): SolarTab — per-zone stats table + 3D curtain overlay"
```

---

### Task 3.3: View tab integration

**Files:**
- Modify: `app/frontend/src/tabs/ViewTab.tsx`

- [ ] **Step 3.3.1: Mirror Task 3.2 in `ViewTab.tsx`**

Apply the identical six edits from Task 3.2.1 to `ViewTab.tsx`. Behavior is the same; only the `runView` call site name differs.

- [ ] **Step 3.3.2: Type-check + smoke**

Same as 3.2.2 but using View. Confirm building-target view also produces stats once Task 1.5 is in place.

- [ ] **Step 3.3.3: Commit**

```bash
git add app/frontend/src/tabs/ViewTab.tsx
git commit -m "feat(zoning): ViewTab — per-zone stats table + 3D curtain overlay"
```

---

### Task 3.4: Landmark tab integration

**Files:**
- Modify: `app/frontend/src/tabs/LandmarkTab.tsx`

- [ ] **Step 3.4.1: Mirror Task 3.2 in `LandmarkTab.tsx`**

Apply the identical edits.

- [ ] **Step 3.4.2: Type-check + smoke**

- [ ] **Step 3.4.3: Commit**

```bash
git add app/frontend/src/tabs/LandmarkTab.tsx
git commit -m "feat(zoning): LandmarkTab — per-zone stats table + 3D curtain overlay"
```

---

## Chunk 4: Final pass

### Task 4.1: Cross-stack consistency snapshot test

**Files:**
- Modify: `tests/app/test_zones.py`

- [ ] **Step 4.1.1: Add a snapshot test for the JS↔Python rasterizer parity**

Pick one non-axis-aligned polygon (4 vertices), hand-compute the expected
cell set against a known small `grid_geom` (the test fixture), and assert
`polygon_lonlat_to_cells` returns exactly that set.

Document in a comment that the JS `polygonToCells` is verified to match
this set by the manual smoke flow (no JS test runner currently configured).

- [ ] **Step 4.1.2: Commit**

```bash
git add tests/app/test_zones.py
git commit -m "test(zoning): snapshot of polygon→cells output"
```

---

### Task 4.2: Final manual end-to-end smoke

- [ ] **Step 4.2.1: Walk the full flow**

1. `python app/run.py` → both servers up.
2. Geocode a city → Generate.
3. Zoning tab: draw 2 rect zones + 1 polygon zone; rename one; delete one.
4. Solar tab (ground): run; verify table values + 3D curtains.
5. Solar tab (building surfaces): run; verify area-weighted means.
6. View tab: run; verify table.
7. Landmark tab: run; verify table.
8. Edit tab: commit a building edit; return to Solar — sim figures cleared (existing behavior); zones preserved.
9. Change target rectangle → zones and all sim figures clear.
10. Reload page → zones gone (frontend-only state, expected).

- [ ] **Step 4.2.2: Final commit if any tweaks were needed**

---

## Done criteria

- All `pytest tests/app/test_zones.py` pass.
- `cd app/frontend && npx tsc --noEmit` is clean.
- Manual smoke in Task 4.2 passes.
- No changes to existing simulation endpoints, request/response shapes, or rendering pipelines.
