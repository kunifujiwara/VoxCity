# App OBJ Import Tab — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a dedicated "Import" tab to the VoxCity web app that uploads a Rhino/OBJ file, positions it over the current model (numeric form + interactive 3D axis-arrow gizmo), and stamps its buildings into the model via `voxcity.importer.add_buildings_from_obj`.

**Architecture:** Two new FastAPI endpoints (`/api/model/import_obj/upload` parses the OBJ and returns groups + lightweight preview geometry; `/api/model/import_obj/commit` runs the importer and replaces `app_state.voxcity`). A new React `ImportTab` holds one shared `Placement` state read/written by a numeric form, a 2D Leaflet footprint overlay, and a 3D `TransformControls` gizmo on an imported-mesh overlay in the R3F `SceneViewer`. Live preview is a client-side visual approximation; the committed result is the exact server-side voxelization.

**Tech Stack:** Python/FastAPI/Pydantic/trimesh (backend), React/TypeScript/Vite/Leaflet/@react-three/fiber/three.js (frontend), pytest + vitest for tests.

**Reference spec:** `docs/superpowers/specs/2026-06-24-app-obj-import-tab-design.md`

---

## File Structure

**Backend (`app/backend/`):**
- Modify `models.py` — add `ImportObjGroup`, `ImportObjPreview`, `ImportObjUploadResponse`, `ImportPlacement`, `ImportObjCommitRequest`, `ImportObjCommitResponse`.
- Modify `main.py` — add `import_obj_store` dict + `/api/model/import_obj/upload` and `/api/model/import_obj/commit` endpoints, plus a `_anchor_lonlat_to_cell` helper.
- Create `test_import_obj.py` — endpoint tests (upload parse, commit apply, auto-elevation, role skip, errors).

**Frontend (`app/frontend/src/`):**
- Create `lib/objPlacement.ts` — `Placement` type, defaults, and the visual preview transform (model XY → scene metres).
- Create `lib/objPlacement.test.ts` — transform + placement-sync unit tests.
- Modify `api.ts` — `uploadImportObj`, `commitImportObj` + their DTO types.
- Create `tabs/ImportTab.tsx` — the tab: uploader, role table, numeric placement form, Advanced section, Import button, embeds map + 3D viewer.
- Create `components/ObjPlacementMap.tsx` — Leaflet footprint overlay (initial-click anchor + draggable footprint), built on the same projection helpers as the Edit tab.
- Modify `three/SceneViewer.tsx` — optional `placementMesh` + gizmo props to render the imported mesh with `TransformControls`.
- Create `three/PlacementGizmo.tsx` — the R3F mesh + `TransformControls` component that emits move/rotation deltas.
- Modify `App.tsx` — register the `import` tab.

---

## Task 1: Backend Pydantic models for OBJ import

**Files:**
- Modify: `app/backend/models.py` (append at end)

- [ ] **Step 1: Add the models**

Append to `app/backend/models.py`:

```python
# ---------------------------------------------------------------------------
# OBJ import (Import tab)
# ---------------------------------------------------------------------------

class ImportObjGroup(BaseModel):
    """One named group parsed from an uploaded OBJ."""
    name: str
    role: str = "building"           # default role from classify_roles
    n_faces: int
    bbox_model: List[List[float]]    # [[xmin,ymin,zmin],[xmax,ymax,zmax]] in model units


class ImportObjPreview(BaseModel):
    """Lightweight geometry for client-side preview (model coordinates)."""
    # Per-group XY footprint outlines (closed rings) in model coords.
    footprints: List[List[List[float]]]
    # Decimated combined mesh for the 3D preview (model coords).
    vertices: List[List[float]]      # [[x,y,z], ...]
    indices: List[List[int]]         # [[a,b,c], ...]


class ImportObjUploadResponse(BaseModel):
    import_id: str
    groups: List[ImportObjGroup]
    model_bounds: List[List[float]]  # [[xmin,ymin,zmin],[xmax,ymax,zmax]]
    preview: ImportObjPreview


class ImportPlacement(BaseModel):
    anchor_lonlat: List[float]                       # [lon, lat]
    anchor_elevation: Optional[float] = None         # None -> auto-sample DEM
    anchor_model_point: List[float] = Field(default_factory=lambda: [0.0, 0.0, 0.0])
    rotation: float = 0.0                            # degrees
    move: List[float] = Field(default_factory=lambda: [0.0, 0.0, 0.0])  # [east, north, up] m
    units: str = "m"
    z_up: bool = True
    swap_yz: bool = False


class ImportObjCommitRequest(BaseModel):
    import_id: str
    placement: ImportPlacement
    roles: Dict[str, str] = Field(default_factory=dict)
    overwrite: bool = True


class ImportObjCommitResponse(BaseModel):
    figure_json: str
    imported_building_ids: List[int]
    n_building_voxels_added: int
    warning: Optional[str] = None
```

- [ ] **Step 2: Verify import compiles**

Run: `& "C:\Users\kunih\miniconda3\Scripts\conda.exe" run -n voxcity python -c "from app.backend.models import ImportObjCommitRequest, ImportObjUploadResponse; print('ok')"`
(Run from the repo root `c:\Users\kunih\OneDrive\00_Codes\python\VoxCity`.)
Expected: prints `ok`.

- [ ] **Step 3: Commit**

```bash
git add app/backend/models.py
git commit -m "feat(app): pydantic models for OBJ import endpoints"
```

---

## Task 2: Backend upload endpoint (`/api/model/import_obj/upload`)

**Files:**
- Modify: `app/backend/main.py`
- Test: `app/backend/test_import_obj.py`

The endpoint stores the uploaded `.obj`, parses it with the importer's `load_obj_groups` + `classify_roles`, and returns group metadata plus decimated preview geometry.

- [ ] **Step 1: Write the failing test**

Create `app/backend/test_import_obj.py`:

```python
"""Tests for the OBJ import endpoints (upload / commit)."""
from __future__ import annotations

import io
import os

import numpy as np
import pytest
import trimesh
from fastapi.testclient import TestClient

from backend.main import app, import_obj_store
from backend.state import app_state

# Reuse the importer's flat-model fixture builder.
from tests.importer.conftest import make_flat_voxcity


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture(autouse=True)
def _model_loaded():
    """Install a small flat model in app_state for the duration of each test."""
    app_state.voxcity = make_flat_voxcity(nx=30, ny=30, nz=12, meshsize=1.0)
    app_state.rectangle_vertices = app_state.voxcity.extras["rectangle_vertices"]
    app_state.land_cover_source = "OpenStreetMap"
    import_obj_store.clear()
    yield
    app_state.voxcity = None
    app_state.rectangle_vertices = None
    import_obj_store.clear()


def _box_obj_bytes() -> bytes:
    """A single 3x3x4 box exported to OBJ, returned as bytes."""
    mesh = trimesh.creation.box(extents=(3.0, 3.0, 4.0))
    mesh.apply_translation((1.5, 1.5, 2.0))  # min corner at origin
    return mesh.export(file_type="obj").encode("utf-8")


def test_upload_returns_groups_and_preview(client):
    files = {"file": ("box.obj", io.BytesIO(_box_obj_bytes()), "text/plain")}
    r = client.post("/api/model/import_obj/upload", files=files)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["import_id"]
    assert len(body["groups"]) >= 1
    assert body["groups"][0]["role"] == "building"
    assert body["groups"][0]["n_faces"] > 0
    assert len(body["preview"]["vertices"]) > 0
    assert len(body["preview"]["indices"]) > 0
    # import_id is registered server-side
    assert body["import_id"] in import_obj_store


def test_upload_rejects_non_obj_garbage(client):
    files = {"file": ("bad.obj", io.BytesIO(b"this is not an obj"), "text/plain")}
    r = client.post("/api/model/import_obj/upload", files=files)
    assert r.status_code == 400


def test_upload_requires_model(client):
    app_state.voxcity = None
    files = {"file": ("box.obj", io.BytesIO(_box_obj_bytes()), "text/plain")}
    r = client.post("/api/model/import_obj/upload", files=files)
    assert r.status_code == 400
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `& "C:\Users\kunih\miniconda3\Scripts\conda.exe" run -n voxcity python -m pytest app/backend/test_import_obj.py -v`
Expected: FAIL — `ImportError: cannot import name 'import_obj_store'` (endpoint not implemented yet).

- [ ] **Step 3: Implement the store + helper + endpoint**

In `app/backend/main.py`, after the `BASE_OUTPUT_DIR` definition (near line 162) add the in-memory store:

```python
# In-memory registry of uploaded OBJ imports: import_id -> stored .obj path.
import_obj_store: Dict[str, str] = {}
```

Add this helper near `_require_model` (around line 2147):

```python
def _anchor_lonlat_to_cell(lon: float, lat: float) -> tuple[int, int]:
    """Project a lon/lat anchor to an (i, j) grid cell using the model grid geom."""
    from voxcity.geoprocessor.draw._common import compute_grid_geometry
    from voxcity.utils.projector import GridProjector

    rect = app_state.rectangle_vertices
    if rect is None and app_state.voxcity is not None and isinstance(app_state.voxcity.extras, dict):
        rect = app_state.voxcity.extras.get("rectangle_vertices")
    if rect is None:
        raise HTTPException(status_code=400, detail="Model has no rectangle_vertices")
    gg = compute_grid_geometry(rect, float(app_state.meshsize))
    i, j = GridProjector(gg).lon_lat_to_cell(float(lon), float(lat))
    return int(i), int(j)
```

Add the endpoint near the other `/api/model/...` routes (e.g. after `apply_edits`, around line 3166):

```python
@app.post("/api/model/import_obj/upload")
async def import_obj_upload(file: UploadFile = File(...)):
    """Parse an uploaded OBJ into groups + preview geometry; register an import_id."""
    _require_model()
    import uuid
    from voxcity.importer.loader import load_obj_groups, classify_roles

    import_id = uuid.uuid4().hex
    dest_dir = os.path.join(BASE_OUTPUT_DIR, "import_obj", import_id)
    os.makedirs(dest_dir, exist_ok=True)
    obj_path = os.path.join(dest_dir, os.path.basename(file.filename) or "model.obj")
    with open(obj_path, "wb") as f:
        f.write(await file.read())

    try:
        groups = load_obj_groups(obj_path)  # [(name, trimesh), ...]; raises ValueError if no mesh
    except (ValueError, FileNotFoundError) as e:
        raise HTTPException(status_code=400, detail=f"Could not parse OBJ: {e}")

    role_map = classify_roles([name for name, _ in groups])

    group_models: List[ImportObjGroup] = []
    footprints: List[List[List[float]]] = []
    all_verts: List[List[float]] = []
    all_faces: List[List[int]] = []
    # Combined-mesh face cap for the preview (decimation: keep every Nth face when large).
    MAX_PREVIEW_FACES = 20000
    total_faces = sum(int(len(m.faces)) for _, m in groups)
    stride = max(1, (total_faces + MAX_PREVIEW_FACES - 1) // MAX_PREVIEW_FACES)

    gxmin = gymin = gzmin = float("inf")
    gxmax = gymax = gzmax = float("-inf")
    for name, mesh in groups:
        verts = np.asarray(mesh.vertices, dtype=float)
        faces = np.asarray(mesh.faces, dtype=int)
        bmin = verts.min(axis=0)
        bmax = verts.max(axis=0)
        gxmin, gymin, gzmin = min(gxmin, bmin[0]), min(gymin, bmin[1]), min(gzmin, bmin[2])
        gxmax, gymax, gzmax = max(gxmax, bmax[0]), max(gymax, bmax[1]), max(gzmax, bmax[2])
        group_models.append(ImportObjGroup(
            name=name,
            role=role_map.get(name, "building"),
            n_faces=int(len(faces)),
            bbox_model=[[float(bmin[0]), float(bmin[1]), float(bmin[2])],
                        [float(bmax[0]), float(bmax[1]), float(bmax[2])]],
        ))
        # Footprint: 2D convex hull of XY projection (closed ring).
        try:
            from scipy.spatial import ConvexHull
            xy = verts[:, :2]
            hull = ConvexHull(xy)
            ring = [[float(xy[v, 0]), float(xy[v, 1])] for v in hull.vertices]
            ring.append(ring[0])
        except Exception:
            ring = [[float(bmin[0]), float(bmin[1])], [float(bmax[0]), float(bmin[1])],
                    [float(bmax[0]), float(bmax[1])], [float(bmin[0]), float(bmax[1])],
                    [float(bmin[0]), float(bmin[1])]]
        footprints.append(ring)
        # Decimated mesh for 3D preview.
        base = len(all_verts)
        all_verts.extend([[float(v[0]), float(v[1]), float(v[2])] for v in verts])
        for fi in range(0, len(faces), stride):
            a, b, c = faces[fi]
            all_faces.append([base + int(a), base + int(b), base + int(c)])

    import_obj_store[import_id] = obj_path
    return ImportObjUploadResponse(
        import_id=import_id,
        groups=group_models,
        model_bounds=[[gxmin, gymin, gzmin], [gxmax, gymax, gzmax]],
        preview=ImportObjPreview(footprints=footprints, vertices=all_verts, indices=all_faces),
    )
```

Ensure the new model names are importable: add them to the `from .models import (...)` block near the top of `main.py` (search for the existing models import list and append `ImportObjGroup, ImportObjPreview, ImportObjUploadResponse, ImportPlacement, ImportObjCommitRequest, ImportObjCommitResponse`). Confirm `UploadFile` and `File` are already imported (they are — used by `/api/epw/upload`).

- [ ] **Step 4: Run the test to verify it passes**

Run: `& "C:\Users\kunih\miniconda3\Scripts\conda.exe" run -n voxcity python -m pytest app/backend/test_import_obj.py -v -k upload`
Expected: PASS (3 upload tests). If `scipy` is unavailable, the bbox-ring fallback path is used — the test still passes because it only checks footprint length indirectly via vertices/indices.

- [ ] **Step 5: Commit**

```bash
git add app/backend/main.py app/backend/test_import_obj.py
git commit -m "feat(app): OBJ import upload endpoint (parse groups + preview geometry)"
```

---

## Task 3: Backend commit endpoint (`/api/model/import_obj/commit`)

**Files:**
- Modify: `app/backend/main.py`
- Test: `app/backend/test_import_obj.py` (add cases)

- [ ] **Step 1: Write the failing test**

Append to `app/backend/test_import_obj.py`:

```python
def _upload_box(client) -> str:
    files = {"file": ("box.obj", io.BytesIO(_box_obj_bytes()), "text/plain")}
    r = client.post("/api/model/import_obj/upload", files=files)
    assert r.status_code == 200, r.text
    return r.json()["import_id"]


def _domain_center_lonlat() -> list[float]:
    rect = app_state.rectangle_vertices
    lons = [p[0] for p in rect]
    lats = [p[1] for p in rect]
    return [sum(lons) / len(lons), sum(lats) / len(lats)]


def test_commit_imports_building(client):
    import_id = _upload_box(client)
    before = int(np.sum(app_state.voxcity.voxels.classes == -3))
    req = {
        "import_id": import_id,
        "placement": {
            "anchor_lonlat": _domain_center_lonlat(),
            "anchor_elevation": None,            # auto-sample DEM
            "anchor_model_point": [0.0, 0.0, 0.0],
            "rotation": 0.0,
            "move": [0.0, 0.0, 0.0],
            "units": "m",
            "z_up": True,
            "swap_yz": False,
        },
        "roles": {},
        "overwrite": True,
    }
    r = client.post("/api/model/import_obj/commit", json=req)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["n_building_voxels_added"] > 0
    assert len(body["imported_building_ids"]) >= 1
    assert body["figure_json"]
    after = int(np.sum(app_state.voxcity.voxels.classes == -3))
    assert after > before


def test_commit_skips_non_building_role(client):
    import_id = _upload_box(client)
    # Find the group name to mark it skip.
    files = {"file": ("box.obj", io.BytesIO(_box_obj_bytes()), "text/plain")}
    name = client.post("/api/model/import_obj/upload", files=files).json()["groups"][0]["name"]
    req = {
        "import_id": import_id,
        "placement": {"anchor_lonlat": _domain_center_lonlat()},
        "roles": {name: "skip"},
        "overwrite": True,
    }
    r = client.post("/api/model/import_obj/commit", json=req)
    assert r.status_code == 200, r.text
    assert r.json()["n_building_voxels_added"] == 0


def test_commit_unknown_import_id_404(client):
    req = {"import_id": "deadbeef", "placement": {"anchor_lonlat": _domain_center_lonlat()}}
    r = client.post("/api/model/import_obj/commit", json=req)
    assert r.status_code == 404
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `& "C:\Users\kunih\miniconda3\Scripts\conda.exe" run -n voxcity python -m pytest app/backend/test_import_obj.py -v -k commit`
Expected: FAIL — 404/405 (commit route not implemented).

- [ ] **Step 3: Implement the commit endpoint**

Add to `app/backend/main.py` after the upload endpoint:

```python
@app.post("/api/model/import_obj/commit", response_model=ImportObjCommitResponse)
async def import_obj_commit(req: ImportObjCommitRequest):
    """Stamp the uploaded OBJ's buildings into the current model and re-render."""
    _require_model()
    from voxcity.importer import add_buildings_from_obj

    obj_path = import_obj_store.get(req.import_id)
    if obj_path is None or not os.path.isfile(obj_path):
        raise HTTPException(status_code=404, detail="Unknown or expired import_id; please re-upload.")

    p = req.placement
    if len(p.anchor_lonlat) != 2:
        raise HTTPException(status_code=400, detail="anchor_lonlat must be [lon, lat]")

    anchor_elev = p.anchor_elevation
    if anchor_elev is None:
        i, j = _anchor_lonlat_to_cell(p.anchor_lonlat[0], p.anchor_lonlat[1])
        dem = np.asarray(app_state.voxcity.dem.elevation)
        nx, ny = dem.shape
        ii = min(max(i, 0), nx - 1)
        jj = min(max(j, 0), ny - 1)
        anchor_elev = float(dem[ii, jj])

    before = int(np.sum(np.asarray(app_state.voxcity.voxels.classes) == -3))
    try:
        out = add_buildings_from_obj(
            app_state.voxcity,
            obj_path,
            anchor_lonlat=(float(p.anchor_lonlat[0]), float(p.anchor_lonlat[1])),
            anchor_elevation=float(anchor_elev),
            anchor_model_point=tuple(float(x) for x in p.anchor_model_point),
            rotation=float(p.rotation),
            move=tuple(float(x) for x in p.move),
            units=p.units,
            roles=req.roles or None,
            z_up=p.z_up,
            swap_yz=p.swap_yz,
            overwrite=req.overwrite,
            backend="trimesh",
        )
    except (ValueError, FileNotFoundError) as e:
        raise HTTPException(status_code=400, detail=str(e))

    app_state.voxcity = out
    app_state.refresh_raw_cache()

    after = int(np.sum(np.asarray(out.voxels.classes) == -3))
    n_added = after - before
    manifest = out.extras.get("imported_buildings") if isinstance(out.extras, dict) else None
    ids: List[int] = []
    if manifest:
        ids = [int(v) for v in (manifest[-1].get("id_map") or {}).values()]
    warning = None if n_added > 0 else (
        "Imported geometry voxelized to 0 cells inside the domain — check anchor/rotation/move/units."
    )
    return ImportObjCommitResponse(
        figure_json=_render_edit_preview(out, title="Imported building"),
        imported_building_ids=ids,
        n_building_voxels_added=int(n_added),
        warning=warning,
    )
```

- [ ] **Step 4: Run the full backend test file**

Run: `& "C:\Users\kunih\miniconda3\Scripts\conda.exe" run -n voxcity python -m pytest app/backend/test_import_obj.py -v`
Expected: PASS (all upload + commit tests).

- [ ] **Step 5: Commit**

```bash
git add app/backend/main.py app/backend/test_import_obj.py
git commit -m "feat(app): OBJ import commit endpoint (stamp buildings + auto elevation)"
```

---

## Task 4: Frontend placement transform + types (`lib/objPlacement.ts`)

**Files:**
- Create: `app/frontend/src/lib/objPlacement.ts`
- Test: `app/frontend/src/lib/objPlacement.test.ts`

This is the testable core for the live preview: a `Placement` type, defaults, and a function that maps a model-space point to scene metres `(x=east, y=north, z=up)` given the placement. It mirrors the *visual* part of `build_placement_transform` (units scale + rotation + move); the exact ground/domain math stays server-side.

- [ ] **Step 1: Write the failing test**

Create `app/frontend/src/lib/objPlacement.test.ts`:

```ts
import { describe, it, expect } from 'vitest';
import { defaultPlacement, unitScale, transformModelPoint, type Placement } from './objPlacement';

describe('unitScale', () => {
  it('maps known units to metres', () => {
    expect(unitScale('m')).toBe(1);
    expect(unitScale('cm')).toBeCloseTo(0.01);
    expect(unitScale('ft')).toBeCloseTo(0.3048);
  });
});

describe('transformModelPoint', () => {
  const base: Placement = { ...defaultPlacement(), units: 'm' };

  it('places the anchor_model_point at move offset (rotation 0)', () => {
    const p = { ...base, move: [5, 7, 2] as [number, number, number] };
    const out = transformModelPoint([0, 0, 0], p); // anchor model point -> move
    expect(out[0]).toBeCloseTo(5); // east
    expect(out[1]).toBeCloseTo(7); // north
    expect(out[2]).toBeCloseTo(2); // up
  });

  it('rotates model +X toward north at rotation=90', () => {
    const p = { ...base, rotation: 90, move: [0, 0, 0] as [number, number, number] };
    const out = transformModelPoint([1, 0, 0], p); // +X, 1 m
    expect(out[0]).toBeCloseTo(0, 5);  // east ~ 0
    expect(out[1]).toBeCloseTo(1, 5);  // north ~ 1
  });

  it('applies unit scale', () => {
    const p = { ...base, units: 'ft' };
    const out = transformModelPoint([1, 0, 0], p);
    expect(out[0]).toBeCloseTo(0.3048, 4);
  });
});
```

- [ ] **Step 2: Run the test to verify it fails**

Run (from `app/frontend`): `npm run test -- objPlacement`
Expected: FAIL — cannot resolve `./objPlacement`.

- [ ] **Step 3: Implement `objPlacement.ts`**

Create `app/frontend/src/lib/objPlacement.ts`:

```ts
/**
 * Client-side OBJ placement model for the Import tab.
 *
 * `Placement` is the single source of truth shared by the numeric form, the 2D
 * footprint map, and the 3D gizmo. `transformModelPoint` maps a model-space
 * point to scene metres (x=east, y=north, z=up) for the *visual* preview only;
 * the committed voxelization uses the exact server-side transform.
 */

export type Units = 'm' | 'cm' | 'mm' | 'ft' | 'in';

export interface Placement {
  anchorLonLat: [number, number] | null; // set by initial map click
  anchorElevation: number | null;        // null -> auto from DEM at commit
  anchorModelPoint: [number, number, number];
  rotation: number;                       // degrees, CCW about up axis
  move: [number, number, number];         // [east, north, up] metres
  units: Units;
  zUp: boolean;
  swapYz: boolean;
}

export function defaultPlacement(): Placement {
  return {
    anchorLonLat: null,
    anchorElevation: null,
    anchorModelPoint: [0, 0, 0],
    rotation: 0,
    move: [0, 0, 0],
    units: 'm',
    zUp: true,
    swapYz: false,
  };
}

const UNIT_SCALE: Record<Units, number> = {
  m: 1, cm: 0.01, mm: 0.001, ft: 0.3048, in: 0.0254,
};

export function unitScale(units: string): number {
  const s = UNIT_SCALE[(units as Units)];
  if (s === undefined) throw new Error(`Unknown units: ${units}`);
  return s;
}

/**
 * Map a model-space point to scene metres relative to the anchor.
 *
 * Mirrors the visual part of voxcity.importer.transform.build_placement_transform:
 *   1. subtract anchorModelPoint, 2. scale by units, 3. rotate `rotation` deg
 *   about the up axis (model +X->east, +Y->north at rotation 0), 4. add move.
 * Returns [east, north, up] metres. Domain rotation + ground offset are applied
 * server-side and intentionally omitted from this visual approximation.
 */
export function transformModelPoint(
  pt: [number, number, number],
  p: Placement,
): [number, number, number] {
  const s = unitScale(p.units);
  const lx = (pt[0] - p.anchorModelPoint[0]) * s;
  const ly = (pt[1] - p.anchorModelPoint[1]) * s;
  const lz = (pt[2] - p.anchorModelPoint[2]) * s;
  const theta = (p.rotation * Math.PI) / 180;
  const cos = Math.cos(theta);
  const sin = Math.sin(theta);
  // east = lx*cos - ly*sin ; north = lx*sin + ly*cos  (CCW, +X->E, +Y->N at 0)
  const east = lx * cos - ly * sin + p.move[0];
  const north = lx * sin + ly * cos + p.move[1];
  const up = lz + p.move[2];
  return [east, north, up];
}
```

- [ ] **Step 4: Run the test to verify it passes**

Run (from `app/frontend`): `npm run test -- objPlacement`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add app/frontend/src/lib/objPlacement.ts app/frontend/src/lib/objPlacement.test.ts
git commit -m "feat(app): client-side OBJ placement model + preview transform"
```

---

## Task 5: Frontend API client functions

**Files:**
- Modify: `app/frontend/src/api.ts` (append after the Edit-tab section)

- [ ] **Step 1: Add DTO types and functions**

Append to `app/frontend/src/api.ts`:

```ts
// ── OBJ import tab ────────────────────────────────────────────

export interface ImportObjGroupDto {
  name: string;
  role: string;
  n_faces: number;
  bbox_model: [number, number, number][]; // [min, max]
}

export interface ImportObjPreviewDto {
  footprints: [number, number][][];
  vertices: [number, number, number][];
  indices: [number, number, number][];
}

export interface ImportObjUploadResult {
  import_id: string;
  groups: ImportObjGroupDto[];
  model_bounds: [number, number, number][];
  preview: ImportObjPreviewDto;
}

export interface ImportPlacementDto {
  anchor_lonlat: [number, number];
  anchor_elevation: number | null;
  anchor_model_point: [number, number, number];
  rotation: number;
  move: [number, number, number];
  units: string;
  z_up: boolean;
  swap_yz: boolean;
}

export interface ImportObjCommitRequestDto {
  import_id: string;
  placement: ImportPlacementDto;
  roles: Record<string, string>;
  overwrite: boolean;
}

export interface ImportObjCommitResult {
  figure_json: string;
  imported_building_ids: number[];
  n_building_voxels_added: number;
  warning: string | null;
}

export async function uploadImportObj(file: File): Promise<ImportObjUploadResult> {
  const form = new FormData();
  form.append('file', file);
  const res = await fetch(`${BASE}/model/import_obj/upload`, { method: 'POST', body: form });
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail || `HTTP ${res.status}`);
  }
  return res.json();
}

export async function commitImportObj(req: ImportObjCommitRequestDto) {
  return request<ImportObjCommitResult>('/model/import_obj/commit', {
    method: 'POST',
    body: JSON.stringify(req),
  });
}
```

- [ ] **Step 2: Verify the frontend type-checks**

Run (from `app/frontend`): `npm run build`
Expected: build succeeds (no TS errors from the new code).

- [ ] **Step 3: Commit**

```bash
git add app/frontend/src/api.ts
git commit -m "feat(app): API client for OBJ import upload/commit"
```

---

## Task 6: ImportTab shell — upload, role table, numeric form, commit

**Files:**
- Create: `app/frontend/src/tabs/ImportTab.tsx`

This task builds the controls + 3D result panels and the full upload→place(numeric)→commit flow. The 2D map (Task 8) and 3D gizmo (Task 9) layer on afterward. It reuses the existing `ThreeViewer` for the committed result so the tab is usable end-to-end before the gizmo lands.

- [ ] **Step 1: Implement the tab**

Create `app/frontend/src/tabs/ImportTab.tsx`:

```tsx
/**
 * Import tab — upload an OBJ, position it, and stamp its buildings into the model.
 *
 * Placement lives in one `Placement` object (lib/objPlacement). The numeric form
 * here writes it; the 2D map (Task 8) and 3D gizmo (Task 9) read/write the same
 * object. Commit calls /api/model/import_obj/commit and renders the result.
 */
import React, { useCallback, useState } from 'react';
import { Upload, Boxes, Check } from 'lucide-react';
import {
  uploadImportObj,
  commitImportObj,
  ImportObjUploadResult,
} from '../api';
import { GuidedSection } from '../components/guided';
import ThreeViewer from '../components/ThreeViewer';
import {
  defaultPlacement,
  Placement,
  Units,
} from '../lib/objPlacement';

interface ImportTabProps {
  hasModel: boolean;
  figureJson: string;
  onFigureChange: (s: string) => void;
  onModelEdited?: () => void;
}

const UNIT_OPTIONS: Units[] = ['m', 'cm', 'mm', 'ft', 'in'];

const ImportTab: React.FC<ImportTabProps> = ({ hasModel, figureJson, onFigureChange, onModelEdited }) => {
  const [upload, setUpload] = useState<ImportObjUploadResult | null>(null);
  const [roles, setRoles] = useState<Record<string, string>>({});
  const [placement, setPlacement] = useState<Placement>(defaultPlacement);
  const [advanced, setAdvanced] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [info, setInfo] = useState<string | null>(null);

  const setMove = (idx: 0 | 1 | 2, v: number) =>
    setPlacement((p) => {
      const move = [...p.move] as [number, number, number];
      move[idx] = v;
      return { ...p, move };
    });

  const handleFile = useCallback(async (file: File | null) => {
    if (!file) return;
    setBusy(true); setError(null); setInfo(null);
    try {
      const res = await uploadImportObj(file);
      setUpload(res);
      setRoles(Object.fromEntries(res.groups.map((g) => [g.name, g.role])));
      setInfo(`Loaded ${res.groups.length} group(s). Set an anchor and import.`);
    } catch (err: any) {
      setError(err.message || 'Upload failed');
    } finally {
      setBusy(false);
    }
  }, []);

  const handleImport = useCallback(async () => {
    if (!upload) return;
    if (!placement.anchorLonLat) { setError('Click the map to set an anchor first.'); return; }
    setBusy(true); setError(null); setInfo(null);
    try {
      const r = await commitImportObj({
        import_id: upload.import_id,
        placement: {
          anchor_lonlat: placement.anchorLonLat,
          anchor_elevation: placement.anchorElevation,
          anchor_model_point: placement.anchorModelPoint,
          rotation: placement.rotation,
          move: placement.move,
          units: placement.units,
          z_up: placement.zUp,
          swap_yz: placement.swapYz,
        },
        roles,
        overwrite: true,
      });
      onFigureChange(r.figure_json);
      onModelEdited?.();
      setInfo(r.warning ?? `Imported ${r.imported_building_ids.length} building(s); ${r.n_building_voxels_added} voxel(s) added.`);
    } catch (err: any) {
      setError(err.message || 'Import failed');
    } finally {
      setBusy(false);
    }
  }, [upload, placement, roles, onFigureChange, onModelEdited]);

  if (!hasModel) {
    return (
      <div className="panel">
        <h2>Import OBJ</h2>
        <div className="alert alert-info">Generate a model first to enable import.</div>
      </div>
    );
  }

  return (
    <div className="three-col">
      <div className="panel edit-control-panel">
        <div className="edit-control-scroll">
          <h2>Import OBJ</h2>

          <GuidedSection index={1} label="UPLOAD">
            <label className="btn btn-secondary" style={{ width: '100%', cursor: 'pointer' }}>
              <Upload size={14} style={{ marginRight: 6 }} />
              {upload ? 'Replace OBJ…' : 'Choose OBJ file…'}
              <input type="file" accept=".obj" style={{ display: 'none' }}
                     onChange={(e) => handleFile(e.target.files?.[0] ?? null)} />
            </label>
          </GuidedSection>

          {upload && (
            <GuidedSection index={2} label="GROUPS / ROLES">
              <table className="role-table" style={{ width: '100%', fontSize: '0.8rem' }}>
                <tbody>
                  {upload.groups.map((g) => (
                    <tr key={g.name}>
                      <td title={`${g.n_faces} faces`}>{g.name}</td>
                      <td style={{ textAlign: 'right' }}>
                        <select value={roles[g.name] ?? 'building'}
                                onChange={(e) => setRoles((r) => ({ ...r, [g.name]: e.target.value }))}>
                          <option value="building">building</option>
                          <option value="skip">skip</option>
                        </select>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </GuidedSection>
          )}

          {upload && (
            <GuidedSection index={3} label="PLACEMENT">
              <div className="guided-tool-hint">
                {placement.anchorLonLat
                  ? `Anchor: ${placement.anchorLonLat[1].toFixed(5)}, ${placement.anchorLonLat[0].toFixed(5)}`
                  : 'Click the map to set the anchor.'}
              </div>
              <div className="form-group">
                <label>Rotation (deg)</label>
                <input type="number" step={1} value={placement.rotation}
                       onChange={(e) => setPlacement((p) => ({ ...p, rotation: parseFloat(e.target.value) || 0 }))} />
              </div>
              <div className="form-group">
                <label>Move east / north / up (m)</label>
                <div style={{ display: 'flex', gap: 6 }}>
                  {[0, 1, 2].map((k) => (
                    <input key={k} type="number" step={0.5} value={placement.move[k]}
                           onChange={(e) => setMove(k as 0 | 1 | 2, parseFloat(e.target.value) || 0)} />
                  ))}
                </div>
              </div>
              <div className="form-group">
                <label>Units</label>
                <select value={placement.units}
                        onChange={(e) => setPlacement((p) => ({ ...p, units: e.target.value as Units }))}>
                  {UNIT_OPTIONS.map((u) => <option key={u} value={u}>{u}</option>)}
                </select>
              </div>

              <details open={advanced} onToggle={(e) => setAdvanced((e.target as HTMLDetailsElement).open)}>
                <summary>Advanced</summary>
                <label className="checkbox-row">
                  <input type="checkbox" checked={placement.zUp}
                         onChange={(e) => setPlacement((p) => ({ ...p, zUp: e.target.checked }))} />
                  Z-up (uncheck for Y-up exports)
                </label>
                <label className="checkbox-row">
                  <input type="checkbox" checked={placement.swapYz}
                         onChange={(e) => setPlacement((p) => ({ ...p, swapYz: e.target.checked }))} />
                  Swap Y/Z
                </label>
                <div className="form-group">
                  <label>Anchor elevation (m, blank = auto from terrain)</label>
                  <input type="number" step={0.5}
                         value={placement.anchorElevation ?? ''}
                         onChange={(e) => setPlacement((p) => ({
                           ...p,
                           anchorElevation: e.target.value === '' ? null : parseFloat(e.target.value),
                         }))} />
                </div>
              </details>
            </GuidedSection>
          )}

          <div className="guided-feedback-slot">
            {error && <div className="alert alert-error">{error}</div>}
            {info && <div className="alert alert-success">{info}</div>}
          </div>
        </div>

        <div className="pending-edit-footer">
          <button className="btn btn-primary pending-update-btn"
                  onClick={handleImport}
                  disabled={!upload || busy || !placement.anchorLonLat}
                  type="button">
            {busy && <span className="spinner" />}
            <Boxes size={14} style={{ marginRight: 6 }} />
            {busy ? 'Importing…' : 'Import building(s)'}
          </button>
        </div>
      </div>

      {/* 2D map placeholder — replaced by ObjPlacementMap in Task 8 */}
      <div className="panel visual-panel">
        <div className="plan-panel-header"><h2>2D placement</h2></div>
        <div className="visual-frame">
          <div className="alert alert-info">Map placement added in a later step.</div>
        </div>
      </div>

      {/* 3D result */}
      <div className="panel visual-panel">
        <div className="plan-panel-header"><h2>3D result</h2></div>
        <div className="visual-frame">
          {figureJson
            ? <ThreeViewer figureJson={figureJson} />
            : <div className="alert alert-info">Import to render the 3D result here.</div>}
        </div>
      </div>
    </div>
  );
};

export default ImportTab;
```

- [ ] **Step 2: Verify it type-checks**

Run (from `app/frontend`): `npm run build`
Expected: build succeeds.

- [ ] **Step 3: Commit**

```bash
git add app/frontend/src/tabs/ImportTab.tsx
git commit -m "feat(app): ImportTab shell (upload, role table, numeric placement, commit)"
```

---

## Task 7: Register the Import tab in App.tsx

**Files:**
- Modify: `app/frontend/src/App.tsx`

- [ ] **Step 1: Add the import and tab entry**

In `app/frontend/src/App.tsx`:
1. Add the import after the other tab imports (line ~8): `import ImportTab from './tabs/ImportTab';`
2. Add `Boxes` to the lucide-react import (line ~15-17).
3. Add a tab entry to `TABS` after the `edit` entry: `{ id: 'import', label: 'Import', Icon: Boxes },`
4. Add `importFigureJson` state alongside the others (line ~38): `const [importFigureJson, setImportFigureJson] = useState('');`
5. Clear it in the rectangle-change effect, `handleModelEdited`, `handleSessionLoaded`, and `onModelReady` (everywhere `setEditFigureJson('')` appears — add `setImportFigureJson('')` next to it).
6. Render the tab after the EditTab block (line ~201):

```tsx
{activeTab === 'import' && (
  <ImportTab
    hasModel={hasModel}
    figureJson={importFigureJson}
    onFigureChange={setImportFigureJson}
    onModelEdited={handleModelEdited}
  />
)}
```

- [ ] **Step 2: Verify build + the tab appears**

Run (from `app/frontend`): `npm run build`
Expected: build succeeds. (Manual check later: `python app/run.py`, the "Import" tab appears and shows the upload UI once a model exists.)

- [ ] **Step 3: Commit**

```bash
git add app/frontend/src/App.tsx
git commit -m "feat(app): register Import tab"
```

---

## Task 8: 2D footprint map (`ObjPlacementMap`) — anchor click + footprint overlay

**Files:**
- Create: `app/frontend/src/components/ObjPlacementMap.tsx`
- Modify: `app/frontend/src/tabs/ImportTab.tsx` (swap the placeholder for the map)

This renders the model's basemap + a footprint outline at the current placement, and sets `anchorLonLat` on click. It reuses `getModelGeo` + the `lonLatToUvM`/cell projection from `lib/grid.ts` and the placement transform from `lib/objPlacement.ts`. Footprint world points are converted back to lon/lat for the Leaflet polygon.

- [ ] **Step 1: Confirm the projection helpers available in lib/grid.ts**

Run (from `app/frontend`): `grep -n "export" src/lib/grid.ts`
Expected: shows the exported projection helpers (e.g. `lonLatToUvM`, `uvMToLonLat`, `polygonToCells`). Use `uvMToLonLat` (scene metres → lon/lat) to place footprint vertices; if it is absent, derive lon/lat from `grid_geom` (`origin` + `u_vec*du*i + v_vec*dv*j`) inline.

- [ ] **Step 2: Implement `ObjPlacementMap`**

Create `app/frontend/src/components/ObjPlacementMap.tsx`:

```tsx
/**
 * Leaflet map for OBJ placement: basemap + footprint outline at the current
 * placement; click sets the anchor lon/lat. Footprint vertices are computed by
 * transformModelPoint() (scene metres) then converted to lon/lat via grid_geom.
 */
import React, { useEffect, useMemo, useRef } from 'react';
import L from 'leaflet';
import { ModelGeoResult } from '../api';
import { Placement, transformModelPoint } from '../lib/objPlacement';

interface Props {
  geo: ModelGeoResult;
  placement: Placement;
  footprints: [number, number][][]; // model-XY rings from upload.preview
  onAnchor: (lonLat: [number, number]) => void;
}

/** scene metres (east,north) -> lon/lat using grid_geom basis vectors. */
function sceneToLonLat(geo: ModelGeoResult, east: number, north: number): [number, number] {
  const { origin, u_vec, v_vec, adj_mesh } = geo.grid_geom;
  // u runs north (axis 0), v runs east (axis 1); see exporter/obj.py convention.
  const j = east / adj_mesh[1];  // east cells (v)
  const i = north / adj_mesh[0]; // north cells (u)
  const lon = origin[0] + u_vec[0] * i + v_vec[0] * j;
  const lat = origin[1] + u_vec[1] * i + v_vec[1] * j;
  return [lon, lat];
}

const ObjPlacementMap: React.FC<Props> = ({ geo, placement, footprints, onAnchor }) => {
  const mapRef = useRef<L.Map | null>(null);
  const layerRef = useRef<L.LayerGroup | null>(null);
  const containerRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    if (!containerRef.current || mapRef.current) return;
    const map = L.map(containerRef.current).setView([geo.center[0], geo.center[1]], 17);
    L.tileLayer('https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}.png', {
      attribution: '© OpenStreetMap, © CARTO',
    }).addTo(map);
    map.on('click', (e: L.LeafletMouseEvent) => onAnchor([e.latlng.lng, e.latlng.lat]));
    layerRef.current = L.layerGroup().addTo(map);
    mapRef.current = map;
    return () => { map.remove(); mapRef.current = null; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Redraw the footprint whenever placement / footprints change.
  useEffect(() => {
    const layer = layerRef.current;
    if (!layer || !placement.anchorLonLat) return;
    layer.clearLayers();
    for (const ring of footprints) {
      const latlngs = ring.map(([mx, my]) => {
        const [east, north] = transformModelPoint([mx, my, 0], placement);
        const [lon, lat] = sceneToLonLat(geo, east, north);
        // offset by the anchor: transformModelPoint is relative to anchor origin
        return L.latLng(placement.anchorLonLat![1] + (lat - geo.grid_geom.origin[1]),
                        placement.anchorLonLat![0] + (lon - geo.grid_geom.origin[0]));
      });
      L.polygon(latlngs, { color: '#e8590c', weight: 2, fillOpacity: 0.25 }).addTo(layer);
    }
  }, [geo, placement, footprints]);

  return <div ref={containerRef} style={{ width: '100%', height: '100%' }} />;
};

export default ObjPlacementMap;
```

> Note for the implementer: the exact anchor-offset composition above must be verified against the live model during manual testing (the footprint should sit centered on the clicked anchor). If the offset drifts, compute the footprint lon/lat by anchoring `transformModelPoint`'s origin at the clicked cell directly (project the anchor lon/lat to a cell via the same `grid_geom`, add the scene-metre deltas in cell space, then back to lon/lat). The placement state and click wiring do not change.

- [ ] **Step 3: Wire it into ImportTab**

In `app/frontend/src/tabs/ImportTab.tsx`:
1. Add imports: `import { getModelGeo, ModelGeoResult } from '../api';` and `import ObjPlacementMap from '../components/ObjPlacementMap';`
2. Add state + load: `const [geo, setGeo] = useState<ModelGeoResult | null>(null);` and a `useEffect` that calls `getModelGeo().then(setGeo)` when `hasModel` is true.
3. Replace the 2D placeholder panel body with:

```tsx
{geo && upload ? (
  <ObjPlacementMap
    geo={geo}
    placement={placement}
    footprints={upload.preview.footprints}
    onAnchor={(lonLat) => setPlacement((p) => ({ ...p, anchorLonLat: lonLat }))}
  />
) : (
  <div className="alert alert-info">Upload an OBJ, then click the map to set the anchor.</div>
)}
```

- [ ] **Step 4: Verify build**

Run (from `app/frontend`): `npm run build`
Expected: build succeeds.

- [ ] **Step 5: Commit**

```bash
git add app/frontend/src/components/ObjPlacementMap.tsx app/frontend/src/tabs/ImportTab.tsx
git commit -m "feat(app): 2D footprint placement map for OBJ import"
```

---

## Task 9: 3D gizmo preview (`PlacementGizmo` + SceneViewer overlay)

**Files:**
- Create: `app/frontend/src/three/PlacementGizmo.tsx`
- Modify: `app/frontend/src/three/SceneViewer.tsx` (accept optional placement-preview props)
- Modify: `app/frontend/src/tabs/ImportTab.tsx` (render `SceneViewer` with the gizmo before commit)

The gizmo renders the decimated imported mesh transformed by the placement and attaches three.js `TransformControls` (translate X/Y/Z + rotate about up). Drag deltas update the shared `Placement` (`move` for translate, `rotation` for rotate). Uses `@react-three/drei`'s `TransformControls` if available, else the `three/examples` control.

- [ ] **Step 1: Confirm TransformControls availability**

Run (from `app/frontend`): `npm ls @react-three/drei three`
Expected: both present. If `@react-three/drei` exposes `TransformControls`, import from there; otherwise import `TransformControls` from `three/examples/jsm/controls/TransformControls.js` and attach it manually in a `useEffect`. The rest of this task assumes the drei component.

- [ ] **Step 2: Implement `PlacementGizmo`**

Create `app/frontend/src/three/PlacementGizmo.tsx`:

```tsx
/**
 * R3F preview of the imported OBJ mesh + a TransformControls gizmo.
 *
 * Translate X/Y/Z maps to placement.move = [east, north, up]; rotate about the
 * up axis maps to placement.rotation. The mesh is built from upload.preview
 * (model coords); object position is derived from move, scaled by unit at render.
 */
import { useEffect, useMemo, useRef } from 'react';
import * as THREE from 'three';
import { TransformControls } from '@react-three/drei';
import { Placement, unitScale } from '../lib/objPlacement';

interface Props {
  vertices: [number, number, number][]; // model coords
  indices: [number, number, number][];
  placement: Placement;
  mode: 'translate' | 'rotate';
  onChange: (next: Partial<Placement>) => void;
}

export function PlacementGizmo({ vertices, indices, placement, mode, onChange }: Props) {
  const meshRef = useRef<THREE.Mesh>(null);

  const geometry = useMemo(() => {
    const g = new THREE.BufferGeometry();
    const pos = new Float32Array(vertices.length * 3);
    vertices.forEach((v, k) => { pos[k * 3] = v[0]; pos[k * 3 + 1] = v[1]; pos[k * 3 + 2] = v[2]; });
    g.setAttribute('position', new THREE.BufferAttribute(pos, 3));
    g.setIndex(indices.flat());
    g.computeVertexNormals();
    return g;
  }, [vertices, indices]);

  // Apply units scale + rotation about up (Z) to the mesh; position from move.
  useEffect(() => {
    const m = meshRef.current;
    if (!m) return;
    const s = unitScale(placement.units);
    m.scale.set(s, s, s);
    m.rotation.set(0, 0, (placement.rotation * Math.PI) / 180);
    m.position.set(placement.move[0], placement.move[1], placement.move[2]);
  }, [placement]);

  const handleObjectChange = () => {
    const m = meshRef.current;
    if (!m) return;
    if (mode === 'translate') {
      onChange({ move: [m.position.x, m.position.y, m.position.z] });
    } else {
      onChange({ rotation: (m.rotation.z * 180) / Math.PI });
    }
  };

  return (
    <TransformControls mode={mode} showX showY showZ onObjectChange={handleObjectChange}>
      <mesh ref={meshRef} geometry={geometry}>
        <meshStandardMaterial color="#e8590c" transparent opacity={0.65} flatShading />
      </mesh>
    </TransformControls>
  );
}
```

- [ ] **Step 3: Add optional placement-preview props to SceneViewer**

In `app/frontend/src/three/SceneViewer.tsx`, extend `SceneViewerProps` with:

```tsx
  /** When set, renders an imported-OBJ placement preview + gizmo. */
  placementPreview?: {
    vertices: [number, number, number][];
    indices: [number, number, number][];
    placement: import('../lib/objPlacement').Placement;
    mode: 'translate' | 'rotate';
    onChange: (next: Partial<import('../lib/objPlacement').Placement>) => void;
  } | null;
```

Inside the `<Canvas>` children (alongside the existing `<MeshLayer/>` chunks), render the gizmo when present:

```tsx
{placementPreview && (
  <PlacementGizmo
    vertices={placementPreview.vertices}
    indices={placementPreview.indices}
    placement={placementPreview.placement}
    mode={placementPreview.mode}
    onChange={placementPreview.onChange}
  />
)}
```

Add the import at the top: `import { PlacementGizmo } from './PlacementGizmo';`

> Note: `TransformControls` from drei automatically disables `OrbitControls`/`CameraControls` while dragging via context; confirm during manual testing that camera orbit and gizmo drag don't fight. If they do, gate the existing `CameraControls` with an `enabled` flag toggled by the gizmo's `dragging-changed` event.

- [ ] **Step 4: Render the live 3D preview in ImportTab**

In `app/frontend/src/tabs/ImportTab.tsx`:
1. Add a gizmo-mode toggle state: `const [gizmoMode, setGizmoMode] = useState<'translate' | 'rotate'>('translate');` with two small buttons (Move / Rotate) in the PLACEMENT section.
2. Replace the 3D result panel so it shows the **live gizmo preview before commit** and the **committed figure after**:

```tsx
<div className="panel visual-panel">
  <div className="plan-panel-header"><h2>3D placement</h2></div>
  <div className="visual-frame">
    {upload && !figureJson ? (
      <SceneViewer
        geometryToken="import-preview"
        placementPreview={{
          vertices: upload.preview.vertices,
          indices: upload.preview.indices,
          placement,
          mode: gizmoMode,
          onChange: (next) => setPlacement((p) => ({ ...p, ...next })),
        }}
      />
    ) : figureJson ? (
      <ThreeViewer figureJson={figureJson} />
    ) : (
      <div className="alert alert-info">Upload an OBJ to place it in 3D.</div>
    )}
  </div>
</div>
```
3. Add the import: `import SceneViewer from '../three/SceneViewer';` (or the correct default/named export — check `three/index.ts`).
4. After a successful commit, the gizmo preview is hidden because `figureJson` becomes set; allow re-placing by clearing `figureJson` when a new file is uploaded (in `handleFile`, call `onFigureChange('')`).

- [ ] **Step 5: Verify build**

Run (from `app/frontend`): `npm run build`
Expected: build succeeds.

- [ ] **Step 6: Run the full frontend test + lint**

Run (from `app/frontend`): `npm run test -- objPlacement && npm run build`
Expected: tests pass, build succeeds.

- [ ] **Step 7: Commit**

```bash
git add app/frontend/src/three/PlacementGizmo.tsx app/frontend/src/three/SceneViewer.tsx app/frontend/src/tabs/ImportTab.tsx
git commit -m "feat(app): 3D placement gizmo (TransformControls) for OBJ import"
```

---

## Final verification

- [ ] **Backend tests:** `& "C:\Users\kunih\miniconda3\Scripts\conda.exe" run -n voxcity python -m pytest app/backend/test_import_obj.py -v` → all pass.
- [ ] **Importer regression:** `& "C:\Users\kunih\miniconda3\Scripts\conda.exe" run -n voxcity python -m pytest tests/importer/ -q` → still 36 passed, 1 skipped.
- [ ] **Frontend tests + build:** from `app/frontend`, `npm run test -- objPlacement` and `npm run build` → pass.
- [ ] **Manual smoke test:** `python app/run.py`, generate a small model, open the **Import** tab, upload `data/voxcity_import_test.obj`, click the map to set the anchor, drag the 3D gizmo (move + rotate), then **Import** — confirm the building appears in the 3D result at the placed location, on the terrain, and that the Edit tab afterward lists/【can delete】the new building.
- [ ] Confirm `git status` is clean and all tasks committed.

---

## Self-review notes (author)

- **Spec coverage:** new tab (Task 7) ✓; multi-group + role table (Tasks 2, 6) ✓; numeric placement (Task 6) ✓; interactive 3D axis-arrow gizmo + live preview (Task 9) ✓; 2D footprint + anchor click (Task 8) ✓; auto-detect groups + roles with sensible defaults / Advanced (Tasks 2, 6) ✓; auto DEM elevation w/ override (Tasks 3, 6) ✓; approximate-preview / exact-commit (Tasks 4, 9 preview vs Task 3 commit) ✓; upload/commit endpoints + errors incl. 404 stale id and off-domain warning (Tasks 2, 3) ✓; `onModelEdited` invalidation (Tasks 6, 7) ✓.
- **Known integration risks flagged inline for the implementer:** (1) the 2D footprint anchor-offset composition in `ObjPlacementMap` (Task 8 Step 2 note) must be validated against the live model; (2) `TransformControls`-vs-camera-controls interaction (Task 9 Step 3 note); (3) exact `SceneViewer` export name and whether camera controls need an `enabled` gate. None change the placement-state contract.
- **Type consistency:** `Placement` (camelCase, frontend) ↔ `ImportPlacementDto`/`ImportPlacement` (snake_case, wire/backend) are converted explicitly in `ImportTab.handleImport` and the commit endpoint; `import_obj_store` name is consistent across Tasks 2/3 and tests.
