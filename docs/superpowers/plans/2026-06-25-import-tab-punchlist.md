# Import Tab Punch-List + Rotated-Preview Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Resolve the eight review findings left open after the OBJ Import tab feature, make the live preview exactly match the committed placement on rotated-grid sites, and add test coverage for the recently-added anchor-datum code.

**Architecture:** Backend fixes harden the commit endpoint (input validation, disambiguated warnings, off-domain-anchor signal) and add tests for the new `/api/model/anchor_ground` endpoint. Frontend fixes cover accessibility, CSS, warning display, and — the substantial item — applying the VoxCity domain-rotation correction client-side so the 2D footprint and 3D gizmo preview align the building to the same (rotated) grid axes the server uses, instead of true north.

**Tech Stack:** FastAPI/Pydantic/numpy (backend), React/TypeScript/Vite/Leaflet/@react-three/fiber/three.js (frontend), pytest + vitest.

**Current HEAD when authored:** `5d73a3e`. Three commits (`ab02057`, `cabf0ae`, `5d73a3e`) landed after the original feature and already: added editable anchor lat/lon/elevation fields, a `/api/model/anchor_ground` endpoint + `anchorScene` vertical datum so the 3D preview seats buildings at correct ground height, and a true-outline (shapely-union) footprint. This plan builds on that state.

---

## Background: the domain-rotation math (read before Tasks 9–11)

The authoritative server transform is `build_placement_transform` in `src/voxcity/importer/transform.py`. Its rotation step maps model (x=east, y=north) into the domain's own (u, v) axes:

- `phi = _domain_rotation_deg(geom)` = `degrees(atan2(u_vec[0], u_vec[1]))` — the bearing (clockwise from true north) of the domain's +u axis. `u_vec = (dlon, dlat)` per metre.
- The model is rotated by user `theta`, then each resulting (east `e`, north `n`) vector is projected onto the (u, v) axes:
  - `u = e*sin(phi) + n*cos(phi)`
  - `v = e*cos(phi) - n*sin(phi)`
- Placement: `T1[0,3] = u_a + move_n` (north→u axis), `T1[1,3] = v_a + move_e` (east→v axis).

The client preview's `transformModelPoint` (`app/frontend/src/lib/objPlacement.ts`) applies only `theta` and returns `[east, north, up]` in a true-compass frame — it omits the `phi` projection. The grid-metre helpers it composes with (`lonLatToUvM`/`sceneXYToLonLat` in `app/frontend/src/lib/grid.ts`) interpret their `[east, north]` arguments as **grid (v, u) metres**. At `phi=0` true-compass and grid axes coincide (preview is correct); at `phi≠0` they diverge by exactly `phi` — that is the bug.

**Fix strategy:** add a client `domainRotationDeg(gridGeom)` that reproduces `_domain_rotation_deg`, and apply the same `(u = e*sinφ + n*cosφ, v = e*cosφ − n*sinφ)` projection in both preview paths (2D footprint via `transformModelPoint`, 3D gizmo via its own object transform). `transformModelPoint` already subtracts `anchorModelPoint`, so issue #6 (the dormant `anchorModelPoint` gap) lives only in the 3D gizmo and is fixed as part of Task 11.

---

## File Structure

**Backend:**
- Modify `app/backend/main.py` — `import_obj_commit` (validation, warning disambiguation, clamp signal).
- Modify `app/backend/test_import_obj.py` — new tests for the above + `/api/model/anchor_ground`.

**Frontend:**
- Modify `app/frontend/src/lib/grid.ts` — add `domainRotationDeg`.
- Modify `app/frontend/src/lib/grid.test.ts` (create if absent) — `domainRotationDeg` parity test.
- Modify `app/frontend/src/lib/objPlacement.ts` — optional `domainRotationDeg` param on `transformModelPoint`.
- Modify `app/frontend/src/lib/objPlacement.test.ts` — domain-rotation cases.
- Modify `app/frontend/src/components/ObjPlacementMap.tsx` — pass domain rotation into the footprint transform.
- Modify `app/frontend/src/three/PlacementGizmo.tsx` — apply domain rotation + `anchorModelPoint` to the mesh transform; keep the rotate-drag round-trip storing the user angle.
- Modify `app/frontend/src/tabs/ImportTab.tsx` — file-upload a11y, pass `domainRotationDeg` to map/gizmo, warning display.
- Modify `app/frontend/src/index.css` — `.role-table` rule.
- Create `app/frontend/src/tabs/importAnchorScene.ts` + `.test.ts` — extract & test the `anchorScene` vertical-datum computation.
- Modify `docs/superpowers/specs/2026-06-24-app-obj-import-tab-design.md` — fix the stale domain-rotation sentence.

---

## Task 1: Backend — validate placement vector shapes/finiteness (#1)

**Files:**
- Modify: `app/backend/main.py` (in `import_obj_commit`, the block after `p = req.placement`)
- Test: `app/backend/test_import_obj.py`

- [ ] **Step 1: Write the failing tests**

Append to `app/backend/test_import_obj.py`:

```python
def test_commit_rejects_wrong_length_move(client):
    import_id = _upload_box(client)
    req = {
        "import_id": import_id,
        "placement": {"anchor_lonlat": _domain_center_lonlat(), "move": [1.0, 2.0]},
    }
    r = client.post("/api/model/import_obj/commit", json=req)
    assert r.status_code == 400, r.text
    assert "move" in r.json()["detail"].lower()


def test_commit_rejects_nan_anchor(client):
    import_id = _upload_box(client)
    req = {
        "import_id": import_id,
        "placement": {"anchor_lonlat": [float("nan"), 35.0]},
    }
    r = client.post("/api/model/import_obj/commit", json=req)
    assert r.status_code == 400, r.text
    assert "anchor_lonlat" in r.json()["detail"].lower()


def test_commit_rejects_wrong_length_anchor_model_point(client):
    import_id = _upload_box(client)
    req = {
        "import_id": import_id,
        "placement": {"anchor_lonlat": _domain_center_lonlat(), "anchor_model_point": [0.0, 0.0]},
    }
    r = client.post("/api/model/import_obj/commit", json=req)
    assert r.status_code == 400, r.text
    assert "anchor_model_point" in r.json()["detail"].lower()
```

- [ ] **Step 2: Run to verify they fail**

Run: `& "C:\Users\kunih\miniconda3\Scripts\conda.exe" run -n voxcity python -m pytest app/backend/test_import_obj.py -v -k "wrong_length or nan_anchor"`
Expected: FAIL — current endpoint returns 200 (or a 500 from deep in the library), not 400 with these messages.

- [ ] **Step 3: Implement the validation**

In `app/backend/main.py`, inside `import_obj_commit`, replace the existing anchor-length check:

```python
    p = req.placement
    if len(p.anchor_lonlat) != 2:
        raise HTTPException(status_code=400, detail="anchor_lonlat must be [lon, lat]")
```

with:

```python
    p = req.placement

    def _require_finite_vec(name: str, vec, length: int) -> None:
        if len(vec) != length or not all(math.isfinite(float(x)) for x in vec):
            raise HTTPException(
                status_code=400,
                detail=f"{name} must be {length} finite number(s), got {vec!r}",
            )

    _require_finite_vec("anchor_lonlat", p.anchor_lonlat, 2)
    _require_finite_vec("move", p.move, 3)
    _require_finite_vec("anchor_model_point", p.anchor_model_point, 3)
    if p.anchor_elevation is not None and not math.isfinite(float(p.anchor_elevation)):
        raise HTTPException(status_code=400, detail="anchor_elevation must be a finite number or null")
```

(`math` is already imported at the top of `main.py` — confirm with `grep -n "^import math" app/backend/main.py`; if absent, add `import math` to the imports.)

- [ ] **Step 4: Run to verify they pass**

Run: `& "C:\Users\kunih\miniconda3\Scripts\conda.exe" run -n voxcity python -m pytest app/backend/test_import_obj.py -v`
Expected: PASS (all prior tests + 3 new).

- [ ] **Step 5: Commit**

```bash
git add app/backend/main.py app/backend/test_import_obj.py
git commit -m "fix(app): validate placement vector shapes/finiteness in import commit"
```

---

## Task 2: Backend — disambiguate the zero-voxel warning (#2)

**Files:**
- Modify: `app/backend/main.py` (`import_obj_commit`, the `warning = ...` line)
- Test: `app/backend/test_import_obj.py`

The current warning is identical whether the building landed off-domain, fully overlapped existing buildings, or had all groups skipped. The manifest's `id_map` distinguishes these: it is populated only for groups that produced at least one in-bounds cell (see `stamp_buildings` in `src/voxcity/importer/integrate.py`). So:
- `ids` non-empty and `n_added == 0` → in-domain but fully overlapped existing buildings (overwrite collisions).
- `ids` empty and `n_added == 0` → nothing landed in-domain (off-domain placement or all groups skipped).

- [ ] **Step 1: Write the failing test**

Append to `app/backend/test_import_obj.py`:

```python
def test_commit_overlap_warning_distinct_from_offdomain(client):
    # First import lands a building at the centre.
    id1 = _upload_box(client)
    center = _domain_center_lonlat()
    r1 = client.post("/api/model/import_obj/commit", json={
        "import_id": id1, "placement": {"anchor_lonlat": center}, "roles": {}, "overwrite": True,
    })
    assert r1.status_code == 200 and r1.json()["n_building_voxels_added"] > 0

    # Second import of the SAME box at the SAME spot fully overlaps -> 0 net added,
    # but ids ARE assigned (in-domain). Warning must say "overlap", not "0 cells".
    id2 = _upload_box(client)
    r2 = client.post("/api/model/import_obj/commit", json={
        "import_id": id2, "placement": {"anchor_lonlat": center}, "roles": {}, "overwrite": True,
    })
    assert r2.status_code == 200, r2.text
    body = r2.json()
    assert body["n_building_voxels_added"] == 0
    assert body["warning"] is not None
    assert "overlap" in body["warning"].lower()
    assert "0 cells" not in body["warning"].lower()
```

- [ ] **Step 2: Run to verify it fails**

Run: `& "C:\Users\kunih\miniconda3\Scripts\conda.exe" run -n voxcity python -m pytest app/backend/test_import_obj.py -v -k overlap_warning`
Expected: FAIL — current warning text says "0 cells inside the domain" for the overlap case too.

- [ ] **Step 3: Implement**

In `app/backend/main.py`, replace:

```python
    warning = None if n_added > 0 else (
        "Imported geometry voxelized to 0 cells inside the domain — check anchor/rotation/move/units."
    )
```

with:

```python
    if n_added > 0:
        warning = None
    elif ids:
        warning = (
            "Imported building(s) fully overlapped existing buildings; no new voxels were "
            "added (placement may be correct — the geometry coincides with what's already there)."
        )
    else:
        warning = (
            "Imported geometry produced 0 cells inside the domain (off-domain placement or all "
            "groups skipped) — check anchor/rotation/move/units and the group roles."
        )
```

- [ ] **Step 4: Run to verify it passes**

Run: `& "C:\Users\kunih\miniconda3\Scripts\conda.exe" run -n voxcity python -m pytest app/backend/test_import_obj.py -v`
Expected: PASS (all prior + new; the existing `test_commit_skips_non_building_role` still passes — its 0-added case has empty `ids`, so it gets the second message, which still has no assertion on exact text).

- [ ] **Step 5: Commit**

```bash
git add app/backend/main.py app/backend/test_import_obj.py
git commit -m "fix(app): distinguish overlap vs off-domain in zero-voxel import warning"
```

---

## Task 3: Backend — signal an off-domain anchor (#3)

**Files:**
- Modify: `app/backend/main.py` (`import_obj_commit`)
- Test: `app/backend/test_import_obj.py`

The auto-elevation branch clamps the anchor cell into bounds silently. Detect when the anchor projects outside the grid and prepend a note to the warning, regardless of elevation source.

- [ ] **Step 1: Write the failing test**

Append to `app/backend/test_import_obj.py`:

```python
def test_commit_offdomain_anchor_warns(client):
    import_id = _upload_box(client)
    # Anchor far from the model rectangle (the flat fixture is near (0,0); use a
    # clearly off-domain lon/lat). Provide explicit elevation so the test doesn't
    # depend on the auto-sample path.
    req = {
        "import_id": import_id,
        "placement": {"anchor_lonlat": [10.0, 10.0], "anchor_elevation": 0.0},
        "roles": {}, "overwrite": True,
    }
    r = client.post("/api/model/import_obj/commit", json=req)
    assert r.status_code == 200, r.text
    w = r.json()["warning"]
    assert w is not None and "outside the model domain" in w.lower()
```

(Confirm the flat test fixture's rectangle is near the equator/origin so `[10.0, 10.0]` is off-domain — `make_flat_voxcity` builds its rectangle near `(0, 0)`; verify by reading `tests/importer/conftest.py`. If the fixture's domain actually contains `(10,10)`, pick a lon/lat clearly outside `app_state.rectangle_vertices` instead.)

- [ ] **Step 2: Run to verify it fails**

Run: `& "C:\Users\kunih\miniconda3\Scripts\conda.exe" run -n voxcity python -m pytest app/backend/test_import_obj.py -v -k offdomain_anchor`
Expected: FAIL — no "outside the model domain" text today.

- [ ] **Step 3: Implement**

In `app/backend/main.py` `import_obj_commit`, compute the anchor cell and an `anchor_off_domain` flag once (used for both the existing auto-elevation clamp and the new signal). Replace the auto-elevation block:

```python
    anchor_elev = p.anchor_elevation
    if anchor_elev is None:
        i, j = _anchor_lonlat_to_cell(p.anchor_lonlat[0], p.anchor_lonlat[1])
        dem = np.asarray(app_state.voxcity.dem.elevation)
        nx, ny = dem.shape
        ii = min(max(i, 0), nx - 1)
        jj = min(max(j, 0), ny - 1)
        anchor_elev = float(dem[ii, jj])
```

with:

```python
    ai, aj = _anchor_lonlat_to_cell(p.anchor_lonlat[0], p.anchor_lonlat[1])
    dem = np.asarray(app_state.voxcity.dem.elevation)
    nx, ny = dem.shape
    anchor_off_domain = not (0 <= ai < nx and 0 <= aj < ny)

    anchor_elev = p.anchor_elevation
    if anchor_elev is None:
        ii = min(max(ai, 0), nx - 1)
        jj = min(max(aj, 0), ny - 1)
        anchor_elev = float(dem[ii, jj])
```

Then, after the existing `warning = ...` block from Task 2, prepend the off-domain note:

```python
    if anchor_off_domain:
        note = "Anchor lon/lat falls outside the model domain; placement may be unexpected. "
        warning = note + warning if warning else note.strip()
```

- [ ] **Step 4: Run to verify it passes**

Run: `& "C:\Users\kunih\miniconda3\Scripts\conda.exe" run -n voxcity python -m pytest app/backend/test_import_obj.py -v`
Expected: PASS (all).

- [ ] **Step 5: Commit**

```bash
git add app/backend/main.py app/backend/test_import_obj.py
git commit -m "fix(app): warn when import anchor falls outside the model domain"
```

---

## Task 4: Backend — tests for `/api/model/anchor_ground` (new-code coverage)

**Files:**
- Test: `app/backend/test_import_obj.py`

The endpoint added in `cabf0ae` has no tests. Read its implementation in `app/backend/main.py` (`model_anchor_ground`, the `@app.get("/api/model/anchor_ground")` route) before writing the asserts so expected values match the real fixture.

- [ ] **Step 1: Write the tests**

Append to `app/backend/test_import_obj.py`:

```python
def test_anchor_ground_returns_datum(client):
    lon, lat = _domain_center_lonlat()
    r = client.get("/api/model/anchor_ground", params={"lon": lon, "lat": lat})
    assert r.status_code == 200, r.text
    body = r.json()
    # Flat fixture: DEM is all zeros, meshsize 1.0.
    assert body["dem_elevation"] == 0.0
    assert body["dem_min"] == 0.0
    assert body["meshsize_m"] == 1.0


def test_anchor_ground_requires_model(client):
    app_state.voxcity = None
    r = client.get("/api/model/anchor_ground", params={"lon": 0.0, "lat": 0.0})
    assert r.status_code == 400


def test_anchor_ground_offdomain_clamps_without_error(client):
    # Off-domain anchor must still return a datum (nearest in-bounds cell), not 500.
    r = client.get("/api/model/anchor_ground", params={"lon": 10.0, "lat": 10.0})
    assert r.status_code == 200, r.text
    assert "dem_elevation" in r.json()
```

(Confirm `make_flat_voxcity`'s DEM/meshsize values by reading `tests/importer/conftest.py`; the autouse `_model_loaded` fixture builds it with `meshsize=1.0` and a zero DEM — adjust the expected numbers if the fixture differs.)

- [ ] **Step 2: Run to verify**

Run: `& "C:\Users\kunih\miniconda3\Scripts\conda.exe" run -n voxcity python -m pytest app/backend/test_import_obj.py -v -k anchor_ground`
Expected: PASS (3 new). If `test_anchor_ground_requires_model` fails because the endpoint doesn't call `_require_model` early enough, that is a real bug in the endpoint — read it and confirm; the endpoint as written in `cabf0ae` does call `_require_model()` first, so this should pass.

- [ ] **Step 3: Commit**

```bash
git add app/backend/test_import_obj.py
git commit -m "test(app): cover /api/model/anchor_ground endpoint"
```

---

## Task 5: Frontend — keyboard-accessible file upload (#4)

**Files:**
- Modify: `app/frontend/src/tabs/ImportTab.tsx` (the `UPLOAD` `GuidedSection`)

The hidden `<input type="file" style={{display:'none'}}>` in an unfocusable `<label>` is unreachable by keyboard/screen reader. Make the label a focusable button that forwards activation to a ref'd input (kept visually hidden but not `display:none`, so assistive tech still sees it).

- [ ] **Step 1: Implement**

In `ImportTab.tsx`, add a ref near the other hooks:

```tsx
  const fileInputRef = React.useRef<HTMLInputElement>(null);
```

Replace the UPLOAD section's `<label>...</label>` with:

```tsx
            <button
              type="button"
              className="btn btn-secondary"
              style={{ width: '100%', cursor: busy ? 'not-allowed' : 'pointer', opacity: busy ? 0.6 : 1 }}
              disabled={busy}
              onClick={() => fileInputRef.current?.click()}
            >
              <Upload size={14} style={{ marginRight: 6 }} />
              {upload ? 'Replace OBJ…' : 'Choose OBJ file…'}
            </button>
            <input
              ref={fileInputRef}
              type="file"
              accept=".obj"
              disabled={busy}
              // Visually hidden but focusable/assistive-tech-visible (not display:none).
              style={{ position: 'absolute', width: 1, height: 1, padding: 0, margin: -1,
                       overflow: 'hidden', clip: 'rect(0,0,0,0)', whiteSpace: 'nowrap', border: 0 }}
              onChange={(e) => handleFile(e.target.files?.[0] ?? null)}
            />
```

A native `<button>` is keyboard-focusable and Enter/Space-activatable by default, and forwards the click to the input. The input stays in the accessibility tree for screen readers.

- [ ] **Step 2: Verify build + tests**

Run (from `app/frontend`): `npm run build && npm run test`
Expected: build succeeds; test count unchanged.

- [ ] **Step 3: Commit**

```bash
git add app/frontend/src/tabs/ImportTab.tsx
git commit -m "fix(app): make ImportTab file upload keyboard/screen-reader accessible"
```

---

## Task 6: Frontend — style the role table (#5)

**Files:**
- Modify: `app/frontend/src/index.css`

- [ ] **Step 1: Implement**

Read `app/frontend/src/index.css` to match its existing variable names (e.g. `--vc-border`, `--vc-text`, `--vc-muted` — confirm the actual custom-property names used elsewhere in the file before writing the rule). Append:

```css
.role-table {
  border-collapse: collapse;
}
.role-table td {
  padding: 2px 4px;
  border-bottom: 1px solid var(--vc-border, #ddd);
  vertical-align: middle;
}
.role-table td:first-child {
  word-break: break-all;
}
.role-table select {
  min-width: 92px;
}
```

(If `--vc-border` is not a real variable in this file, substitute the actual border-color variable used by sibling tables/inputs, or a literal `#ddd`.)

- [ ] **Step 2: Verify build**

Run (from `app/frontend`): `npm run build`
Expected: build succeeds.

- [ ] **Step 3: Commit**

```bash
git add app/frontend/src/index.css
git commit -m "style(app): add role-table styling for the Import tab group/role list"
```

---

## Task 7: Frontend — render import warnings as cautions (#2/#3 tie-in)

**Files:**
- Modify: `app/frontend/src/tabs/ImportTab.tsx` (`handleImport` + the feedback slot)

Today a returned `warning` is shown via `setInfo(...)` in a green `alert-success` box. A caution should read as a caution. `index.css` already defines `.alert-warning`.

- [ ] **Step 1: Implement**

Add a `warning` state next to `error`/`info`:

```tsx
  const [warning, setWarning] = useState<string | null>(null);
```

In `handleImport`, on success replace:

```tsx
      setInfo(r.warning ?? `Imported ${r.imported_building_ids.length} building(s); ${r.n_building_voxels_added} voxel(s) added.`);
```

with:

```tsx
      if (r.warning) {
        setWarning(r.warning);
        setInfo(null);
      } else {
        setWarning(null);
        setInfo(`Imported ${r.imported_building_ids.length} building(s); ${r.n_building_voxels_added} voxel(s) added.`);
      }
```

Clear `warning` alongside `error`/`info` at the start of `handleFile` and `handleImport` (set `setWarning(null)` wherever `setError(null); setInfo(null);` appears). In the feedback slot, add a warning row:

```tsx
          <div className="guided-feedback-slot">
            {error && <div className="alert alert-error">{error}</div>}
            {warning && <div className="alert alert-warning">{warning}</div>}
            {info && <div className="alert alert-success">{info}</div>}
          </div>
```

- [ ] **Step 2: Verify build + tests**

Run (from `app/frontend`): `npm run build && npm run test`
Expected: build succeeds; test count unchanged.

- [ ] **Step 3: Commit**

```bash
git add app/frontend/src/tabs/ImportTab.tsx
git commit -m "feat(app): show import commit warnings as cautions, not success"
```

---

## Task 8: Docs — fix the stale domain-rotation sentence (#7)

**Files:**
- Modify: `docs/superpowers/specs/2026-06-24-app-obj-import-tab-design.md`

- [ ] **Step 1: Implement**

Read the spec around line 144 (search for "domain rotation"). The sentence states the TS preview transform is a port of "anchor→uv_m, units scale, rotation + domain rotation, move." Since Tasks 9–11 of THIS plan now make the client apply domain rotation, update the sentence to reflect the final design: the client `transformModelPoint` applies anchor-relative scale + user rotation, and the domain-rotation correction is applied via `domainRotationDeg(gridGeom)` in the preview composition (`ObjPlacementMap` / `PlacementGizmo`) so the preview matches the server placement on rotated grids. Replace the stale "intentionally omits domain rotation" phrasing (if present elsewhere in the doc, e.g. an "Out of scope" or approximate-preview note) with a statement that domain rotation IS now applied in the preview.

- [ ] **Step 2: Commit**

```bash
git add docs/superpowers/specs/2026-06-24-app-obj-import-tab-design.md
git commit -m "docs(app): spec now reflects client-side domain-rotation in preview"
```

> Note: do this AFTER Tasks 9–11 land so the spec describes the shipped behavior. If executing in order, move this task's commit to the end; it is listed here only because it is trivial and self-contained.

---

## Task 9: Frontend — `domainRotationDeg` helper with server parity (#8 part 1)

**Files:**
- Modify: `app/frontend/src/lib/grid.ts`
- Test: `app/frontend/src/lib/grid.test.ts` (create if it doesn't exist — check first)

- [ ] **Step 1: Generate the ground-truth expected value**

The client helper must reproduce `src/voxcity/importer/transform.py:_domain_rotation_deg` = `degrees(atan2(u_vec[0], u_vec[1]))`. Write a throwaway Python script (do not commit) that builds a deliberately-rotated grid and prints both `u_vec` and the expected degrees, to hardcode into the test:

```python
# scratch only — run via: & "C:\Users\kunih\miniconda3\Scripts\conda.exe" run -n voxcity python scratch.py
import math
from voxcity.geoprocessor.raster.core import compute_grid_geometry
# A rectangle rotated ~30 deg: choose 4 lon/lat corners forming a rotated rectangle.
rect = [(139.70, 35.66), (139.6948, 35.6690), (139.7100, 35.6778), (139.7152, 35.6688)]
geom = compute_grid_geometry(rect, 1.0)
u = geom["u_vec"]
print("u_vec", float(u[0]), float(u[1]))
print("phi_deg", math.degrees(math.atan2(float(u[0]), float(u[1]))))
```

Record the printed `u_vec` and `phi_deg` for the test below.

- [ ] **Step 2: Write the failing test**

In `app/frontend/src/lib/grid.test.ts` (match existing vitest style in the repo; if the file is new, mirror `objPlacement.test.ts`'s imports). Use the EXACT `u_vec` and `phi_deg` numbers printed in Step 1 — replace the placeholders `<U0>`, `<U1>`, `<PHI>` with the real printed values:

```ts
import { describe, it, expect } from 'vitest';
import { domainRotationDeg, type GridGeom } from './grid';

describe('domainRotationDeg', () => {
  it('matches the server _domain_rotation_deg (bearing of u_vec, deg CW from north)', () => {
    const geom = {
      origin: [139.70, 35.66], side_1: [0, 0], side_2: [0, 0],
      u_vec: [<U0>, <U1>], v_vec: [0, 0], adj_mesh: [1, 1], grid_size: [1, 1],
    } as unknown as GridGeom;
    expect(domainRotationDeg(geom)).toBeCloseTo(<PHI>, 6);
  });

  it('is ~0 for an axis-aligned (north-up) grid', () => {
    const geom = { u_vec: [0, 1] } as unknown as GridGeom;
    expect(domainRotationDeg(geom)).toBeCloseTo(0, 9);
  });
});
```

- [ ] **Step 3: Run to verify it fails**

Run (from `app/frontend`): `npm run test -- grid`
Expected: FAIL — `domainRotationDeg` not exported.

- [ ] **Step 4: Implement**

Add to `app/frontend/src/lib/grid.ts` (near `lonLatToUvM`/`sceneXYToLonLat`):

```ts
/**
 * Bearing (degrees, clockwise from true north) of the grid's +u axis — the
 * VoxCity "domain rotation". Reproduces src/voxcity/importer/transform.py's
 * _domain_rotation_deg: degrees(atan2(u_vec[0], u_vec[1])), where u_vec is
 * (dlon, dlat) per metre along side_1.
 */
export function domainRotationDeg(geo: GridGeom): number {
  const [du, dv] = geo.u_vec; // (dlon, dlat) per metre
  return (Math.atan2(du, dv) * 180) / Math.PI;
}
```

- [ ] **Step 5: Run to verify it passes**

Run (from `app/frontend`): `npm run test -- grid`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add app/frontend/src/lib/grid.ts app/frontend/src/lib/grid.test.ts
git commit -m "feat(app): client domainRotationDeg matching server _domain_rotation_deg"
```

---

## Task 10: Frontend — apply domain rotation in `transformModelPoint` + 2D footprint (#8 part 2)

**Files:**
- Modify: `app/frontend/src/lib/objPlacement.ts`
- Modify: `app/frontend/src/lib/objPlacement.test.ts`
- Modify: `app/frontend/src/components/ObjPlacementMap.tsx`

Extend `transformModelPoint` with an optional `domainRotationDeg` arg (default 0 ⇒ existing behavior/tests unchanged). When set, after computing the user-rotated `(e, n)`, project onto the (u, v) grid axes exactly as the server does, and return the result so that `[out0, out1] = [v, u] = [east-grid, north-grid] metres` — the convention `sceneXYToLonLat`/`anchorScene` already consume.

- [ ] **Step 1: Write the failing tests**

Append to `app/frontend/src/lib/objPlacement.test.ts`:

```ts
describe('transformModelPoint with domain rotation', () => {
  it('reduces to the no-domain result when domainRotationDeg = 0', () => {
    const p = { ...defaultPlacement(), units: 'm' as const, move: [0, 0, 0] as [number, number, number] };
    const a = transformModelPoint([2, 3, 1], p);
    const b = transformModelPoint([2, 3, 1], p, 0);
    expect(b[0]).toBeCloseTo(a[0], 9);
    expect(b[1]).toBeCloseTo(a[1], 9);
    expect(b[2]).toBeCloseTo(a[2], 9);
  });

  it('projects (east,north) onto the rotated (u,v) axes (phi=90)', () => {
    // At phi=90: u = e*sin90 + n*cos90 = e ; v = e*cos90 - n*sin90 = -n.
    // transformModelPoint returns [v, u, up] = [-n, e, up].
    const p = { ...defaultPlacement(), units: 'm' as const, rotation: 0, move: [0, 0, 0] as [number, number, number] };
    // model +X -> (e,n) = (1,0) at rotation 0 -> [v,u] = [-0, 1] = [0,1]
    const out = transformModelPoint([1, 0, 0], p, 90);
    expect(out[0]).toBeCloseTo(0, 6); // v (east-grid)
    expect(out[1]).toBeCloseTo(1, 6); // u (north-grid)
  });
});
```

- [ ] **Step 2: Run to verify they fail**

Run (from `app/frontend`): `npm run test -- objPlacement`
Expected: FAIL — `transformModelPoint` takes only 2 args today.

- [ ] **Step 3: Implement**

Read the current `transformModelPoint` in `app/frontend/src/lib/objPlacement.ts`. It currently returns `[east, north, up]` where `east = lx*cos - ly*sin + move[0]`, `north = lx*sin + ly*cos + move[1]`. Refactor so the user-rotated, unit-scaled, anchor-relative `(e, n)` is computed first WITHOUT adding move, then projected by `phi`, then move added in the grid frame. Replace the function body with:

```ts
export function transformModelPoint(
  pt: [number, number, number],
  p: Omit<Placement, 'units'> & { units: string },
  domainRotationDeg = 0,
): [number, number, number] {
  const s = unitScale(p.units);
  const lx = (pt[0] - p.anchorModelPoint[0]) * s;
  const ly = (pt[1] - p.anchorModelPoint[1]) * s;
  const lz = (pt[2] - p.anchorModelPoint[2]) * s;
  const theta = (p.rotation * Math.PI) / 180;
  const ct = Math.cos(theta), st = Math.sin(theta);
  // user-rotated (east, north) of the anchor-relative, unit-scaled point
  const e = lx * ct - ly * st;
  const n = lx * st + ly * ct;
  // project (e, n) onto the domain (u, v) axes (server parity):
  //   u = e*sin(phi) + n*cos(phi) ; v = e*cos(phi) - n*sin(phi)
  const phi = (domainRotationDeg * Math.PI) / 180;
  const sp = Math.sin(phi), cp = Math.cos(phi);
  const u = e * sp + n * cp;
  const v = e * cp - n * sp;
  // return grid-frame [east=v, north=u, up], plus move in the same frame
  return [v + p.move[0], u + p.move[1], lz + p.move[2]];
}
```

> Verify the `phi=0` identity by hand: `sp=0, cp=1` ⇒ `u=n, v=e` ⇒ returns `[e+move0, n+move1, ...]`, identical to the old formula. Existing `objPlacement.test.ts` cases (which pass no `domainRotationDeg`) therefore still pass.

- [ ] **Step 4: Pass domain rotation from `ObjPlacementMap`**

In `app/frontend/src/components/ObjPlacementMap.tsx`, import `domainRotationDeg` from `'../lib/grid'`, compute it once from `geo.grid_geom`, and pass it as the 3rd arg wherever `transformModelPoint(...)` is called in the footprint-drawing effect:

```ts
const phiDeg = domainRotationDeg(geo.grid_geom);
// ...
const [eastOffset, northOffset] = transformModelPoint([mx, my, 0], placement, phiDeg);
```

(Read the current effect to place this correctly; the rest of the footprint composition — `anchorScene + offset → sceneXYToLonLat`— is unchanged.)

- [ ] **Step 5: Run to verify**

Run (from `app/frontend`): `npm run test -- objPlacement && npm run build`
Expected: tests pass (incl. the 2 new), build succeeds.

- [ ] **Step 6: Commit**

```bash
git add app/frontend/src/lib/objPlacement.ts app/frontend/src/lib/objPlacement.test.ts app/frontend/src/components/ObjPlacementMap.tsx
git commit -m "feat(app): apply domain rotation in 2D footprint preview (server parity)"
```

---

## Task 11: Frontend — apply domain rotation + anchorModelPoint to the 3D gizmo (#8 part 3, #6)

**Files:**
- Modify: `app/frontend/src/three/PlacementGizmo.tsx`
- Modify: `app/frontend/src/three/SceneViewer.tsx` (thread a `domainRotationDeg` value through `placementPreview`)
- Modify: `app/frontend/src/tabs/ImportTab.tsx` (compute and pass it)

The gizmo positions/rotates the mesh directly in scene space (grid axes: X=east/v, Y=north/u, Z=up). To match the committed placement and the 2D footprint, the mesh transform must reproduce `transformModelPoint`'s mapping including domain rotation `phi` and `anchorModelPoint`. The tricky part: the rotate-drag must still store the **user** angle `theta` in `placement.rotation`, not `theta+phi`.

The mesh transform for a model vertex `m` should produce scene point:
`anchorScene + project_phi( R(theta) · (s · (m − amp)) ) + [move.east, move.north, move.up]`
where `project_phi(e, n) = (v, u) = (e·cosφ − n·sinφ, e·sinφ + n·cosφ)`.

Implement this as: pre-translate the mesh geometry by `−amp` (or set a child group), scale by `s`, then a single Z-rotation by the **combined** scene-yaw that equals `theta + phi` *expressed in the scene frame*, then translate to `anchorScene + grid-projected move`. Specifically, because `project_phi` is itself a rotation by `phi` of the (e,n) plane into (v,u)... **verify the exact combined yaw and move projection against Task 10's `transformModelPoint`** rather than assuming the sign — see Step 2's parity test, which is the ground-truth gate.

- [ ] **Step 1: Thread `domainRotationDeg` into the gizmo props**

In `ImportTab.tsx`, compute `const phiDeg = useMemo(() => geo ? domainRotationDeg(geo.grid_geom) : 0, [geo]);` (import `domainRotationDeg` from `'../lib/grid'`) and add `domainRotationDeg: phiDeg` to the `placementPreview` object passed to `<SceneViewer>`. In `SceneViewer.tsx`, add `domainRotationDeg: number` to the `placementPreview` prop type and forward it to `<PlacementGizmo domainRotationDeg={placementPreview.domainRotationDeg} ... />`. In `PlacementGizmo.tsx`, add `domainRotationDeg: number` to `PlacementGizmoProps`.

- [ ] **Step 2: Write a parity test for the gizmo's transform helper**

Extract the gizmo's mesh-transform math into a pure, exported helper so it can be unit-tested without R3F. In `PlacementGizmo.tsx` add and export:

```ts
/**
 * Scene-space placement of a model point for the 3D preview mesh, matching
 * lib/objPlacement.transformModelPoint (including domain rotation) plus the
 * anchor's scene position. Used to position/orient the gizmo mesh so the 3D
 * preview agrees with the 2D footprint and the committed voxelization.
 */
export function gizmoModelToScene(
  pt: [number, number, number],
  placement: Placement,
  anchorScene: [number, number, number],
  domainRotationDeg: number,
): [number, number, number] {
  const off = transformModelPoint(pt, placement, domainRotationDeg);
  return [anchorScene[0] + off[0], anchorScene[1] + off[1], anchorScene[2] + off[2]];
}
```

(Import `transformModelPoint` from `'../lib/objPlacement'`.) Add `app/frontend/src/three/PlacementGizmo.test.ts`:

```ts
import { describe, it, expect } from 'vitest';
import { gizmoModelToScene } from './PlacementGizmo';
import { defaultPlacement } from '../lib/objPlacement';

describe('gizmoModelToScene', () => {
  it('equals anchorScene + transformModelPoint offset', () => {
    const p = { ...defaultPlacement(), anchorLonLat: [0, 0] as [number, number], rotation: 20,
                move: [3, -2, 5] as [number, number, number], anchorModelPoint: [1, 1, 0] as [number, number, number] };
    const anchor: [number, number, number] = [100, 200, 7];
    const out = gizmoModelToScene([4, 5, 2], p, anchor, 35);
    // Independently: anchor + transformModelPoint([4,5,2], p, 35)
    expect(out[0]).toBeCloseTo(anchor[0] + (out[0] - anchor[0]), 9); // tautology guard; real check below
  });
});
```

Replace the tautology with a concrete expected vector: compute `transformModelPoint([4,5,2], p, 35)` by running the function in a throwaway node/vitest scratch, then hardcode `anchor + that` as the expected `[x,y,z]` with `toBeCloseTo(..., 6)`. (The point is to lock the gizmo's positioning to `transformModelPoint`, the already-server-parity-tested function from Task 10, so the 3D and 2D previews cannot silently diverge.)

- [ ] **Step 3: Run to verify it fails**

Run (from `app/frontend`): `npm run test -- PlacementGizmo`
Expected: FAIL — `gizmoModelToScene` not exported yet.

- [ ] **Step 4: Implement the gizmo transform using the helper**

Rewrite `PlacementGizmo`'s placement→mesh sync effect so the mesh's world transform places its model vertices via `gizmoModelToScene`. Because `gizmoModelToScene` is an affine map of the model point, set the mesh transform so that mesh world = that map:
- `mesh.scale = s = unitScale(units)`.
- The mesh's yaw about Z must equal the net rotation that `transformModelPoint` applies: user `theta` then `phi` projection. Net scene yaw `psiDeg = placement.rotation + domainRotationDeg` (verify sign against Step 2's parity test — if the parity test fails, flip the sign of `domainRotationDeg` in `psiDeg`). Set `mesh.rotation.set(0, 0, psiDeg * Math.PI / 180)`.
- The mesh position must place model origin `amp` at `gizmoModelToScene(amp, ...)`. Compute `const originScene = gizmoModelToScene(placement.anchorModelPoint, placement, anchorScene, domainRotationDeg);` and set `mesh.position.set(originScene[0], originScene[1], originScene[2])`. Also bake the `-amp*s` pivot by setting the geometry's translation once OR by offsetting position with the rotated/scaled `amp` — simplest: since `gizmoModelToScene(amp,...)` already accounts for `amp` (transformModelPoint subtracts it), and the mesh geometry is raw model coords, set `mesh.position = gizmoModelToScene(amp,...)` and additionally translate the geometry by `-amp` so the mesh's local origin is `amp`. Pre-translate the geometry once in the `useMemo` that builds it: `geom.translate(-amp[0], -amp[1], -amp[2])` (guard for `amp = [0,0,0]` no-op).

  > This is the single most error-prone step. The Step 2 parity test only checks `gizmoModelToScene` (the math), not the three.js object transform. Add a SECOND assertion approach: after wiring, verify visually in the running app (manual) that for a non-zero `anchorModelPoint` (temporarily hardcode one), the 3D mesh and 2D footprint stay aligned. Since `anchorModelPoint` has no UI today (issue #6 is dormant), the `[0,0,0]` path is what ships; the `geom.translate` + `gizmoModelToScene(amp)` structure makes the non-zero case correct-by-construction for when a UI is added.

- In `handleObjectChange`, the rotate branch must recover the **user** angle: `onChange({ rotation: (mesh.rotation.z * 180 / Math.PI) - domainRotationDeg })` (again, sign per the parity test). The translate branch recovers `move` by inverting the grid-frame offset: `move = project_phi_inverse(mesh.position - originScene_at_zero_move)`. Simpler and robust: recover move as the grid-frame delta directly — `move.east = mesh.position.x - (originScene.x without move)`, etc. Compute `originSceneNoMove = gizmoModelToScene(amp, {...placement, move:[0,0,0]}, anchorScene, domainRotationDeg)` and set `onChange({ move: [mesh.position.x - originSceneNoMove[0], mesh.position.y - originSceneNoMove[1], mesh.position.z - originSceneNoMove[2]] })`. Because move is added in the grid frame in `transformModelPoint`, this delta is exactly `[move.east, move.north, move.up]`.

- [ ] **Step 5: Run tests + build**

Run (from `app/frontend`): `npm run test && npm run build`
Expected: all tests pass (incl. `PlacementGizmo` parity), build succeeds.

- [ ] **Step 6: Manual verification (cannot be unit-tested)**

Run `python app/run.py`, generate a model on a **rotated** AOI (a rectangle clearly not axis-aligned), open Import, upload `data/voxcity_import_test.obj`, set an anchor, and confirm: (a) the 2D footprint and the 3D gizmo mesh point the same way, and (b) after **Import**, the committed building matches the preview's orientation/position. On an axis-aligned AOI confirm no regression. Report what you observed (this is the real acceptance gate for #8).

- [ ] **Step 7: Commit**

```bash
git add app/frontend/src/three/PlacementGizmo.tsx app/frontend/src/three/PlacementGizmo.test.ts app/frontend/src/three/SceneViewer.tsx app/frontend/src/tabs/ImportTab.tsx
git commit -m "feat(app): apply domain rotation + anchorModelPoint to 3D placement gizmo"
```

---

## Task 12: Frontend — extract & test `anchorScene` vertical datum (new-code coverage)

**Files:**
- Create: `app/frontend/src/tabs/importAnchorScene.ts`
- Create: `app/frontend/src/tabs/importAnchorScene.test.ts`
- Modify: `app/frontend/src/tabs/ImportTab.tsx` (use the extracted helper)

The `anchorScene` `useMemo` in `ImportTab.tsx` (added in `cabf0ae`) computes the preview's vertical datum (`effElev - dem_min + meshsize`) and the horizontal anchor metres. It is untested React-embedded logic. Extract the pure part and test it.

- [ ] **Step 1: Write the failing test**

Create `app/frontend/src/tabs/importAnchorScene.test.ts`:

```ts
import { describe, it, expect } from 'vitest';
import { anchorSceneUp } from './importAnchorScene';

describe('anchorSceneUp', () => {
  it('returns 0 when no ground datum is available', () => {
    expect(anchorSceneUp(null, null)).toBe(0);
  });
  it('seats move_up=0 at (effElev - dem_min) + meshsize, auto elevation', () => {
    // ground.dem_elevation=12, dem_min=4, meshsize=2 -> (12-4)+2 = 10
    expect(anchorSceneUp(null, { dem_elevation: 12, dem_min: 4, meshsize_m: 2 })).toBeCloseTo(10, 9);
  });
  it('uses the manual elevation override when set', () => {
    // override=20, dem_min=4, meshsize=2 -> (20-4)+2 = 18
    expect(anchorSceneUp(20, { dem_elevation: 12, dem_min: 4, meshsize_m: 2 })).toBeCloseTo(18, 9);
  });
});
```

- [ ] **Step 2: Run to verify it fails**

Run (from `app/frontend`): `npm run test -- importAnchorScene`
Expected: FAIL — module doesn't exist.

- [ ] **Step 3: Implement the helper**

Create `app/frontend/src/tabs/importAnchorScene.ts`:

```ts
import type { AnchorGroundResult } from '../api';

/**
 * Scene-Z (up, metres) at which the imported building's model z=0 sits so that
 * move_up = 0 seats it on the ground — mirroring the commit transform's
 * `(anchor_elevation - dem_min) + meshsize`. `anchorElevation` is the user's
 * manual override (or null to auto-use the DEM sample at the anchor). Returns 0
 * until the ground datum is available.
 */
export function anchorSceneUp(
  anchorElevation: number | null,
  ground: AnchorGroundResult | null,
): number {
  if (!ground) return 0;
  const effElev = anchorElevation ?? ground.dem_elevation;
  return effElev - ground.dem_min + ground.meshsize_m;
}
```

- [ ] **Step 4: Use it in `ImportTab.tsx`**

In the `anchorScene` `useMemo`, replace the inline `up` computation with `const up = anchorSceneUp(placement.anchorElevation, anchorGround);` (import `anchorSceneUp` from `'./importAnchorScene'`). Keep the east/north computation as-is.

- [ ] **Step 5: Run tests + build**

Run (from `app/frontend`): `npm run test && npm run build`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add app/frontend/src/tabs/importAnchorScene.ts app/frontend/src/tabs/importAnchorScene.test.ts app/frontend/src/tabs/ImportTab.tsx
git commit -m "test(app): extract & cover the anchorScene vertical-datum computation"
```

---

## Final verification

- [ ] **Backend:** `& "C:\Users\kunih\miniconda3\Scripts\conda.exe" run -n voxcity python -m pytest app/backend/ tests/importer/ -v` → all pass (importer suite unchanged at 36 passed / 1 skipped; backend suite = prior + new import tests).
- [ ] **Frontend:** from `app/frontend`, `npm run build` then `npm run test` → build clean, all vitest pass.
- [ ] **Manual (#8 acceptance):** rotated-AOI preview-vs-commit alignment confirmed (Task 11 Step 6).
- [ ] **`git status`** clean except pre-existing unrelated dirty files.

---

## Self-review (author)

- **Coverage of the 8 issues:** #1 → Task 1; #2 → Task 2; #3 → Task 3; #4 → Task 5; #5 → Task 6; #6 → folded into Task 11 (gizmo `anchorModelPoint` handling); #7 → Task 8; #8 → Tasks 9–11. Recent-code tests → Tasks 4 (anchor_ground) + 12 (anchorScene). All accounted for.
- **Highest risk:** Task 11 (3D gizmo domain-rotation + drag round-trip + the user-vs-display rotation separation). The plan pins the gizmo math to `transformModelPoint` (Task 10, server-parity-tested) via `gizmoModelToScene`, and includes a mandatory manual rotated-site acceptance check, because the three.js object-transform sign/compose cannot be fully unit-tested. Implementer is explicitly told to flip the `domainRotationDeg` sign in `psi`/recovery if the parity test fails, rather than guess.
- **Type/name consistency:** `domainRotationDeg` (grid.ts export) is reused by `transformModelPoint`'s 3rd param, `ObjPlacementMap`, and the gizmo prop chain (`ImportTab` → `SceneViewer.placementPreview.domainRotationDeg` → `PlacementGizmo` prop). `gizmoModelToScene` and `anchorSceneUp` are the two new exported helpers. `anchorScene` semantics (grid-frame [east=v, north=u, up]) are consistent with `transformModelPoint`'s new grid-frame return.
- **Sequencing note:** Task 8 (spec doc) describes behavior delivered by Tasks 9–11; commit it last if executing strictly in order.
